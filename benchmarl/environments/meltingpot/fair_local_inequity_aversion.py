from __future__ import annotations

import importlib

from typing import Dict, List, Mapping, Sequence

import torch
import copy

from tensordict import TensorDict, TensorDictBase

from torchrl.envs.libs.meltingpot import MeltingpotWrapper, _remove_world_prefix, _filter_global_state_from_dict
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType
import numpy as np
from scipy.spatial.distance import cosine

_has_meltingpot = importlib.util.find_spec("meltingpot") is not None

class MeltingpotFairLocalIAEnv(MeltingpotWrapper):
    """
        MeltingPot wrapper with Fair & Local Inequity Aversion.
        It is very similar to MeltingpotEnv by torchrl with only difference
        on _reset and _step methods.
        This version has three modifications to IA;
        - Normalized temporally smoothed rewards
        - Agent-based weighting
        - Localization of temporall smoothed rewards
    """

    def __init__(
        self,
        substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa
        *,
        max_steps: int | None = None,
        categorical_actions: bool = True,
        group_map: MarlGroupMapType | Dict[str, List[str]] = MarlGroupMapType.ONE_GROUP_PER_AGENT,
        ia_alpha=0,
        ia_beta=0,
        ia_lambda=0,
        ia_gamma=0,
        social_drive_modifiers=[],
        **kwargs,
    ):
        if not _has_meltingpot:
            raise ImportError(
                f"meltingpot python package was not found. Please install this dependency. "
                f"More info: {self.git_url}."
            )

        super().__init__(
            substrate=substrate,
            max_steps=max_steps,
            categorical_actions=categorical_actions,
            group_map=group_map,
            **kwargs
        )

        self.substrate = substrate
        self.ia_alpha = ia_alpha
        self.ia_beta = ia_beta
        self.ia_gamma = ia_gamma
        self.ia_lambda = ia_lambda
        self.social_drive_modifiers = social_drive_modifiers if len(social_drive_modifiers) > 0 else [0] * len(self.agent_names)

        self.temporal_smoothed_reward_bounds = {}

    def _check_kwargs(self, kwargs: Dict):
        if "substrate" not in kwargs:
            raise TypeError("Could not find environment key 'substrate' in kwargs.")

    def _build_env(
        self,
        substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa
    ) -> "meltingpot.utils.substrates.substrate.Substrate":  # noqa
        from meltingpot import substrate as mp_substrate

        if isinstance(substrate, str):
            substrate_config = mp_substrate.get_config(substrate)
        else:
            substrate_config = substrate

        return super()._build_env(
            env=mp_substrate.build_from_config(
                substrate_config, roles=substrate_config.default_player_roles
            )
        )

    def _is_in_view(self, p1, p2, p1_zapped=False, p2_zapped=False):
        if p1_zapped or p2_zapped:
            return False
        return (abs(p1[0] - p2[0]) <= 5) and (abs(p1[1] - p2[1]) <= 5)

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        tensordict_out = super()._reset(tensordict=tensordict, **kwargs)

        self.temporal_smoothed_rewards = {}
        N = len(self.agent_names)
        for i in range(N):
            self.temporal_smoothed_rewards[i] = 0

        self.normalized_temporal_smoothed_rewards_pov = {}
        self.normalized_temporal_smoothed_rewards_pov_age = {}
        for i in range(N):
            self.normalized_temporal_smoothed_rewards_pov[i] = {j: 0 for j in range(N)}
            self.normalized_temporal_smoothed_rewards_pov_age[i] = {j: 0 for j in range(N)}

        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action_dict = {}
        for group, agents in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_np = self.full_action_spec[group, "action"].to_numpy(
                group_action
            )
            for index, agent in enumerate(agents):
                action_dict[agent] = group_action_np[index]

        actions = [action_dict[agent] for agent in self.agent_names]
        timestep = self._env.step(actions)
        self.num_cycles += 1

        rewards = timestep.reward
        done = timestep.last() or (
            (self.num_cycles >= self.max_steps) if self.max_steps is not None else False
        )
        obs = timestep.observation

        td = TensorDict(
            {
                "done": self.full_done_spec["done"].encode(done),
                "terminated": self.full_done_spec["terminated"].encode(done),
            },
            batch_size=self.batch_size,
        )
        # Global state
        td.update(
            _remove_world_prefix(_filter_global_state_from_dict(obs[0], world=True))
        )

        # update temporal smoothed rewards
        N = len(rewards)
        ext_rewards = copy.deepcopy(rewards)
        for i in range(N):
            obs[i]['EXT_REWARD'] = ext_rewards[i]

        for i in range(N):
            self.temporal_smoothed_rewards[i] = self.ia_gamma * self.ia_lambda * self.temporal_smoothed_rewards[i] + ext_rewards[i]
            if i not in self.temporal_smoothed_reward_bounds.keys():
                self.temporal_smoothed_reward_bounds[i] = {}
                self.temporal_smoothed_reward_bounds[i]['max'] = self.temporal_smoothed_rewards[i]
                self.temporal_smoothed_reward_bounds[i]['min'] = self.temporal_smoothed_rewards[i]

            self.temporal_smoothed_reward_bounds[i]['max'] = max(self.temporal_smoothed_reward_bounds[i]['max'], self.temporal_smoothed_rewards[i])
            self.temporal_smoothed_reward_bounds[i]['min'] = min(self.temporal_smoothed_reward_bounds[i]['min'], self.temporal_smoothed_rewards[i])

        # apply normalization based on each agent's potential
        normalized_tsr = {}
        for i, tsr in self.temporal_smoothed_rewards.items():
            tsr_min = self.temporal_smoothed_reward_bounds[i]['min']
            tsr_max = self.temporal_smoothed_reward_bounds[i]['max']
            normalized_tsr[i] = (( (tsr - tsr_min) / (tsr_max - tsr_min) ) if (tsr_max - tsr_min) != 0 else 1) + 1e-8 # Small amount to avoid nonzero division

        next_normalized_pov = {i: dict(self.normalized_temporal_smoothed_rewards_pov[i]) for i in range(N)}
        next_pov_age = {i: dict(self.normalized_temporal_smoothed_rewards_pov_age[i]) for i in range(N)}

        # gossip about others by using older values
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                p_i = obs[i]['POSITION']
                p_j = obs[j]['POSITION']
                p_i_zapped = int(obs[i]['PLAYER_IS_ZAPPED']) == 1 if 'PLAYER_IS_ZAPPED' in obs[i].keys() else False
                p_j_zapped = int(obs[j]['PLAYER_IS_ZAPPED']) == 1 if 'PLAYER_IS_ZAPPED' in obs[j].keys() else False

                if self._is_in_view(p_i, p_j, p_i_zapped, p_j_zapped):
                    for k in range(N):
                        if k == i or k == j:
                            continue

                        # gossip about what j knew about k in previous step
                        gossip_value = self.normalized_temporal_smoothed_rewards_pov[j][k]
                        gossip_age = self.normalized_temporal_smoothed_rewards_pov_age[j][k]
                        current_age = next_pov_age[i].get(k, float('inf'))

                        if gossip_age < current_age:
                            next_normalized_pov[i][k] = gossip_value
                            next_pov_age[i][k] = gossip_age

        # update self
        for i in range(N):
            next_normalized_pov[i][i] = normalized_tsr[i]
            next_pov_age[i][i] = 0

        # exchange up-to-date values with agents within view
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                p_i = obs[i]['POSITION']
                p_j = obs[j]['POSITION']
                p_i_zapped = int(obs[i]['PLAYER_IS_ZAPPED']) == 1 if 'PLAYER_IS_ZAPPED' in obs[i].keys() else False
                p_j_zapped = int(obs[j]['PLAYER_IS_ZAPPED']) == 1 if 'PLAYER_IS_ZAPPED' in obs[j].keys() else False

                if self._is_in_view(p_i, p_j, p_i_zapped, p_j_zapped):
                    next_normalized_pov[i][j] = normalized_tsr[j]
                    next_pov_age[i][j] = 0
                else:
                    next_pov_age[i][j] += 1


        # Commit the buffered updates after all are computed
        self.normalized_temporal_smoothed_rewards_pov = next_normalized_pov
        self.normalized_temporal_smoothed_rewards_pov_age = next_pov_age

        # apply fair&local inequity aversion
        for i in range(N):
            total_age = sum(self.normalized_temporal_smoothed_rewards_pov_age[i].values())
            for j in range(N):
                if i != j:
                    rewards[i] -= (self.ia_alpha * self.social_drive_modifiers[i] / (N - 1)) * max(self.normalized_temporal_smoothed_rewards_pov[i][j] - self.normalized_temporal_smoothed_rewards_pov[i][i], 0)
                    rewards[i] -= (self.ia_beta * self.social_drive_modifiers[i] / (N - 1)) * max(self.normalized_temporal_smoothed_rewards_pov[i][i] - self.normalized_temporal_smoothed_rewards_pov[i][j], 0)

            if 'TSR_RANGE' in obs[i].keys():
                obs[i]['TSR_RANGE'] = self.temporal_smoothed_reward_bounds[i]['max'] - self.temporal_smoothed_reward_bounds[i]['min']

            if 'TOTAL_AGE' in obs[i].keys():
                obs[i]['TOTAL_AGE'] = total_age

        for group, agent_names in self.group_map.items():
            agent_tds = []
            for index_in_group, agent_name in enumerate(agent_names):
                global_index = self.agent_names_to_indices_map[agent_name]
                agent_obs = self.observation_spec[group, "observation"][
                    index_in_group
                ].encode(_filter_global_state_from_dict(obs[global_index], world=False))

                agent_reward = self.full_reward_spec[group, "reward"][
                    index_in_group
                ].encode(rewards[global_index])
                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                        "reward": agent_reward,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                agent_tds.append(agent_td)
            agent_tds = torch.stack(agent_tds, dim=0)
            td.set(group, agent_tds)
        return td
