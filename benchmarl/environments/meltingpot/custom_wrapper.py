from __future__ import annotations

import importlib

from typing import Dict, List, Mapping, Sequence

import torch
import copy

from tensordict import TensorDict, TensorDictBase

from torchrl.envs.libs.meltingpot import MeltingpotWrapper, _remove_world_prefix, _filter_global_state_from_dict
from torchrl.envs.utils import _classproperty, check_marl_grouping, MarlGroupMapType

_has_meltingpot = importlib.util.find_spec("meltingpot") is not None

class MeltingpotCustomEnv(MeltingpotWrapper):
    def __init__(
        self,
        substrate: str | "ml_collections.config_dict.ConfigDict",  # noqa
        *,
        max_steps: int | None = None,
        categorical_actions: bool = True,
        group_map: MarlGroupMapType | Dict[str, List[str]] = MarlGroupMapType.ONE_GROUP_PER_AGENT,
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

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        return super()._reset(tensordict=tensordict, **kwargs)

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

        N = len(rewards)
        ext_rewards = copy.deepcopy(rewards)
        for i in range(N):
            obs[i]['EXT_REWARD'] = ext_rewards[i]

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
