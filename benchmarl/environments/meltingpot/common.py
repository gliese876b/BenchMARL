#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import copy
from typing import Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase

from torchrl.data import Composite
from torchrl.envs import (
    DoubleToFloat,
    DTypeCastTransform,
    EnvBase,
    FlattenObservation,
    Transform,
    ExcludeTransform
)

from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from torchrl.envs.utils import MarlGroupMapType

from .custom_wrapper import MeltingpotCustomEnv
from .inequity_aversion import MeltingpotIAEnv
from .social_value_orientation import MeltingpotSVOEnv

from .fair_local_inequity_aversion import MeltingpotFairLocalIAEnv
from .fair_local_social_value_orientation import MeltingpotFairLocalSVOEnv

class MeltingPotClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        from torchrl.envs.libs.meltingpot import MeltingpotEnv

        config = copy.deepcopy(self.config)

        inequity_aversion = config.pop("inequity_aversion", False)
        social_value_orientation = config.pop("social_value_orientation", False)

        fair_local_svo = config.pop("fair_local_svo", False)
        fair_local_ia = config.pop("fair_local_ia", False)

        if inequity_aversion:
            ia_alpha = config.pop("ia_alpha", 0)
            ia_beta = config.pop("ia_beta", 0)
            ia_lambda = config.pop("ia_lambda", 0)
            ia_gamma = config.pop("ia_gamma", 0)

            # builds the same substrate but wraps it with the version that applies inequity aversion
            return lambda: MeltingpotIAEnv(
                substrate=self.name.lower().replace('_ia', ''),
                categorical_actions=True,
                device=device,
                group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT,
                ia_alpha=ia_alpha,
                ia_beta=ia_beta,
                ia_lambda=ia_lambda,
                ia_gamma=ia_gamma,
                **config,
            )
        elif social_value_orientation:
            svo_weight = config.pop("svo_weight", 0)
            target_svos = config.pop("target_svos", [0])
            svo_lambda = config.pop("svo_lambda", 0)
            svo_gamma = config.pop("svo_gamma", 0)

            # builds the same substrate but wraps it with the version that applies social value orientation
            return lambda: MeltingpotSVOEnv(
                substrate=self.name.lower().replace('_svo', ''),
                categorical_actions=True,
                device=device,
                group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT,
                svo_weight=svo_weight,
                target_svos=target_svos,
                svo_lambda=svo_lambda,
                svo_gamma=svo_gamma,
                **config,
            )
        elif fair_local_svo:
            svo_weight = config.pop("svo_weight", 0)
            target_svos = config.pop("target_svos", [0])
            svo_lambda = config.pop("svo_lambda", 0)
            svo_gamma = config.pop("svo_gamma", 0)
            social_drive_modifiers = config.pop("social_drive_modifiers", 0)

            return lambda: MeltingpotFairLocalSVOEnv(
                substrate=self.name.lower().replace('_flsvo', ''),
                categorical_actions=True,
                device=device,
                group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT,
                svo_weight=svo_weight,
                target_svos=target_svos,
                svo_lambda=svo_lambda,
                svo_gamma=svo_gamma,
                social_drive_modifiers=social_drive_modifiers,
                **config,
            )
        elif fair_local_ia:
            ia_alpha = config.pop("ia_alpha", 0)
            ia_beta = config.pop("ia_beta", 0)
            ia_lambda = config.pop("ia_lambda", 0)
            ia_gamma = config.pop("ia_gamma", 0)
            social_drive_modifiers = config.pop("social_drive_modifiers", 0)

            return lambda: MeltingpotFairLocalIAEnv(
                substrate=self.name.lower().replace('_flia', ''),
                categorical_actions=True,
                device=device,
                group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT,
                ia_alpha=ia_alpha,
                ia_beta=ia_beta,
                ia_lambda=ia_lambda,
                ia_gamma=ia_gamma,
                social_drive_modifiers=social_drive_modifiers,
                **config,
            )
        return lambda: MeltingpotCustomEnv(
            substrate=self.name.lower(),
            categorical_actions=True,
            device=device,
            group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 100)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        interaction_inventories_keys = [
            (group, "observation", "INTERACTION_INVENTORIES")
            for group in self.group_map(env).keys()
            if (group, "observation", "INTERACTION_INVENTORIES")
            in env.observation_spec.keys(True, True)
        ]
        return [DoubleToFloat()] + (
            [
                FlattenObservation(
                    in_keys=interaction_inventories_keys,
                    first_dim=-2,
                    last_dim=-1,
                )
            ]
            if len(interaction_inventories_keys)
            else []
        )

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        keys_to_exclude = []
        for key in env.observation_spec.keys(True, True):
            if "observation" in key and "RGB" not in key:
                keys_to_exclude.append(key)
                keys_to_exclude.append(("next", ) + key)

        keys_to_exclude.append(('RGB'))
        keys_to_exclude.append(('next', 'RGB'))

        return [
            ExcludeTransform(*keys_to_exclude, inverse=True),
            DTypeCastTransform(
                dtype_in=torch.uint8,
                dtype_out=torch.float,
                in_keys=[
                    (group, "observation", "RGB"),
                    ("next", group, "observation", "RGB"),
                ],
                in_keys_inv=[],
            )
        ]

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()

        keys_to_exclude = [key for key in observation_spec.keys(True, True) if "observation" in key and "RGB" not in key]
        for key in keys_to_exclude:
            del observation_spec[key]

        for group in self.group_map(env):
            del observation_spec[group]
        if list(observation_spec.keys()) != ["RGB"]:
            raise ValueError(
                f"More than one global state key found in observation spec {observation_spec}."
            )
        return observation_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()

        # keep only RGB as the observation key
        keys_to_exclude = [key for key in observation_spec.keys(True, True) if "observation" in key and "RGB" not in key]
        for key in keys_to_exclude:
            del observation_spec[key]

        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
            else:
                group_obs_spec = observation_spec[group_key]["observation"]
                del group_obs_spec["RGB"]
        return observation_spec

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec

    @staticmethod
    def env_name() -> str:
        return "meltingpot"

    @staticmethod
    def render_callback(experiment, env: EnvBase, data: TensorDictBase):
        return data.get("RGB")


class MeltingPotTask(Task):
    """Enum for meltingpot tasks."""

    PREDATOR_PREY__ALLEY_HUNT = None
    CLEAN_UP = None
    COLLABORATIVE_COOKING__CIRCUIT = None
    FRUIT_MARKET__CONCENTRIC_RIVERS = None
    COLLABORATIVE_COOKING__FIGURE_EIGHT = None
    PAINTBALL__KING_OF_THE_HILL = None
    FACTORY_COMMONS__EITHER_OR = None
    PURE_COORDINATION_IN_THE_MATRIX__ARENA = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__CRAMPED = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ARENA = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__REPEATED = None
    TERRITORY__OPEN = None
    STAG_HUNT_IN_THE_MATRIX__REPEATED = None
    CHICKEN_IN_THE_MATRIX__REPEATED = None
    GIFT_REFINEMENTS = None
    PURE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    COLLABORATIVE_COOKING__FORCED = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__ARENA = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__ARENA = None
    CHEMISTRY__TWO_METABOLIC_CYCLES_WITH_DISTRACTORS = None
    COMMONS_HARVEST__PARTNERSHIP = None
    PREDATOR_PREY__OPEN = None
    TERRITORY__ROOMS = None
    HIDDEN_AGENDA = None
    COOP_MINING = None
    DAYCARE = None
    PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA = None
    TERRITORY__INSIDE_OUT = None
    BACH_OR_STRAVINSKY_IN_THE_MATRIX__REPEATED = None
    COMMONS_HARVEST__CLOSED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS = None
    STAG_HUNT_IN_THE_MATRIX__ARENA = None
    PAINTBALL__CAPTURE_THE_FLAG = None
    COLLABORATIVE_COOKING__CROWDED = None
    ALLELOPATHIC_HARVEST__OPEN = None
    COLLABORATIVE_COOKING__RING = None
    COMMONS_HARVEST__OPEN = None
    COINS = None
    PREDATOR_PREY__ORCHARD = None
    PREDATOR_PREY__RANDOM_FOREST = None
    COLLABORATIVE_COOKING__ASYMMETRIC = None
    RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__REPEATED = None
    CHEMISTRY__THREE_METABOLIC_CYCLES = None
    RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ONE_SHOT = None
    CHEMISTRY__TWO_METABOLIC_CYCLES = None
    CHICKEN_IN_THE_MATRIX__ARENA = None
    BOAT_RACE__EIGHT_RACES = None
    EXTERNALITY_MUSHROOMS__DENSE = None

    # Symmetric Harvest Env
    ASYMMETRIC_COMMONS_HARVEST__DEFAULT = None

    ASYMMETRIC_COMMONS_HARVEST__DEFAULT_IA = None
    ASYMMETRIC_COMMONS_HARVEST__DEFAULT_SVO = None
    ASYMMETRIC_COMMONS_HARVEST__DEFAULT_FLSVO = None
    ASYMMETRIC_COMMONS_HARVEST__DEFAULT_FLIA = None

    # Harvest with asymmetry in apple rewards
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_2COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_4COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_6COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_8COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_10COOP = None

    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_IA = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_SVO = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_FLSVO = None
    ASYMMETRIC_COMMONS_HARVEST__5HIGH_5LOW_REWARD_FLIA = None

    # Harvest with asymmetry in zap radius
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_2COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_4COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_6COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_8COOP = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_10COOP = None

    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_IA = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_SVO = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_FLIA = None
    ASYMMETRIC_COMMONS_HARVEST__5STANDARD_5WIDE_ZAPPER_FLSVO = None

    # Symmetric Coins Env
    ASYMMETRIC_COINS__DEFAULT = None

    ASYMMETRIC_COINS__DEFAULT_IA = None
    ASYMMETRIC_COINS__DEFAULT_SVO = None

    # Coins with asymmetry in coin rewards
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_HIGH_COOP = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_LOW_COOP = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_BOTH_COOP = None

    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_IA = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_SVO = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_FLIA = None
    ASYMMETRIC_COINS__1HIGH_1LOW_REWARD_FLSVO = None

    # Coins with asymmetry in coin spawn
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_STANDARD_COOP = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_SPAWN_BIASED_COOP = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_BOTH_COOP = None

    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_IA = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_SVO = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_FLIA = None
    ASYMMETRIC_COINS__1STANDARD_1SPAWN_BIASED_FLSVO = None


    @staticmethod
    def associated_class():
        return MeltingPotClass
