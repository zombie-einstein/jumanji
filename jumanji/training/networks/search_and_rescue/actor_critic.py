# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from math import prod
from typing import Sequence, Tuple, Union

import chex
import haiku as hk
import jax.numpy as jnp

from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.types import Observation
from jumanji.training.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from jumanji.training.networks.parametric_distribution import (
    ContinuousActionSpaceNormalDistribution,
)


def make_actor_critic_search_and_rescue(
    search_and_rescue: SearchAndRescue,
    layers: Sequence[int],
) -> ActorCriticNetworks:
    n_actions = prod(search_and_rescue.action_spec.shape)
    parametric_action_distribution = ContinuousActionSpaceNormalDistribution(n_actions)
    policy_network = make_network_mlp(critic=False, layers=layers, n_actions=n_actions)
    value_network = make_network_mlp(critic=True, layers=layers, n_actions=n_actions)

    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_mlp(critic: bool, layers: Sequence[int], n_actions: int) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
        views = observation.searcher_views  # (B, A, V)
        x = views.reshape(views.shape[0], -1)  # (B, N)

        if critic:
            value = hk.nets.MLP([*layers, 1])(x)  # (B, 1)
            return jnp.squeeze(value, axis=-1)

        else:
            means = hk.nets.MLP([*layers, n_actions])(x)  # (B, A)
            log_stds = hk.get_parameter(
                "log_stds", shape=means.shape[1:], init=hk.initializers.Constant(0.1)
            )  # (A,)
            log_stds = jnp.broadcast_to(log_stds, means.shape)  # (B, A)
            return means, log_stds

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
