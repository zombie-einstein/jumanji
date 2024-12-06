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

import abc
from typing import Tuple

import chex
import jax.numpy as jnp
from esquilax.transforms import spatial

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.common.updates import angular_width, view, view_reduction
from jumanji.environments.swarms.search_and_rescue.types import State, TargetState


class ObservationFn(abc.ABC):
    def __init__(
        self,
        view_shape: Tuple[int, ...],
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        self.view_shape = view_shape
        self.num_vision = num_vision
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size

    @abc.abstractmethod
    def __call__(self, state: State) -> chex.Array:
        """
        Generate agent view/observation from state

        Args:
            state: Current simulation state

        Returns:
            Array of individual agent views
        """


class AgentObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        super().__init__(
            (1, num_vision),
            num_vision,
            vision_range,
            view_angle,
            agent_radius,
            env_size,
        )

    def __call__(self, state: State) -> chex.Array:
        searcher_views = spatial(
            view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return searcher_views[:, jnp.newaxis]


def target_view(
    _key: chex.PRNGKey,
    params: Tuple[float, float],
    searcher: AgentState,
    target: TargetState,
    *,
    n_view: int,
    i_range: float,
    env_size: float,
) -> chex.Array:
    view_angle, agent_radius = params
    rays = jnp.linspace(
        -view_angle * jnp.pi,
        view_angle * jnp.pi,
        n_view,
        endpoint=True,
    )
    d, left, right = angular_width(
        searcher.pos,
        target.pos,
        searcher.heading,
        i_range,
        agent_radius,
        env_size,
    )
    checks = jnp.logical_and(target.found, jnp.logical_and(left < rays, rays < right))
    obs = jnp.where(checks, d, -1.0)
    return obs


class AgentAndTargetObservationFn(ObservationFn):
    def __init__(
        self,
        num_vision: int,
        vision_range: float,
        view_angle: float,
        agent_radius: float,
        env_size: float,
    ) -> None:
        self.vision_range = vision_range
        self.view_angle = view_angle
        self.agent_radius = agent_radius
        self.env_size = env_size
        super().__init__(
            (2, num_vision),
            num_vision,
            vision_range,
            view_angle,
            agent_radius,
            env_size,
        )

    def __call__(self, state: State) -> chex.Array:
        searcher_views = spatial(
            view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.searchers,
            pos=state.searchers.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        target_views = spatial(
            target_view,
            reduction=view_reduction,
            default=-jnp.ones((self.num_vision,)),
            include_self=False,
            i_range=self.vision_range,
            dims=self.env_size,
        )(
            state.key,
            (self.view_angle, self.agent_radius),
            state.searchers,
            state.targets,
            pos=state.searchers.pos,
            pos_b=state.targets.pos,
            n_view=self.num_vision,
            i_range=self.vision_range,
            env_size=self.env_size,
        )
        return jnp.hstack([searcher_views[:, jnp.newaxis], target_views[:, jnp.newaxis]])
