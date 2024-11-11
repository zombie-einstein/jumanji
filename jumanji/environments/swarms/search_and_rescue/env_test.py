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
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.environments.swarms.common.types import AgentState
from jumanji.environments.swarms.search_and_rescue import SearchAndRescue
from jumanji.environments.swarms.search_and_rescue.types import (
    Observation,
    State,
    TargetState,
)
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
)
from jumanji.types import StepType, TimeStep

SEARCHER_VISION_RANGE = 0.2
TARGET_CONTACT_RANGE = 0.05
AGENT_RADIUS = 0.05


@pytest.fixture
def env() -> SearchAndRescue:
    return SearchAndRescue(
        num_searchers=10,
        num_targets=20,
        searcher_vision_range=SEARCHER_VISION_RANGE,
        target_contact_range=TARGET_CONTACT_RANGE,
        num_vision=11,
        agent_radius=AGENT_RADIUS,
        searcher_max_rotate=0.2,
        searcher_max_accelerate=0.01,
        searcher_min_speed=0.01,
        searcher_max_speed=0.05,
        searcher_view_angle=0.5,
        max_steps=25,
    )


def test_env_init(env: SearchAndRescue) -> None:
    """
    Check newly initialised state has expected array shapes
    and initial timestep.
    """
    k = jax.random.PRNGKey(101)
    state, timestep = env.reset(k)
    assert isinstance(state, State)

    assert isinstance(state.searchers, AgentState)
    assert state.searchers.pos.shape == (env.num_searchers, 2)
    assert state.searchers.speed.shape == (env.num_searchers,)
    assert state.searchers.speed.shape == (env.num_searchers,)

    assert isinstance(state.targets, TargetState)
    assert state.targets.pos.shape == (env.num_targets, 2)
    assert state.targets.found.shape == (env.num_targets,)
    assert jnp.array_equal(state.targets.found, jnp.full((env.num_targets,), False, dtype=bool))
    assert state.step == 0

    assert isinstance(timestep.observation, Observation)
    assert timestep.observation.searcher_views.shape == (
        env.num_searchers,
        env.num_vision,
    )
    assert timestep.step_type == StepType.FIRST


def test_env_step(env: SearchAndRescue) -> None:
    """
    Run several steps of the environment with random actions and
    check states (i.e. positions, heading, speeds) all fall
    inside expected ranges.
    """
    key = jax.random.PRNGKey(101)
    n_steps = 22

    def step(
        carry: Tuple[chex.PRNGKey, State], _: None
    ) -> Tuple[Tuple[chex.PRNGKey, State], Tuple[State, TimeStep[Observation]]]:
        k, state = carry
        k, k_search = jax.random.split(k)
        actions = jax.random.uniform(k_search, (env.num_searchers, 2), minval=-1.0, maxval=1.0)
        new_state, timestep = env.step(state, actions)
        return (k, new_state), (state, timestep)

    init_state, _ = env.reset(key)
    (_, final_state), (state_history, timesteps) = jax.lax.scan(
        step, (key, init_state), length=n_steps
    )

    assert isinstance(state_history, State)

    assert state_history.searchers.pos.shape == (n_steps, env.num_searchers, 2)
    assert jnp.all((0.0 <= state_history.searchers.pos) & (state_history.searchers.pos <= 1.0))
    assert state_history.searchers.speed.shape == (n_steps, env.num_searchers)
    assert jnp.all(
        (env.searcher_params.min_speed <= state_history.searchers.speed)
        & (state_history.searchers.speed <= env.searcher_params.max_speed)
    )
    assert state_history.searchers.speed.shape == (n_steps, env.num_searchers)
    assert jnp.all(
        (0.0 <= state_history.searchers.heading) & (state_history.searchers.heading <= 2.0 * jnp.pi)
    )

    assert state_history.targets.pos.shape == (n_steps, env.num_targets, 2)
    assert jnp.all((0.0 <= state_history.targets.pos) & (state_history.targets.pos <= 1.0))


def test_env_does_not_smoke(env: SearchAndRescue) -> None:
    """Test that we can run an episode without any errors."""
    env.max_steps = 10

    def select_action(action_key: chex.PRNGKey, _state: Observation) -> chex.Array:
        return jax.random.uniform(action_key, (env.num_searchers, 2), minval=-1.0, maxval=1.0)

    check_env_does_not_smoke(env, select_action=select_action)


# def test_env_specs_do_not_smoke(env: SearchAndRescue) -> None:
#     """Test that we can access specs without any errors."""
#     check_env_specs_does_not_smoke(env)
#
#
# @pytest.mark.parametrize(
#     "predator_pos, predator_heading, predator_view, prey_pos, prey_heading, prey_view",
#     [
#         # Both out of view range
#         ([[0.8, 0.5]], [jnp.pi], [(0, 0, 1.0)], [[0.2, 0.5]], [0.0], [(0, 0, 1.0)]),
#         # In predator range but not prey
#         ([[0.35, 0.5]], [jnp.pi], [(0, 5, 0.75)], [[0.2, 0.5]], [0.0], [(0, 0, 1.0)]),
#         # Both view each other
#         ([[0.25, 0.5]], [jnp.pi], [(0, 5, 0.25)], [[0.2, 0.5]], [0.0], [(0, 5, 0.5)]),
#         # Prey facing wrong direction
#         (
#             [[0.25, 0.5]],
#             [jnp.pi],
#             [(0, 5, 0.25)],
#             [[0.2, 0.5]],
#             [jnp.pi],
#             [(0, 0, 1.0)],
#         ),
#         # Prey sees closest predator
#         (
#             [[0.35, 0.5], [0.25, 0.5]],
#             [jnp.pi, jnp.pi],
#             [(0, 5, 0.75), (0, 16, 0.5), (1, 5, 0.25)],
#             [[0.2, 0.5]],
#             [0.0],
#             [(0, 5, 0.5)],
#         ),
#         # Observed around wrapped edge
#         (
#             [[0.025, 0.5]],
#             [jnp.pi],
#             [(0, 5, 0.25)],
#             [[0.975, 0.5]],
#             [0.0],
#             [(0, 5, 0.5)],
#         ),
#     ],
# )
# def test_view_observations(
#     env: PredatorPrey,
#     predator_pos: List[List[float]],
#     predator_heading: List[float],
#     predator_view: List[Tuple[int, int, float]],
#     prey_pos: List[List[float]],
#     prey_heading: List[float],
#     prey_view: List[Tuple[int, int, float]],
# ) -> None:
#     """
#     Test view model generates expected array with different
#     configurations of agents.
#     """
#
#     predator_pos = jnp.array(predator_pos)
#     predator_heading = jnp.array(predator_heading)
#     predator_speed = jnp.zeros(predator_heading.shape)
#
#     prey_pos = jnp.array(prey_pos)
#     prey_heading = jnp.array(prey_heading)
#     prey_speed = jnp.zeros(prey_heading.shape)
#
#     state = State(
#         predators=AgentState(pos=predator_pos, heading=predator_heading, speed=predator_speed),
#         prey=AgentState(pos=prey_pos, heading=prey_heading, speed=prey_speed),
#         key=jax.random.PRNGKey(101),
#     )
#
#     obs = env._state_to_observation(state)
#
#     assert isinstance(obs, Observation)
#
#     predator_expected = jnp.ones(
#         (
#             predator_heading.shape[0],
#             2 * env.num_vision,
#         )
#     )
#     for i, idx, val in predator_view:
#         predator_expected = predator_expected.at[i, idx].set(val)
#
#     assert jnp.all(jnp.isclose(obs.predators, predator_expected))
#
#     prey_expected = jnp.ones(
#         (
#             prey_heading.shape[0],
#             2 * env.num_vision,
#         )
#     )
#     for i, idx, val in prey_view:
#         prey_expected = prey_expected.at[i, idx].set(val)
#
#     assert jnp.all(jnp.isclose(obs.prey[0], prey_expected))


def test_search_and_rescue_render(monkeypatch: pytest.MonkeyPatch, env: SearchAndRescue) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    step_fn = jax.jit(env.step)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    action = env.action_spec.generate_value()
    state, timestep = step_fn(state, action)
    env.render(state)
    env.close()


def test_search_and_rescue__animation(env: SearchAndRescue, tmpdir: py.path.local) -> None:
    """Check that the animation method creates the animation correctly and can save to a gif."""
    step_fn = jax.jit(env.step)
    state, _ = env.reset(jax.random.PRNGKey(0))
    states = [state]
    action = env.action_spec.generate_value()
    state, _ = step_fn(state, action)
    states.append(state)
    animation = env.animate(states, interval=200, save_path=None)
    assert isinstance(animation, matplotlib.animation.Animation)

    path = str(tmpdir.join("/anim.gif"))
    animation.save(path, writer=matplotlib.animation.PillowWriter(fps=10), dpi=60)
