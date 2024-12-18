# 🚁 Search & Rescue

[//]: # (TODO: Add animated plot)

Multi-agent environment, modelling a group of agents searching a 2d environment
for multiple targets. Agents are individually rewarded for finding a target
that has not previously been detected.

Each agent visualises a local region around itself, represented as a simple segmented
view of locations of other agents and targets in the vicinity. The environment
is updated in the following sequence:

- The velocity of searching agents are updated, and consequently their positions.
- The positions of targets are updated.
- Targets within detection range and an agents view cone are marked as found.
- Agents are rewarded for locating previously unfound targets.
- Local views of the environment are generated for each search agent.

The agents are allotted a fixed number of steps to locate the targets. The search
space is a uniform square space, wrapped at the boundaries.

Many aspects of the environment can be customised:

- Agent observations can include targets as well as other searcher agents.
- Rewards can be shared by agents, or can be treated completely individually for individual agents.
- Target dynamics can be customised to model various search scenarios.

## Observations

- `searcher_views`: jax array (float) of shape `(num_searchers, channels, num_vision)`.
  Each agent generates an independent observation, an array of values representing the distance
  along a ray from the agent to the nearest neighbour or target, with  each cell representing a
  ray angle (with `num_vision` rays evenly distributed over the agents field of vision).
  For example if an agent sees another agent straight ahead and `num_vision = 5` then
  the observation array could be

  ```
  [-1.0, -1.0, 0.5, -1.0, -1.0]
  ```

  where `-1.0` indicates there is no agents along that ray, and `0.5` is the normalised
  distance to the other agent. Channels in the segmented view are used to differentiate
  between different agents/targets and can be customised. By default, the view has three
  channels representing other agents, found targets, and unfound targets.
- `targets_remaining`: float in the range `[0, 1]`. The normalised number of targets
  remaining to be detected (i.e. 1.0 when no targets have been found).
- `Step`: int in the range `[0, time_limit]`. The current simulation step.

## Actions

Jax array (float) of `(num_searchers, 2)` in the range `[-1, 1]`. Each entry in the
array represents an update of each agents velocity in the next step. Searching agents
update their velocity each step by rotating and accelerating/decelerating, where the
values are `[rotation, acceleration]`. Values are clipped to the range `[-1, 1]`
and then scaled by max rotation and acceleration parameters, i.e. the new values each
step are given by

```
heading = heading + max_rotation * action[0]
```

and speed

```
speed = speed + max_acceleration * action[1]
```

Once applied, agent speeds are clipped to velocities within a fixed range of speeds.

## Rewards

Jax array (float) of `(num_searchers,)`. Rewards are generated for each agent individually.
Agents are rewarded +1 for locating a target that has not already been detected. It is possible
for multiple agents to detect a target inside a step, as such rewards can either be shared
by the locating agents, or each agent can get the full reward.
