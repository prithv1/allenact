import math
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, cast

import gym
import copy
import numpy as np

from core.base_abstractions.misc import RLStepResult
from core.base_abstractions.sensor import Sensor
from core.base_abstractions.task import Task
from plugins.robothor_plugin.robothor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from utils.system import get_logger

EPS = 1e-8


class PointNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.task_info[
            "taken_actions"
        ] = []  # Keep a log of actions taken by the agent per-step in an episode
        self.task_info[
            "action_success"
        ] = []  # Track if per-step actions were successful or not
        self.task_info["shaped_rew"] = []  # Track the per-step shaped rewards
        self.task_info[
            "goal_in_range"
        ] = []  # Track if the goal was in range w.r.t. an undertaken action
        self.task_info[
            "far_from_goal"
        ] = []  # Track the distance to goal w.r.t. an undertaken action
        self.task_info[
            "took_end_action"
        ]: bool = False  # Whether or not the end action was called
        self.task_info["ep_rewards"] = []  # Track reward per step

        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]
        self.task_info["taken_actions"].append(action_str)

        # Store this meta-data
        if self.env._motor_failure:
            self.task_info["motor_failure"] = True
            self.task_info["failed_action"] = self.env._failed_action

        if self.env._drift:
            self.task_info["drift_dir"] = self.env._drift_dir
            self.task_info["drift_deg"] = self.env._drift_deg

        if action_str == END:
            self._took_end_action = True
            self.task_info["took_end_action"] = self._took_end_action
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_success"].append(self.last_action_success)
        else:
            # Functionality for motor failure
            if self.env._motor_failure:
                if action_str == self.env._failed_action:
                    action_str = "Pass"

            if self.env._drift:
                if action_str == "MoveAhead":
                    curr_pos = self.env.last_event.metadata["agent"]["position"]
                    curr_rot = self.env.last_event.metadata["agent"]["rotation"]
                    drift_deg = self.env._drift_deg
                    if self.env._drift_dir == "Right":
                        drift_deg = self.env._drift_deg + curr_rot["y"]
                    else:
                        drift_deg = self.env._drift_deg - curr_rot["y"]
                    self.env.step(
                        {
                            "action": "TeleportFull",
                            "x": curr_pos["x"],
                            "y": curr_pos["y"],
                            "z": curr_pos["z"],
                            "rotation": {
                                "x": curr_rot["x"],
                                "y": drift_deg,
                                "z": curr_rot["z"],
                            },
                        }
                    )

            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            self.task_info["action_success"].append(self.last_action_success)
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1

        self.task_info["shaped_rew"].append(self.shaping())
        self.task_info["goal_in_range"].append(self._is_goal_in_range())
        self.task_info["far_from_goal"].append(self.dist_to_target())

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
            self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        self.task_info["ep_rewards"].append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env.config["gridSize"]
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def soft_progress(self):
        """
        Compute soft-progress made towards the goal
        """
        d_init = self.optimal_distance
        d_T = self.dist_to_target()
        if d_init >= 0 and d_T >= 0:
            succ_fac = max(0, 1.0 - (d_T / (d_init + 1e-8)))
        else:
            succ_fac = 0.0
        return succ_fac

    def soft_spl(self):
        """
        Compute the soft-SPL metric
        """
        succ_fac = self.soft_progress()
        li = self.optimal_distance
        pi = self.num_moves_made * self.env.config["gridSize"]
        pl_ratio = li / (max(pi, li) + 1e-8)
        res = succ_fac * pl_ratio
        return res

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()
        soft_progress = self.soft_progress()
        soft_spl = self.soft_spl()

        return {
            **super(PointNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "soft_progress": soft_progress,
            "soft_spl": soft_spl,
        }


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)
    # _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)  # Checking

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self.last_geodesic_distance = self.env.distance_to_object_type(
            self.task_info["object_type"]
        )

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.task_info[
            "taken_actions"
        ] = []  # Keep a log of actions taken by the agent per-step in an episode
        self.task_info[
            "action_success"
        ] = []  # Track if per-step actions were successful or not
        self.task_info["shaped_rew"] = []  # Track the per-step shaped rewards
        self.task_info[
            "goal_in_range"
        ] = []  # Track if the goal was in range w.r.t. an undertaken action
        self.task_info[
            "far_from_goal"
        ] = []  # Track the distance to goal w.r.t. an undertaken action
        self.task_info[
            "took_end_action"
        ]: bool = False  # Whether or not the end action was called
        self.task_info["ep_rewards"] = []  # Track reward per step

        self.num_moves_made = 0
        self.optimal_distance = self.last_geodesic_distance
        self.closest_geo_distance = self.last_geodesic_distance

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if self.mirror:
            if action_str == ROTATE_RIGHT:
                action_str = ROTATE_LEFT
            elif action_str == ROTATE_LEFT:
                action_str = ROTATE_RIGHT

        self.task_info["taken_actions"].append(action_str)

        # Store this meta-data
        if self.env._motor_failure:
            self.task_info["motor_failure"] = True
            self.task_info["failed_action"] = self.env._failed_action

        if self.env._drift:
            self.task_info["drift_dir"] = self.env._drift_dir
            self.task_info["drift_deg"] = self.env._drift_deg

        if action_str == END:
            self._took_end_action = True
            self.task_info["took_end_action"] = self._took_end_action
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_success"].append(self.last_action_success)
        else:
            # Functionality for motor failure
            if self.env._motor_failure:
                if action_str == self.env._failed_action:
                    action_str = "Pass"

            if self.env._drift:
                if action_str == "MOVE_AHEAD":
                    curr_pos = self.env.last_event.metadata["agent"]["position"]
                    curr_rot = self.env.last_event.metadata["agent"]["rotation"]
                    drift_deg = self.env._drift_deg
                    if self.env._drift_dir == "Right":
                        drift_deg = self.env._drift_deg + curr_rot["y"]
                    else:
                        drift_deg = self.env._drift_deg - curr_rot["y"]
                    self.env.step(
                        action="TeleportFull",
                        x=curr_pos["x"],
                        y=curr_pos["y"],
                        z=curr_pos["z"],
                        rotation={
                            "x": curr_rot["x"],
                            "y": drift_deg,
                            "z": curr_rot["z"],
                        },
                    )

            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            self.task_info["action_success"].append(self.last_action_success)
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1

        self.task_info["shaped_rew"].append(self.shaping())
        self.task_info["goal_in_range"].append(self._is_goal_in_range())
        self.task_info["far_from_goal"].append(
            self.env.distance_to_object_type(self.task_info["object_type"])
        )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = frame[:, ::-1, :].copy()  # horizontal flip
            # print("mirrored render")
        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.env.distance_to_object_type(
            self.task_info["object_type"]
        )

        # Ensuring the reward magnitude is not greater than the total distance moved
        max_reward_mag = 0.0
        if len(self.path) >= 2:
            p0, p1 = self.path[-2:]
            max_reward_mag = math.sqrt(
                (p0["x"] - p1["x"]) ** 2 + (p0["z"] - p1["z"]) ** 2
            )

        if self.reward_configs.get("positive_only_reward", False):
            if geodesic_distance > 0.5:
                rew = max(self.closest_geo_distance - geodesic_distance, 0)
        else:
            if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
            ):  # (robothor limits)
                rew += self.last_geodesic_distance - geodesic_distance

        self.last_geodesic_distance = geodesic_distance
        self.closest_geo_distance = min(self.closest_geo_distance, geodesic_distance)

        return (
            max(min(rew, max_reward_mag), -max_reward_mag,)
            * self.reward_configs["shaping_weight"]
        )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_configs["goal_success_reward"]
            else:
                reward += self.reward_configs["failed_stop_reward"]

        self._rewards.append(float(reward))
        self.task_info["ep_rewards"].append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env.config["gridSize"]
        res = li / (max(pi, li) + 1e-8)
        return res

    def soft_progress(self):
        """
        Compute soft-progress made towards the goal
        """
        d_init = self.optimal_distance
        d_T = self.env.distance_to_object_type(self.task_info["object_type"])
        if d_init >= 0 and d_T >= 0:
            succ_fac = max(0, 1.0 - (d_T / (d_init + 1e-8)))
        else:
            succ_fac = 0.0
        return succ_fac

    def soft_spl(self):
        """
        Compute the soft-SPL metric
        """
        succ_fac = self.soft_progress()
        li = self.optimal_distance
        pi = self.num_moves_made * self.env.config["gridSize"]
        pl_ratio = li / (max(pi, li) + 1e-8)
        res = succ_fac * pl_ratio
        return res

    def get_observations(self, **kwargs) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    if (
                        len(obs[o].shape) == 3
                    ):  # heuristic to determine this is a visual sensor
                        obs[o] = obs[o][:, ::-1, :].copy()  # horizontal flip
                    elif len(obs[o].shape) == 2:  # perhaps only two axes for depth?
                        obs[o] = obs[o][:, ::-1].copy()  # horizontal flip
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        dist2tget = self.env.distance_to_object_type(self.task_info["object_type"])

        # if self._success:
        #     li = self.optimal_distance
        #     pi = self.num_moves_made * self.env.config["gridSize"]
        #     if min(pi, li) < 0:
        #         spl = -1.0
        #     elif li == pi:
        #         spl = 1.0
        #     else:
        #         spl = li / (max(pi, li))
        # else:
        #     spl = 0.0

        spl = self.spl()
        soft_progress = self.soft_progress()
        soft_spl = self.soft_spl()

        metrics = {
            **super(ObjectNavTask, self).metrics(),
            "success": self._success,
            "total_reward": np.sum(self._rewards),
            "dist_to_target": dist2tget,
            "spl": spl,
            "soft_spl": soft_spl,
            "soft_progress": soft_progress,
        }
        # if spl >= 0:
        #     metrics["spl"] = spl
        return metrics

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        if end_action_only:
            return 0, False
        else:
            self.env.step(
                {
                    "action": "ObjectNavExpertAction",
                    "objectType": self.task_info["object_type"],
                }
            )
            if self.env.last_action_success:
                expert_action: Optional[str] = self.env.last_event.metadata[
                    "actionReturn"
                ]
                if isinstance(expert_action, str):
                    if self.mirror:
                        if expert_action == "RotateLeft":
                            expert_action = "RotateRight"
                        elif expert_action == "RotateRight":
                            expert_action = "RotateLeft"

                    return self.action_names().index(expert_action), True
                else:
                    # This should have been caught by self._is_goal_in_range()...
                    return 0, False
            else:
                return 0, False
