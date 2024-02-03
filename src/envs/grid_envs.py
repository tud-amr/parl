from __future__ import annotations

from abc import ABC
from enum import IntEnum
from operator import add


from gymnasium.core import ActType, ObsType
from typing import Any
from minigrid.core.world_object import *
from minigrid.core.constants import *
from gymnasium import spaces, Env
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, WorldObj
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import random

OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 11,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "switch": 4,
}
NUM_OBJECTS = len(OBJECT_TO_IDX)

class WorldObj2Act(WorldObj, ABC):
    def __init__(self, type: str, color: str):
        super().__init__('box', color)
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v


class Switch(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains
        self.color = color
        self.state = 0
        self.type = "switch"

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color] #if self.state else (0, 0, 0)  # Use specified color if on, black if off

        # Switch body (rectangle)
        fill_coords(img, point_in_rect(0.45, 0.55, 0.3, 0.7), c)

        # Switch handle (rectangle)
        handle_color = (255, 255, 255) if not self.state else (100, 100, 100)  # White if on, gray if off
        fill_coords(img, point_in_rect(0.45, 0.55, 0.15, 0.3), handle_color)

    def toggle(self, env, pos):
        # Toggle the switch
        self.state = 1
        self.color = "green" if not self.state else "yellow"
        return True

    def encode(self) -> tuple[int, int, int]:
        """Encode the description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.state)

class LightFloor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color = None):
        if color is None:
            color = random.choice(COLOR_NAMES)
        assert color in COLOR_NAMES
        self.color = color
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def change_color(self):
        self.color = random.choice(COLOR_NAMES)

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type],0, COLOR_TO_IDX[self.color])

class DoubleActMiniGridEnv(MiniGridEnv):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # Toggle A
        toggle1 = 3
        # Toggle B
        toggle2 = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=NUM_OBJECTS,
            shape=self.observation_space.spaces["image"].shape,
            dtype="uint8",
        )
        self.actions = DoubleActMiniGridEnv.Actions
        # Action enumeration for this environment
        self.action_space = spaces.Discrete(len(self.actions))

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Toggle 1
        elif action == self.actions.toggle1:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Toggle 2
        elif action == self.actions.toggle2:
            if fwd_cell:
                # fwd_cell.toggle(self, fwd_pos)
                # Not implemented yet
                pass


        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
            "switch": "S",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        str = ""

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c is None:
                    str += "  "
                    continue

                if c.type == "door":
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue

                if c.type == "switch":
                    if c.state:
                        str += "S" + c.color[0].upper()
                    else:
                        str += "s" + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"

        return str



class DistShiftEnv2(DoubleActMiniGridEnv):

    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=2,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (width - 2, 1)
        self.strip2_row = strip2_row

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the lava rows
        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Lava())
            self.grid.set(3 + i, self.strip2_row, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class SlipperyDistShift(DistShiftEnv2):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actions = SlipperyDistShift.Actions
        # Action enumeration for this environment
        self.action_space = spaces.Discrete(len(self.actions))
        self.alpha = 0.35
        self.deterministic_path = [(1,1), (1,2), (1,3), (1,4), (1,5),
                                   (2,5),(3,5),(4,5),(5,5),(6,5),(7,5),
                                   (7,4),(7,3),(7,2),(7,1)]

        self.foggy_areas = {0.08:[(1,1), (2,1), (2,2), (2,3), (3,3),
                                   (4,3),(5,3),(6,3),(6,2),(6,1)],
                            0.04:[(1,2), (2,4), (3,4), (4,4),
                                   (5,4),(6,4),(7,2)]}

        self.noise_dict = {}
        # Iterate over rows and columns of the grid
        for i in range(self.width):
            for j in range(self.height):
                pos = (i, j)
                value = 1
                if pos in self.deterministic_path:
                    value = 0
                self.noise_dict[pos] = value

        self.observation_space = spaces.Box(
            low=0,
            high=max([self.grid.width, self.grid.height]),
            shape=(3,),
            dtype="float32",
        )


    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the lava rows
        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Lava())
            self.grid.set(3 + i, 2, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _dist_to_lava(self):
        x_dist = max(
            max(3-self.agent_pos[0],0),
            max(self.agent_pos[0]-self.width + 3 ,0),
        )
        if x_dist == 0:
            x_dist = 100
        y_dist = max(self.agent_pos[1]-2,0)
        if y_dist == 0:
            y_dist = 100
        return min(x_dist, y_dist)

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        image = np.concatenate([np.expand_dims(image[:, :, 0],-1), np.expand_dims(image[:, :, 2],-1)], axis=2)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = 5 #self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)

        else:
            raise ValueError(f"Unknown action: {action}")

        if np.random.rand() < self.alpha*self.noise_dict[self.agent_pos]:
            self.agent_dir = random.choice([0,1,2,3])

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = np.asarray((self.agent_pos[0], self.agent_pos[1], self.agent_dir), dtype="float32")

        return obs, reward, terminated, truncated, {'true_entropy': 1.3*self.noise_dict[self.agent_pos]}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = np.asarray((self.agent_pos[0], self.agent_pos[1], self.agent_dir),
                         dtype="float32")

        return obs, {'true_entropy': 1.3}

class DynamicObstaclesSwitchEnv(DoubleActMiniGridEnv):

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_obstacles=1,
        max_steps: int | None = None,
        penalty=0.1,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.penalty = penalty

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        # self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)
        self.switch_pos = None
        self.observation_space = spaces.Box(
            low=0,
            high=max([self.grid.width, self.grid.height]),
            shape=(6,),
            dtype="float32",
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place a switch in the bottom-left corner
        self.switch_pos = (1, 0)
        self.grid.set(self.switch_pos[0], self.switch_pos[1], Switch("green"))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        # switch_in_front = front_cell and front_cell.type == "switch"
        # Get switch state
        switch_state_old = self.grid.get(*self.switch_pos).state
        obs, reward, terminated, truncated, info = super().step(action)
        switch_state = self.grid.get(*self.switch_pos).state
        not_clear = front_cell and front_cell.type == "ball"

        # Update obstacle positions
        if switch_state == 0:
            info["true_entropy"] = 2.07
            for i_obst in range(len(self.obstacles)):
                old_pos = self.obstacles[i_obst].cur_pos
                top = tuple(map(add, old_pos, (-1, -1)))
                try:
                    self.place_obj(
                        self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                    )
                    self.grid.set(old_pos[0], old_pos[1], None)
                except Exception:
                    pass
        else:
            info["true_entropy"] = 0
            # # Replace obstacles with empty cells
            # for i_obst in range(len(self.obstacles)):
            #     old_pos = self.obstacles[i_obst].cur_pos
            #     self.grid.set(old_pos[0], old_pos[1], None)
        if switch_state != switch_state_old:
            for i_obst in range(len(self.obstacles)):
                old_pos = self.obstacles[i_obst].cur_pos
                # self.grid.set(old_pos[0], old_pos[1], None)
                # self.obstacles[i_obst].cur_pos = (self.height-1, 1)
                try:
                    self.place_obj(
                        self.obstacles[i_obst], top=(1,self.height-2), size=(1,1), max_tries=100
                    )
                    self.grid.set(old_pos[0], old_pos[1], None)
                except Exception:
                    pass
            obs = self.gen_obs()

        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info
        if reward > 0:
            reward = 1-switch_state_old*self.penalty

        return obs, reward*5, terminated, truncated, info

    def gen_obs(self):
        switch_state = self.grid.get(*self.switch_pos).state
        obs = np.asarray((switch_state, self.agent_pos[0], self.agent_pos[1], self.agent_dir,
                          self.obstacles[0].cur_pos[0],self.obstacles[0].cur_pos[1]), dtype="float32")
        return obs

    def _reset_switch(self):
        self.grid.set(self.switch_pos[0], self.switch_pos[1], Switch("green"))

    def _reward(self) -> float:
        return 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_switch()

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {'true_entropy': 2.07}
