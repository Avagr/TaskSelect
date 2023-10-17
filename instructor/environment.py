import random
from typing import Callable

import gym
import numpy as np
from gym import spaces

from datasets.shapes import Shape


class FindAllShapesEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    action_width = 6
    task_width = 6

    REJECT = 0
    CHECK_COLOR = 1
    CHECK_SHAPE = 2
    ACCEPT = 3
    STOP = 4

    action_dict = {0: "REJECT", 1: "CHECK_COLOR", 2: "CHECK_SHAPE", 3: "ACCEPT", 4: "STOP"}

    possible_colors = ["red", "green", "blue"]
    possible_shapes = ["circle", "triangle", "square"]

    def __init__(self, shapes_provider: Callable[..., list[Shape]], memory_size=2):
        super().__init__()
        self.shapes_provider = shapes_provider
        self.memory_size = memory_size

        self.init_state()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.state.shape, dtype=np.float32)

    def init_state(self):
        self.extracted_shapes = set()
        self.action_sequence = []
        self.questioned_current = True
        self.current = None
        self.state = np.zeros(self.task_width + self.action_width * self.memory_size, dtype=np.float32)

        self.spent_on_current = 0

        self.shapes = self.shapes_provider()
        self.req_color, self.req_shape = random.choice(self.possible_colors), random.choice(self.possible_shapes)

        self.correct_shapes = set((s for s in self.shapes if s.name == self.req_shape and s.color == self.req_color))
        match self.req_shape:
            case "circle":
                self.state[0] = 1
            case "triangle":
                self.state[1] = 1
            case "square":
                self.state[2] = 1
            case _:
                raise ValueError("Invalid shape in task")
        match self.req_color:
            case "red":
                self.state[3] = 1
            case "green":
                self.state[4] = 1
            case "blue":
                self.state[5] = 1
            case _:
                raise ValueError("Invalid color in task")

    def handle_action(self, action: int) -> np.ndarray:
        res = np.zeros(self.action_width)

        if 0 <= action <= 3:
            res[action] = 1

        if (self.current is None or self.current >= len(self.shapes)) and 1 <= action <= 3:
            return res

        match action:
            case self.REJECT:
                if self.current is None:
                    self.current = -1
                self.current += 1

                if self.current < len(self.shapes):
                    res[4] = 1
                else:
                    res[5] = 1

            case self.CHECK_COLOR:
                if self.shapes[self.current].color == self.req_color:
                    res[4] = 1
                else:
                    res[5] = 1

            case self.CHECK_SHAPE:
                if self.shapes[self.current].name == self.req_shape:
                    res[4] = 1
                else:
                    res[5] = 1
            # test if moving next after extract improves performance
            case self.ACCEPT:
                self.extracted_shapes.add(self.shapes[self.current])
                self.current += 1
                if self.current < len(self.shapes):
                    res[4] = 1
                else:
                    res[5] = 1

        return res

    def step(self, action: int):

        if 1 <= action <= 2:
            self.spent_on_current += 1
        else:
            self.spent_on_current = 0

        reward = max(-0.1 * 1.2 ** max(0, self.spent_on_current - 2), -2)  # Existence is suffering

        if self.current is not None and self.current >= len(self.shapes) and 0 <= action <= 3:
            reward -= 0.5 * (self.current - len(self.shapes) + 1)

        if self.current is None and 1 <= action <= 3:
            reward -= 0.5

        if len(self.action_sequence) > 0 and self.action_sequence[-1] == action:
            reward -= 0.5
        match action:
            case self.STOP:
                if len(self.extracted_shapes) == 0 and len(self.correct_shapes) == 0:
                    return self.state, 5, True, {}
                iou = len(self.extracted_shapes.intersection(self.correct_shapes)) / len(
                    self.extracted_shapes.union(self.correct_shapes))
                return self.state, iou, True, {'iou': iou}

            case self.REJECT | self.ACCEPT:
                if not self.questioned_current:
                    reward -= 5
                self.questioned_current = False

            case self.ACCEPT if self.current is not None and self.current < len(self.shapes):
                current_shape = self.shapes[self.current]
                if current_shape.name == self.req_shape and current_shape.color == self.req_color:
                    reward += 8
                else:
                    reward -= 1

            case self.CHECK_COLOR | self.CHECK_SHAPE:
                self.questioned_current = True

        # Shift the memory cells
        self.state[self.task_width:-self.action_width] = self.state[self.task_width + self.action_width:]
        self.state[-self.action_width:] = self.handle_action(action)

        self.action_sequence.append(action)
        self.action_sequence = self.action_sequence[-50:]
        return self.state, reward, False, {}

    def reset(self):
        self.init_state()
        return self.state

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        print(f"\nCurrent task: Find all {self.req_color} {self.req_shape}s")
        if self.current is not None and self.current < len(self.shapes):
            print(f"Current shape {self.shapes[self.current]} [{(self.current + 1)}/{len(self.shapes)}]")
        else:
            print("No current shape")
        print(f'Extracted: {", ".join(map(str, self.extracted_shapes))}\n')
