import gymnasium as gym
from gymnasium import spaces
import numpy as np

class QuadraticEnv(gym.Env):
    """
    Gymnasium environment for solving a quadratic equation by predicting its maximum root.

    Observation: coefficients [a, b, c]
    Action: predicted maximum root [x] as a vector of length 1
    Reward: negative squared error between predicted and true maximum root.
    Episode ends after one step.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        # Observation: coefficients a, b, c
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
        )
        # Action: predict maximum root as a 1-d vector
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        self._rng = None
        self.seed()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.last_pred = 0.0
        self.max_steps = 3
        if seed is not None:
            self.seed(seed)
        # Sample two real roots uniformly in [-1, 1]
        root = self._rng.uniform(0, 1.0)
        # Determine max root
        self.true_root = float(np.max(root))
        self.initial_error = abs(self.true_root - self.last_pred)
        r1, r2 = 0,root
        # Sample coefficient a in [-1,1], avoid zero
        a = 1.0
        self.a = float(a)
        # Compute b and c from original roots
        self.b = -self.a * (r1 + r2)
        self.c = self.a * r1 * r2
        obs = np.array([self.a, self.b, self.c,self.last_pred,0.0], dtype=np.float64)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        # Flatten action vector and extract scalar
        pred = float(np.asarray(action, dtype=np.float64).flatten()[0])
        # Compute squared error against true max root
        curr_err = abs(pred - self.true_root)
        #sq_err = (pred - self.true_root) ** 2
        self.last_error = abs(self.true_root - self.last_pred)
        reward = self.last_error - curr_err
        obs = np.array([self.a, self.b, self.c,pred,self.step_count/self.max_steps], dtype=np.float64)
        self.last_pred = pred
        self.last_error = curr_err
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = {"true_root": self.true_root,"raw_reward": reward}
        if self.step_count == self.max_steps:
            reward -= self.initial_error
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(
            f"Quadratic equation: {self.a} x^2 + {self.b} x + {self.c} = 0; "
            f"true max root: {self.true_root}"
        )

# Register the environment
from gymnasium.envs.registration import register
register(
    id="Quadratic-v2",
    entry_point="my_envs.quadratic_env:QuadraticEnv",
)
