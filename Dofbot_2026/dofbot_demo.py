"""关节与 TCP 滑块交互演示。"""

import time

import pybullet as p

from dofbot import DofbotEnv


if __name__ == "__main__":
    env = DofbotEnv()
    env.reset()
    try:
        while p.isConnected(env.physics_client_id):
            env.step_with_sliders()
            time.sleep(0.01)
    finally:
        env.close()
