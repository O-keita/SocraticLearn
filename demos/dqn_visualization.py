import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from environment.custom_env import SocraticTutorEnv
from stable_baselines3 import DQN, PPO
from environment.rendering import Renderer

import numpy as np
import imageio
import pygame
import time

# Configuration for the saved video
DURATION_SECONDS = 30            # desired video length
FPS = 5                          # frames per second (match renderer.clock.tick)
OUTPUT_PATH = "simulation_30s.mp4"

# Load the trained model
model = PPO.load("../models/ppo_engagement/final_model.zip")

# Create the environment and renderer
env = SocraticTutorEnv()
renderer = Renderer()

# Ensure pygame display is initialized (Renderer likely does this internally)
# but we guard in case Renderer doesn't create a display surface immediately.
if not pygame.get_init():
    pygame.init()

# Prepare video writer (requires ffmpeg; imageio will use system ffmpeg)
writer = imageio.get_writer(OUTPUT_PATH, fps=FPS, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])

# Metrics storage
engagement_list = []
confusion_list = []
effort_list = []
reward_list = []

obs, _ = env.reset()
done = False
step = 0
frame_count = 0
max_frames = DURATION_SECONDS * FPS

start_time = time.time()

try:
    while not done and frame_count < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = env._get_state_dict()
        engagement_list.append(state.get("engagement", 0.0))
        confusion_list.append(state.get("confusion", 0.0))
        effort_list.append(state.get("effort", 0.0))
        reward_list.append(reward)

        # Render to screen
        renderer.render(state, action, step=step, reward=reward)

        # Capture the current display surface to a numpy frame
        try:
            surface = pygame.display.get_surface()
            if surface is None:
                raise RuntimeError("No pygame display surface available to capture.")
            frame = pygame.surfarray.array3d(surface)  # shape (width, height, 3)
            frame = np.transpose(frame, (1, 0, 2))     # convert to (height, width, 3)
            # imageio expects HxWx3 uint8
            writer.append_data(frame)
            frame_count += 1
        except Exception as e:
            # If capturing fails, log and continue (video will be shorter)
            print(f"[frame capture warning] step={step} could not capture frame: {e}")

        step += 1
        # Tick the renderer clock to control framerate
        renderer.clock.tick(FPS)

    # If we exited early due to episode end, optionally pad final frame to reach desired length
    if frame_count < max_frames:
        try:
            surface = pygame.display.get_surface()
            if surface is not None:
                frame = pygame.surfarray.array3d(surface)
                frame = np.transpose(frame, (1, 0, 2))
                # Repeat last frame to reach duration
                for _ in range(max_frames - frame_count):
                    writer.append_data(frame)
                frame_count = max_frames
        except Exception:
            pass

finally:
    writer.close()
    # Quit pygame display to release resources
    try:
        pygame.display.quit()
    except Exception:
        pass

end_time = time.time()

print("\n=== Episode Summary ===")
if engagement_list:
    print(f"Average Engagement: {np.mean(engagement_list):.2f}")
    print(f"Average Confusion: {np.mean(confusion_list):.2f}")
    print(f"Average Effort: {np.mean(effort_list):.2f}")
    print(f"Total Reward: {np.sum(reward_list):.2f}")
else:
    print("No metric data was collected.")

print(f"\nSaved video to: {os.path.abspath(OUTPUT_PATH)}")
print(f"Frames written: {frame_count} (target {max_frames})")
print(f"Elapsed wall time: {end_time - start_time:.2f}s")