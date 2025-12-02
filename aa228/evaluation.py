import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3 import PPO

# import roundabout_env  # noqa: F401
from highway_env.envs.aa228_env import AA228Env

TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("aa228-v0", render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model
    model_str = "ppo"

    # Run the trained model and record video
    if model_str == "dqn":
        model = DQN.load("roundabout_dqn/model", env=env)
    elif model_str == "ppo":
        model = PPO.load("roundabout_ppo/model", env=env)
    env = RecordVideo(
        env, video_folder="roundabout_" + model_str + "/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()