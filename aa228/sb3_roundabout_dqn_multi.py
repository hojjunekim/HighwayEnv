import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3 import PPO

# import roundabout_env  # noqa: F401
from highway_env.envs.roundabout_env import RoundaboutEnv


TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("roundabout-v0", render_mode="rgb_array",
                   config={
                       "controlled_vehicles": 2,
                       "vehicles_count" : 1,
                    }
    )
    obs, info = env.reset()

    # Create the model
    model_str = "ppo"

    if model_str == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="roundabout_dqn/",
        )
    elif model_str == "ppo":
        n_cpu = 6
        batch_size = 32
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.9,
            verbose=2,
            tensorboard_log="roundabout_" + model_str + "/",
        )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e2))
        # model.learn(total_timesteps=int(2e4))
        model.save("roundabout_" + model_str + "/model")
        del model

    # Run the trained model and record video
    if model_str == "dqn":
        model = DQN.load("roundabout_" + model_str + "/model", env=env)
    elif model_str == "ppo":
        model = PPO.load("roundabout_" + model_str + "/model", env=env)
    env = RecordVideo(
        env, video_folder="roundabout_" + model_str + "/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.config.update({
        "action" : {
            "type": "MultiAgentAction",
            "action_config" : {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 8, 16]
            }
        }
    })
    env.reset()

    env.unwrapped.set_record_video_wrapper(env)

    for videos in range(3):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step((action, action))
            # Render
            env.render()
    env.close()
