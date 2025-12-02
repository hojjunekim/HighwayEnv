import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import WandbCallback # <-- New Import
from wandb.integration.sb3 import WandbCallback
import wandb # <-- New Import

# import roundabout_env  # noqa: F401
from highway_env.envs.aa228_env import AA228Env

TRAIN = True

if __name__ == "__main__":
    # Create the environment
    env = gym.make("aa228-v0", render_mode="rgb_array",
                #    config={
                #        "action" : {
                #             # "type": "ContinuousAction",
                #             # "type": "DiscreteMetaAction",
                #        }
                #    }
    )
    obs, info = env.reset()

    # Create the model
    model_str = "dqn"

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
        # Initialize wandb run
        run = wandb.init(
            project="aa228-highway-env",  # Replace with your desired project name
            sync_tensorboard=True,        # Sync SB3's TensorBoard logs
            monitor_gym=True,             # Monitor gym/gymnasium environments
            save_code=True                # Save the script file
        )

        # Define the WandbCallback
        wandb_callback = WandbCallback(
            model_save_path="roundabout_" + model_str + "/wandb_models", # Folder to save models to
            verbose=2,
        )

        # Pass the callback to model.learn()
        model.learn(total_timesteps=int(e2), callback=wandb_callback)
        # model.learn(total_timesteps=int(2e4), callback=wandb_callback)
        
        # Save the final model outside of the callback
        model.save("roundabout_" + model_str + "/model")
        
        # Finish the wandb run
        run.finish()
        
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