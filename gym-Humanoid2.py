import time
import gymnasium as gym
from stable_baselines3 import SAC
#model_path = 'models/sac/Humanoid-v4_1/Humanoid-v4.zip'
model_path = 'models/sac/Humanoid-v4_4/best_model.zip'
env = gym.make('Humanoid-v4', render_mode='human', max_episode_steps=5000)
model = SAC.load(model_path, env=env)
observation, info = env.reset(seed=42)
steps, done = 0, False
#for _ in range(3000):
while not done:
#    time.sleep(0.01)
    action, _states = model.predict(observation,deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if steps % 1000 == 0:
        print('Step:', steps, reward)
    done = terminated or truncated
    steps += 1
#   if terminated or truncated:
#        observation, info = env.reset()
env.close()
