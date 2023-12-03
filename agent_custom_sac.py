"""
Example random agent script using the gym3 API to demonstrate that procgen works
"""
import torch
from procgen import ProcgenGym3Env
from sac import Agent
import gym
import numpy as np
from procgen import ProcgenEnv
import time


def make_env(run_name, gamma, capture_video=False):
    envs = ProcgenEnv(num_envs=1, env_name="starpilot", num_levels=0, start_level=0, distribution_mode="easy", render_mode="rgb_array")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs

run_name = f"starpilot_{int(time.time())}"
env = make_env(run_name, 0.999, False)
print(env.single_action_space.n)
print(env.single_observation_space.shape)
# env = ProcgenGym3Env(num=1, env_name="starpilot", render_mode="rgb_array")

agent = Agent(input_dims=env.single_observation_space.shape, env=env, n_actions=env.single_action_space.n)

score_history = []
load_checkpoint = False
step = 0
episodes = 250
best_score = 0

if load_checkpoint:
    agent.load_models()

for i in range(100):
    obs = env.reset()
    done = False
    obs = torch.Tensor(obs)
    score = 0
    while not done:
        action = agent.choose_action(obs)
        # print("######")
        # print(action)
        # print(action.shape)
        obs_, reward, done, info = env.step(action)
        score += reward
        agent.remember(obs, action, reward, obs_, done)
        if not load_checkpoint:
            agent.learn()
        obs = obs_
    score_history.append(score)
    score_history = score_history[-100:]
    avg_score = np.mean(score_history)

    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
    print(score, avg_score)
    print(F"Episode: {i} Score: {score} Avg Score: {avg_score:.2f}")

# while True:
#     with torch.no_grad():
#         action, _, _, _ = agent.get_action_and_value(next_obs)
#     env.render()
#     # TRY NOT TO MODIFY: execute the game and log data.
#     next_obs, reward, done, info = env.step(action.cpu().numpy())
#     next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
#     # print(f"step {step} reward {rew} first {first}")
#     if step == 50000:
#         break
#     step += 1
