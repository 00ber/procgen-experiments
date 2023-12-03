import random
import gym
import gym3
import numpy as np
import torch
from procgen import ProcgenEnv
from cleanrl.cleanrl.ppg_procgen import Agent

 # TRY NOT TO MODIFY: seeding
seed = 16
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# env setup
envs = ProcgenEnv(num_envs=1, env_name="starpilot", num_levels=0, start_level=0, distribution_mode="easy", render_mode="rgb_array")
envs = gym3.ViewerWrapper(envs)
envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
envs.single_action_space = envs.action_space
envs.single_observation_space = envs.observation_space["rgb"]
envs.is_vector_env = True
envs = gym.wrappers.RecordEpisodeStatistics(envs)
# envs = gym.wrappers.RecordVideo(envs, f"videos/latest-video")
envs = gym.wrappers.NormalizeReward(envs, gamma=0.99)
envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

agent = Agent(envs).to(device)
# agent.load_state_dict(torch.load("/Users/karkisushant/workspace/procgen/models/agent.pt"), strict=False)
agent.load_state_dict(torch.load("/Users/karkisushant/workspace/procgen/models/agent.pt"))

# import pickle

# open a file, where you stored the pickled data
# _model = open('cleanrl/wandb/latest-run/run-2eg5ekfk.wandb', 'rb')

# dump information to that file
# data = pickle.load(_model)
# print(data.keys())
next_obs = envs.reset()
next_obs = torch.Tensor(next_obs).to(device)
step = 0
while True:
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(next_obs)
    envs.render()
    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, info = envs.step(action.cpu().numpy())
    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
    # print(f"step {step} reward {rew} first {first}")
    if step == 50000:
        break
    step += 1
    

