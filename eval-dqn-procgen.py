import procgen
import time
from gym3 import ViewerWrapper
from algorithms.dqn_procgen import QNetwork as Agent
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None,
        help="the path to the model checkpoint")
    parser.add_argument("--num-timesteps", type=int, default=10000,
        help="number of timesteps to run simulation for")
    args = parser.parse_args()

    return args

def main(args):
    if not args.model_path:
        raise Exception("Must provide path to valid checkpoint")
    env = procgen.ProcgenGym3Env(num=1, env_name="starpilot", render_mode="rgb_array")
    env = ViewerWrapper(env=env, info_key="rgb")
    env.single_observation_space = env.ob_space["rgb"]
    env.single_action_space = env.ac_space
    env.single_action_space.n = 15
    agent = Agent(env).to("cpu")
    agent.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    next_obs = None
    
    
    start = time.time()
    for i in range(args.num_timesteps):
        if i == 0:
            action = np.array([0])
        else:
            q_values = agent(next_obs)
            action = np.array([torch.argmax(q_values, dim=1).item()])
        env.act(action)
        next_obs = torch.FloatTensor(env.observe()[1]["rgb"])
        print("step", i, i / (time.time() - start))


if __name__ == "__main__":
    args = parse_args()
    main(args)
