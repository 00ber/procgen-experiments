import procgen
import time
from gym3 import ViewerWrapper
from algorithms.dqn_procgen import QNetwork as Agent
import torch
import numpy as np

def main():
    env = procgen.ProcgenGym3Env(num=1, env_name="starpilot", render_mode="rgb_array")
    env = ViewerWrapper(env=env, info_key="rgb")
    env.single_observation_space = env.ob_space["rgb"]
    env.single_action_space = env.ac_space
    env.single_action_space.n = 15
    agent = Agent(env).to("cpu")
    agent.load_state_dict(torch.load("./models/dqn-agent.pt", map_location=torch.device('cpu')))

    # next_obs = torch.Tensor(env.reset())
    # next_obs = env.callmethod("get_state")
    # next_obs = torch.Tensor(next_obs)
    next_obs = None
    
    
    start = time.time()
    for i in range(10000):
        # print(env.single_action_space.sample())
        if i == 0:
            action = np.array([0])
        else:
            q_values = agent(next_obs)
            action = np.array([torch.argmax(q_values, dim=1).item()])
            print(action)
            # action, _, _, _ = agent.get_action_and_value(next_obs)
        env.act(action)
        next_obs = torch.FloatTensor(env.observe()[1]["rgb"])
        print("step", i, i / (time.time() - start))


if __name__ == "__main__":
    main()
