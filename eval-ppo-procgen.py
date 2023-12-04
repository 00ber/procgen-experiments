import procgen
import time
from gym3 import ViewerWrapper, types_np
from algorithms.ppo_procgen import Agent
import torch

def main():
    env = procgen.ProcgenGym3Env(num=1, env_name="starpilot", render_mode="rgb_array")
    env = ViewerWrapper(env=env, info_key="rgb")
    print(env.ob_space)
    env.single_observation_space = env.ob_space["rgb"]
    env.single_action_space = env.ac_space
    print(env.single_action_space)
    print(dir(env.single_action_space))
    env.single_action_space.n = 15
    agent = Agent(env).to("cpu")
    agent.load_state_dict(torch.load("./models/ppo-agent.pt", map_location=torch.device('cpu')))

    # next_obs = torch.Tensor(env.reset())
    # next_obs = env.callmethod("get_state")
    # next_obs = torch.Tensor(next_obs)
    next_obs = None
    
    
    start = time.time()
    for i in range(10000):
        if i == 0:
            action = torch.Tensor([0])
        else:
            action, _, _, _ = agent.get_action_and_value(next_obs)
        env.act(action.cpu().numpy())
        next_obs = torch.FloatTensor(env.observe()[1]["rgb"])
        print("step", i, i / (time.time() - start))


if __name__ == "__main__":
    main()
