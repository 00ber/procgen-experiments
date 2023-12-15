# Procgen Experiments
Exploration of different RL algorithms within the procgen environments.


## Environment
The results are based on training for 5M steps in the Starpilot environment in the following system:
- Macbook Pro (Sonoma 14.0)
- Python 3.9 with Miniconda
- CPU

## Setup
```
pip install -r requirements-base.txt
pip install -r requirements-procgen.txt
```

## Training

- DQN
```
python algorithms/dqn_procgen.py --env-id starpilot --num-envs 1 --track --capture-video --total-timesteps 5000000
``````
- PPO
```
python algorithms/ppo_procgen.py --env-id starpilot --track --capture-video --total-timesteps 5000000
``````
- PPG
```
python algorithms/ppg_procgen.py --env-id starpilot --num-envs 64 --track --total-timesteps 5000000
```
