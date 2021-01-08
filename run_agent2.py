import gym
import numpy as np
import ray.rllib.agents.ppo as ppo


"""
checkpoint_path='/home/peterpirog/PycharmProjects/Ray_tests/ray_results/PPO/PPO_CartPole-v0_b2244_00002_2_lr=0.0001_2021-01-07_16-59-25/checkpoint_13'

config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework":"tf2",
    }
agent = ppo.PPOTrainer(config=config,env="CartPole-v0")
agent.restore(checkpoint_path)

print('agent=',agent)



########################################

"""
# instantiate env class
env = gym.make("CartPole-v0")

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    #action = agent.compute_action(obs)
    action=np.random.randint(2)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    print('episode_reward=',episode_reward)
    
