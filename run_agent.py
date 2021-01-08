
import gym
import ray.rllib.agents.ppo as ppo

#checkpoint_path='/home/peterpirog/PycharmProjects/Ray_tests/ray_results/PPO/PPO_CartPole-v0_f82cf_00002_2_lr=0.0001_2021-01-07_15-48-11/checkpoint_5'


import ray
from ray import tune

ray.init(num_cpus=8,
         num_gpus=1,
         include_dashboard=True,
         dashboard_host='0.0.0.0')


config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework":"tf2",
    }




analysis=tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    mode='max',
    checkpoint_at_end=True,
    config=config,
)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")


print('checkpoints=',checkpoints)
checkpoint_path,reward=checkpoints[0]
print('checkpoint_path=',checkpoint_path)



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
import gym

# instantiate env class
env = gym.make("CartPole-v0")

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    print('episode_reward=',episode_reward)