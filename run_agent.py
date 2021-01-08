import gym
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.schedulers import ASHAScheduler

# checkpoint_path='/home/peterpirog/PycharmProjects/Ray_tests/ray_results/PPO/PPO_CartPole-v0_f82cf_00002_2_lr=0.0001_2021-01-07_15-48-11/checkpoint_5'


#### STEP 1 - CONFIGURE CLUSTER HEAD
# DOCS IN LINK: https://docs.ray.io/en/master/package-ref.html
ray.init(num_cpus=8,
         num_gpus=1,
         include_dashboard=True,
         dashboard_host='0.0.0.0')

#### STEP 3 - CONFIGURE TUNE SCHEDULER
# DOCS IN LINK: https://docs.ray.io/en/master/tune/api_docs/schedulers.html?highlight=ASHAScheduler#asha-tune-schedulers-ashascheduler

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',  # use number of iterations as limiter
    metric='episode_reward_mean',  # parameter to optimize
    # mode='max',                     #find maximum, do notdefine here if you define in tune.run
    max_t=50,  # maximum number of iterations -  stop trainng after max_t iterations
    grace_period=3,  # stop after grace_period iterations without result improvement
    reduction_factor=3,
    brackets=1)
"""

#Available agents (training algorithms): https://docs.ray.io/en/master/rllib-algorithms.html
config_agent={
        "env": "CartPole-v0",   #name of environment from gym library, it can be defined by user, example:
        "num_gpus": 0,
        "num_workers": 2, #every trainer has theese number of workers
        #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework":"tf2",  #configuration to use tensorflow 2 as a main framework
    }

agent = ppo.PPOTrainer(env="CartPole-v0",config=config_agent)#

"""
#### STEP 2 - CONFIGURE TRAINER CONFIG DICTIONARY
# DOCS IN LINK: https://docs.ray.io/en/master/rllib-training.html#evaluating-trained-policies
# DOCS ABOUT SEARCHING OPTIONS HERE: https://docs.ray.io/en/master/tune/api_docs/search_space.html
config = {
    "env": "CartPole-v0",  # name of environment from gym library, it can be defined by user, example:
    "num_gpus": 0,
    "num_workers": 2,  # every trainer has theese number of workers
    # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "lr": tune.uniform(0.0001, 0.01),
    "framework": "tf2",  # configuration to use tensorflow 2 as a main framework
}
#### STEP 3 - CONFIGURE RAY TUNE
# DOCS IN LINK: https://docs.ray.io/en/master/tune/api_docs/execution.html#tune-run

analysis = tune.run(
    run_or_experiment="PPO",  # check if your environment is continous or discreete before choosing training algorithm
    scheduler=asha_scheduler,
    keep_checkpoints_num=3,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    stop={"episode_reward_mean": 100},  # stop training if this value is reached
    mode='max',  # find maximum vale as a target
    reuse_actors=True,
    num_samples=10,  # number of trial simulation value important if you have mane combinations of values
    # local_dir='./ray_results',  #directory to store results, be careful if you use docker containers
    config=config,
)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

print('checkpoints=', checkpoints)
checkpoint_path, reward = checkpoints[0]
print('checkpoint_path=', checkpoint_path)

config = {
    "env": "CartPole-v0",
    "num_gpus": 0,
    "num_workers": 1,
    "framework": "tf2",
}
agent = ppo.PPOTrainer(config=config, env="CartPole-v0")
agent.restore(checkpoint_path)

print('agent=', agent)

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
    print('episode_reward=', episode_reward)
