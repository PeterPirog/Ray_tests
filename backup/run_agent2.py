import gym
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2

import tensorflow as tf
print(f'\n Is cuda available: {tf.test.is_gpu_available()} \n')

env_name='CartPole-v0' #'BipedalWalkerHardcore-v3'

#### STEP 1 - CONFIGURE CLUSTER HEAD
# DOCS IN LINK: https://docs.ray.io/en/master/package-ref.html
ray.init(num_cpus=8, ##if you use docker use docker run --cpus 8
         num_gpus=1, #if you use docker use docker run --gpus all
         include_dashboard=True, #if you use docker use docker run -p 8265:8265 -p 6379:6379
         dashboard_host='0.0.0.0')
        # if you use docker  remember about allocate memory use docker run --shm-size=16g (it means allocate for docker 16 GB of physical memory)

#### STEP 3 - CONFIGURE TUNE SCHEDULER
# DOCS IN LINK: https://docs.ray.io/en/master/tune/api_docs/schedulers.html?highlight=ASHAScheduler#asha-tune-schedulers-ashascheduler

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',  # use number of iterations as limiter
    metric='episode_reward_mean',  # parameter to optimize
    # mode='max',                     #find maximum, do notdefine here if you define in tune.run
    max_t=500,  # maximum number of iterations -  stop trainng after max_t iterations
    grace_period=3,  # stop after grace_period iterations without result improvement
    reduction_factor=3,
    brackets=1)


#### STEP 2 - CONFIGURE TRAINER CONFIG DICTIONARY
# DOCS IN LINK: https://docs.ray.io/en/master/rllib-training.html#evaluating-trained-policies
# DOCS ABOUT SEARCHING OPTIONS HERE: https://docs.ray.io/en/master/tune/api_docs/search_space.html
config = {
    "env": env_name,  # name of environment from gym library, it can be defined by user, example:
    "num_gpus": 0, #number of GPUs for trainer, remember even with GPU trainer needs 1 CPU
    "num_workers": 1,  #  number of workers for one trainer
    "lr": tune.grid_search([0.01, 0.001, 0.0001]), # or "lr": tune.uniform(0.0001, 0.01)
    "framework": "tf2",  # configuration to use tensorflow 2 as a main framework

    # Model options for the Q network(s).
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [tune.grid_search([32,64,128,256]), tune.grid_search([32,64,128,256])], #"fcnet_hiddens": [256, 256],
    },
    # Model options for the policy function.
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [tune.grid_search([32,64,128,256]), tune.grid_search([32,64,128,256])], #"fcnet_hiddens": [256, 256]
    },
}
#### STEP 3 - CONFIGURE RAY TUNE
# DOCS IN LINK: https://docs.ray.io/en/master/tune/api_docs/execution.html#tune-run

analysis = tune.run(
    run_or_experiment="SAC",  # check if your environment is continous or discreete before choosing training algorithm https://docs.ray.io/en/master/rllib-algorithms.html#soft-actor-critic-sac
    scheduler=asha_scheduler,
    keep_checkpoints_num=3,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    stop={"episode_reward_mean": 200},  # stop training if this value is reached
    mode='max',  # find maximum vale as a target
    reuse_actors=True,
    num_samples=20,  # number of trial simulation value important if you have mane combinations of values
    # local_dir='~/ray_results',  #directory to store results,  if you use docker containers this directory must be mounted docker run -v
    config=config,
    verbose=3, #0 = silent, 1 = only status updates, 2 = status and brief trial results, 3 = status and detailed trial results. Defaults to 3
)

#### STEP 3 - FIND PATH TO CHECKPOINTS
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

print('checkpoints=', checkpoints)
checkpoint_path, reward = checkpoints[0]
print('checkpoint_path=', checkpoint_path)

config = {
    "env": env_name,
    "num_gpus": 0,
    "num_workers": 1,
    "framework": "tf2",
}


#agent = ppo.PPOTrainer(config=config, env=env_name)
agent = sac.SACTrainer(config=config, env=env_name)
agent.restore(checkpoint_path)

print('agent=', agent)

########################################
import gym

# instantiate env class
env = gym.make(env_name)

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    print('episode_reward=', episode_reward)

