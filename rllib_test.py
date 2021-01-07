"""
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


config={"env": "CartPole-v0",
        "log_level":"INFO",
        "framework":"tf2",
        "num_workers":7,
        "num_gpus": 1,
        "num_envs_per_worker": 1
        }



tune.run(PPOTrainer, config=config)  # "log_level": "INFO" for verbose,
                                                     # "framework": "tfe"/"tf2" for eager,
                                                     # "framework": "torch" for PyTorch
                                                     #"monitor":True
"""
#https://docs.ray.io/en/master/package-ref.html
#https://docs.ray.io/en/master/tune/api_docs/execution.html
#https://docs.ray.io/en/master/rllib-training.html

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

ray.init(num_cpus=8,
         num_gpus=1,
         include_dashboard=True,
         dashboard_host='0.0.0.0')


sched = ASHAScheduler(time_attr="training_iteration", max_t=100, grace_period=10)


tune.run(
    "PPO",
    keep_checkpoints_num=3,
    checkpoint_freq=3,
    scheduler=sched,
    mode="max",
    reuse_actors=True,
    stop={"episode_reward_mean": 195},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,   # number GPU for single trainer
        "num_workers": 7,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework":"tf2",
    },
)