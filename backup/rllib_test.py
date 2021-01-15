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


#sched = ASHAScheduler(time_attr="training_iteration",max_t=5,grace_period=3)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='episode_reward_mean',
    mode='max',
    max_t=5,
    grace_period=2,
    reduction_factor=3,
    brackets=1)

tune.run(
    "PPO",
    keep_checkpoints_num=3,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    scheduler=asha_scheduler,
   #mode="max",
    reuse_actors=True,
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 1,   # number GPU for single trainer
        "num_workers": 7,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "framework":"tf2",
    },
)



#rllib train --run=PG --env=CartPole-v0 --restore=$HOME/ray_results/default/PG_CartPole-v0_0_2019-04-05_16-43-02s_gcpmkl/checkpoint_9/checkpoint-9