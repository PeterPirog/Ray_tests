import ray
from ray.tune import run, sample_from
from ray.tune.schedulers.pb2 import PB2
import random

# Create the PB2 scheduler.
pb2_scheduler = PB2(
        time_attr="timesteps_total",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=50000,
        quantile_fraction=0.25,  # copy bottom % with top % (weights)
        # Specifies the hyperparam search space
        hyperparam_bounds={
            "lambda": [0.9, 1.0],
            "clip_param": [0.1, 0.5],
            "lr": [1e-3, 1e-5],
            "train_batch_size": [1000, 60000]
        })

# Run PPO algorithm experiment on BipedalWalker with PB2.
analysis = run(
        "PPO",
        name="ppo_pb2_bipedal",
        scheduler=pb2_scheduler,
        verbose=1,
        num_samples=4,  # population size
        stop={"timesteps_total": 1000000},
        config={
            "env": "BipedalWalker-v2",
            "log_level": "INFO",
            "kl_coeff": 1.0,
            "num_gpus": 0,
            "horizon": 1600,
            "observation_filter": "MeanStdFilter",
            "model": {
                "fcnet_hiddens": [32,32],
                "free_log_std": True
            },
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,
            "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
            "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
            "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
            "train_batch_size": sample_from(
                lambda spec: random.randint(1000, 60000))
        })