
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

config={"env": "CartPole-v0",
        "log_level":"INFO",
        "framework":"tf2",
        "num_workers":2}



tune.run(PPOTrainer, config=config)  # "log_level": "INFO" for verbose,
                                                     # "framework": "tfe"/"tf2" for eager,
                                                     # "framework": "torch" for PyTorch