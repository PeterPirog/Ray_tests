#https://docs.ray.io/en/master/configure.html#ray-ports

"""
# To start a head node.
$ ray start --head --num-cpus=<NUM_CPUS> --num-gpus=<NUM_GPUS>

# To start a non-head node.
$ ray start --address=<address> --num-cpus=<NUM_CPUS> --num-gpus=<NUM_GPUS>

# Specifying custom resources
ray start [--head] --num-cpus=<NUM_CPUS> --resources='{"Resource1": 4, "Resource2": 16}'

"""
import os
#ray start --address='192.168.1.16:6379' --redis-password='5241590000000000'

cmd='ray start --head --num-cpus=1 --num-gpus=8'
os.system(cmd)