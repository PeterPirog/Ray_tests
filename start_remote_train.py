import os
#Set 1
docker_cmd='docker run -v `./ray_results/`:`/root/ray_results/` -w `pwd` --cpus 8 --gpus all -it -p 8265:8265 --rm peterpirogtf/ray_tf2'
rllib_cmd='rllib train --run=TD3 --env=BipedalWalker-v3'


#Set2
docker_cmd='sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'
rllib_cmd='python rllib_test.py'

cmd=docker_cmd + ' ' + rllib_cmd


os.system(cmd)

