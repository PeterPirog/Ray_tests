import os
####

container_id='peterpirogtf/ray_tf2:latest'
script_name='tune_keras.py'
#docker_cmd=f"sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm {container_id} python 'main_ddpg.py'"


#docker_cmd=f"sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm {container_id} python '{script_name}'"

#docker_cmd=f"s sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm peterpirogtf/ray_tf2:latest python 'tune_keras.py'"
#docker_cmd=f"sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm peterpirogtf/ray_tf2:latest python '03_tune_keras_functional_cnn.py'"
docker_cmd=f"sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm peterpirogtf/ray_tf2:latest python 'simple_mnist_cnn.py'"
os.system(docker_cmd)


#TODO:
#convert actor model to Functional or sequential form to save and load as one file
# add nosie for observations
#monit number of steps before fail
# sudo docker run -v `pwd`:`pwd` -w `pwd` --gpus all -it --rm peterpirogtf/ray_tf2:latest python 'tune_keras.py'

# sudo docker run -p 6379:6379 --gpus all -it --rm peterpirogtf/ray_tf2:latest ray start --head --num-cpus=8 --num-gpus=1