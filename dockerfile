#Use existing docker image as a base
#FROM tensorflow/tensorflow:2.3.1-gpu
FROM  tensorflow/tensorflow:latest-gpu-py3

#Download and install dependencies
RUN export python=python3
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
#RUN apt-get install ffmpeg

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
#RUN pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install scikit-learn


RUN pip install modin
RUN pip install -U ray[all]
RUN ray install-nightly

RUN pip install argparse
#libraries for gym

RUN pip install gym[atari]
RUN pip install box2d
RUN pip install gym[box2d]

#RUN ray start --head
# Container start command
CMD ["sh"]


#command to build new image:
#sudo docker build -t peterpirogtf/ray_tf2:latest .

#How to push docker image to hub

#login by:
# docker login
# docker push peterpirogtf/ray_tf2:latest
#https://stackoverflow.com/questions/41984399/denied-requested-access-to-the-resource-is-denied-docker

#https://github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst

# sudo docker run --cpus 8 --gpus all -it -p 8265:8265 --rm peterpirogtf/ray_tf2 rllib train --run=TD3 --env=BipedalWalker-v3
# sudo docker run --cpus 8 --gpus all -it -p 8265:8265 --rm peterpirogtf/ray_tf2 rllib train --run=TD3 --env=BipedalWalkerHardcore-v3

#rllib train --env=PongDeterministic-v4 --run=A2C --config '{"num_workers": 2, "monitor": true}'
#rllib train --run DQN --env CartPole-v0 --eager --config '{"num_workers": 2, "monitor": true, "num_gpus" : 1}'
#rllib train --run=TD3 --env=BipedalWalkerHardcore-v3 --eager --ray-num-gpus 1 --checkpoint-freq 10 --export-formats model
