#Use existing docker image as a base
#FROM tensorflow/tensorflow:2.3.1-gpu
FROM  tensorflow/tensorflow:latest-gpu-py3

#Download and install dependencies
RUN export python=python3
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
#RUN pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install scikit-learn


RUN pip install modin
RUN pip install ray[all]
RUN ray install-nightly

RUN pip install -U ray
RUN pip install argparse
#libraries for gym

RUN pip install gym[atari]
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