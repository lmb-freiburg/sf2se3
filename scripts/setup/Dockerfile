ARG IMAGE_NAME
FROM nvidia/cuda:11.4.2-base-ubuntu20.04
# base (CUDA), runtime (CUDA, SHARED_LIBS), devel (CUDA, SHARED_LIBS, DEV_TOOLS)

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

# for python packages from git
RUN apt-get install -y git

# remove delay of package configuration
#RUN apt-get install apt-utils

# opencv libgl import
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN mkdir /home/docker-user/
#RUN mkdir /home/docker-user/sflow2rigid3d
COPY / /home/docker-user/sflow2rigid3d
WORKDIR /home/docker-user/sflow2rigid3d

RUN python3 -m pip install -r scripts/setup/req_python_essential.txt

ENTRYPOINT ["python3.8", "eval.py"]
