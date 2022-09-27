
#chmod +x ./scripts/setup/libs/install/cuda/cuda_11_4.sh
#./scripts/setup/libs/install/cuda/cuda_11_4.sh

source scripts/setup/libs/add_env_local_cuda_11_4.sh
export CUDA_HOME=$CUDA_ROOT

sudo apt-get update
sudo apt-get install -y software-properties-common gcc
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3-distutils python3-pip python3-apt

# for python packages from git
sudo apt-get install -y git

# remove delay of package configuration
#RUN apt-get install apt-utils

# opencv libgl import
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install -y ffmpeg libsm6 libxext6

python3.8 -m venv venv_py38_sflow2se3
source venv_py38_sflow2se3/bin/activate

python3 -m pip install -r scripts/setup/req_python_essential.txt