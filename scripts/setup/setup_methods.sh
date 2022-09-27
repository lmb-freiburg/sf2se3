# ./setup/setup_methods.sh -v venv_py38_2004
venv=venv
cluster=true
python=python3.8
cuda=10

while getopts v:c:p: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
        c) cluster=${OPTARG};;
        p) python=${OPTARG};;
    esac
done

echo "venv $venv"
echo "cluster $cluster"
echo "python $python"

if [ "$cluster" = "true" ]; then
    venvs_dir=/misc/student/sommerl/venvs/
    ubuntu_version=$(lsb_release -sr)
    if [ "$ubuntu_version" = "18.04" ]; then
      python=python3.6
      venv=venv_py36_1804
      venv=$venvs_dir$venv
      source /misc/software/cuda/add_environment_cuda10.1.243_cudnnv7.6.4.sh
      cuda="10"
    elif [ "$ubuntu_version" = "20.04" ]; then
      python=python3.8
      venv=venv_py38_2004
      venv=$venvs_dir$venv
      # note: cuda 11.2 is required because oldest compiler is gcc-9 g++-9
      source /misc/student/sommerl/driver/cuda/add_env_cuda_11_2.sh
      cuda="11"
    fi
else
  source setup/libs/add_env_local_cuda_11_4.sh
fi

export CUDA_HOME=$CUDA_ROOT

REQUIRED_PKG="$python"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

REQUIRED_PKG="$python-venv"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

# git clone git@github.com:psiori/zmq-ffmpeg-video-streaming.git
# git submodule  update --init --recursive --remote
# git submodule update --init
# git clone git@github.com:limpbot/sflow2rigid3d.git
git submodule update --init --recursive
# git pull --recurse-submodules

if [ -d "$venv" ]; then
    echo "venv $venv exists."
else
    echo "create venv $venv"
    $python -m venv $venv
fi

source $venv/bin/activate
pip3 install pip --upgrade
pip3 install -r setup/req_python_essential.txt
pip3 install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.0

#
if [ "$cuda" = 11 ]; then
  echo "install torch with cuda 11.1"
  pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  #pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
fi

chmod +x ./setup/raft3d/setup_raft3d.sh
./setup/raft3d/setup_raft3d.sh -v $venv

chmod +x ./setup/rigidmask/setup_rigidmask.sh
./setup/rigidmask/setup_rigidmask.sh -v $venv