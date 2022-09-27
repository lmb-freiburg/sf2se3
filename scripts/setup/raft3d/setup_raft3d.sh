venv=venv
password=blablabla
while getopts v: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
        p) password=${OPTARG};;
    esac
done

REQUIRED_PKG="python3.8-dev"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

REQUIRED_PKG="libsuitesparse-dev"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

REQUIRED_PKG="libcholmod3"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

source $venv/bin/activate

weights_dir='third_party/RAFT3D/models'
weights_files='raft3d_laplacian.pth raft3d_kitti.pth raft3d.pth'
mkdir $weights_dir

REQUIRED_PKG="curl"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

for weights_file in $weights_files; do
  weights_filepath=$weights_dir/$weights_file
  if [ -f "$weights_filepath" ]; then
      echo "$weights_filepath exists."
  else
    echo "download $weights_filepath"
    curl -u cs:${password} https://sommer-space.de/nextcloud/remote.php/dav/files/cs/neural_nets/weights/raft3d/$weights_file > $weights_filepath
  fi
done

pip3 install git+https://github.com/princeton-vl/lietorch.git
pip3 install tensorboard
pip3 install scikit-sparse
# sudo apt-get install python-scipy libsuitesparse-dev
# raft3d problems:
#     1. pad -> pad such that divisible by 8
#     2. se3 for each pixel (no masks)