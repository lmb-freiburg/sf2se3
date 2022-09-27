venv=venv
password=blablabla
while getopts v:p: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
        p) password=${OPTARG};;
    esac
done
source $venv/bin/activate

weights_dir='third_party/rigidmask/weights'
weights_files='sf.pth kitti.pth'
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
    curl -u cs:${password} https://sommer-space.de/nextcloud/remote.php/dav/files/cs/neural_nets/weights/rigidmask/$weights_file > $weights_filepath
  fi
done

chmod +x ./setup/rigidmask/setup_opencv.sh
./setup/rigidmask/setup_opencv.sh -v $venv
pip3 install opencv-python

chmod +x ./setup/rigidmask/setup_dcn.sh
./setup/rigidmask/setup_dcn.sh -v $venv

chmod +x ./setup/rigidmask/setup_ngransac.sh
./setup/rigidmask/setup_ngransac.sh -v $venv

# rigidmask problems:
#         memory problem