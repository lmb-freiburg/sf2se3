venv=venv
while getopts v: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
    esac
done

REQUIRED_PKG="python3.8-dev"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

source $venv/bin/activate

# 1. install opencv ./setup/setup_opencv.sh
# 2. adapt setup.py script
#    opencv_inc_dir = './venv/include/opencv4/opencv2'
#    opencv_lib_dir = './venv/lib'
# 3. install ngransac

opencv_inc_dir="$venv/include/opencv4/opencv2"
opencv_lib_dir="$venv/lib"

# using globally installed libopencv_core / libopencv_calib3d
#opencv_inc_dir="/path/to/nowhere"
#opencv_lib_dir="/path/to/nowhere"


export opencv_inc_dir
export opencv_lib_dir

echo "opencv_inc_dir: $opencv_inc_dir"
echo "opencv_lib_dir: $opencv_lib_dir"
#pip3 install ninja
pip3 uninstall -y ngransac
rm -rf third_party/rigidmask/models/build/*
rm -rf third_party/rigidmask/models/dist/*

cd third_party/rigidmask/models/ngransac/; python3 setup.py install; cd -