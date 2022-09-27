venv=venv
opencv_version=3.4.2 # 3.4.2  4.5.0
while getopts v:o: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
        o) opencv_version=${OPTARG};;
    esac
done
source $venv/bin/activate

REQUIRED_PKG="cmake"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG
fi

pip3 uninstall -y opencv-python
pip3 install numpy

mkdir third_party
cd third_party

# git submodule add --depth 1 git@github.com:opencv/opencv_contrib.git
# git submodule add --depth 1 git@github.com:opencv/opencv.git
cd opencv_contrib
opencv_contrib_dir=`pwd`
git checkout .
git checkout $opencv_version
cd ..

cd opencv
opencv_dir=`pwd`
git checkout .
git checkout $opencv_version

cd $venv

mkdir tmp
cd tmp

mkdir build
cd build

rm -rf *

# sudo apt-get install libjpeg-dev

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D OPENCV_EXTRA_MODULES_PATH=${opencv_contrib_dir}/modules \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D OPENCV_ENABLE_NONFREE=ON \
-D ENABLE_CXX11=ON \
-D WITH_GSTREAMER=OFF \
-D WITH_FFMPEG=ON \
-D BUILD_opencv_ml=OFF \
-D BUILD_EXAMPLES=OFF ${opencv_dir}

#-D PYTHON_EXECUTABLE=$(which python3) \
#-D PYTHON3_EXECUTABLE=$(which python3) \
#-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \

#-D BUILD_opencv_objdetect=OFF \
#-D BUILD_opencv_stitching=ON \
#-D BUILD_opencv_calib3d=ON \
#-D BUILD_opencv_dnn=OFF \
#-D BUILD_opencv_flann=ON \
#-D BUILD_opencv_ts=OFF \
#-D BUILD_opencv_video=OFF \
#-D BUILD_opencv_gapi=OFF \
#-D BUILD_opencv_photo=ON \
#-D BUILD_opencv_features2d=ON \
#-D BUILD_opencv_xfeatures2d=ON \

make -j$(nproc)
make install
ldconfig

cd ${opencv_dir}
cd ../..