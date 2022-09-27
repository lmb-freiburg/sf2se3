
# add key
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

# add repository
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# install libraries
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils

# install for extensions
#sudo apt-get install librealsense2-dev
#sudo apt-get install librealsense2-dbg

pip install pip --upgrade
pip install pyrealsense2
pip install numpy
pip install opencv-python
pip install pyzmq

# install camera drive in realsense-viewer
