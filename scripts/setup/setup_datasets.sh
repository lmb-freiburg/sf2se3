
#### Sintel ####
# download form http://sintel.is.tue.mpg.de/downloads

# oflow
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
unzip MPI-Sintel-complete.zip -d MPI-Sintel-oflow
rm -f MPI-Sintel-complete.zip

# depth + camera extrinsics
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip
unzip MPI-Sintel-depth-training-20150305.zip -d MPI-Sintel-depth
rm -f MPI-Sintel-depth-training-20150305.zip

# disprity
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip
unzip MPI-Sintel-stereo-training-20150305.zip -d MPI-Sintel-disp
rm -f MPI-Sintel-stereo-training-20150305.zip

# segmentation
wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-segmentation-training-20150219.zip
unzip MPI-Sintel-segmentation-training-20150219.zip -d MPI-Sintel-segmentation
rm -f MPI-Sintel-segmentation-training-20150219.zip


# ft3d : https://academictorrents.com/userdetails.php?id=9551

# run python3 flyingthings3d_dispnet_2_seq_data.py

# scene flow (for camera information) : https://academictorrents.com/userdetails.php?id=9551

# run python3 flyingthings3d_full_2_dispnet_seq_data.py