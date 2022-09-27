umount /media/driveD/sflow2rigid3d/results_cs
umount /media/driveD/datasets/FlyingThings3D
umount /media/driveD/datasets/datasets_cs
#umount /media/driveD/object_poses
#umount /media/driveD/test

# results_cs
sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/results /media/driveD/sflow2rigid3d/results_cs

# FlyingThings3D
sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/datasets/FlyingThings3D /media/driveD/datasets/FlyingThings3D

#datasets_cs
sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/datasets /media/driveD/datasets_cs
#sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid19/datasets/FlyingThings3D-ObjectPoses /media/driveD/object_poses


sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/full_data/object_index /media/driveD/test
