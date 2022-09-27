

ssh sommerl@lmblogin.informatik.uni-freiburg.de -p 2122  
ssh sommerl@10.8.167.159 -p 2122  
ssh sommerl@10.8.167.159 -p 2122 -v -X  
ssh sommerl@terence

#### cluster 1804
    config_setup=configs/setup/cluster_cs.yaml
    venvs_dir=/misc/student/sommerl/venvs/
    venv=venv_py36_1804
    venv=$venvs_dir$venv
    source $venv/bin/activate
    dataset=flyingthings3d_flownet3d
    method=rigidmask
    echo $dataset $method
    source /misc/software/cuda/add_environment_cuda10.1.243_cudnnv7.6.4.sh
    python eval.py --config-setup $config_setup \
               --config-data configs/data/$dataset.yaml --config-sflow2se3 configs/sflow2se3/$method.yaml \
               --config-sflow2se3-data-dependent configs/sflow2se3/$dataset/$method.yaml


#### cluster 2004
    config_setup=configs/setup/cluster_cs.yaml
    venvs_dir=/misc/student/sommerl/venvs/
    venv=venv_py38_2004
    venv=$venvs_dir$venv
    source $venv/bin/activate
    dataset=sintel
    method=rigidmask
    echo $dataset $method
    source /misc/software/cuda/add_environment_cuda10.1.243_cudnnv7.6.4.sh
    source /misc/student/sommerl/driver/cuda/add_env_cuda_11_2.sh
    python eval.py --config-setup $config_setup \
               --config-data configs/data/$dataset.yaml --config-sflow2se3 configs/sflow2se3/$method.yaml \
               --config-sflow2se3-data-dependent configs/sflow2se3/$dataset/$method.yaml


#### tower 
    config_setup=configs/setup/tower_2080ti_part.yaml
    venv=/media/driveD/venv_py38_2004
    source $venv/bin/activate
    dataset=bonn_rgbd
    method=classic
    echo $dataset $method
    python eval.py --config-setup $config_setup \
               --config-data configs/data/$dataset.yaml --config-sflow2se3 configs/sflow2se3/$method.yaml \
               --config-sflow2se3-data-dependent configs/sflow2se3/$dataset/$method.yaml

# cluster  
machines with gpus: dacky ducky

`qsub -l nodes=X:ppn=Y:gpus=Z:FEATURE1:FEATURE2,mem=A,walltime=T,other=R -q QUEUE SCRIPTNAME`
`qsub -l nodes=1:ppn=1 -I`

`qsub -l hostlist=^track,nodes=1:ppn=1:gpus=1:ubuntu2004:nvidiaMin11GB,mem=16gb,walltime=01:00:00 -q student sflow2rigid3d/run.sh`  


### Console:  

#### Ubuntu 18.04

`qsub -l hostlist=^quack,nodes=1:ppn=1:gpus=1:ubuntu1804:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student -I`  

#### Ubuntu 20.04

`qsub -l hostlist=^track,nodes=1:ppn=1:gpus=1:ubuntu2004:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student -I`  


### Script

#### Ubuntu 18.04 (dacky, quack, nicky, mario)

`qsub -l hostlist=^quack,nodes=1:ppn=1:gpus=1:ubuntu1804:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student sflow2rigid3d/eval.sh`  

#### Ubuntu 20.04 (ducky, track)

`qsub -l hostlist=^elmo,nodes=1:ppn=1:gpus=1:ubuntu2004:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student sflow2rigid3d/eval.sh`  


show jobs
`qstat -a | grep sommerl`

kill job
`qdel 1234567.lmbtorque`


# Remove the submodule entry from .git/config
git submodule deinit -f path/to/submodule

# Remove the submodule directory from the superproject's .git/modules directory
rm -rf .git/modules/path/to/submodule

# Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
git rm -f path/to/submodule



git submodule add git@github.com:limpbot/rigidmask.git
git remote set-url origin git@github.com:limpbot/rigidmask.git

df -h
df -i

/usr/local/cuda-X.X
# ~/.bashrc
export PATH=<CUDA>/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<CUDA>/lib64/




full:  
mkdir /media/driveD/datasets/FlyingThings3D/full  
sshfs -o allow_other,default_permissions -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/full_data /media/driveD/datasets/FlyingThings3D/full
ln -s /misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/full_data/* /misc/lmbraid21/sommerl/datasets/Brox_SceneFlow/full

full restructured:  
mkdir /media/driveD/datasets/Brox_SceneFlow/full_restructured  
sudo sshfs -o allow_other,default_permissions -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/combined_data /media/driveD/datasets/Brox_SceneFlow/full_restructured  

dispnet:
mkdir /media/driveD/datasets/FlyingThings3D/dispnet  
sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/dispnet_data /media/driveD/datasets/FlyingThings3D/dispnet  

sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/combined_data /media/driveD/datasets/FlyingThings3D/full_dispnet_comb  

results  
`sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/results /media/driveD/sflow2rigid3d/results_cs  
`  
datasets:  
`sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/datasets /media/driveD/datasets_cs  
`  

FlyingThings3D  
sshfs -o follow_symlinks,allow_other -p 2122 sommerl@lmblogin.informatik.uni-freiburg.de:/misc/lmbraid21/sommerl/datasets/FlyingThings3D /media/driveD/datasets/FlyingThings3D
  


datasets:
cd /misc/lmbraid21/schroepp/datasets/orig/FlyingThings3D/full_data
tar -xf flyingthings3d__object_index.tar.bz2 flyingthings3d__object_index.tar
tar -xf flyingthings3d__object_index.tar.bz2 
chmod o+r op
run config:

tail --line=+2 metrics.csv >> metrics.csv

# results
## 1 download:
`curl -u cs:${password} https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv > results/metrics_old.csv`  

## 2 append:
`tail --line=+2 results/metrics.csv >> results/metrics_old.csv`  
`mv results/metrics_old.csv results/metrics.csv`  

## 3 upload:
`curl -u cs:${password} -T results/metrics.csv https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv`  
 

# run
`python eval.py --configs configs/exps/home/sintel/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/sceneflow/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/kitti/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/bonn_rgbd/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/tum_rgbd_fr1/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/tum_rgbd_fr2/classic/configs.yaml`  
`python eval.py --configs configs/exps/home/tum_rgbd_fr3/classic/configs.yaml`  
