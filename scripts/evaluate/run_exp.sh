datasets='flyingthings3d_dispnet' # bonn_rgbd flyingthings3d_dispnet
methods='classic'
#datasets='bonn_rgbd tum_rgbd_fr2 tum_rgbd_fr3 sintel flyingthings3d_dispnet'
#methods='classic'

#cd ../..
#exp_dir='configs/exps/downscale_factor/'
#exp_dir='configs/exps/rigid_dist_dev_max/'
#exp_dir='configs/exps/sigma_min/'
#exp_dir='configs/exps/sigma_max/'
#exp_dir='configs/exps/sigma_factor_pt_pair_invalid/'
#exp_dir='configs/exps/sigma_geo_2d/'
#exp_dir='configs/exps/sigma_geo_depth_rel/'
#exp_dir='configs/exps/delta_geo_global_ratio/'
#exp_dir='configs/exps/prob_contribute_min/'
#exp_dir='configs/exps/prob_se3_overlap_max/'
#exp_dir='configs/exps/prob_se3geo_overlap_max/'
#exp_dir='configs/exps/delta_overlap_max/'
#exp_dir='configs/exps/extraction_cycles/'
#exp_dir='configs/exps/refinement_cycles/'
#exp_dir='configs/exps/extraction_refinement_cycles/'

exp_dir='configs/exps/ft3d/'

venv=/media/driveD/venv_py38_2004
cluster=true
online=true
password=blablabla

while getopts v:c:o:p flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
        c) cluster=${OPTARG};;
        o) online=${OPTARG};;
        p) password=${OPTARG};;
    esac
done


if [ "$cluster" = "true" ]; then
  echo "changing directory ~/sflow2rigid3d"
  cd ~/sflow2rigid3d
  if [ "$online" = "true" ]; then
    config_setup="configs/setup/cluster_cs.yaml"
  else
    config_setup="configs/setup/cluster_cs_no_wandb.yaml"
  fi
  results_dir="/misc/lmbraid21/sommerl/results"
  venvs_dir="/misc/student/sommerl/venvs/"

  ubuntu_version=$(lsb_release -sr)
  if [ "$ubuntu_version" = "18.04" ]; then
    venv=venv_py36_1804
    venv=$venvs_dir$venv
    source /misc/software/cuda/add_environment_cuda10.1.243_cudnnv7.6.4.sh
  elif [ "$ubuntu_version" = "20.04" ]; then
    venv=venv_py38_2004
    venv=$venvs_dir$venv
    # bug: for DCNv2 we still need CUDA10.1
    source /misc/software/cuda/add_environment_cuda10.1.243_cudnnv7.6.4.sh
    source /misc/student/sommerl/driver/cuda/add_env_cuda_11_3.sh
  fi
else
  echo "changing directory `dirname $0`/../.."
  cd "`dirname $0`/../.."

  if [ "$online" = "true" ]; then
    config_setup="configs/setup/tower_2080ti.yaml"
  else
    config_setup="configs/setup/tower_2080ti_no_wandb.yaml"
  fi

  results_dir="/media/driveD/sflow2rigid3d/results"
  source setup/libs/add_env_local_cuda_11_4.sh
fi
export CUDA_HOME=$CUDA_ROOT

# add path for opencv libs installed locally at venv
export LD_LIBRARY_PATH="${venv}/lib:${LD_LIBRARY_PATH}"

echo "activate venv $venv"
source $venv/bin/activate
#config_ablation="configs/setup/config_ablation.yaml"
#dataset=kitti
#method=rigidmask

ablations=(`ls $exp_dir`)
ablations=${ablations[*]}
echo $ablations
for ablation in $ablations; do
  echo "$exp_dir$ablation"
done
#ablations='ablation_07.yaml ablation_08.yaml ablation_09.yaml ablation_10.yaml'
#ablations='ablation_11.yaml ablation_12.yaml ablation_13.yaml ablation_14.yaml ablation_15.yaml'
#ablations='configs/ablation.yaml'

for dataset in $datasets; do
  for method in $methods; do
    for ablation in $ablations; do
      #rm -f ${results_dir}/metrics.csv
      echo $dataset $method

      echo "eval.py --config-setup $config_setup \
             --config-data configs/data/$dataset.yaml --config-sflow2se3 configs/sflow2se3/$method.yaml \
             --config-sflow2se3-data-dependent configs/sflow2se3/$dataset/$method.yaml --config-ablation=$exp_dir$ablation"
      python eval.py --config-setup $config_setup \
             --config-data configs/data/$dataset.yaml --config-sflow2se3 configs/sflow2se3/$method.yaml \
             --config-sflow2se3-data-dependent configs/sflow2se3/$dataset/$method.yaml --config-ablation=$exp_dir$ablation

      if [ "$online" = "true" ]; then
        if curl -o /dev/null --head --silent --fail -u cs:${password} https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv; then
                echo 'download metrics from nextcloud...'
                ## 1 download
                curl -u cs:${password} https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv > ${results_dir}/metrics_sync.csv
                ## 2 append:
                echo 'append metrics...'
                tail --line=+2 ${results_dir}/metrics_not_sync.csv >> ${results_dir}/metrics_sync.csv
                rm ${results_dir}/metrics_not_sync.csv
                ## 3 upload:
                echo 'upload metrics to nextcloud...'
                curl -u cs:${password} -T ${results_dir}/metrics_sync.csv https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv
        else
                echo "info: could not find metrics online."
                ## 3 upload:
                echo 'upload metrics to nextcloud...'
                mv ${results_dir}/metrics_not_sync.csv ${results_dir}/metrics_sync.csv
                curl -u cs:${password} -T ${results_dir}/metrics_sync.csv https://sommer-space.de/nextcloud/remote.php/dav/files/cs/results/metrics.csv
        fi
      fi
    done
  done
done

