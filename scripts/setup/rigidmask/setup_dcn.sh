venv=venv
while getopts v: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
    esac
done
source $venv/bin/activate
pip3 install timm
#pip3 install torch torchvision

#pip3 install torch==1.7.1 torchvision==0.8.2

# workaround for torch==1.8
#cd third_party/rigidmask/models/networks
#rm -rf DCNv2
#git rm -r --cached DCNv2
#git submodule add git@github.com:jinfagang/DCNv2_latest.git
#git mv DCNv2_latest DCNv2
#cd ../../../..
# note: requires small adapaptions in import

cd third_party/rigidmask/models/networks/DCNv2/; python3 setup.py install; cd -