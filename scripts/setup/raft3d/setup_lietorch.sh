venv=venv
while getopts v: flag
do
    case "${flag}" in
        v) venv=${OPTARG};;
    esac
done

source $venv/bin/activate

git submodule --init --recursive

cd third_party/lietorch/; python3 setup.py install; cd -