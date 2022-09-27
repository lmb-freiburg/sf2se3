
from options.parser import get_args
import datasets.seq_dataset
import datasets.sintel
import sys

def test_seqdataset():
    #--configs
    #configs / configs.yaml
    #'--configs', 'configs/configs.yaml'

    sys.argv = sys.argv[1:]
    #sys.argv = sys.argv[:1]
    sys.argv.append('--configs')
    sys.argv.append('configs/configs.yaml')

    args = get_args()

    # args.setup_dataset_dir = os.path.join(
    #    args.setup_datasets_dir, args.setup_dataset_subdir
    # )

    dataloader = datasets.sintel.dataloader_from_args(args)

    for (id, data) in enumerate(dataloader):
        print(id, data.keys())
