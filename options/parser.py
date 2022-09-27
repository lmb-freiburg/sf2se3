import sys
import configargparse
import options.config
import options.setup
import options.data
import options.sflow2se3
import options.eval
import argparse
from datetime import datetime

def get_args():
    # Parse arguments
    parser = configargparse.ArgumentParser(
        description="UFlow Training",
        config_file_parser_class=configargparse.ConfigparserConfigFileParser,
        # YAMLConfigFileParser | ConfigparserConfigFileParser,
    )

    parser = options.config.parser_add_args(parser)

    args_config, _ = parser.parse_known_args()
    args_config_list = []
    for key, val in vars(args_config).items():
        if val is not None:
            args_config_list.append("--" + str(key).replace("_", "-"))
            args_config_list.append(str(val))

    sys.argv += args_config_list

    print(sys.argv)

    parser = options.setup.parser_add_args(parser)
    parser = options.data.parser_add_args(parser)
    parser = options.sflow2se3.parser_add_args(parser)
    parser = options.eval.parser_add_args(parser)

    args = parser.parse_args()

    args.data_dataset_tags = args.data_dataset_tags.split("-")

    args.setup_debug_mode = "_pydev_bundle.pydev_log" in sys.modules.keys()
    if args.setup_debug_mode:
        print("DEBUG MODE")
        args.setup_dataloader_num_workers = 0

    #args.setup_dataloader_device = None  # self.device
    args.setup_dataloader_pin_memory = False

    if args.setup_dataloader_num_workers > 0 and args.setup_dataloader_device == "cpu":
        args.setup_dataloader_pin_memory = True

    args.datetime = datetime.now()
    return args
