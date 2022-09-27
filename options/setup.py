import configargparse
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parser_add_args(parser):

    parser.add_argument(
        "--setup-datasets-dir",
        default=None,
        required=True,
        type=str,
        metavar="DIRS",
        help="path to datasets",
    )

    parser.add_argument(
        "--setup-results-dir",
        default=None,
        required=True,
        type=str,
        metavar="DIRS",
        help="path to results",
    )

    parser.add_argument(
        "--setup-dataloader-num-workers",
        default=0,
        type=int,
        help="dataloader-num-workers (default: 0)",
    )

    parser.add_argument(
        "--setup-dataloader-device",
        default="cuda",
        type=str,
        help="setup-dataloader-device (default: 'cuda')",
    )

    parser.add_argument(
        "--setup-wandb-entity",
        default=None,
        required=True,
        type=str,
        help="wandb entity",
    )

    parser.add_argument(
        "--setup-wandb-project",
        default=None,
        required=True,
        type=str,
        help="wandb project",
    )

    parser.add_argument(
        "--setup-wandb-log",
        default=False,
        type=str2bool,
        help="wandb log enabled? (default: False)",
    )

    return parser
