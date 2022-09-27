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

    """eval-sflow2se3-visualize-ref-se3-assign: False
eval-sflow2se3-visualize-ref-se3-updated: False
eval-sflow2se3-visualize-ref-geo-assign: False
eval-sflow2se3-visualize-ref-geo-updated: False
eval-sflow2se3-visualize-ref-objs-selected: False
"""

    parser.add_argument(
        "--eval-live",
        default=False,
        type=str2bool,
        help="eval-live (default: False)",
    )

    parser.add_argument(
        "--eval-visualize-paper",
        default=False,
        type=str2bool,
        help="eval-visualize-paper (default: False)",
    )

    parser.add_argument(
        "--eval-remote-frame-encode",
        default=True,
        type=str2bool,
        help="eval-remote-frame-encode (default: True)",
    )

    parser.add_argument(
        "--eval-remote-frame-width",
        default=640,
        type=int,
        help="eval-remote-frame-width (default: 640)",
    )

    parser.add_argument(
        "--eval-remote-frame-height",
        default=480,
        type=int,
        help="eval-remote-frame-height (default: 480)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-ref-se3-assign",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-ref-se3-assign (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-ref-se3-updated",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-ref-se3-updated (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-ref-geo-assign",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-ref-geo-assign (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-ref-geo-updated",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-ref-geo-updated (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-ref-objs-selected",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-ref-objs-selected (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-pt3d-pair-valid",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-pt3d-pair-valid (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-se3-extr",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-se3-extr (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-se3-extr-sparse",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-se3-extr-sparse (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-se3-extr-selected",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-se3-extr-selected (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-se3-extr-fused",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-se3-extr-fused (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-geo-added",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-geo-added (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-objs-splitted",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-objs-splitted (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-objs-selected",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-objs-selected (default: False)",
    )

    parser.add_argument(
        "--eval-sflow2se3-visualize-extr-objs-fused",
        default=False,
        type=str2bool,
        help="eval-sflow2se3-visualize-extr-objs-fused (default: False)",
    )

    parser.add_argument(
        "--eval-visualization-keys",
        default=[],
        required=False,
        action='append',
        type=str,
        help="visualizations (default: [])"
    )

    parser.add_argument(
        "--eval-visualize-result",
        default=False,
        type=str2bool,
        help="eval-visualize-result (default: False)",
    )

    parser.add_argument(
        "--eval-visualize-dataset",
        default=False,
        type=str2bool,
        help="eval-visualize-dataset (default: False)",
    )

    parser.add_argument(
        "--eval-visualize-pred-sflow",
        default=False,
        type=str2bool,
        help="eval-visualize-pred-sflow (default: False)",
    )

    parser.add_argument(
        "--eval-visualize-pred-se3",
        default=False,
        type=str2bool,
        help="eval-visualize-pred-se3 (default: False)",
    )

    parser.add_argument(
        "--eval-visualize-width-max",
        default=1920,
        type=int,
        help="eval-visualize-width-max (default: 1920)",
    )

    parser.add_argument(
        "--eval-visualize-height-max",
        default=1080,
        type=int,
        help="eval-visualize-height-max (default: 1080)",
    )

    return parser
