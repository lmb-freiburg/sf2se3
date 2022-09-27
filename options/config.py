def parser_add_args(parser):

    parser.add_argument(
        "--configs",
        required=False,
        is_config_file=True,
        help="Path to config file for configs parameters",
    )

    parser.add_argument(
        "--config-eval",
        required=False,
        default="configs/eval.yaml",
        is_config_file=True,
        help="Path to config file for eval parameters",
    )


    parser.add_argument(
        "--config-sflow2se3",
        required=False,
        is_config_file=True,
        help="Path to config file for sflow2se3 parameters",
    )

    parser.add_argument(
        "--config-sflow2se3-data-dependent",
        required=False,
        is_config_file=True,
        help="Path to config file for sflow2se3 parameters",
    )


    parser.add_argument(
        "--config-setup",
        required=True,
        is_config_file=True,
        help="Path to config file for setup parameters",
    )

    parser.add_argument(
        "--config-data",
        required=False,
        is_config_file=True,
        help="Path to config file for data parameters",
    )

    parser.add_argument(
        "--config-ablation",
        required=False,
        default="configs/ablation.yaml",
        is_config_file=True,
        help="Path to config file for eval parameters",
    )
    return parser
