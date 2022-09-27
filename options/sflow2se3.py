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
        "--sflow2se3-approach",
        default="classic",
        type=str,
        help="sflow2se3-approach: classic | raft3d | rigidmask (default: classic)",
    )

    parser.add_argument(
        "--sflow2se3-sflow-use-depth-if-available",
        default=False,
        type=str2bool,
        help="sflow2se3-sflow-use-depth-if-available (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-sflow-use-oflow-if-available",
        default=False,
        type=str2bool,
        help="sflow2se3-sflow-use-oflow-if-available (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-downscale-factor",
        default=0.05,
        type=float,
        help="sflow2se3-downscale-factor (default: 0.05)",
    )

    parser.add_argument(
        "--sflow2se3-downscale-mode",
        default="nearest_v2",
        type=str,
        help="sflow2se3-downscale-mode (default: nearest_v2)",
    )

    parser.add_argument(
        "--sflow2se3-depth-min",
        default=0.1,
        type=float,
        help="sflow2se3-depth-min (default: 0.1)",
    )

    parser.add_argument(
        "--sflow2se3-depth-max",
        default=10.0,
        type=float,
        help="sflow2se3-depth-max (default: 10.)",
    )

    parser.add_argument(
        "--sflow2se3-depth-complete-invalid",
        default=False,
        type=str2bool,
        help="sflow2se3-depth-complete-invalid (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-occl-warp-dssim-max",
        default=0.2,
        type=float,
        help="sflow2se3-occl-warp-dssim-max (default: 0.2)",
    )

    parser.add_argument(
        "--sflow2se3-disp-occl-source",
        default="fwdbwd",
        type=str,
        help="sflow2se3-disp-occl-source (default: fwdbwd)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-std-auto",
        default=True,
        type=str2bool,
        help="sflow2se3-oflow-disp-std-auto (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-trusted-perc",
        default=0.25,
        type=float,
        help="sflow2se3-oflow-disp-trusted-perc (default: 0.25)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-std-valid-min-perc",
        default=0.10,
        type=float,
        help="sflow2se3-oflow-disp-std-valid-min-perc (default: 0.10)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-std-correct-truncation",
        default=True,
        type=str2bool,
        help="sflow2se3-oflow-disp-std-correct-truncation (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-std-abs-min",
        default=1,
        type=float,
        help="sflow2se3-oflow-disp-std-abs-min (default: 1)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-disp-std-abs-max",
        default=5,
        type=float,
        help="sflow2se3-oflow-disp-std-abs-max (default: 5)",
    )

    parser.add_argument(
        "--sflow2se3-oflow-occl-source",
        default="fwdbwd",
        type=str,
        help="sflow2se3-oflow-occl-source (default: fwdbwd)",
    )

    parser.add_argument(
        "--sflow2se3-min-samples",
        default=5,
        type=int,
        help="sflow2se3-min-samples (default: 5)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-type",
        default="pdf",
        type=str,
        help="sflow2se3-model-se3-likelihood-type (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-std-factor-occ",
        default=3,
        type=float,
        help="sflow2se3-model-se3-std-factor-occ (default: 3.)",
    )
    parser.add_argument(
        "--sflow2se3-model-se3-disp-std-use-fwdbwd-dev-if-larger",
        default=False,
        type=str2bool,
        help="sflow2se3-model-se3-disp-std-use-fwdbwd-dev-if-larger (default: False)",
    )
    parser.add_argument(
        "--sflow2se3-model-se3-oflow-std-use-fwdbwd-dev-if-larger",
        default=False,
        type=str2bool,
        help="sflow2se3-model-se3-oflow-std-use-fwdbwd-dev-if-larger (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-sflow-abs-std",
        default=0.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-sflow-abs-std (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-sflow-rel-std",
        default=0.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-sflow-rel-std (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-oflow-abs-std",
        default=0.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-oflow-abs-std (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-use-oflow",
        default=True,
        type=str2bool,
        help="sflow2se3-model-se3-likelihood-use-oflow (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-oflow-invalid-pairs",
        default=-1.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-oflow-invalid-pairs (default: -1.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-disp-abs-std",
        default=0.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-disp-abs-std (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-use-disp",
        default=True,
        type=str2bool,
        help="sflow2se3-model-se3-likelihood-use-disp (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-disp-invalid-pairs",
        default=1.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-disp-invalid-pairs (default: 1.)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-use-sflow",
        default=False,
        type=str2bool,
        help="sflow2se3-model-se3-likelihood-use-sflow (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-likelihood-sflow-invalid-pairs",
        default=0.0,
        type=float,
        help="sflow2se3-model-se3-likelihood-sflow-invalid-pairs (default: 0.)",
    )

    parser.add_argument(
        "--sflow2se3-model-inlier-hard-threshold",
        default=0.0455,
        type=float,
        help="sflow2se3-model-inlier-hard-threshold (default: 0.0455)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-inlier-oflow-hard-threshold",
        default=0.0455,
        type=float,
        help="sflow2se3-model-se3-inlier-oflow-hard-threshold (default: 0.0455)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-inlier-disp-hard-threshold",
        default=0.0455,
        type=float,
        help="sflow2se3-model-se3-inlier-disp-hard-threshold (default: 0.0455)",
    )

    parser.add_argument(
        "--sflow2se3-model-se3-inlier-sflow-hard-threshold",
        default=0.0455,
        type=float,
        help="sflow2se3-model-se3-inlier-sflow-hard-threshold (default: 0.0455)",
    )

    parser.add_argument(
        "--sflow2se3-raft3d-cluster",
        default=False,
        type=str2bool,
        help="--sflow2se3-raft3d-cluster (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-model-euclidean-nn-rel-depth-dev-std",
        default=0.13,
        type=float,
        help="sflow2se3-model-euclidean-nn-rel-depth-dev-std (default: 0.13)",
    )

    parser.add_argument(
        "--sflow2se3-model-euclidean-nn-uv-dev-rel-to-width-std",
        default=0.12,
        type=float,
        help="sflow2se3-model-euclidean-nn-uv-dev-rel-to-width-std (default: 0.12)",
    )

    parser.add_argument(
        "--sflow2se3-model-euclidean-nn-dist-inlier-hard-threshold",
        default=0.0455,
        type=float,
        help="sflow2se3-model-euclidean-nn-dist-inlier-hard-threshold (default: 0.0455)",
    )

    parser.add_argument(
        "--sflow2se3-model-euclidean-nn-connect-global-dist-range-ratio-max",
        default=0.25,
        type=float,
        help="--sflow2se3-model-euclidean-nn-connect-global-dist-range-ratio-max (default: 0.25)",
    )

    parser.add_argument(
        "--sflow2se3-model-euclidean-nn-pts-count-max",
        default=1000,
        type=int,
        help="sflow2se3-model-euclidean-nn-pts-count-max (default: 1000)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-refinement-cycles",
        default=1,
        type=int,
        help="sflow2se3-extraction-refinement-cycles (default: 1)"
    )

    parser.add_argument(
        "--sflow2se3-refinements-per-extraction",
        default=0,
        type=int,
        help="sflow2se3-refinements-per-extraction (default: 0)"

    )

    parser.add_argument(
        "--sflow2se3-extraction-cluster-types",
        default=[],
        required=False,
        action='append',
        type=str,
        help="sflow2se3-extraction-cluster-types (default: [])"
    )

    parser.add_argument(
        "--sflow2se3-extraction-cluster-pts-valid-count-max",
        default=100,
        type=int,
        help="--sflow2se3-extraction-cluster-pts-valid-count-max (default: 100)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-cluster-eucl-min-inlier-perc",
        default=0.5,
        type=float,
        help="sflow2se3-extraction-cluster-eucl-min-inlier-perc (default: 0.5)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-cluster-eucl-oflow-dev-abs-max",
        default=3,
        type=float,
        help="sflow2se3-extraction-cluster-eucl-oflow-dev-abs-max (default: 3)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-cluster-eucl-oflow-dev-rel-max",
        default=0.05,
        type=float,
        help="sflow2se3-extraction-cluster-eucl-oflow-dev-rel-max (default: 0.05)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-se3-fit-pt3d-oflow-method",
        default="cpu-epnp",
        type=str,
        help="sflow2se3-extraction-se3-fit-pt3d-oflow-method (default: cpu-epnp)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-se3-refine",
        default=False,
        type=str2bool,
        help="sflow2se3-extraction-se3-refine (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-se3-refine-pt3d-oflow-method",
        default="cpu-iterative-continue",
        type=str,
        help="sflow2se3-extraction-se3-refine-pt3d-oflow-method (default: cpu-iterative-continue)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-select-objects-source",
        default="inlier",
        type=str,
        help="sflow2se3-extraction-select-objects-source (default: inlier)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-select-method",
        default="greedy",
        type=str,
        help="sflow2se3-refinement-select-method (default: greedy)",
    )

    "sflow2se3-refinement-select-method"

    parser.add_argument(
        "--sflow2se3-refinement-se3-fit-assign-only-inlier",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-se3-fit-assign-only-inlier (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-fit-assign-hard",
        default=True,
        type=str2bool,
        help="sflow2se3-refinement-se3-fit-assign-hard (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-geo-fit-assign-only-inlier",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-geo-fit-assign-only-inlier (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-geo-fit-assign-hard",
        default=True,
        type=str2bool,
        help="sflow2se3-refinement-geo-fit-assign-hard (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-fit",
        default=True,
        type=str2bool,
        help="sflow2se3-refinement-se3-fit (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-fit-data",
        default="pt3d-oflow",
        type=str,
        help="sflow2se3-refinement-se3-fit-data (default: pt3d-oflow)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-fit-pt3d-oflow-method",
        default="cpu-epnp",
        type=str,
        help="sflow2se3-refinement-se3-fit-pt3d-oflow-method (default: cpu-epnp)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-refine",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-se3-refine (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-se3-refine-pt3d-oflow-method",
        default="cpu-iterative-continue",
        type=str,
        help="sflow2se3-refinement-se3-refine-pt3d-oflow-method (default: cpu-iterative-continue)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-geo-source",
        default="se3+geo",
        type=str,
        help="sflow2se3-refinement-geo-source (default: se3+geo)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-split-objects-geo-apply",
        default=True,
        type=str2bool,
        help="sflow2se3-refinement-split-objects-geo-apply (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-split-objects-geo-fuse",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-split-objects-geo-fuse (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-split-objects-geo-fuse-req-global-connected",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-split-objects-geo-fuse-req-global-connected (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-refinement-split-objects-geo-fuse-keep-splitted",
        default=False,
        type=str2bool,
        help="sflow2se3-refinement-split-objects-geo-fuse-keep-splitted (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-split-objects-geo-apply",
        default=True,
        type=str2bool,
        help="sflow2se3-extraction-split-objects-geo-apply (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-extraction-split-objects-geo-fuse",
        default=True,
        type=str2bool,
        help="sflow2se3-extraction-split-objects-geo-fuse (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-split-objs-eucl-fuse",
        default=True,
        type=str2bool,
        help="sflow2se3-split-objs-eucl-fuse (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-split-objects-geo-method",
        default="agglomerative",
        type=str,
        help="sflow2se3-split-objects-geo-method (default: agglomerative)",
    )

    parser.add_argument(
        "--sflow2se3-assign-objects-upsampling-depth-interpolation-apply",
        default=False,
        type=str2bool,
        help="sflow2se3-assign-objects-upsampling-depth-interpolation-apply (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-assign-objects-upsampling-depth-triangulation-apply",
        default=False,
        type=str2bool,
        help="sflow2se3-assign-objects-upsampling-depth-triangulation-apply (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-accumulation-req-neighbor",
        default=False,
        type=str2bool,
        help="sflow2se3-rigid-clustering-accumulation-req-neighbor (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-add-sflow-bound",
        default=False,
        type=str2bool,
        help="sflow2se3-rigid-clustering-add-sflow-bound (default: False)",
    )


    parser.add_argument(
        "--sflow2se3-rot-deg-max",
        default=45.,
        type=float,
        help="sflow2se3-rot-deg-max (default: 45.)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-dist-dev-max",
        default=0.03,
        type=float,
        help="sflow2se3-rigid-dist-dev-max (default: 0.03)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-dist-dev-rel-max",
        default=0.01,
        type=float,
        help="sflow2se3-rigid-dist-dev-rel-max (default: 0.01)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-dist-min",
        default=-0.05,
        type=float,
        help="sflow2se3-rigid-dist-min (default: -0.05)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-dist-max",
        default=99999.0,
        type=float,
        help="sflow2se3-rigid-dist-max (default: 99999)",
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-method",
        default="agglomerative",
        type=str,
        help="sflow2se3-rigid-clustering-method (default: agglomerative)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-repetitions",
        default=1,
        type=int,
        help="--sflow2se3-rigid-clustering-repetitions (default: 1)"
    )

    parser.add_argument(
        "--sflow2se3-model-se3-fit-return-largest-k",
        default=-1,
        type=int,
        help="--sflow2se3-model-se3-fit-return-largest-k (default: -1)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-accumulation-max-samples",
        default=2,
        type=int,
        help="sflow2se3-rigid-clustering-accumulation-max-samples (default: 2)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-accumulation-use-range",
        default=False,
        type=str2bool,
        help="sflow2se3-rigid-clustering-accumulation-use-range (default: False)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-accumulation-largest-k",
        default=-1,
        type=int,
        help="sflow2se3-rigid-clustering-accumulation-largest-k (default: -1)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-agglomerative-linkage",
        default="complete",
        type=str,
        help="sflow2se3-rigid-clustering-agglomerative-linkage (default: complete)"
    )

    parser.add_argument(
        "--sflow2se3-rigid-clustering-agglomerative-max-dist",
        default=0.5,
        type=float,
        help="sflow2se3-rigid-clustering-agglomerative-max-dist (default: 0.5)"
    )

    parser.add_argument(
        "--sflow2se3-se3filter-prob-gain-min",
        default=5.0,
        type=float,
        help="sflow2se3-se3filter-prob-gain-min (default: 5.0)",
    )

    parser.add_argument(
        "--sflow2se3-se3filter-prob-same-mask-max",
        default=0.5,
        type=float,
        help="sflow2se3-se3filter-prob-same-mask-max (default: 0.5)",
    )

    parser.add_argument(
        "--sflow2se3-se3geofilter-prob-same-mask-max",
        default=0.1,
        type=float,
        help="sflow2se3-se3geofilter-prob-same-mask-max (default: 0.1)",
    )

    parser.add_argument(
        "--sflow2se3-sim-dist-max",
        default=0.1,
        type=float,
        help="sflow2se3-sim-dist-max (default: 0.1)",
    )

    parser.add_argument(
        "--sflow2se3-sim-angle-max",
        default=5.0,
        type=float,
        help="sflow2se3-sim-angle-max (default: 5.)",
    )

    parser.add_argument(
        "--sflow2se3-mask-recall-max",
        default=0.5,
        type=float,
        help="sflow2se3-mask-recall-max (default: 0.5)",
    )

    parser.add_argument(
        "--sflow2se3-visualize-mask-se3-progression",
        default=False,
        type=str2bool,
        help="sflow2se3-visualize-mask-se3-progression (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-disp-net-archictecture",
        default="raft",
        type=str,
        help="raft or leaststereo (default: raft)",
    )

    parser.add_argument(
        "--sflow2se3-leaststereo-train-dataset",
        default="sceneflow",
        type=str,
        help="sceneflow or kitti (default: sceneflow)",
    )

    parser.add_argument(
        "--sflow2se3-raft-train-dataset",
        default="flyingthings3d",
        type=str,
        help="flyingthings3d or kitti (default: flyingthings3d)",
    )

    parser.add_argument(
        "--sflow2se3-raft3d-train-dataset",
        default="flyingthings3d",
        type=str,
        help="flyingthings3d or kitti (default: flyingthings3d)",
    )

    parser.add_argument(
        "--sflow2se3-raft3d-architecture-bilaplacian",
        default=True,
        type=str2bool,
        help="only for sceneflow this can be false (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-rigidmask-train-dataset",
        default="sceneflow",
        type=str,
        help="sceneflow or kitti (default: sceneflow)",
    )

    parser.add_argument(
        "--sflow2se3-labels-source-argmax",
        default="prior*inlier",
        type=str,
        help="sflow2se3-labels-source-argmax (default: prior*inlier)",
    )

    parser.add_argument(
        "--sflow2se3-pt3d-valid-req-disp-nocc",
        default=False,
        type=str2bool,
        help="sflow2se3-pt3d-valid-req-disp-nocc (default: False)",
    )

    parser.add_argument(
        "--sflow2se3-pt3d-valid-req-pxl-lower-than",
        default=1.0,
        type=float,
        help="--sflow2se3-pt3d-valid-req-pxl-lower-than (default: 1.0)",
    )

    parser.add_argument(
        "--sflow2se3-se3-inlier-req-pt3d-0-valid",
        default=True,
        type=str2bool,
        help="sflow2se3-pt3d-valid-req-disp-nocc (default: True)",
    )

    parser.add_argument(
        "--sflow2se3-assignment-object-method",
        default="bayesian",
        type=str,
        help="sflow2se3-assignment-object-method (default: bayesian)"
    )

    return parser
