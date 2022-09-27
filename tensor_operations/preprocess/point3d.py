
from tensor_operations.preprocess import depth as  o4pp_depth
from tensor_operations.geometric import pinhole as o4geo_pin

def complete(pt3d, pt3d_valid, reproj_mat):
    """ given invalid points complete points by completing depth using recursive neighbor interpolation and re-project

    Parameters
    ----------
    pt3d torch.Tensor: Bx3xHxW, float
    pt3d_valid torch.Tensor: Bx1xHxW, bool
    reprojection_matrix torch.Tensor: Bx3x3, float

    Returns
    -------
    pt3d torch.Tensor: Bx3xHxW, float
    pt3d_valid torch.Tensor: Bx1xHxW, bool
    """

    depth = pt3d[:, 2:]

    depth, pt3d_valid = o4pp_depth.complete(depth, pt3d_valid)

    pt3d = o4geo_pin.depth_2_pt3d(depth, reproj_mats=reproj_mat)

    return pt3d, pt3d_valid

