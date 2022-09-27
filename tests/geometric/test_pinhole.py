import matplotlib.pyplot as plt
import torch

import tensor_operations.visual._2d as o4visual
import tensor_operations.geometric.pinhole as o4geo_pinhole
from tensor_operations.vision import warp as o4warp

import kornia.utils

def test_visualize_plane3d():
    #
    # 16 / 4
    # 15 / 3

    H_start = 28
    W_start = 28
    # 28 = 7 * 4
    # 27 = 9 * 3


    pxl3d = o4geo_pinhole.shape_2_pxl3d(B=0, H= H_start, W = W_start, dtype=float, device='cpu')

    H_down = 4
    W_down = 4
    #H_out=H_down, W_out=W_down
    pxl3d_down = o4visual.resize(pxl3d, H_out=H_down, W_out=W_down , mode='nearest_v2', align_corners=False)

    print(pxl3d_down.shape)
    o4visual.visualize_pxl2d(pxl3d_down[:2].flatten(1), H_start, W_start, fpath='results/visuals/nearest_sampling_align_corners_false.png')
    #K = torch.Tensor([])
    #o4visual.visualize_pts3d([pxl3d, pxl3d_down], change_viewport=False, fpath='results/visuals/sampling.png')

def test_visualize_grid_sampling():

    H_src = 3
    W_src = 3

    H_dst = 3
    W_dst = 3
    noise = torch.rand(size=(2, 3, 3)) / 4
    map_pos_dst_2_src = (o4geo_pinhole.shape_2_pxl2d(B=0, H=H_dst, W=W_dst, dtype=float, device='cpu')) / 2.0 + 0.5 + noise
    flow = (o4geo_pinhole.shape_2_pxl2d(B=0, H=H_src, W=W_src, dtype=float, device='cpu') + 0.5 - H_src / 2.0) * 3
    flow_rgb = o4visual.flow2rgb(flow, size_wheel=H_src * 3)
    flow_rgb_pos = o4visual.draw_pixels(flow_rgb, pxls = map_pos_dst_2_src.flatten(1).T)

    o4visual.visualize_img(flow_rgb_pos, fpath='results/visuals/warp_src_orig.png')
    dst = o4warp.interpolate2d(flow_rgb[None], map_pos_dst_2_src[None], mode="nearest")[0]

    upscale_factor = 100
    map_pos_dst_2_src_up = map_pos_dst_2_src * upscale_factor + upscale_factor // 2

    flow_rgb_up_pos = o4visual.resize_nearest_v2(flow_rgb[None], H_out=upscale_factor*H_src, W_out=upscale_factor*W_src, align_corners=False)[0]
    #flow_rgb_up
    for y in range(H_dst):
        for x in range(W_dst):
            if x < W_dst - 1:
                flow_rgb_up_pos = kornia.utils.draw_line(flow_rgb_up_pos, map_pos_dst_2_src_up[:, y, x], map_pos_dst_2_src_up[:, y, x+1], torch.Tensor([0, 0, 0.]))
            if y < H_dst - 1:
                flow_rgb_up_pos = kornia.utils.draw_line(flow_rgb_up_pos, map_pos_dst_2_src_up[:, y, x], map_pos_dst_2_src_up[:, y+1, x], torch.Tensor([0, 0, 0.]))
    flow_rgb_up_pos = o4visual.draw_pixels(flow_rgb_up_pos, pxls=map_pos_dst_2_src_up.flatten(1).T, radius_in=0, radius_out=7)
    #, colors=torch.Tensor([0., 0., 0.]).repeat(H_dst*W_dst, 1)
    dst_up = o4visual.resize_nearest_v2(dst[None], H_out=upscale_factor*H_dst, W_out=upscale_factor*W_dst, align_corners=False)[0]
    o4visual.visualize_img(flow_rgb_up_pos, fpath='results/visuals/warp_src.png')
    o4visual.visualize_img(dst_up, fpath='results/visuals/warp_dst.png')


    #o4geo_pinhole.pxl

    #target = torch.
    #o4warp.interpolate2d()

def test_triangulation():
    from tensor_operations import elemental as o4
    from tensor_operations.geometric import epipolar as o4geo_epi

    method = "midpoint"
    cycles = 3
    pxl_std = 6. #3
    inlier_thresh = 0.1

    method = "midpoint" # dlt midpoint
    dlt_framework = "opencv"
    dlt_framework = "kornia" # "opencv" "kornia"
    midpoint_z_positive = True

    device = "cpu" # cuda:0
    cdim=1
    depth = torch.rand(size=[4, 1, 374, 1238]).to(device) * 10 + 1.0

    intr = torch.Tensor(
        [[[718.3351,   0.0000, 600.3891],
         [  0.0000, 718.3351, 181.5122]]]).to(device)

    intr_inv = torch.Tensor(
        [[[0.0014, 0.0000, -0.8358],
        [0.0000, 0.0014, -0.2527],
        [0.0000, 0.0000, 1.0000]]]).to(device)

    transf_pts_t1_t2 = torch.Tensor(
        [[[ 9.9999e-01,  9.8987e-04,  3.2440e-03,  5.4737e-03],
         [-9.8295e-04,  1.0000e+00, -2.1340e-03,  4.1142e-03],
         [-3.2461e-03,  2.1308e-03,  9.9999e-01, -6.3363e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
        [[ 9.9926e-01, -1.0040e-02,  3.7124e-02, -2.7715e-01],
         [ 9.8157e-03,  9.9993e-01,  6.2186e-03, -8.3504e-02],
         [-3.7184e-02, -5.8496e-03,  9.9929e-01, -1.6771e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
        [[ 9.9954e-01,  1.6120e-04,  3.0440e-02, -3.9978e-01],
         [-1.0905e-04,  1.0000e+00, -1.7151e-03, -2.2546e-03],
         [-3.0440e-02,  1.7109e-03,  9.9954e-01, -3.9144e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
        [[ 9.9992e-01, -4.7804e-03,  1.1815e-02,  4.9403e-01],
         [ 4.7849e-03,  9.9999e-01, -3.5337e-04,  1.6331e-02],
         [-1.1813e-02,  4.0988e-04,  9.9993e-01, -4.8250e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]).to(device)


    extr_cam2 = torch.linalg.inv(transf_pts_t1_t2)

    pt3d_1 = o4geo_pinhole.depth_2_pt3d(depth, reproj_mats=intr_inv.repeat(4, 1, 1))

    hpt4d_1 = o4geo_pinhole.pt3d_2_hpt4d(pt3d_1, cdim=cdim)

    pt3d_2 = o4.multiply_matrix_vector(transf_pts_t1_t2, hpt4d_1, cdim=cdim)[:, :3]

    pxl2d_1 = o4geo_pinhole.pt3d_2_pxl2d(pt3d_1, proj_mats=intr)
    pxl2d_2 = o4geo_pinhole.pt3d_2_pxl2d(pt3d_2, proj_mats=intr)
    oflow = pxl2d_2 - pxl2d_1

    #intr = intr[:1]
    #intr_inv = intr_inv[:1]
    #extr_cam2 = extr_cam2[:1]
    #oflow = oflow[:1]
    #pt3d_1 = pt3d_1[:1]

    from time import time

    durations = []

    for k in range(cycles):
        time_start = time()


        pt3d_1_triang, pt3d_1_pxl_2d_inlier_prob = o4geo_epi.triangulate_single_se3(
            intr_inv_cam2=intr_inv,
            intr_cam2=intr,
            extr_cam2=extr_cam2,
            oflow=oflow,
            pxl_std=pxl_std,
            method=method,
            dlt_framework=dlt_framework,
            midpoint_z_positive=midpoint_z_positive
        )

        durations.append(round(time() - time_start, 2))

    print("durations:", durations)
    print("mean inlier probability: ", str(round(pt3d_1_pxl_2d_inlier_prob.mean().item() * 100)) + "%")
    print("pt3d error for inlier probability: ", str(round(inlier_thresh * 100)) + "%", (pt3d_1_triang - pt3d_1).norm(dim=cdim, keepdim=True)[pt3d_1_pxl_2d_inlier_prob > inlier_thresh].mean())


def test_visualize_inlier_probability():
    #plt.fill_between(ages, total_population, 25000000, alpha=0.30)

    import matplotlib.pyplot as plt
    import torch

    distr_normal = torch.distributions.normal.Normal(loc=0, scale=1.)

    range = 6
    samples = 1000
    step_width = range / samples
    xs = torch.linspace(-range/2., range/2., samples)
    ys = torch.exp(distr_normal.log_prob(xs))
    ys_acc = distr_normal.cdf(xs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    #ys_inlier = 2 * (1. - torch.cumsum(ys * step_width, dim=0))


    error = 1.
    xs_partial_left = xs[xs < -error]
    ys_partial_left = torch.exp(distr_normal.log_prob(xs_partial_left))
    xs_partial_right = xs[xs > error]
    ys_partial_right = torch.exp(distr_normal.log_prob(xs_partial_right))


    #ax1.fill_between(xs, ys, alpha=1.0)
    ax1.plot(xs_partial_left, ys_partial_left, 'r')
    ax1.fill_between(xs_partial_left, ys_partial_left, color='r', alpha=0.5)
    ax1.plot(xs_partial_right, ys_partial_right, 'r')
    ax1.fill_between(xs_partial_right, ys_partial_right, color='r', alpha=0.5)
    ax1.plot(xs, ys, label="f(x)")

    ax1.plot([-error, error], [0., 0.], 'k|', markersize=200, linewidth=5)
    ax1.legend()

    ys_inlier = 2 * (1. - ys_acc)
    ys_inlier[:samples //2] = ys_inlier[samples //2:].flip(dims=[0])
    ax2.plot(xs, ys_inlier, 'g', label="P(x)")
    ax2.legend()
    plt.savefig('results/visuals/inlier_probability.svg')
    plt.show()



def test_plot_estimate_std_with_trunc_data():

    from tensor_operations.probabilistic.models.gaussian import calc_sigma_optimum, \
        calc_sigma_rel_optimum, calc_ratio_sigma_sigma_part, \
        calc_sigma_rel_implicit_proposals, calc_sigma_rel_implicit
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    alphas = torch.Tensor([0.1, 1, 1.5, 1.8, 2.0, 2.5, 3.0, 9])
    sigma_trunc = torch.Tensor([1])  # 1
    delta = torch.Tensor([1]) #.sqrt() #2.sqrt()

    sigma_rel_min = 0.1
    sigma_rel_max = 1. # 10
    resolution = 1000
    y_min = -0.2
    y_max = 0.2
    thresh = 1e-3
    #plt.figure(1, 5, 6)
    plt.figure(figsize=(6.5, 4.3))
    for alpha in alphas:
        sigma_rel, sigma_rel_implicit_proposals = calc_sigma_rel_implicit_proposals(delta * alpha, sigma_trunc,
                                                                            sigma_rel_min=sigma_rel_min, sigma_rel_max=sigma_rel_max,
                                                                            resolution=resolution)

        sigma_optimum = calc_sigma_optimum(delta* alpha, sigma_trunc,
                                           sigma_rel_min=sigma_rel_min, sigma_rel_max=sigma_rel_max,
                                           resolution=resolution, thresh=thresh)
        sigma_rel_optimum = calc_sigma_rel_optimum(delta* alpha, sigma_trunc,
                                                   sigma_rel_min=sigma_rel_min, sigma_rel_max=sigma_rel_max,
                                                   resolution=resolution, thresh=thresh)

        plt.plot(sigma_rel, sigma_rel_implicit_proposals,
                 label=r"$\frac{\delta}{\sigma_{trunc}}$="+str(round(alpha.item(),1)) + r"$ \rightarrow $" + r"$\frac{\sigma_{trunc}}{\sigma}$=" + str(round(sigma_rel_optimum.item(),1)))


        sigma_rel_implicit_optimum = calc_sigma_rel_implicit(delta* alpha, sigma_trunc, sigma_optimum)

        plt.plot(sigma_rel_optimum, sigma_rel_implicit_optimum, "ro")
        plt.vlines(sigma_rel_optimum, y_min, sigma_rel_implicit_optimum, linestyle="dashed")
        #plt.annotate(str(round(y[i].item(), 2)), xy=(x_angle_deg[i]-28, y[i]+0.05), fontsize=12)
        #plt.hlines(y, 0, x_angle_deg, linestyle="dashed")
        #for i in range(len(x_angle_deg)):

    plt.xlim([sigma_rel_min, sigma_rel_max+1])
    plt.ylim([-0.2, 0.2])
    #plt.plot(sigma_optimum / sigma_trunc , sigma_optimum / sigma_trunc * 0, "ro", markersize=10)
    plt.xlabel(r'$ \frac{\sigma_{trunc}}{\sigma}$', fontsize=20)
    plt.ylabel(r"$ r(\frac{\sigma_{trunc}}{\sigma})$", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.savefig("results/visuals/sigma_trunc_correction.svg")
    plt.show()

    """
    ys_sigma_sigma_part = calc_ratio_sigma_sigma_part(xs_delta_sigma)

    ys_sigma_sigma_part_2 = (1. / (delta * sigma_part)) * ((ys_sigma_sigma_part * sigma_part)**2) * xs_delta_sigma

    plt.plot(xs_delta_sigma, ys_sigma_sigma_part, linewidth=3, label="equation 1")
    plt.plot(xs_delta_sigma, ys_sigma_sigma_part_2, linewidth=3, label="equation 2")
    plt.xlim([0, 10])
    plt.ylim([0, 5])
    plt.xlabel(r'$\delta / \sigma$')
    plt.ylabel(r"$\sigma / \sigma_{part}$")
    plt.plot(xs_delta_sigma_optimum, ys_sigma_sigma_part_optimum, "ro", markersize=10, label="optimum")# "*")
    plt.legend()
    print("\n\nevtl found sigma", sigma_optimum, "\n\n")
    plt.savefig('results/visuals/std_est_trunc_correct_factor.svg')
    plt.show()

    #ys_acc = distr_normal.cdf(xs)
    #distr_normal = torch.distributions.normal.Normal(loc=0, scale=1.)
    """

def test_plot_rel_sflow_bound_over_angle():
    import numpy as np
    samples = 1000
    xs_angles_deg = torch.linspace(0, 360., samples) #%/  5.
    ys = torch.sqrt((torch.sin(xs_angles_deg / 360 * 2 * np.pi))**2 + ((1 - torch.cos(xs_angles_deg / 360 * 2 * np.pi))**2))

    plt.plot(xs_angles_deg, ys, linewidth=3, label="equation 1")
    plt.xlabel(r'$\alpha$ in [$^{\circ}$]', fontsize=18)
    plt.ylabel(r"$\max \frac{||s_2 - s_1||}{||p_2 - p_1||} $", fontsize=18)
    plt.xticks(fontsize=12)#, rotation=90)
    plt.yticks(fontsize=12)
    x_angle_deg = torch.Tensor([25, 45])
    y = torch.sqrt((torch.sin(x_angle_deg / 360 * 2 * np.pi))**2 + ((1 - torch.cos(x_angle_deg / 360 * 2 * np.pi))**2))
    plt.plot(x_angle_deg, y, "ro")
    plt.vlines(x_angle_deg, 0, y, linestyle="dashed")
    plt.hlines(y, 0, x_angle_deg, linestyle="dashed")
    for i in range(len(x_angle_deg)):
        plt.annotate(str(round(y[i].item(), 2)), xy=(x_angle_deg[i]-28, y[i]+0.05), fontsize=12)
    plt.tight_layout()
    plt.savefig('results/visuals/sflow_continuity_bound_alpha.svg')
    plt.show()
