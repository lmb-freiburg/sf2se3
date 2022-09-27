import torch
import torchvision
import numpy as np
import cv2
import PIL

def write_oflow(oflow, fname="test.png"):
    # oflow 2xHxW

    oflow_copy = oflow.clone()
    _, H, W = oflow.shape
    ones = torch.ones(size=(1, H, W), dtype=oflow.dtype).to(oflow.device).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
    oflow = ((oflow * 64) + 2 ** 15).clamp(0, np.inf).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)

    oflow = np.concatenate((oflow, ones), axis=2)

    oflow = oflow[:, :, ::-1]

    cv2.imwrite(filename=fname, img=oflow)#, cv2.IMREAD_UNCHANGED)
    #from tensor_operations.visual import _2d as o4visual2d
    #oflow_read, oflow_valid_read = read_flow(fname, device="cpu")

    #print(oflow_read - oflow_copy.to("cpu"))
    #o4visual2d.visualize_img(o4visual2d.flow2rgb(flow))
    #pass

def write_disp(disp, fname="test.png"):
    # disp 1xHxW
    disp = (disp * 256).abs().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint16)
    disp = disp[: , :, 0]
    cv2.imwrite(filename=fname, img=disp)#, cv2.IMREAD_ANYDEPTH)

def read_flow(flow_fn, device):

        flow = cv2.imread(flow_fn, cv2.IMREAD_UNCHANGED)
        # H x W x 3 : dtype=np.uint16
        flow = flow[:, :, ::-1]
        # numpy.ndarray: HxWx3

        flow_valid = torch.from_numpy(flow[:, :, 2].astype(bool)).to(device)[
            None,
        ]
        # torch.bool: HxW

        flow_uv = (
            torch.from_numpy(flow[:, :, :2].astype(np.int32))
            .to(device)
            .permute(2, 0, 1)
            - 2 ** 15
        ) / 64.0
        # torch.float32: 2xHxW

        # flow_inside = o4mask.oflow_2_mask_inside(flow_uv[None,])[0]
        # o4mask.pxl2d_2_mask_inside()

        # flow_uv = o4visual.resize(flow_uv, H_out=self.height, W_out=self.width, mode='nearest', vals_rescale=True)
        # flow_valid = o4visual.resize(flow_valid, H_out=self.height, W_out=self.width, mode='nearest')

        return flow_uv, flow_valid
"""
def read_flow(flow_fn, device):

    flow = cv2.imread(flow_fn, cv2.IMREAD_UNCHANGED)
    # H x W x 3 : dtype=np.uint16
    flow = flow[:, :, ::-1]
    # numpy.ndarray: HxWx3

    flow_valid = torch.from_numpy(flow[:, :, 2].astype(bool)).to(device)[
        None,
    ]
    # torch.bool: HxW

    flow_uv = (
                      torch.from_numpy(flow[:, :, :2].astype(np.int32))
                      .to(device)
                      .permute(2, 0, 1)
                      - 2 ** 15
              ) / 64.0
    # torch.float32: 2xHxW

    # flow_inside = o4mask.oflow_2_mask_inside(flow_uv[None,])[0]
    # o4mask.pxl2d_2_mask_inside()

    # flow_uv = o4visual.resize(flow_uv, H_out=self.height, W_out=self.width, mode='nearest', vals_rescale=True)
    # flow_valid = o4visual.resize(flow_valid, H_out=self.height, W_out=self.width, mode='nearest')

    return flow_uv, flow_valid

"""

def read_rgb(img_fn, device, target_width=None, target_height=None):
    img = PIL.Image.open(img_fn)

    torch_transform = torchvision.transforms.ToTensor()
    img = torch_transform(img).to(device)

    if target_width is not None and target_height is not None:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )[0]

    return img

def read_disp(disp_fn, device):
    # TODO: check if cv2.imread(disp_fn, cv2.IMREAD_UNCHANGED) reads as np.uint16
    disp = cv2.imread(disp_fn, cv2.IMREAD_ANYDEPTH)
    # H x W : dtype=np.uint16 note: maximum > 256
    disp = torch.from_numpy(disp.astype(np.float32)).to(device) / 256.0

    disp = disp.unsqueeze(0)

    # disp = o4visual.resize(disp, H_out=self.height, W_out=self.width, mode='nearest', vals_rescale=True)

    disp_mask = disp > 0.0

    return disp, disp_mask

def read_mask_objects(mask_objects_fn, device):
    mask_objects = cv2.imread(mask_objects_fn, cv2.IMREAD_UNCHANGED).astype(
        np.uint8
    )

    mask_objects = torch.from_numpy(mask_objects).to(device)
    mask_objects = mask_objects.unsqueeze(0)
    # shape: 1 x H x W range: [0, num_objects_max], type: torch.uint8

    # mask_objects = o4visual.resize(mask_objects, H_out=self.height, W_out=self.width, mode='nearest')
    # (self.height, self.width),
    return mask_objects

def read_calib(calib_fp, device, target_width=None, target_height=None):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(calib_fp, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    # indices 0, 1, 2, 3  = left-gray, right-gray, left-rgb, right-rgb
    # note: left-rgb, right-rgb have same width, height, fx, fy, cx, cy

    # width = data['S_rect_02'][0]
    # height = data['S_rect_02'][1]

    if target_width is not None:
        sx = target_width / data["S_rect_02"][0]
    else:
        sx = 1.0

    if target_height is not None:
        sy = target_height / data["S_rect_02"][1]
    else:
        sy = 1.0

    fx = data["P_rect_02"][0] * sx
    fy = data["P_rect_02"][5] * sy

    cx = data["P_rect_02"][2] * sx
    cy = data["P_rect_02"][6] * sy

    baseline = torch.from_numpy(
        np.array([data["T_02"][0] - data["T_03"][0]], dtype=np.float32)
    ).to(device)
    # 3D-2D Projection:
    # u = (fx*x + cx * z) / z
    # v = (fy*y + cy * y) / z
    # shift on plane: delta_x = (fx * bx) / z
    #                 delta_y = (fy * by) / z
    # uv = (P * xyz) / z
    # P = [ fx   0  cx]
    #     [ 0   fy  cy]
    projection_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy]], dtype=np.float32)

    # 2D-3D Re-Projection:
    # x = (u/fx - cx/fx) * z
    # y = (v/fy - cy/fy) * z
    # z = z
    # xyz = (RP * uv1) * z
    # RP = [ 1/fx     0  -cx/fx ]
    #      [    0  1/fy  -cy/fy ]
    #      [    0      0      1 ]
    reprojection_matrix = np.array(
        [[1 / fx, 0.0, -cx / fx], [0.0, 1 / fy, -cy / fy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    projection_matrix = torch.from_numpy(projection_matrix).to(device)
    reprojection_matrix = torch.from_numpy(reprojection_matrix).to(device)

    return projection_matrix, reprojection_matrix, baseline

