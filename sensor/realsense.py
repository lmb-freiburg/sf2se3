## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import zmq.ssh
import time
from datetime import datetime

def draw_text_in_rgb(img, text='title0'):
    # 3xHxW
    _, H, W = img.shape

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (10, 50)  # left, top
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    img = img / 255.0
    #img = torch.from_numpy(img).permute(2, 0, 1)
    #img = img.to(device)

    return img

def main():
    port_rep = "2308"
    frame_encode = True
    live = False
    phase="start" # "record" "process

    if live is False:
        print("press r for recording")
    max_fps = 10
    win_scale = 1. # 2.3

    date = datetime.now()
    fname = str(date.year) + "-" + str(date.month) + "-" + str(date.day) + "_" + str(date.hour) + "-" + str(date.minute)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter("C:\\Users\\User\\Documents\\master_thesis_presi\\" + fname + ".mp4", fourcc, max_fps, (int(640 * win_scale), int(480 * win_scale)))
    #if send_to_server:
    # True | False
    context = zmq.Context()
    print("Connecting to server...")
    socket_req = context.socket(zmq.REQ)

    #socket_req.connect("tcp://sommer-space.de:%s" % port_rep)
    #socket_req.connect("tcp://192.168.0.107:%s" % port_rep)

    zmq.ssh.tunnel_connection(socket_req, "tcp://bud:%s" % port_rep, server="sommerl@lmblogin.informatik.uni-freiburg.de:2122",
                              password="", keyfile="C:\\Users\\User\\.ssh\\id_rsa")

    # ssh sommerl@lmblogin.informatik.uni-freiburg.de -p 2122
    # qsub -l hostlist=^bud,nodes=1:ppn=1:gpus=1:ubuntu2004:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student sflow2rigid3d/eval_remote_rgbd.sh
    # qsub -l hostlist=^bud,nodes=1:ppn=1:gpus=1:ubuntu2004:nvidiaMin11GB,mem=16gb,walltime=24:00:00 -q student sflow2rigid3d/eval_remote_rgbd_slow.sh

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    #pipeline.get_active_profile().get_stream() #(RS2_STREAM_DEPTH). as < rs2::video_stream_profile >

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    #hole_filling = rs.hole_filling_filter()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming

    profile = pipeline.start(config)

    """
    profile_depth = profile.get_stream(rs.stream.depth)
    intr_depth = profile_depth.as_video_stream_profile().get_intrinsics()
    profile_color = profile.get_stream(rs.stream.color)
    intr_color = profile_color.as_video_stream_profile().get_intrinsics()
    print(intr_color, intr_depth)
    """

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    rgb_enc_stack = []
    depth_enc_stack = []
    playback_i = 0
    while True:

        if live or phase != "process":
            # Wait for a coherent pair of frames: depth and color
            for i in range(1):
                frames = pipeline.wait_for_frames()
            #frames = pipeline.wait_for_frames()

            time_received = time.time()

            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays

            #depth_frame = hole_filling.process(depth_frame)
            depth = np.asanyarray(depth_frame.get_data())
            rgb = np.asanyarray(color_frame.get_data())

            if frame_encode:
                #encoding_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                _, rgb_enc = cv2.imencode('.png', rgb)#, encoding_params)
                _, depth_enc = cv2.imencode('.png', depth)#, encoding_params)
            else:
                rgb_enc = rgb
                depth_enc = depth
        else:
            if playback_i < len(rgb_enc_stack):
                print(playback_i, ":", len(rgb_enc_stack))
                rgb_enc = rgb_enc_stack[playback_i]
                depth_enc = depth_enc_stack[playback_i]
                playback_i = playback_i + 1
            else:
                break

        if live or phase == "process":
            socket_req.send(rgb_enc.tobytes(), flags=zmq.SNDMORE)
            socket_req.send(depth_enc.tobytes())

            rgb_recv = socket_req.recv()
            rgb_enc = np.frombuffer(rgb_recv, np.uint8)
        else:
            if phase == "record":
                rgb_enc_stack.append(rgb_enc)
                depth_enc_stack.append(depth_enc)
        #else:
        #    rgb_enc = np.frombuffer(rgb_recv, np.uint8)
        if frame_encode:
            rgb_dec = cv2.imdecode(rgb_enc, flags=-1)
        else:
            rgb_dec = rgb_enc.reshape(480, 640, 3)

        #rgb_dec = cv2.cvtColor(rgb_dec, cv2.COLOR_BGR2RGB)

        duration_min = 1. / max_fps
        duration = time.time() - time_received
        if duration_min > duration:
            duration_wait = int(1000 * (duration_min - duration))
        else:
            duration_wait = int(0)

        if phase == "start":
            if cv2.waitKey(1 + duration_wait) == ord('r'):
                print("recording...")
                print("press p for processing")
                phase = "record"

        if phase == "record":
            if cv2.waitKey(1 + duration_wait) == ord('p'):
                print("processing...")
                phase = "process"

        duration = time.time() - time_received
        fps = round(1/duration, 2)

        if live:
            rgb_dec = draw_text_in_rgb(rgb_dec, "fps: " + str(fps))
        else:
            rgb_dec = draw_text_in_rgb(rgb_dec, "fps: " + str(max_fps))

        rgb_dec_res = cv2.resize(rgb_dec, (int(640*win_scale), int(480*win_scale)))

        if live is False and phase == "process":
            #w, h = rgb_dec_res.size
            writer.write((rgb_dec_res * 255).astype(np.uint8))

        cv2.imshow("live", rgb_dec_res) #* np.clip(depth[:, :, None] * depth_scale / 10, 0., 1.))
        cv2.waitKey(1)

        #m4_io.send_array(socket, rgbd, copy=True)
        #rgbd = m4_io.recv_array(socket)

        #print(color_image)
        #print(depth_image)

    if live is False:
        writer.release()
if __name__ == "__main__":
    main()