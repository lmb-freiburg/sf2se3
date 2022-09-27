Having a look at the environment of error:

1. print to get reason for error
2. set breakpoint only in case of reason
3. look at environment

# Frequently Asked Questions

- [Pose Estimate Empty Masks](#pose-estimate-empty-masks)



## magma cuda

python: /opt/conda/conda-bld/magma-cuda102_1583546904148/work/interface_cuda/interface.cpp:901: void magma_queue_create_from_cuda_internal(magma_device_t, cudaStream_t, cublasHandle_t, cusparseHandle_t, magma_queue**, const char*, const char*, int): Assertion `queue->dCarray__ != __null' failed.
-> after set sflow2se3 meta

            print("INFO :: eval :: objs center pred se3 pt3d_1")
            #data_pred_se3['objs_center_3d_0'][0] = torch.Tensor([0., 0.1, 0.]).type(data_pred_se3['objs_center_3d_0'].dtype).to(data_pred_se3['objs_center_3d_0'].device)
            data_pred_se3['objs_center_3d_1'][0] = o4geo_se3_transf.pts3d_transform(
                data_pred_se3['objs_center_3d_0'][:1, :, None, None], torch.linalg.inv(data_pred_se3['objs_params']['se3']['se3'])[0, :1])[0, :, 0, 0]



## Pose Estimate Empty Masks
    third_party/rigidmask/utils/dydepth.py :: pose_estimate: 

    if num_samp > 0:
        submask = np.random.choice(range(tmphp0.shape[1]), num_samp)
        tmphp0 = tmphp0[:,submask]
        tmphp1 = tmphp1[:,submask]
    else:
        tmphp0 = np.array([])
        tmphp1 = np.array([])
## Pose Refinement Too Few Points:

#### error
      File "/home/leo/sflow2rigid3d/eval.py", line 586, in main
        data_pred_se3.update(net_rigidmask.forward(data_pred_sflow))
      File "/home/leo/sflow2rigid3d/venv/lib/python3.8/site-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
        return func(*args, **kwargs)
      File "/home/leo/sflow2rigid3d/tensor_operations/retrieval/sflow2rigid/rigid_rigidmask.py", line 229, in forward
        scene_type, T01_c, R01, RTs = ddlib.rb_fitting(bgmask_np, polarmask_label_np, disp, flow, occ, K0, K1, bl, parallax_th=4,
      File "/home/leo/sflow2rigid3d/third_party/rigidmask/utils/dydepth.py", line 498, in rb_fitting
        _,rvec, T01_cx=cv2.solvePnP(reg_flow_P.T[obj_mask,np.newaxis],
    cv2.error: OpenCV(4.5.3) /tmp/pip-req-build-afu9cjzs/opencv/modules/calib3d/src/solvepnp.cpp:831: error: (-215:Assertion failed) ( (npoints >= 4) || (npoints == 3 && flags == SOLVEPNP_ITERATIVE && useExtrinsicGuess) || (npoints >= 3 && flags == SOLVEPNP_SQPNP) ) && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) in function 'solvePnPGeneric'

#### solution in: sflow2rigid3d/third_party/rigidmask/utils/dydepth.py :: rb_fitting
    if obj_mask.sum() >= 4:
                # extra checking because of aligned_mask (aligned triangulation) there is another restriction of points
                _,rvec, T01_cx=cv2.solvePnP(reg_flow_P.T[obj_mask,np.newaxis],
                                       hp1[:2].T[obj_mask,np.newaxis], K0, 0,
                                       flags=cv2.SOLVEPNP_DLS)
                _,rvec, T01_cx=cv2.solvePnP(reg_flow_P.T[obj_mask,np.newaxis],
                                           hp1[:2].T[obj_mask,np.newaxis], K0, 0,rvec, T01_cx,useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
                R01x = cv2.Rodrigues(rvec)[0].T
                T01_cx = -R01x.dot(T01_cx)[:,0]
                if T01_cx is None:
                    RT01=None
                else:
                    RT01 = [R01x, T01_cx]
            else:
                RT01=None

## Different Sequence Length:
#### error
    File "/home/sommerl/sflow2rigid3d/tensor_operations/eval.py", line 141, in eval_data
        se3_mat = o4geo_se3_reg.calc_pointsets_registration_from_corresp3d(data_pred_seq_poses_xyz[None,], data_gt_seq_poses_xyz[None,])[0]
      File "/home/sommerl/sflow2rigid3d/tensor_operations/geometric/se3/registration.py", line 411, in calc_pointsets_registration_from_corresp3d
        U, S, V = torch.svd(torch.matmul(pts2_norm.permute(0, 2, 1), pts1_norm))

## python3.6 does not allow circular imports

## DEBUG RAFT3D does not work:
RAFT3D sflow 100: 
-> NVIDIA TITAN X (WITHOUT GTX) 

## DEBUG NGRANSAC REQUIRES OPENCV
ubuntu1804
libopencv-core3.2
ubuntu2004
libopencv-core4.2

Installing 3.4.2
## rigid

File "/home/sommerl/sflow2rigid3d/third_party/rigidmask/models/submodule.py", line 709, in F_ngransac  
import ngransac  
ImportError: libopencv_core.so.4.5: cannot open shared object file: No such file or directory  

-> reinstall 3.4.2 and rm -rf /third_party/rigidmask/models/ngransac/build/*  

Installing 4.5.0  
File "/home/sommerl/sflow2rigid3d/third_party/rigidmask/models/VCNplus.py", line 672, in forward  
    rotx,transx,Ex = F_ngransac(hp0x,hp1x,Kinv.inverse(),rand,unc_occ, Kn = Kinv_n.inverse(),cv=False)  
  File "/home/sommerl/sflow2rigid3d/third_party/rigidmask/models/submodule.py", line 709, in F_ngransac  
    import ngransac  
ImportError: /misc/student/sommerl/venvs/venv_py38_2004/lib/python3.8/site-packages/ngransac-0.0.0-py3.8-linux-x86_64.egg/ngransac.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN2cv3MatC1Ev  


## cuda does not support gcc/g++ version:  

`export CC=gcc-8`  
`export CXX=g++-8`  


## cant open libcudart.so.10.1  

it is not:  
-> cuda version of torch 10.2 / 11.1  
-> python3.8  
-> cuda 11.X  

it could be:
-> ubuntu2004
-> 

nvcc --version=11.2
torch.version.cuda=10.2
    -> 

## debug libopencv_core 3.2. not found

-> we dont want it to use 3.2. but our 4.5.x
probably lib not found with ld command is issue


## cv2 arrow wrong type:
cv2.error: OpenCV(4.5.3) :-1: error: (-5:Bad argument) in function 'arrowedLine'
> Overload resolution failed:
>  - Can't parse 'pt2'. Sequence item with index 0 has a wrong type
>  - Can't parse 'pt2'. Sequence item with index 0 has a wrong type

-> problem float("NaN") in arguments

## eof file erro: ran out of input using torch.load

models were probably not downloaded 

## Key Error se3_objects_count

Error:  
>   metrics["se3_objects_count"],  
> KeyError: 'se3_objects_count'  

Problem:  
    data was not read


## rigidmask

B/0026 3:9 (if ran again error does not appear)
/home/leo/sflow2rigid3d/third_party/rigidmask/utils/dydepth.py:451: RuntimeWarning: Mean of empty slice.
  print('[BG Fitting] mean pp/flow: %.1f/%.1f px'%(parallax_mag[bgmask_pred].mean(), flow_mag[bgmask_pred].mean()))
/home/leo/sflow2rigid3d/venv/lib/python3.8/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "/home/leo/sflow2rigid3d/eval.py", line 1042, in <module>
    main()
  File "/home/leo/sflow2rigid3d/eval.py", line 593, in main
    data_pred_se3.update(net_rigidmask.forward(data_pred_sflow))
  File "/home/leo/sflow2rigid3d/venv/lib/python3.8/site-packages/torch/autograd/grad_mode.py", line 28, in decorate_context
    return func(*args, **kwargs)
  File "/home/leo/sflow2rigid3d/tensor_operations/retrieval/sflow2rigid/rigid_rigidmask.py", line 268, in forward
    polarmask_not_assigned[:, 0] += polarmask_onehot[None, len(se3_objs)-1].to(device)
IndexError: index 1 is out of bounds for dimension 1 with size 1
labels tensor([0])
count tensor([518400])
[BG Fitting] mean pp/flow: nan/nan px
[BG Update] cam trans mag: 1.00
label -1 tensor(0, device='cuda:0') 

-> ensure that min label = 0, so that number masks / number se3s is not messed up



##  virtualenv not in home
pip, pip3 and pip3.8 are installed in '/home/sommerl/.local/bin' which is not on PATH.
WARNING: The scripts jupyter, jupyter-migrate and jupyter-troubleshoot are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script jsonschema is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script pygmentize is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script jupyter-trust is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts jupyter-kernel, jupyter-kernelspec and jupyter-run are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts iptest, iptest3, ipython and ipython3 are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
 WARNING: The script send2trash is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script jupyter-nbconvert is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script tqdm is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts f2py, f2py3 and f2py3.8 are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts jupyter-bundlerextension, jupyter-nbextension, jupyter-notebook and jupyter-serverextension are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script jupyter-server is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script pyjson5 is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script pybabel is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script wheel is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts convert-caffe2-to-onnx and convert-onnx-to-caffe2 are installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script tabulate is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script jupyter-nbclassic is installed in '/home/sommerl/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.

  ERROR: Failed to restore /home/sommerl/.local/bin/convert-caffe2-to-onnx
  ERROR: Failed to restore /home/sommerl/.local/bin/convert-onnx-to-caffe2
  ERROR: Failed to restore /home/sommerl/.local/lib/python3.8/site-packages/caffe2/


error: from pytorch3d import _C  : undefined symbol : _ZNK2at6Tensor7is_cudaEv 

solution: 
pip3 uninstall pytorch3d
pip3 install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.0

error: zmq module not found (but installed pyzmq)
-> probably install of pyzmq is somewhere else then ipython/tornado

solution: reinstall python3 python3-pip solved it