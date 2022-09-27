try:
    import torch
except ModuleNotFoundError:
    # Error handling
    pass

try:
    import zmq
except ModuleNotFoundError:
    # Error handling
    pass

import numpy as np
import json


def readlines_txt(fpath):
    a_file = open(fpath, "r")
    lines = a_file.read().splitlines()
    return lines

def save_json(fpath, val):
    a_file = open(fpath, "w")
    json.dump(val, a_file)
    a_file.close()

def read_json(fpath):
    a_file = open(fpath, "r")
    return json.load(a_file)

def save_torch_as_nptxt(torch_out, fpath):
    np_out = torch_out.numpy()
    np.savetxt(fpath, np_out)


def read_nptxt_as_torch(fpath):
    np_in = np.loadtxt(fpath)
    torch_in = torch.from_numpy(np_in)
    return torch_in


def dtype_numpy2torch(dtype_numpy):

    numpy_to_torch_dtype_dict = {
        np.bool: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }

    return numpy_to_torch_dtype_dict[dtype_numpy]


def dtype_torch2numpy(dtype_torch):
    torch_to_numpy_dtype_dict = {
        torch.bool: bool,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }

    return torch_to_numpy_dtype_dict[dtype_torch]


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A.copy(), flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def send_tensor(socket, A, flags=0, copy=True, track=False):
    A = A.detach().cpu.numpy()
    send_array(socket, A, flags=0, copy=True, track=False)

def recv_tensor(socket, flags=0, copy=True, track=False):
    A = recv_array(socket, flags=0, copy=True, track=False)
    A = torch.from_numpy(A)
    return A