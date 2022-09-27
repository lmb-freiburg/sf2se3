import zmq
import time
import sys
import m4_io.m4 as m4_io
import numpy as np

port = "2308"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

while True:
    #  Wait for next request from client
    message = m4_io.recv_array(socket)
    #message = socket.recv()
    print("Received request: ", message)

    time.sleep(1)
    m4_io.send_array(socket, message)
    #socket.send_string("World from %s" % port)