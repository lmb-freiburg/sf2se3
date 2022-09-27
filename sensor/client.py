
import zmq
import time
import sys
import m4_io.m4 as m4_io
import numpy as np

port = "2308"
context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REQ)
#socket.connect("tcp://sommer-space.de:%s" % port)
socket.connect("tcp://localhost:%s" % port)

#  Do 10 requests, waiting each time for a response
for request in range(1,10):
    print("Sending request ", request,"...")
    #socket.send_string("Hello")
    A = np.ones(shape=(30, 20))
    message = m4_io.send_array(socket, A)
    #  Get the reply.
    message = m4_io.recv_array(socket)
    print("Received reply ", request, "[", message, "]")


import numpy

