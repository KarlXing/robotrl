import numpy as np
import cv2
import struct


def recv_bytes(sock):
    bytes_size = (320*240+3)*4;  #image with 320*240 + 3 sonar readings
    data = b''
    while(len(data)<bytes_size):
        # print("length of data is: ", len(data))
        part = sock.recv(bytes_size-len(data))
        data += part
    return data

def decode_bytes(data):
    sonars = struct.unpack('>fff',data[:12])
    img = np.zeros((240,320,3), np.uint8)
    for i in range(240):
        for j in range(320):
            img[i,j] = [data[(i*320+j)*4+3+12], data[(i*320+j)*4+2+12], data[(i*320+j)*4+1+12]]
    return sonars, img

def decode_bytes_rgb(data):
    sonars = struct.unpack('>fff',data[:12])
    img = np.zeros((240,320,3), np.uint8)
    for i in range(240):
        for j in range(320):
            img[i,j] = [data[(i*320+j)*4+1+12], data[(i*320+j)*4+2+12], data[(i*320+j)*4+3+12]]
    return sonars, img

def get_obs(sock):
    data = recv_bytes(sock)
    return decode_bytes(data)

def get_obs_rgb(sock):
    data =recv_bytes(sock)
    return decode_bytes_rgb(data)


def updateNN(model):
    return

def calc_rwd(img):
    return 1

def get_action(img, model):
    return 1500

def save_model(model, path):
    return

def load_model(path):
    return