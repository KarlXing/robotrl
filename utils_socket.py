import numpy as np
import cv2
import struct


def recv_bytes(sock, h, w):
    bytes_size = (w*h)*4;  #image with 320*240 + 3 sonar readings
    data = b''
    while(len(data)<bytes_size):
        # print("length of data is: ", len(data))
        part = sock.recv(bytes_size-len(data))
        data += part
    return data

def decode_bytes(data, h, w):
    #sonars = struct.unpack('>fff',data[:12])
    img = np.zeros((h,w,3), np.uint8)
    for i in range(h):
        for j in range(w):
            img[i,j] = [data[(i*w+j)*4+3], data[(i*w+j)*4+2], data[(i*w+j)*4+1]]
    return None, img
    #return sonars, img

def decode_bytes_rgb(data, h, w):
    #sonars = struct.unpack('>fff',data[:12])
    img = np.zeros((h,w,3), np.uint8)
    for i in range(h):
        for j in range(w):
            img[i,j] = [data[(i*w+j)*4+1], data[(i*w+j)*4+2], data[(i*w+j)*4+3]]
    return None, img
    #return sonars, img

def get_obs(sock, h, w):
    data = recv_bytes(sock, h, w)
    return decode_bytes(data, h, w)

def get_obs_rgb(sock, h, w):
    data =recv_bytes(sock, h, w)
    return decode_bytes_rgb(data, h, w)


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