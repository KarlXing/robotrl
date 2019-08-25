import cv2
import numpy as np
import threading
import json
from PIL import Image
from PIL import ImageOps

class Rewarder:
    def __init__(self, w, h):
        super(Rewarder, self).__init__()
        # self.img_w = 80  # assume image size(240x320)
        # self.img_h = 60

        # self.reward_w = 20  # for calculating rewards
        # self.reward_h = 8
        self.img_w = w
        self.img_h = h

        if self.img_w == 320:
            self.reward_w = 80
            self.reward_h = 32
        elif self.img_w == 80:
            self.reward_w = 20
            self.reward_h = 8
        else:
            print("Invalid rewarder initialization\n")

        self.sonar_thresh = 8
        self.grass_thresh = 0.5

        self.reward_grass = -1.0
        self.reward_obstacle = -2.0
        self.reward_road = 0.5
        self.reward_center = 0.5


    def reward(self, img, sonars):
        for sonar in sonars:
            if sonar <self.sonar_thresh:
                return self.reward_obstacle
        # assume image size (240, 320)
        target_img = img[self.img_h-self.reward_h:, int((self.img_w - self.reward_w)/2): int((self.img_w + self.reward_w)/2)]
        if np.sum(target_img) < (self.grass_thresh * 255 * self.reward_w * self.reward_h):
            return self.reward_grass
        else:
            num_road = np.sum(img > 0)
            ratio_left_road = np.sum(img[:, :int(self.img_w/2)] > 0)/num_road
            ratio_right_road = np.sum(img[:, int(self.img_w/2):] > 0)/num_road

            road_bonus = self.reward_center * num_road/(self.img_w * self.img_h) * (1 - abs(ratio_left_road - ratio_right_road))
            #center_bonus = (1 - np.sum(mirror_img != img) / (self.img_w * self.img_h)) ** 2 * self.reward_center
            return self.reward_road + road_bonus


class ImgProcessor:
    def __init__(self, path = None):
        super(ImgProcessor, self).__init__()
        self.l1 = np.array([0,0,80])  
        self.h1 = np.array([50,50,254])

        self.l2 = np.array([90,30,5])
        self.h2 = np.array([120,255,120])

        self.l3 = np.array([80,0,100])
        self.h3 = np.array([175,60,170])

        self.l4 = np.array([160, 160, 160])
        self.h4 = np.array([255, 255, 255])

        self.img_path = path if path is not None else "/Users/karl/Documents/Notebooks/RobotRL/Record/"

    # def save_img(self, img, step):
    #     cv2.imwrite(img_path+str(step)+".png", img)

    def process_img(self, img):
        img = self.blur(img)  #blur
        img = self.bgr2hsv(img)  #hsv
        img = self.hsv2gray(self.cut(img, self.l1, self.h1) + self.cut(img, self.l2, self.h2) + self.cut(img, self.l3, self.h3) + self.cut(img, self.l4, self.h4)) #gray
        img = self.erode(self.dilate(img, 2), 2) #fill

        _, thresh = cv2.threshold(img, 0, 255, 0)  # return biggest tour
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        res_img = np.zeros_like(img)
        if (len(contours) != 0):
            contour = max(contours, key = cv2.contourArea)
            res_img = cv2.drawContours(res_img, [contour], 0, (255), thickness=cv2.FILLED)
        return res_img


    def bgr2hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def rgb2hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def dilate(self, img, k, r=1):
        kernel = np.ones((3,3), np.uint8)/r
        return cv2.dilate(img, kernel, iterations=k) 
    
    def erode(self, img, k, r=1):
        kernel = np.ones((3,3), np.uint8)/r
        return cv2.erode(img, kernel, iterations=k) 
    
    def blur(self, img, size=5):
        return cv2.GaussianBlur(img,(5,5),-1)
    
    def show(self, imgs):
        cv2.startWindowThread()
        for i in range(len(imgs)):
            cv2.imshow('image'+str(i),imgs[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def cut(self, img, lower_bound, upper_bound):
        mask = cv2.inRange(img, lower_bound, upper_bound)
        return cv2.bitwise_and(img,img, mask= mask)
    
    def hsv2gray(self, img):
        return img[:,:,2]

class Saver(threading.Thread):
    def __init__(self, path, img, sonars):
        super(Saver, self).__init__()
        self.path = path
        self.img = img
        self.sonars = sonars

    def run(self):
        data = {}
        img = {}
        for k, v in self.img.items():
            img[k] = v.tolist()
        data['img'] = img
        data['sonar'] = self.sonars
        with open(self.path, 'w') as f:
            json.dump(data, f)


def show(imgs):
    cv2.startWindowThread()
    for i in range(len(imgs)):
        cv2.imshow('image'+str(i),imgs[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


