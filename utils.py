import matplotlib.pyplot as plt
import cv2
import numpy
import torch
import sys


def getColorImage(file):
    image = numpy.array(cv2.imread(file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
    return image

def showDepthImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("depth", gray)

def getDepthImage(file):
    image = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 
    return image

def accurate(predicted, truth):
    print(predicted,"=======\n",truth)
    input("test")
    return True
