#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV
import numpy as np

#there is no lable in 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ian Waters", "Harry Waters", "Jamie Waters", "Claire Waters"]

#function to detect face using OpenCV
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#load OpenCV face detector, I am using LBP which is fast
#there is also a more accurate but slow: Haar classifier
face_cascade = cv2CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

'''
Let's detect multiscale images(some images may be closer to camera than
others. Result is a list of faces
'''
faces = faces_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5):

#if no faces are detected then return to origanal img
    if (len(faces) == 0):
        return None, none

    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image

    return gray[y:y+w, x:x9+h], faces[0]
'''
This function will read all a persons' training images, detect a face from each image
and will return two lists of exactly same size, one list of faces and another list
of lables for each face
'''
def prepare_training_data(data_folder_path):
    
    #get directories in data folder
    dirs = os.listdir(data_folder_path)
    ### setup github

