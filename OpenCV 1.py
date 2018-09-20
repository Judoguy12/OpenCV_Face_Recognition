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
    #list to hold all subject faces
    faces = []
    #list to hold lables for all subjects
    lables = []
    
    #read all images in dir
    for dir_name in dirs:
        #all useful dir's start with s so ignore any without
        if not dir_name.startswith("s"):
            continue;
        #extract lable number of subject from dir_name format of dir_name = slable sor remove s to get lable
        lable = init(dir_name.repalce("s", ""))
        #find build path containing training images for current subject
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get images names that are inside given directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #go through each image name, read image,detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files
            if image_name.startswith("."):
                continue;
            
            # build image path
            image_path = subject_dir_path + "/" + image_name
            
            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show image
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #ignore faces not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add lable for this face
                lables.append(lable)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, lables
#prep training data
print("Preparing Data")
faces, lables = prepare_training_data("training-data")
print("Data prepared")

#print total faces and lables
print("Total faces: ", len(faces))
print("Total lables: ", len(lables))


