#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV
import numpy as np
import matplotlib.pyplot as plt

#there is no lable in 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ian Waters", "Harry Waters", "Jamie Waters", "Claire Waters"]

#function to detect face using OpenCV
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    '''
    Let's detect multiscale images(some images may be closer to camera than
    others. Result is a list of faces
    '''
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return to origanal img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image

    return gray[y:y+w, x:x+h], faces[0]
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
    labels = []
    
    #read all images in dir
    for dir_name in dirs:
        #all useful dir's start with s so ignore any without
        if not dir_name.startswith("s"):
            continue;
        #extract lable number of subject from dir_name format of dir_name = slable sor remove s to get lable
        label = int(dir_name.replace("s", ""))
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
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
#prep training data
print("Preparing Data")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and lables
print("Total faces: ", len(faces))
print("Total lables: ", len(labels))

#create our face recogniser
face_recogniser = cv2.face.createLBPHFaceRecogniser()
#train our face recogniser
face_recogniser.train(faces, np.array(lables))
#function to draw rectangle on image according to given (x, y) corrdinants and given width & height
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
#this recognises person then draws rectangle arround it then adds text with name
    
def predict(test_img):
    #make copy to not alter origional
    img = test_img.copy()
    #detect face
    face, rect = detect_face(img)
    #predict face
    lable = face_recogniser.predict(face)
    #get name of lable returned
    lable_text = subjects[lable]
    #draw rectangle arround face
    draw_rectangle(img, rect)
    #draw name of person
    draw_text(img, lable_text, rect[0], rect[1]-5)
    return img


print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
