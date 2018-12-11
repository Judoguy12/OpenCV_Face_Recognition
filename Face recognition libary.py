import face_recognition
import picamera
import numpy as np

camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)

print("Loading known face image(s)")
jamie_image = face_recognition.load_image_file("/home/pi/OpenCV_Face_Recognition/training-data/s3/1.JPG")
jamie_image_encoding = face_recognition.face_encodings(jamie_image)[0]

#int var
face_locations = []
face_encodings = []

while True:
    print("Capturing image.")
    camers.capture(output, format="rgb")

    face_locations = face_recognitions.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([jamie_image_encoding], face_encoding)
        name = "<Unknown Person>"

        if match[0]:
            name = "Jamie Waters"

        print("I see someone named {}!".format(name))
