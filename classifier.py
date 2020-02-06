import os
import local_config as lc
import pickle
import numpy as np
import face_recognition
import cv2

print('Loading Model..')
players_face_encoding = pickle.load(open(os.path.join(lc.MODEL_DIR, lc.MODEL_FILE), "rb"))
players_name_encoding = pickle.load(open(os.path.join(lc.MODEL_DIR, lc.PLAYERS), "rb"))


def allowed_file(filename):
    '''
    Checks if a given file `filename` is of type image with 'png', 'jpg', or 'jpeg' extensions
    '''
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif'])
    return (('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS))

def prepare_image(image_path):
    '''
    Loads image from the given `image_path` parameter
    '''
    return face_recognition.load_image_file(image_path)

def make_prediction(filename):
    '''
    Predicts a given `filename` file
    '''
    print('Filename is ', filename)
    fullpath = os.path.join(lc.OUTPUT_DIR, filename)

    # Face detection
    img = cv2.imread(fullpath)
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    face = faces[0]
    face_locations_test = [(face[1]+20, face[0]+face[2]-20, face[1]+face[3]-20, face[0]+20)]

    test_data = prepare_image(fullpath)
    encoding_test = face_recognition.face_encodings(test_data, face_locations_test)[0]
    predictions = face_recognition.compare_faces(players_face_encoding, encoding_test)
    return players_name_encoding[np.argmax(predictions)]
