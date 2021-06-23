import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance 
import math
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings
from sklearn import preprocessing
import time
from playsound import playsound
import geocoder
from twilio.rest import Client
import os
from datetime import datetime
import time
import os

import sys
import warnings
warnings.filterwarnings("ignore")

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


##loading the model from the saved file
with open('model.bin', 'rb') as f_in:
    model1 = pickle.load(f_in)

##functions

def eye_aspect_ratio(eye):
    """Calculates EAR -> ratio length and width of eyes"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculates MAR -> ratio length and width of mouth"""
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    """Calculates PUC -> low perimeter leads to lower pupil"""
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area / (p**2)

def mouth_over_eye(eye):
    """Calculates the MOE -> ratio of MAR to EAR"""
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def sound_alarm(path):
    """play an alarm sound"""
    playsound(path)
    print('Drowsiness Detected!')
    print('Voice Alert Activated!')

def get_current_location(g_maps_url):
    g = geocoder.ip('me')
    #lat = g.latlng[0] + 2.64
    lat = g.latlng[0]
    #long = g.latlng[1] + 1.3424
    lon = g.latlng[1]
    #print(lat, long)
    current_location =  g_maps_url.format(lat, lon)
    return current_location


def send_alert_message(num, current_location):
    # twilio credentials
    account_sid = 'ACf48e37d09d3e70cf8b7629de4b84fc95'
    auth_token = '5852f7598fe260cf19d409b87a747898'
    sender = "+18136027620"
    message = "The driver doesn't seem okay!Last known location: {}".format(current_location)

    client = Client(account_sid, auth_token)
    client.messages.create(
            to=num,
            from_=sender,
            body=message
        )


g_maps_url = "http://maps.google.com/?q={},{}"
	
def model(landmarks):

    features = pd.DataFrame(columns=["EAR","MAR","Circularity","MOE"])

    eye = landmarks[36:68]
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    cir = circularity(eye)
    mouth_eye = mouth_over_eye(eye)

    df = features.append({"EAR":ear,"MAR": mar,"Circularity": cir,"MOE": mouth_eye},ignore_index=True)

    df["EAR_N"] = (df["EAR"]-mean["EAR"])/ std["EAR"]
    df["MAR_N"] = (df["MAR"]-mean["MAR"])/ std["MAR"]
    df["Circularity_N"] = (df["Circularity"]-mean["Circularity"])/ std["Circularity"]
    df["MOE_N"] = (df["MOE"]-mean["MOE"])/ std["MOE"]

    Result = model1.predict(df)
    if Result == 1:
        Result_String = " "
        n=1
        n=n+1
    else:
        Result_String = "Not Drowsy"
        n=0


    return Result_String, df.values,n

def calibration():
    data = []
    cap = cv2.VideoCapture(0)

    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(image, 0)

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            data.append(shape)
            cv2.putText(image,"Calibrating...", bottomLeftCornerOfText, font, fontScale, fontColor,lineType)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("Output", image)
        #time.sleep(10)
        
      	
        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    features_test = []
    for d in data:
        eye = d[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        features_test.append([ear, mar, cir, mouth_eye])
    
    features_test = np.array(features_test)
    x = features_test
    y = pd.DataFrame(x,columns=["EAR","MAR","Circularity","MOE"])
    df_means = y.mean(axis=0)
    df_std = y.std(axis=0)
    
    return df_means,df_std

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


def live():
    cap = cv2.VideoCapture(0)
    data = []
    result = []
    count=0
    alarmc=0
    n=0
    prevn=2
    gpsc=0
    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(image, 0)
        
        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            Result_String, features,n = model(shape)
            #count=count+n
            #print(count)
            print(n)
            cv2.putText(image,Result_String, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)
            data.append (features)
            result.append(Result_String)
            
          

            

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        if n==2 & prevn==2:
            alarmc=alarmc+1
        else: 
            alarmc=0
        prevn=n
        if alarmc>2:
            cv2.putText(image, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            sound_alarm('soundfiles/warning.mp3')
            #sound_alarm('sound files/warning_yawn.mp3')
            alarmc=0
            gpsc=gpsc+1
            if(gpsc>2):
                gpsc=0
                print("Something wrong?")
                SEND_MESSAGE = True
                if SEND_MESSAGE:
                #print("driver", driver)
                # send message to the person's 3 immediate contacts
                    current_location = get_current_location(g_maps_url)
                 # get the contact list of the person
                    num= "+91"+str(sys.argv[1])
                    send_alert_message(num, current_location)

                
                

            

        # Show the image
        cv2.imshow("Output", image)

        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    
    return data,result

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


mean,std = calibration()

features, result = live()
