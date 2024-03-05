# %%
import numpy as np
import cv2
import sys
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
from PIL import ImageTk
import tkinter as tk
from tkinter import *

#opencv initialization

face_cascade = cv2.CascadeClassifier(r"C:\Users\DIVYA SRI\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)
show_text=[0]
emotion= ''

#face expression recognizer initialization
from tensorflow.keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights



emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
if(sys.argv[1]=='m'):
    emoji_dist={'angry':"./emojis1/angry.png",'disgust':"./emojis1/disgusted.png",'fear':"./emojis1/fearful.png",'happy':"./emojis1/happy.png",'neutral':"./emojis1/neutral.png",'sad':"./emojis1/sad.png",'surprise':"./emojis1/surpriced.png"}
elif(sys.argv[1]=='w'):
    emoji_dist={'angry':"./emojis2/angry.jpeg",'disgust':"./emojis2/angry.jpeg",'fear':"./emojis2/fearful.jpeg",'happy':"./emojis2/happy.jpeg",'neutral':"./emojis2/neutral.jpeg",'sad':"./emojis2/sad.jpeg",'surprise':"./emojis2/surpriced.jpeg"}

while(True):
    ret, img = cap.read()
#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#print(faces) #locations of detected faces

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image

        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)

        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels) #store probabilities of 7 expressions

        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        show_text[0]=max_index

        emotion = emotions[max_index]
        #print(emotion)

        #write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    #print(emotion)  
        frame2=cv2.imread(emoji_dist[emotion])
        cv2.imshow('image', frame2)
    #print(frame2)
    

#process on detected face end


        cv2.imshow('img',img)

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break

#kill open cv things
cap.release()
cv2.destroyAllWindows()


# %%