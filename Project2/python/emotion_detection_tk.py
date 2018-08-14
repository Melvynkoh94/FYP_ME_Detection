###set "KERAS_BACKEND=tensorflow"
###python python/facial-expression-recognition-from-stream.py

import numpy as np
import cv2
from keras.preprocessing import image
import h5py
from datetime import datetime
import os
import tkinter as tk #GUI package
import tkinter.filedialog
from tkinter import *

#----------------------------- GUIDE:https://www.youtube.com/watch?v=JrWHyqonGj8
#Tkinter GUI
window = tk.Tk()
window.title("Melvyn FYP Micro-Expression Detection")
window.geometry("800x800")

#LABEL
title = tk.Label(text="HELLO")
title.grid(column=0, row=0)

#BUTTON 1
button1 = tk.Button(text="Choose Video or Image File")
button1.grid(column=0, row=1)

#BUTTON 2
button2 = tk.Button(text="Run Webcam")
button2.grid(column=1, row=1)

window.mainloop() #THIS MUST ALWAYS BE AT THE BOTTOM!!

def select_image():
	#grab a reference to an image panel
	global panelA

	#open a file chooser dialog for user to select an input image
	path = filedialog.askopenfilename()

	#ensure a file path was selected
	if(len(path) > 0):
		new_image = cv2.imread(path)
		new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

		if panelA is None:
			panelA = Label(image=new_image_gray)


#-----------------------------
#opencv initialization
face_cascade = cv2.CascadeClassifier('C:/Users/User/Anaconda3/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0) #real-time streaming using webcam

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json 
model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())	#json format for keras is just the architecture strucutre of the model 
model.load_weights('model/facial_expression_model_weights.h5') #load weights
#HDF5 or h5py is the file type that contains a model/weights in keras
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.mkdir(directory)
      #print("%s created in %s" % (new_folder, new_folder_dir))
	except OSError:
		print ("Error creating directory for "+directory)

cwd = os.getcwd()
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
new_folder = "frames_captured_{0}".format(time_now)
new_folder_dir = "{0}\{1}\{2}".format(cwd, 'python', new_folder)
createFolder(new_folder_dir)

#counter for filename later for faces captured in each frame 
i = 0 

while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	#locations of detected faces

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
		detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
		detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
		detected_face_mini = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
		img_pixels = image.img_to_array(detected_face_mini)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
		
		predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
		max_index = np.argmax(predictions[0])
		
		emotion_captured = emotions[max_index]
		
		#write emotion text above rectangle
		cv2.putText(img, emotion_captured, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
		#saving of frames captured into frames_captured folder
		i+=1
		FaceFileName = "./python/{0}/frame_{1}_{2}.jpg".format(new_folder, i, emotion_captured)
		cv2.imwrite(FaceFileName, detected_face)
		#process on detected face end
		#-------------------------

	cv2.imshow('Real Time Facial Expression Recognition',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()