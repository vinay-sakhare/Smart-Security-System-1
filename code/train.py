# -*- coding: utf-8 -*-
import face_recognition
import tkinter as tk
from tkinter import *
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import dlib
import argparse
from imutils import paths
import pickle
from imutils.video import VideoStream
import imutils


window = tk.Tk()
window.title("Face_Recogniser")
window.geometry('1920x1080')
window.configure(background='#D9FFDC')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

ap = argparse.ArgumentParser()

ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
               help='path to weights file')
args = ap.parse_args()


#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))

message = tk.Label(window, text="Smart Security System" ,bg="#176AE3"  ,fg="white"  ,width=60  ,height=3,font=('monserrat', 30)) 
message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="white"  ,bg="#189CED" ,font=('monserrat', 15) ) 
lbl.place(x=400, y=200)

txt = tk.Text(window,height=2, width=30,bg="#189CED" ,fg="white",font=('monserrat', 15))
txt.place(x=700, y=200)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="white"  ,bg="#189CED"    ,height=2 ,font=('monserrat', 15)) 
lbl2.place(x=400, y=300)

txt2 = tk.Text(window,height=2, width=30,bg="#189CED"  ,fg="white",font=('monserrat', 15))
txt2.place(x=700, y=300)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="white"  ,bg="#189CED"  ,height=2 ,font=('monserrat', 15)) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="#189CED"  ,fg="white"  ,width=30  ,height=2, activebackground = "#189CED" ,font=('monserrat', 15)) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Presence : ",width=20  ,fg="white"  ,bg="#189CED"  ,height=2 ,font=('monserrat', 15)) 
lbl3.place(x=400, y=500)


message2 = tk.Label(window, text="" ,fg="white"   ,bg="#189CED",activeforeground = "white",width=30  ,height=5  ,font=('monserrat', 15)) 
message2.place(x=700, y=500)


def clear():
	txt.delete(0, 'end')    
	res = ""
	message.configure(text= res)

def clear2():
	txt2.delete(0, 'end')    
	res = ""
	message.configure(text= res)    
	
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass 
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass 
	return False
 
def TakeImages():        
	Id=(txt.get("1.0",'end-1c'))
	name=(txt2.get("1.0",'end-1c'))
	os.makedirs(os.path.join('dataset', name ))
	if(is_number(Id) and name.isalpha()):
		cam = cv2.VideoCapture(0)
		#harcascadePath = "haarcascade_frontalface_default.xml"
		#detector=cv2.CascadeClassifier(harcascadePath)
		sampleNum=0
		while(True):
			ret, img = cam.read()
			
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			hog_face_detector = dlib.get_frontal_face_detector()
			#faces = detector.detectMultiScale(gray, 1.3, 5)
			faces_hog = hog_face_detector(gray, 1)
			for face in faces_hog:
				x = face.left()
				y = face.top()
				w = face.right() - x
				h = face.bottom() - y

				# draw box over face
				cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)        
				#incrementing sample number 
				sampleNum=sampleNum+1
				#saving the captu face in the dataset folder TrainingImage
				str_nm=name+"_"+Id+'_'+str(sampleNum)+".jpg"
				cv2.imwrite(os.path.join('dataset',name,str_nm), img)
				#display the frame
				cv2.imshow('frame',img)    
                            
                                
                                
			#wait for 5 miliseconds 
			if cv2.waitKey(5) & 0xFF == ord('q'):
				break
			# break if the sample number is morethan 200
			elif sampleNum>99:
				break
		cam.release()
		cv2.destroyAllWindows() 
		res = "Images Saved for ID : " + Id +" Name : "+ name
		row = [Id , name]
		filename=os.path.join('PersonDetails','PersonDetails.csv')
		with open(filename,'a+') as csvFile:
			headers = ['Id','Name']
			writer = csv.DictWriter(csvFile, delimiter=',', fieldnames=headers)
			writer = csv.writer(csvFile)
			writer.writerow(row)
		csvFile.close()
		message.configure(text= res)
	else:
		if(is_number(Id)):
			res = "Enter Numeric Id"
			message.configure(text= res)
		if(name.isalpha()):
			res = "Enter Alphabetical Name"
			message.configure(text= res)
	
def TrainImages():
	# construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        #ap.add_argument("-i", "--dataset", required=True,
        #	help="path to input directory of faces + images")
        #ap.add_argument("-e", "--encodings", required=True,
        #	help="path to serialized db of facial encodings")
        ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
        args = vars(ap.parse_args())
        dataset="dataset"
        #encodings="encodings.pickle"
        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(dataset))

        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(i + 1,
                        len(imagePaths)))
                res="[INFO] processing image {}/{}".format(i + 1,
                        len(imagePaths))
                
                name = imagePath.split(os.path.sep)[-2]

                # load the input image and convert it from RGB (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                boxes = face_recognition.face_locations(rgb,
                        model=args["detection_method"])

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)

                # loop over the encodings
                for encoding in encodings:
                        # add each encoding + name to our set of known names and
                        # encodings
                        knownEncodings.append(encoding)
                        knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

        res = "Image Trained"#+",".join(str(f) for f in Id)
        message.configure(text= res)
        print("Images Trained")

def TrackImages():
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        #ap.add_argument("-e", "--encodings", required=True,
        #	help="path to serialized db of facial encodings")
        #ap.add_argument("-o", "--output", type=str,
        #	help="path to output video")
        ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
        ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
        args = vars(ap.parse_args())
        output="output/demo_webcam.avi"
        # load the known faces and embeddings
        print("[INFO] loading encodings...")
        data = pickle.loads(open("encodings.pickle", "rb").read())

        # initialize the video stream and pointer to output video file, then
        # allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        writer = None
        time.sleep(2.0)
        frame_number=0
        # loop over frames from the video file stream
        while True:
                print(frame_number)
                frame_number=frame_number+1
                # grab the frame from the threaded video stream
                frame = vs.read()
                
                
                # convert the input frame from BGR to RGB then resize it to have
                # a width of 750px (to speedup processing)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame, width=750)
                r = frame.shape[1] / float(rgb.shape[1])

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb,
                        model=args["detection_method"])
                
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                
                
                
                names = []
                # loop over the facial embeddings
                for encoding in encodings:
                        # attempt to match each face in the input image to our known
                        # encodings
                        matches = face_recognition.compare_faces(data["encodings"],
                                encoding)
                        name = "Unknown"
                        

                        # check to see if we have found a match
                        if True in matches:
                                # find the indexes of all matched faces then initialize a
                                # dictionary to count the total number of times each face
                                # was matched
                                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                                counts = {}
                                
                        
                                # loop over the matched indexes and maintain a count for
                                # each recognized face face
                                for i in matchedIdxs:
                                        name = data["names"][i]
                                        counts[name] = counts.get(name, 0) + 1
                                        

                                # determine the recognized face with the largest number
                                # of votes (note: in the event of an unlikely tie Python
                                # will select first entry in the dictionary)
                                
                                name = max(counts, key=counts.get)
                                
                                
                                #print(counts)
                                if counts[name]<95:
                                        name="unknown"
                                
                        
                        
                        #loop
                        else:
                                name="unknown"
                                print(name)

                        # update the list of names
                        names.append(name)
                        

                # loop over the recognized faces
                for ((top, right, bottom, left), name) in zip(boxes, names):
                        # rescale the face coordinates
                        top = int(top * r)
                        right = int(right * r)
                        bottom = int(bottom * r)
                        left = int(left * r)

                        # draw the predicted face name on the image
                        cv2.rectangle(frame, (left, top), (right, bottom),
                                (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)

                # if the video writer is None *AND* we are supposed to write
                # the output video to disk initialize the writer
                if writer is None and output is not None:
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(output, fourcc, 20,
                                (frame.shape[1], frame.shape[0]), True)

                # if the writer is not None, write the frame with recognized
                # faces t odisk
                if writer is not None:
                        writer.write(frame)

                # check to see if we are supposed to display the output frame to
                # the screen
                if args["display"] > 0:
                        cv2.imshow("Frame", frame)
                        key = cv2.waitKey(1) & 0xFF

                        # if the `q` key was pressed, break from the loop
                        if key == ord("q"):
                                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        # check to see if the video writer point needs to be released
        if writer is not None:
                writer.release()

        ts = time.time()      
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        monserrattamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour,Minute,Second=monserrattamp.split(":")
        fileName="Security_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
        #attendance.to_csv(os.path.join("Security",fileName),index=False)

        

        # Final insert statement
        #cam.release()
        cv2.destroyAllWindows()
	#print(attendance)
       # res=attendance
       # message2.configure(text= res)

  
clearButton = tk.Button(window, text="X", command=clear  ,fg="white"  ,bg="#189CED"  ,width=1 ,height=1 ,activebackground = "#189CED" ,font=('monserrat', 15))
clearButton.place(x=1100, y=200)
clearButton2 = tk.Button(window, text="X", command=clear2  ,fg="white"  ,bg="#189CED"  ,width=1  ,height=1, activebackground = "#189CED" ,font=('monserrat', 15))
clearButton2.place(x=1100, y=300)    
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="white"  ,bg="#189CED"  ,width=20  ,height=3, activebackground = "#189CED" ,font=('monserrat', 15))
takeImg.place(x=200, y=700)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="white"  ,bg="#189CED"  ,width=20  ,height=3, activebackground = "#189CED" ,font=('monserrat', 15))
trainImg.place(x=500, y=700)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="white"  ,bg="#189CED"  ,width=20  ,height=3, activebackground = "#189CED" ,font=('monserrat', 15))
trackImg.place(x=800, y=700)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="#189CED"  ,width=20  ,height=3, activebackground = "#189CED" ,font=('monserrat', 15))
quitWindow.place(x=1100, y=700)
 
window.mainloop()
