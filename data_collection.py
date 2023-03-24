#Python code to add new asanas

#Importing necessary libraries
import mediapipe as mp 
import numpy as np 
import cv2 

def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

 #Opens webcam
cap = cv2.VideoCapture(0)

#Name of new asana
name = input("Enter the name of the Asana : ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils


X = []
data_size = 0

#Creating empty list and appending co-ordinates and angle of each asana
while True:
	lst = []

	_, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - res.pose_landmarks.landmark[0].x)
			lst.append(i.y - res.pose_landmarks.landmark[0].y)

		X.append(lst)
		data_size = data_size+1
    #Prompt  if object is not in screen
	else: 
		cv2.putText(frm, "Make Sure Full body visible", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frm)

    #dealy to add pose of new asana
	if cv2.waitKey(1) == 27 or data_size>80:
		cv2.destroyAllWindows()
		cap.release()
		break

#TO save the new label
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)