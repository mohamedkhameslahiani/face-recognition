import cv2
import pandas as pd
import numpy as np
import face_recognition
import os
from datetime import datetime , date
import time
# from PIL import ImageGrab
 
path = r'C:\Users\CHINKO\Desktop\Face\Employees_Images'
Employees_Images = []
Employees_Names = []
myList = os.listdir(path)
#print(myList)


#1- Getting the picture from the root directory :
for picture in myList:
    Normal_Image = cv2.imread(f'{path}/{picture}')
    Employees_Images.append(Normal_Image)
    Employees_Names.append(os.path.splitext(picture)[0])  # splittext to split in root and extension
print('Your employees are : ', Employees_Names)



#2- Find the Encodings of the Employees_Images :
def findEncodings(Employees_Images):
    encodeList = []
    for img in Employees_Images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # COnvert image to RGB 
        encode = face_recognition.face_encodings(img)[0] 
        encodeList.append(encode)
    return encodeList
 
    
def get_attendance_datetime():
    # Use current date to get a csv file name.
    return "Real Time Attendance of " + str(date.today().strftime("%A, %d. %B %Y")) 
    
attendance_today = get_attendance_datetime()
# Building the live Attendance Tracking Sheet :
def MarkAttendance(name):
    with open(r"C:\Users\CHINKO\Desktop\Face/"+ attendance_today + ".csv" ,'a+') as csv: #a+ :to create it if not already done 
        myDataList = csv.readlines()
        names_list = []
        for line in myDataList:
            entry = line.split(',')
            names_list.append(entry[0])
        if name not in names_list:  # if the name is not already present , get me his name and time of arrival,(to avoid writing employee twice)
            now = datetime.now()
            date_String = now.strftime('%H:%M:%S')
            csv.writelines(f'\n{name},{date_String}')
 
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr
 
encodeListKnown = findEncodings(Employees_Images)
print('The Encoding is Complete for the ' , len(encodeListKnown) , ' Images of Employees Successefully ! ')




# Starting the Live Video Capturing : 
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Cap_DSHOW : DirectShow (via videoInput) 
 
while True:
    success, img = video.read()
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # Help speed up the process : (0,0) for pixel size
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)  # Find lcoations of faces when having multiple faces in the video
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame) # Build the encodings of the detected faces from their locations
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame): # grab one face location and its encoding:
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        Face_Distance = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(Face_Distance)
        winning_match_Index = np.argmin(Face_Distance)
 
       
            
        # Set condition for Unknown people :
        if Face_Distance[winning_match_Index] < 0.60:
            name = Employees_Names[winning_match_Index].upper()
            MarkAttendance(name)
        else: name = 'Unknown Person'
        
        print(name)
        
        y1,x2,y2,x1 = faceLoc
        #y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(90,90,216),3)   # (0,255,0) for the color and 2 for the thickness
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(90,90,216),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) #name have to be string , (255,255,255) for color of text and 2 for thikness
        MarkAttendance(name)
        #time.sleep(5)
 
    cv2.imshow('Real Time Attendance with Employee Recognition',img)
    # To break Streaming press : q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    
    
    
        
        
        
 # 3 :  Shutdiown Camera:
video.release()


cv2.destroyAllWindows()



attendance = pd.read_csv(r"C:\Users\CHINKO\Desktop\Face/"+ attendance_today + ".csv", 
header=None)

attendance.columns = ['Name Of The Employee' , 'Time Of Arrival']

#attendance.drop_duplicates(keep='first', inplace=True)

att_pivot = attendance.pivot(columns='Name Of The Employee' ,  values='Time Of Arrival')

att_pivot_final = att_pivot.apply(lambda x: pd.Series(x.dropna().values)).fillna('')

att_pivot_final.drop_duplicates(keep='first' , inplace=True)

att_pivot_final.to_excel( attendance_today + '.xlsx' ,  index=False , sheet_name = 'Employee Attendance')

os.remove(r"C:\Users\CHINKO\Desktop\Face/"+ attendance_today + ".csv" )