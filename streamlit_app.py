import streamlit as st
import cv2
import pandas as pd
import numpy as np
import face_recognition
from datetime import datetime, date
import os
from threading import Thread
from queue import Queue

# Function to find the encodings of employee images
def find_encodings(employees_images):
    encode_list = []
    for img in employees_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Function to mark attendance
def mark_attendance(name, attendance_today):
    with open(f"./{attendance_today}.csv", 'a+') as csv:
        my_data_list = csv.readlines()
        names_list = [entry.split(',')[0] for entry in my_data_list]
        if name not in names_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            csv.writelines(f'\n{name},{date_string}')

# Function to get the current date as a string
def get_attendance_datetime():
    return date.today().strftime("%Y-%m-%d")

# Streamlit app
def main():
    st.title("Employee Recognition Attendance System")

    # Load employee images
    path = r'C:\Users\CHINKO\Desktop\Face\Employees_Images'
    employees_images = [cv2.imread(os.path.join(path, picture)) for picture in os.listdir(path)]

    # Find encodings of employee images
    encode_list_known = find_encodings(employees_images)

    # Variable for attendance_today
    attendance_today = get_attendance_datetime()

    # Getting the picture from the root directory:
    Employees_Images = []
    Employees_Names = []

    for picture in os.listdir(path):
        Normal_Image = cv2.imread(f'{path}/{picture}')
        Employees_Images.append(Normal_Image)
        Employees_Names.append(os.path.splitext(picture)[0])

    print('Your employees are: ', Employees_Names)

    # Video capture
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Create a queue for communication between threads
    frame_queue = Queue()

    # Function to update the frame queue
    def update_frame_queue():
        while True:
            success, img = video.read()
            if not success or img is None:
                st.warning("Error capturing video frame. Please check your camera.")
                break

            frame_queue.put(img)

    # Start the thread to update the frame queue
    update_thread = Thread(target=update_frame_queue)
    update_thread.start()

    # Streamlit app loop
    show_one_frame = st.checkbox("Show One Frame")
    if show_one_frame:
        # Get the latest frame from the queue
        img = frame_queue.get()

        # Perform face recognition and update Streamlit
        img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(img_s)
        encodes_cur_frame = face_recognition.face_encodings(img_s, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_list_known, encode_face)
            face_distance = face_recognition.face_distance(encode_list_known, encode_face)
            winning_match_index = np.argmin(face_distance)

            if face_distance[winning_match_index] < 0.60:
                name = Employees_Names[winning_match_index].upper()
                mark_attendance(name, attendance_today)
            else:
                name = 'Unknown Person'

            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 216), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (90, 90, 216), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        st.image(img, channels="BGR", use_column_width=True, caption="Real Time Attendance with Employee Recognition")

if __name__ == "__main__":
    main()
