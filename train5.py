# -*- coding: utf-8 -*-

import queue
import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import numpy as np
import pandas as pd
import datetime
import time
import tkinter.font as font
import sqlite3
from PIL import Image, ImageTk
import threading
from tkinter import messagebox

# Database setup
conn = sqlite3.connect('students.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    Id INTEGER PRIMARY KEY,
                    Name TEXT NOT NULL
                )''')
conn.commit()  # Ensure the changes are saved

# Tkinter window setup
window = tk.Tk()
window.title("Face Recogniser")
window.geometry('1000x600')
window.configure(background='teal')
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

message = tk.Label(window, text="Smart-Attendance-System-with-Face Recognition", bg="lime", fg="black", width=50, height=3, font=('times', 25, 'italic bold underline'))
message.place(x=23, y=20)

lbl = tk.Label(window, text="Enter ID", width=15, height=2, fg="black", bg="yellow", font=('times', 15, 'bold'))
lbl.place(x=100, y=180)

txt = tk.Entry(window, width=20, bg="yellow", fg="black", font=('times', 15, 'bold'))
txt.place(x=320, y=195)

lbl2 = tk.Label(window, text="Enter Name", width=15, fg="black", bg="yellow", height=2, font=('times', 15, 'bold'))
lbl2.place(x=100, y=280)

txt2 = tk.Entry(window, width=20, bg="yellow", fg="black", font=('times', 15, 'bold'))
txt2.place(x=320, y=295)

lbl3 = tk.Label(window, text="Notification : ", width=15, fg="black", bg="yellow", height=2, font=('times', 15, 'bold underline'))
lbl3.place(x=100, y=380)

notification_message = tk.Label(window, text="", bg="yellow", fg="black", width=35, height=2, activebackground="yellow", font=('times', 15, 'bold'))
notification_message.place(x=320, y=380)

lbl4 = tk.Label(window, text="Attendance : ", width=15, fg="black", bg="yellow", height=2, font=('times', 15, 'bold underline'))
lbl4.place(x=100, y=540)

message2 = tk.Label(window, text="", fg="black", bg="yellow", activeforeground="green", width=36, height=2, font=('times', 15, 'bold'))
message2.place(x=315, y=540)

# Function to clear ID input
def clear():
    txt.delete(0, 'end')
    notification_message.configure(text="")

# Function to clear Name input
def clear2():
    txt2.delete(0, 'end')
    notification_message.configure(text="")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

Attendance = 0
# Global variable to safely pass data between threads
image_queue = queue.Queue()

def TakeImages():
    Id = txt.get().strip()  # ID from the entry widget
    name = txt2.get().strip()  # Name from the entry widget

    if not Id or not name:
        messagebox.showwarning("Input Error", "Please enter both ID and Name")
        return

    # Insert ID and Name into the database
    try:
        cursor.execute("INSERT INTO attendance (Id, Name) VALUES (?, ?)", (Id, name))
        conn.commit()  # Save the changes
        print(f"Inserted ID: {Id}, Name: {name} into the database")  # Debugging print
    except sqlite3.IntegrityError:
        messagebox.showwarning("Database Error", f"ID {Id} already exists. Please use a unique ID.")
        return

    sampleNum = 0
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def capture_images():
        nonlocal sampleNum
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Open the webcam

        if not cam.isOpened():
            messagebox.showerror("Camera Error", "Failed to access the camera")
            return

        start_time = time.time()

        while True:
            ret, img = cam.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to capture image")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                print(f"Image saved: {name}.{Id}.{sampleNum}.jpg")  # Debugging print

            if sampleNum >= 100 or (time.time() - start_time) > 5:  # Stop after 5 seconds or 100 images
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"Images saved for ID: {Id}, Name: {name}")

    # Start the image capture in a separate thread
    threading.Thread(target=capture_images, daemon=True).start()

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        # Get the filename
        filename = os.path.split(imagePath)[-1]
        parts = filename.split(".")

        # Ensure there are enough parts to extract ID and handle cases where it fails
        if len(parts) >= 3:
            try:
                Id = int(parts[1])  # Extract ID from the second part
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                faces.append(imageNp)
                Ids.append(Id)
            except ValueError:
                print(f"Warning: Could not convert ID from filename '{filename}'")
        else:
            print(f"Warning: Filename '{filename}' does not have the expected format.")

    # Convert Ids list to numpy array only if it's not empty
    if len(faces) == 0 or len(Ids) == 0:
        print("Warning: No faces or IDs were found for training.")
        return [], []

    return faces, np.array(Ids)  # Ensure Ids are returned as a numpy array

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Ids = getImagesAndLabels("TrainingImage")
    
    if len(faces) == 0 or len(Ids) == 0:
        print("Error: No valid training data found.")
        return  # Exit if there's no data to train on

    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    notification_message.configure(text="Image Trained")
    print("Training completed successfully.")

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Attendance']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                ts = time.time()
                aa = cursor.execute("SELECT [Name] FROM attendance WHERE Id = ?", (Id,))
                aa = cursor.fetchone()
                if aa is not None:
                    tt = str(Id) + "-" + aa[0]
                else:
                    tt = str(Id) + "-Unknown"
            else:
                tt = 'Unknown'
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
            attendance = attendance.drop_duplicates(subset=['Id'], keep='last')

        cv2.imshow('im', im)
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # Exit if the 'Esc' key is pressed
            break

    attendance.to_csv("Attendance.csv")  # Save attendance to a CSV file
    cam.release()
    cv2.destroyAllWindows()

# Buttons to trigger functions
# Buttons to trigger functions
clearButton = tk.Button(window, text="Clear ID", command=clear, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
clearButton.place(x=550, y=180)

clearButton2 = tk.Button(window, text="Clear Name", command=clear2, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
clearButton2.place(x=550, y=280)

takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
takeImg.place(x=60, y=460)

trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
trainImg.place(x=280, y=460)

trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
trackImg.place(x=500, y=460)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="yellow", width=15, height=2, activebackground="Red", font=('times', 15, 'bold'))
quitWindow.place(x=720, y=460)



# Close the database connection when the program ends
def on_closing():
    conn.close()  # Close the connection before exiting
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
