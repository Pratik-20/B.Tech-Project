import cv2
from tkinter import *
from PIL import ImageTk, Image


thres = 0.45  # Threshold to detect object
"""
cap = cv2.VideoCapture(1)

cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)
"""
# Create an instance of tkinter window
win = Tk()

# Define the geometry of the window
win.geometry("700x500")

frame = Frame(win, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)


classNames = []
classFile = '/storage/emulated/0/Android/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('n')

    configPath = '/storage/emulated/0/Android/tflite_graph.pbtxt'
    weightsPath = '/storage/emulated/0/Android/tflite_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

img = cv2.imread('/storage/emulated/0/Download/IMG_20220405_154644-removebg-preview.png')
classIds, confs, bbox = net.detect(img, confThreshold=thres)
print(classIds)
print(classIds, bbox)
if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                	cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                # cv2.putText(img, str(classNames[classId - 1]), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(classId - 1,2)), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('image7.jpeg',img)
img = ImageTk.PhotoImage(Image.open("image7.jpeg"))

# Create a Label Widget to display the text or Image
label = Label(frame, image = img)
label.pack()

win.mainloop()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
"""        
        
import cv2
from tkinter import *
from PIL import ImageTk, Image

# Create an instance of tkinter window
win = Tk()

# Define the geometry of the window
win.geometry("700x500")

frame = Frame(win, width=600, height=400)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('/storage/emulated/0/Zip/0/DCIM/Camera/IMG_20220319_180654.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

   
face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite('image2.jpeg',img)
img = ImageTk.PhotoImage(Image.open("image2.jpeg"))

# Create a Label Widget to display the text or Image
label = Label(frame, image = img)
label.pack()

win.mainloop()     
"""   