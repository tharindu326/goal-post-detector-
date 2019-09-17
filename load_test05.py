from keras.models import load_model
import cv2
from keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import math
import array as arr
import matplotlib.pyplot as plt

videoFile = "France v Croatia - 2018 FIFA World Cup™ FINAL - HIGHLIGHTS.mp4"
imagesFolder = "G:/Uni/7th sem/finalYr Proj/load_model_test01/final/"
result_possitive_folder = 'G:/Uni/7th sem/finalYr Proj/load_model_test01/results_positive/'
result_negative_folder = 'G:/Uni/7th sem/finalYr Proj/load_model_test01/results_negative/'

img_width, img_height = 224, 224

cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
print(frameRate)
frame_count = cap.get(7)# total no.of frames
print(frame_count)
frameRate_selected = cap.get(5)*1   #getting frmaes with a time gap of frameRate*3 / frames in each 3 seconds
print(frameRate_selected)
frame_count_selected = round(frame_count / frameRate_selected ) #number of rames we are selecting/ how many frames that we are getting in each 3secs
print(frame_count_selected)
array_frameId = []
array_time = []
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    #print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate_selected) == 0):
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 180, 250, cv2.THRESH_BINARY)
        cv2.imwrite(filename, threshold)
        cv2.destroyAllWindows()
        cv2.waitKey(0)
        # load the trained model
        model = tf.keras.models.load_model("goalpost30_390.model")
        # model = load_model(model_path)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        #img = image.load_img('images/cat.jpeg', target_size=(227, 227))

        img = image.load_img(filename, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        classes = model.predict_classes(img, batch_size=16)
        print(classes)
        if classes == 0:
            filename1 = result_negative_folder + "/image_" + str(int(frameId)) + ".jpg"
            cv2.imwrite(filename1, threshold)
        elif classes == 1:
            filename2 = result_possitive_folder + "/image_" + str(int(frameId)) + ".jpg"
            cv2.imwrite(filename2, threshold)
            array_time.append(frameId/frameRate)
            array_frameId.append(frameId)
cap.release()
print(array_time)
array_frameId_0 = array_frameId
print(array_frameId)
x = array_time
y = array_frameId
plt.scatter(x, y, label="stars", color="green",
            marker="*", s=30)
plt.legend()
plt.show()
k = len(array_time) - 1
diff = []
margin = []
for a in range(k):
    A = a+1
    d = array_time[A] - array_time[a]
    diff.append(d)
    if d >= 6:
        k = [array_time[a], array_time[A]]
        margin.append(k)

print(diff)
print(margin)
length_time_stamp = []
for j in range(len(margin)-1):
    m = [margin[j][1], margin[j + 1][0]]
    length_time_stamp.append(m)
    m_l = margin[j + 1][1]

m_f = [array_time[0], margin[0][0]]
m_l = [m_l, array_time[len(array_time)-1]]
length_time_stamp = [m_f, length_time_stamp, m_l]
length_time_stamp = np.vstack(length_time_stamp)
print(length_time_stamp)

array_time_removed = []

for i in range(len(length_time_stamp)):
    q = length_time_stamp[i][0]
    q1 = array_time.index(q)
    r= length_time_stamp[i][1]
    r1 = array_time.index(r)
    s= len(array_time[q1:r1+1])
    if s <= 3 :
        array_time_r = array_time[q1:r1+1]
        array_time_removed.append(array_time_r)

print(array_time_removed)

for i in range(len(array_time_removed)):
    q = array_time_removed[i][0]
    q1 = array_time.index(q)
    r = array_time_removed[i][-1]
    r1 = array_time.index(r)
    for e in range(q1,r1+1):
        array_frameId[e] = 0


print(array_frameId)
x = array_time
y = array_frameId
plt.scatter(x, y, label="stars", color="red",
            marker="*", s=30)
plt.legend()
plt.show()

