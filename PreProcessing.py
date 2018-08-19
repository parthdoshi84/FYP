import os
import cv2
import shutil
import numpy as np
from PIL import Image

emotion_dirs = os.listdir("E:\FYP\Expression Recognition\CK+ Dataset\Emotion")
labels = dict()
empty_dirs = []
count = 0
train_data = []


def organizeData():

    for emotion_dir in emotion_dirs:

        dirs = os.listdir("E:\FYP\Expression Recognition\CK+ Dataset\Emotion\\" + emotion_dir)
        for dir in dirs:

            emotions = os.listdir("E:\FYP\Expression Recognition\CK+ Dataset\Emotion\\" + emotion_dir + "\\" + dir)
            if emotions:
                emotion = emotions[0]
                f = open("E:\FYP\Expression Recognition\CK+ Dataset\Emotion\\" + emotion_dir + "\\" + dir + "\\" + emotion , "rb")
                name = emotion.split("_emotion")[0]
                shutil.copy2("E:\FYP\Expression Recognition\CK+ Dataset\cohn-kanade-images\\" + emotion_dir + "\\" + dir + "\\" + name + ".png", "E:\FYP\Expression Recognition\MyData\Images\\" + name + ".png")

                for line in f:
                    value = line.split(".")
                    labels[emotion] = int(value[0])


            else:
                print emotion_dir + "\\" + dir
                empty_dirs.append(emotion_dir +"\\"+dir )
        print "count = " + str(count)
        count = count + 1

    np.save("E:\FYP\Expression Recognition\MyData\labels.npy", labels)
    np.save("E:\FYP\Expression Recognition\MyData\empty_dirs.npy", empty_dirs)

def face_detection():
    count  = 0
    images = os.listdir("E:\FYP\Expression Recognition\myImage")
    for image in images:

        f = cv2.imread("E:\FYP\Expression Recognition\myImage\\" + image)

        face_cascade = cv2.CascadeClassifier("C:\Users\Parth123\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
        W, H, D = f.shape
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        new_arr = []
        for (x, y, w, h) in faces:
            crop_img = gray[y:y + h, x:x + w]
            resized_image = cv2.resize(crop_img, (48, 48))

            cv2.imwrite("E:\FYP\Expression Recognition\MyData\CroppedResizedImages\\" + image, resized_image)
            reshaped = np.resize(resized_image, 48*48)
            mean = np.mean(reshaped)
            std = np.std(reshaped)
            new_arr = (reshaped - mean)/256
            #print new_arr
            train_data.append(new_arr)

            print "Count = " + str(count)
            count = count + 1
            break

    new_data = np.array(train_data)
    np.save("E:\FYP\Expression Recognition\MyData\my_data_train_data.npy",new_data)


def main():
    face_detection()
    '''images = os.listdir("E:\FYP\Expression Recognition\MyData\Images")
    print len(images)
    #face_detection()
    counts = [0,0,0,0,0,0,0]
    labels = np.load("E:\FYP\Expression Recognition\MyData\labels.npy").item()
    for k,v in labels.items():
        if v == 0:
            counts[0]+=1
        elif v == 1:
            counts[1]+=1
        elif v == 2:
            counts[2]+=1
        elif v == 3:
            counts[3]+=1
        elif v == 4:
            counts[4]+=1
        elif v == 5:
            counts[5]+=1
        elif v == 6:
            counts[6]+=1
    print counts'''

main()

