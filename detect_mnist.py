import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
import image_slicer
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.yolov4 import read_class_names
from yolov3.configs import *


images = image_slicer.slice('04.jpg',4)
images = list(images)
for i in range(0, len(images)):
    image = str(images[i])
    name = image.split()
    name = name[3]
    name = name[:-1]

    # resizing the image
    image_path = name
    image_original = cv2.imread(image_path)
    image_resized = cv2.resize(image_original, (864, 864), interpolation = cv2.INTER_AREA)
    cv2.imwrite("04_resized.jpg", image_resized)

    #giving the saved image to model
    image_path_new = "04_resized.jpg"

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights
    image, bboxes = detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

## Algorithim for representation of detected numbers

    NUM_CLASS = read_class_names(TRAIN_CLASSES)
    num_classes = len(NUM_CLASS)

    score_list = []
    detected_number_list = []
    x_middle_points = []
    y_middle_points =[]

    for i, bbox in enumerate(bboxes):

        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        x = (x1+x2)/2
        y = (y1+y2)/2

        x_middle_points.append(x)
        y_middle_points.append(y)

        score_str = " {:.2f}".format(score)
        label = "{}".format(NUM_CLASS[class_ind])

        score_list.append(score_str)
        detected_number_list.append(label)

    final_scores = []
    final_numbers = []
    final_middle_points_x = []
    final_middle_points_y = []

    for i in range(0, len(detected_number_list)):

        if float(score_list[i]) >= 0.80:
            final_scores.append(score_list[i])
            final_numbers.append(detected_number_list[i])
            final_middle_points_x.append(x_middle_points[i])
            final_middle_points_y.append(y_middle_points[i])

    print("Scores: ", final_scores)
    print("Detected_Numbers: ", final_numbers)
    print("Boxes X cordinate: ", final_middle_points_x)
    print("Boxes Y cordinate: ", final_middle_points_y)

"""
TO_DO:

* Develop this algorithm further so that we are able to represent the detected numbers by the deep leraning
  model accurately and correct sequence. We have the bboxes which contains the box confidence scores, labels
  and the cordinates for the bounding-boxes.
* The main concern here is to use these obtained values from the model to represent the detected sequence of
  numbers in a accurate manner. We need to develop a algorithm that is able to accurately give the written
  series of numbers with the help of these cordinates and labels. The thresholding of the confidence scores
  will also reduce the missdetection from the model which is also done above, but further development is
  required. 

NOTE: The theresholding of the confidence scores will reduce the problem
      of missdetection in digits. Also, if the number of epochs for the training of the model is increased from
      30 to higher number like 100 then also the model becomes more accurate.

"""
