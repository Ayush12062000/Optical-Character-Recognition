import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import shutil
import random
from zipfile import ZipFile

SIZE_WIDTH = 864
SIZE_HEIGHT = 864
images_num_train = 1000
images_num_test = 200

image_sizes = [5]

add_path = "mnist"
if os.getcwd().split(os.sep)[-1] != "mnist":
    add_path = "mnist"
    os.chdir(add_path)
else:
    add_path = ""

path_noise = "noise_images/"
list_noise = os.listdir(path_noise)

def compute_iou(box1, box2):
    # xmin, ymin, xmax, ymax
    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, xmin, ymin, ratio=1):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(32*ratio), int(32*ratio)))
    h, w, c = image.shape

    while True:

        xmin = xmin + int(w/2.4) #np.random.randint(0, SIZE_WIDTH-w, 1)[0]
        ymin = ymin #np.random.randint(0, SIZE_WIDTH-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]

        if max(iou) <= 0.234:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    return blank


for file in ["train", "test"]:
    if not os.path.exists("mnist/{}".format(file)):
        with ZipFile("mnist/{}.zip".format(file), 'r') as zip:

            print('Extracting all {} files now...'.format(file))
            zip.extractall()
            shutil.move(file, "mnist")
            print('Done!')

for file in ['train','test']:
    images_path = os.getcwd()+"/mnist_{}".format(file)
    labels_txt = os.getcwd()+"/mnist_{}.txt".format(file)

    if file == 'train': images_num = images_num_train
    if file == 'test': images_num = images_num_test

    if os.path.exists(images_path): shutil.rmtree(images_path)
    os.mkdir(images_path)

    image_paths  = [os.path.join(os.path.realpath("."), os.getcwd()+"/mnist/{}/".format(file) + image_name)
                           for image_name in os.listdir(os.getcwd()+"/mnist/{}".format(file))]

    with open(labels_txt, "w") as wf:
        image_num = 0
        while image_num < images_num:
            image_path = os.path.realpath(os.path.join(images_path, "%06d.jpg" %(image_num+1)))

            annotation = image_path

            index = np.random.randint(0,17)
            noise_image = cv2.imread(path_noise + list_noise[index])
            blanks = cv2.resize(noise_image, (SIZE_WIDTH, SIZE_HEIGHT), interpolation = cv2.INTER_AREA)

            bboxes = [[0,0,1,1]]
            labels = [0]
            data = [blanks, bboxes, labels]
            bboxes_num = 0

            ratios = [[0.9]]
            count = 0
            xmin = np.random.randint(0, SIZE_WIDTH-300, 1)[0]
            ymin = np.random.randint(0, SIZE_WIDTH-30, 1)[0]
            for i in range(len(ratios)):
                N = random.randint(4,20)
                if N !=0: bboxes_num += 1
                for _ in range(N):
                    if count >= 7:
                        count = 0
                        xmin = np.random.randint(0, SIZE_WIDTH-300, 1)[0]
                        ymin = np.random.randint(0, SIZE_WIDTH-30, 1)[0]
                    ratio = random.choice(ratios[i])
                    idx = random.randint(0, len(image_paths)-1)
                    data[0] = make_image(data, image_paths[idx], xmin, ymin, ratio)
                    count += 1

            if bboxes_num == 0: continue
            data[0] = cv2.GaussianBlur(data[0],(3,3), 0)
            cv2.imwrite(image_path, data[0])
            for i in range(len(labels)):
                if i == 0: continue
                xmin = str(bboxes[i][0])
                ymin = str(bboxes[i][1])
                xmax = str(bboxes[i][2])
                ymax = str(bboxes[i][3])
                class_ind = str(labels[i])
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            image_num += 1
            print("=> %s" %annotation)
            wf.write(annotation + "\n")

if add_path != "": os.chdir("..")
