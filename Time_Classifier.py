import cv2, os, glob, random
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np


def load_datasetf(image_dir):  # function to load images
    im_list = []
    image_types = ["day", "night"]  # separate the two folders

    for im_type in image_types:
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            im = mimg.imread(file)  # Read in the image

            if not im is None:
                im_list.append((im, im_type))  # add image if it exists

    return im_list


def resize_img(image):  # resize images
    stimg = cv2.resize(image, (1000, 700))
    return stimg


def encode(time):  # encode the images based on time of day
    temp = 0
    if (time == 'day'):
        temp = 1
    return (temp)


def standardize(image_list):
    l = []
    for i in image_list:
        img = i[0]
        label = i[1]
        l.append([resize_img(img), encode(label)])

    return (l)


def classify(image): #if the images
    threshold = 110
    predicted = 0
    if (np.average(image) > threshold):
        predicted = 1
    return (predicted)


def get_missed_classified_image(lst):
    missed_classified = []
    for i in lst:
        image = i[0]
        true = i[1]
        predict = classify(image)
        if (true != predict):
            missed_classified.append([image, predict, true])
    return (missed_classified)


# our images path

training_path = "samples/training"
test_path = "samples/test"

# load them
training = load_datasetf(training_path)
test = load_datasetf(test_path)

# standardize the images to list
std1 = standardize(training)
std2 = standardize(test)

img = std1[2][0]
img1 = np.copy(img)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

# separate image into HSV(Hue,Sat,Value) channels
plt.imshow(std1[1][0])
[h, s, v] = [img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]]
plt.imshow(v, cmap='gray')

print("Average brightness: " + "{:.2f}".format(100 * np.average(v) / 255) + "%") #

random.shuffle(std2)  # randomize the order of images in testing


missed_class = get_missed_classified_image(std2)
total = len(std2)
miss = len(missed_class) # get the number of missed images
print("Number of misclassified images:", miss)
print("Total testing samples", total)

accuracy = ((total - miss) / total) * 100

print("Accuracy: ", accuracy)

