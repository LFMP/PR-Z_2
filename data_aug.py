import imgManip.image_manip as im
from imgManip.dataAug import zanoni_aug, manual_data_aug
import os
import numpy as np
import cv2

INPUT = "../testes/teste6/in/"
DA_OUTPUT = "../testes/teste6/out/"

#INPUT = "../GC_Images/"
#DA_OUTPUT = "../Dataset_Variations/OLD_DA/MDA_16/"


if not os.path.isdir(DA_OUTPUT):
    os.mkdir(DA_OUTPUT)

count = 1
for sample in os.listdir(INPUT):
    print(sample,count)
    img = cv2.imread(INPUT+sample, -1)
    #wid, hei, chann = img.shape
    #img = np.reshape(img, (1, wid, hei, chann))

    imgs = np.array([img for i in range(16)])
    #images_aug = zanoni_aug().augment_images(imgs)
    images_aug = manual_data_aug(img, n=16)

    sample_name = DA_OUTPUT + sample.split('.')[0]

    cv2.imwrite(sample_name + "$0.jpg", img)

    i = 1
    for new_img in images_aug:
        cv2.imwrite('{}${}.jpg'.format(sample_name,i), new_img)
        i+=1

    count += 1

'''
size_t=[(32, 24), (48, 36), (64, 48)]
RESIZE_OUTPUT = "../Dataset_Variations/MDA_16_resized_"
nam = ["32x24/", "48x36/", "64x48/"]

im.resize(DA_OUTPUT, RESIZE_OUTPUT+nam[0], size=size_t[0])
im.resize(DA_OUTPUT, RESIZE_OUTPUT+nam[1], size=size_t[1])
im.resize(DA_OUTPUT, RESIZE_OUTPUT+nam[2], size=size_t[2])


#resize("../Dataset_Variations/MDA_16/","../Dataset_Variations/MDA_16_resized_04/", size=0.04)
#general("../testes/teste5/IN/", "../testes/teste5/OUT1/",kernel_sz=5)
#split("../GC_Images/","../Dataset_Variations/split_04_05/",  size=0.04, proportion=0.05)
'''