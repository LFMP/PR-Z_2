import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2, os
from imgManip.image_manip import pseudo_coloring, remove_background,  GradientFilters

def matsushita_aug():
    return iaa.SomeOf((1, 4), [
            iaa.Noop(),
            iaa.Superpixels(p_replace=0.5, n_segments=64),
            iaa.Add((-50, 50), per_channel=True),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.GaussianBlur(sigma=(0.1, 1.0)),
            iaa.ContrastNormalization((0.5, 2), per_channel=0.5),
            iaa.AdditiveGaussianNoise(scale=0.075 * 255, per_channel=0.5),
            iaa.CoarseDropout((0.1, 0.25), size_percent=(0.02, 0.25), per_channel=0.5),
            iaa.OneOf([
                iaa.Emboss(alpha=0.25),
                iaa.Sharpen(alpha=0.5),
                iaa.Invert(1, per_channel=1.0),
            ]),
            iaa.Affine(
                scale={'x':(0.84, 1.16), 'y': (0.84, 1.16)},
                mode=['edge', 'wrap']),
            iaa.Affine(
                translate_percent={'x':(-0.16, 0.16), 'y': (-0.16, 0.16)},
                mode=['edge', 'wrap']),
            iaa.Affine(
                rotate=(-16,16),
                mode=['edge', 'wrap']),
            iaa.Affine(
                shear=(-16,16),
                mode=['edge', 'wrap']),
            iaa.Fliplr(1)
        ])

def zanoni_aug():
    return iaa.SomeOf((1,4), [
            # OKS
            iaa.Noop(),
            iaa.Grayscale(alpha=(0.25, 1.0)), 
            iaa.Flipud(0.45),
            iaa.Fliplr(0.45),
            iaa.AddToHueAndSaturation((-40, 40)),
            iaa.ContrastNormalization((0.8, 1.2), per_channel=0.33),
            iaa.AdditiveGaussianNoise(scale=0.02*255),
            iaa.GaussianBlur(sigma=(0.1, 1.0)),
            iaa.OneOf([
                #iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"), 
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="BGR")
                #iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="YCrCb"),
                #iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSL")
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 0.33), strength=(0.02, 0.25)), 
                iaa.Sharpen(alpha=(0.0, 0.33), lightness=(0.02, 0.25)) 
            ]),
            iaa.OneOf([
                iaa.Add((-15, 15), per_channel=0.25), 
                iaa.Multiply((0.5, 2.0), per_channel=0.25)
            ])
            #iaa.CoarseDropout((0.1, 0.25), size_percent=(0.5, 0.75), per_channel=0.5)
    ])


def manual_data_aug(img, n=16):
    if n>3 and n<72:
        imgs = [img, cv2.flip(img, 1), cv2.flip(img, 0), cv2.flip(cv2.flip(img, 1), 0)]
        gf = GradientFilters(uint8=True)

        for i in range(len(imgs)):
            imgs.append(gf.sobel(imgs[i]))
            imgs.append(gf.scharr(imgs[i]))

        # add GS and HSV
        for i in range(len(imgs)):
            gray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
            imgs.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            imgs.append(cv2.applyColorMap(gray, 9))

        
        #for i in range(len(imgs)):
        #    imgs.append(remove_background(imgs[i]))
        
        return imgs

    else:
        return img
