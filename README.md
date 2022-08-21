# Pictures classifier

(work in progress...)

## Overview

This repository sorts 'good' and 'bad' pictures. As this is a very blurry concept :) let's define 'bad' as: 
- blurry pictures
- pictures where someone has eyes closed.
- (more things to come)


# Methods

## 1. Blurry sort
The blur detector is based in 3 different detectors:
### Laplacian
Edge detector, it computes the second derivatives of an image, measuring the rate at which the first derivatives change. Comes with open-cv, [cv2.Laplacian](https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html).
### CPBD
CPBD is a perceptual-based no-reference objective image sharpness metric based on the cumulative probability of blur detection. Took it from [this pypi library](https://pypi.org/project/cpbd/).
### Wavelet
The Gabor-wavelet analysis allows a rapid estimation of image flow vectors with low spatial resolution. It's used for movement blur detection. Used the library from [this author](https://github.com/pedrofrodenas/blur-Detection-Haar-Wavelet).

## 2. Open-Closed eye detector
### Step 0: Sort blurry pictures
Sort pictures depending on the blurriness. It leaves pictures that have blurry surroundings with a focused area.

### Step 1: Retrieve-Store faces from each picture
Retrieve faces using [pypi face-recognition](https://pypi.org/project/face-recognition/) (uses state of the art face recognition using deeplearning).

### Step 2: Sort faces (valid, sunglasses/~~not-valid/blurry~~)
This step is using a model trained previously for sunglasses detection. It also uses blur detection.

### Step 3: Retrieve eyes
It takes right and left eyes directly croping the left and right top corners. Then it flips left eye in order to get a 'second' right eye.

### Step 4: Classify eyes/faces/pictures
Classifies open or closed eyes using a previously trained model for 'right eyes'.

### Step 5: Sort eyes (open, closed, unknown), ~~faces again (open, closed, unknown) and pictures (closed eyes)~~
Sorts the images in different folders.

### Step 6: Check open/closed eyes folder by hand
This folder is supposed to contain the correct labeling of open and closed eyes. Have to check by hand if all of them are correct in order to use them to train the model again.

### Step 7: Manual labeling unknown eyes folder
In the unknown folder I will have to manually label the eyes. I created a function for this purpose:
`lib.manual_labeling_lib.label_eyes_from_folder('output/eyes/unknown')`

### ~~Step 8: Retrain the model adding the new dataset~~
to-do

## Tools

- Keras with Tensorflow for the eyes detection model created using MobileNetV2 and fine tunning
- pypi face-recognition (uses state of the art face recognition using deeplearning)
- pypi cpbd for blur detection
- cv2 laplacian for blur detection
- blur-detection wavelet using 
