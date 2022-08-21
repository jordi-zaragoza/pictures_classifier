from matplotlib import pyplot as plt

# from lib import eye_lib, face_lib, manual_labeling_lib, model_lib
from lib import general_lib
directory = '../Other'


# Step 0: Sort blurry pictures
# lib.blurr_lib.blurr_classifier_folder('../Other',grid_size=[3,3])
# lib.blurr_lib.blurr_sort('../Other', threshold=[150,200,200])


# Step 1: Retrieve-Store faces from each picture
# face_lib.store_faces_from_directory(directory)

# Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
# lib.blurr_lib.blurr_classifier_folder('output/faces', file_name='faces_blur.csv',grid_size=[1,1])
# lib.blurr_lib.blurr_sort('output/faces', file_name='faces_blur.csv', threshold=[150,200,200])
# face_lib.sort_sunglasses()
 
 
# Step 3: Retrieve eyes
# eye_lib.store_eyes_from_directory()


# Step 4: Classify eyes/faces/pictures
# eye_lib.classify_eyes()
# face_lib.classify_faces()


# Step 5: Sort eyes (open, closed, unknown)
# eye_lib.sort_eyes()

# Step 6: Check open/closed eyes folder by hand
# Step 7: Manual labeling unknown eyes folder
# manual_labeling_lib.label_eyes_from_folder('output/eyes/unknown')

# Step 8: Retrain the model adding the new dataset
general_lib.train_test_split(path='output/eyes')
# model_lib.model_trainer(PATH='output/eyes', MODEL_NAME='model_eye', BATCH_SIZE=32, IMG_SIZE=(160, 160))
