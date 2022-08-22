from matplotlib import pyplot as plt

#from lib import eye_lib, face_lib, manual_labeling_lib, model_lib
from lib import general_lib, blurr_lib

directory = '../data/test'

# Step 0: Sort blurry pictures
# print(blurr_lib.blurr_classifier_folder(directory, grid_size=[3,3]))
blurr_lib.blurr_sort(directory, threshold=[150,200,200])


# Step 1: Retrieve-Store faces from each picture
# face_lib.store_faces_from_directory(directory)

# Step 2: Sort faces (valid, blurry/not-valid/sunglasses)
# lib.blurr_lib.blurr_classifier_folder(directory+'/output/faces', file_name='faces_blur.csv',grid_size=[1,1])
# lib.blurr_lib.blurr_sort(directory+'/output/faces', file_name='faces_blur.csv', threshold=[150,200,200])
# face_lib.sort_sunglasses(directory)
 
 
# Step 3: Retrieve eyes
# eye_lib.store_eyes_from_directory(directory)


# Step 4: Classify eyes/faces/pictures
# eye_lib.classify_eyes(directory)
# face_lib.classify_faces(directory)


# Step 5: Sort eyes (open, closed, unknown)
# eye_lib.sort_eyes(directory)

# Step 6: Check open/closed eyes folder by hand
# Step 7: Manual labeling unknown eyes folder
# manual_labeling_lib.label_eyes_from_folder(directory + '/output/eyes/unknown')

# Step 8: Retrain the model adding the new dataset
# general_lib.train_test_split(path=path+'/output/eyes')
# model_lib.model_trainer(PATH=directory+'/output/eyes', MODEL_NAME='model_eye', BATCH_SIZE=32, IMG_SIZE=(160, 160))
