from lib import picture_lib, general_lib

path_folders = '/media/jzar/CAROL SSD/220603_Anna_i_Carles/'
path_save = '../data/work/boda_anna_carles/'

# Retrieve pictures from usb drive in jpg for each folder
# picture_lib.get_pictures_from_folders(path_folders, path_save)

# Classify all folders
time_classifier = general_lib.timeit(picture_lib.classify_folders)
time_classifier(path_save)

# Classify a folder
# time_classifier = general_lib.timeit(picture_lib.classifier)
# time_classifier(path_save+'/01_Firma/pictures')

# Step 6: Check open/closed eyes folder by hand
# Step 7: Manual labeling unknown eyes folder
# manual_labeling_lib.label_eyes_from_folder(directory + '/output/eyes/unknown')

# Step 8: Retrain the model adding the new dataset
# general_lib.train_test_split(path=path+'/output/eyes')
# model_lib.model_trainer(PATH=directory+'/output/eyes', MODEL_NAME='model_eye', BATCH_SIZE=32, IMG_SIZE=(160, 160))
