from lib import general_lib, classifier_lib, picture_lib
from lib import model_lib

path_folders = '/media/jzar/CAROL SSD/220603_Anna_i_Carles'
path_save = '../data/work/boda_anna_carles'

# # Steps -1, Get jpgs from raw -------------------------------------------
# Retrieve pictures from usb drive in jpg for each folder
# picture_lib.get_pictures_from_folders(path_folders, path_save)

# # Steps 0 to 6, classifier ----------------------------------------------
# Classify all folders
# time_classifier = general_lib.timeit(classifier_lib.classify_folders)
# time_classifier(path_save)

# Classify a folder
time_classifier = general_lib.timeit(classifier_lib.classifier)
time_classifier(path_save+'/06_Aperitivo')

# Step 7: rate the raw files for
picture_lib.rate_pictures(path_result=path_save+'/06_Aperitivo/results/pictures_classification.csv',
                          path_raw_folder=path_folders+'/06_Aperitivo')

# # steps 8 to 10, retrain the model --------------------------------------
# # Step 8: Check all datasets
# # Step 9: Manual labeling unknown eyes folder and generate train-test
# manual_labeling_lib.label_eyes_from_folder(directory + '/datasets_extracted/eyes/unknown')
# general_lib.train_test_split(path='directory + /datasets_extracted/eyes')

# Step 10: Retrain the models adding the new datasets
# model_lib.retrain_all_models(PATH='../data/datasets/model')
