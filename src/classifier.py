from lib import general_lib, classifier_lib, picture_lib
from lib import model_lib

path_folder = r'../data/production/02 Wim'
path_save = r'../data/production/transformed/02 Wim'

# # Steps -1, Get jpgs from raw and rename -------------------------------------------
# Retrieve pictures from usb drive in jpg

# general_lib.rename_all_files_underscore(path_folder)

# for multiple folders
# picture_lib.get_pictures_from_folders(path_folders, path_save)

# for single folder
# time_classifier = general_lib.timeit(picture_lib.retrieve_files)
# time_classifier(path_folder, path_save)

# # Steps 0 to 6, classifier ----------------------------------------------

# Classify all folders
# time_classifier = general_lib.timeit(classifier_lib.classify_folders)
# time_classifier(path_save)

# Classify single folder
time_classifier = general_lib.timeit(classifier_lib.classifier)
time_classifier(path_save)

# Step 7: rate the raw files ----------------------------------------------
# picture_lib.rate_pictures(path_result=path_save+'/fotos/results/pictures_classification.csv',
#                           path_raw_folder=path_save+'/xmp')

# # steps 8 to 10, retrain the model --------------------------------------
# # Step 8: Check all datasets
# # Step 9: Move files to the production dataset for modeling
# Step 10: Retrain the models adding the new datasets
# model_lib.retrain_all_models(PATH='../data/datasets/model')
