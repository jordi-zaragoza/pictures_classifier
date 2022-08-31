from lib import picture_lib, general_lib

path_folders = '/media/jzar/CAROL SSD/220603_Anna_i_Carles/'
path_save = '../data/work/boda_anna_carles'

# Retrieve pictures from usb drive in jpg for each folder
# picture_lib.get_pictures_from_folders(path_folders, path_save)

# Classify all folders
# time_classifier = general_lib.timeit(picture_lib.classify_folders)
# time_classifier(path_save)

# Classify a folder
time_classifier = general_lib.timeit(picture_lib.classifier)
time_classifier(path_save+'/02_Carles')