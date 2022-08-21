from matplotlib import pyplot as plt

from lib import eye_lib, face_lib, manual_labeling_lib

directory = '../datasets/boda'

# faces_classifier_lib.store_faces_from_directory(directory)
# faces_classifier_lib.sort_faces()
# eye_lib.store_eyes_from_directory()
# eye_lib.classify_eyes()
# face_lib.classify_faces()
# eye_lib.sort_eyes()

# Manual labeling part
manual_labeling_lib.label_eyes_from_folder('output/eyes/unknown')

# Model trainer