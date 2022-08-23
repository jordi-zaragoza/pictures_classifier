from lib import picture_lib

picture_lib.classifier('../data/work')

# Step 6: Check open/closed eyes folder by hand
# Step 7: Manual labeling unknown eyes folder
# manual_labeling_lib.label_eyes_from_folder(directory + '/output/eyes/unknown')

# Step 8: Retrain the model adding the new dataset
# general_lib.train_test_split(path=path+'/output/eyes')
# model_lib.model_trainer(PATH=directory+'/output/eyes', MODEL_NAME='model_eye', BATCH_SIZE=32, IMG_SIZE=(160, 160))
