from lib import model_lib

PATH = '../data/datasets/model/eye_use'
MODEL_NAME = 'model_eye'

# PATH = '../data/datasets/model/sunglasses_use'
# MODEL_NAME = 'model_sunglasses'

# PATH = '../data/datasets/model/blurry_use'
# MODEL_NAME = 'model_blurry'

# PATH = '../data/datasets/model/profile_use'
# MODEL_NAME = 'model_profile'

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

model_lib.model_trainer(PATH, MODEL_NAME, BATCH_SIZE, IMG_SIZE)


# Step 9: Retrain the models adding the new datasets
# model_lib.retrain_all_models(PATH='../data/datasets/model')

