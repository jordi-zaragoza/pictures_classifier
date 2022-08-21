from lib import model_lib

PATH = '../datasets/eye_use'
MODEL_NAME = 'model_eye'

# PATH = '../datasets/sunglasses_use'
# MODEL_NAME = 'model_sunglasses'

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

model_lib.model_trainer(PATH, MODEL_NAME, BATCH_SIZE, IMG_SIZE)


