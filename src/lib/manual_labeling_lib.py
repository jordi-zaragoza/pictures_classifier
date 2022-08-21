import cv2
from lib.eye_lib import *
from lib.sunglasses_lib import *


def single_code_to_label(num):
    return {
        '0': 'closed',
        '1': 'open'
    }.get(num, 'not_valid')


def ask_user_to_label(image):
    image = np.array(image)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 600, 600)
    cv2.imshow("Image", image[:, :, ::-1])

    key = cv2.waitKey(100)  # pauses for 3 seconds before fetching next image
    if key == 27:  # if ESC is pressed, exit loop
        cv2.destroyAll

    code_input = input("Insert Closed or Open eye (0-1):")
    cv2.destroyAllWindows()
    return code_input


def label_eye(image_name, path):
    eye_image = tf.keras.preprocessing.image.load_img(path + '/' + image_name)
    code_input = ask_user_to_label(eye_image)
    classification = single_code_to_label(code_input)
    general_lib.move_file(image_name, path, path + '/' + classification)


def label_eyes_from_folder(path):
    image_list = os.listdir(path)
    image_list = general_lib.filter_images(image_list)
    for image_name in image_list:
        label_eye(image_name, path)
