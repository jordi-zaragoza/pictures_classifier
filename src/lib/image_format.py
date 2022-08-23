# Convert raw images into jpg images
import cv2
import rawpy
import os
from lib import general_lib
import matplotlib.image as mpimg


def black_white_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def resize_image(image, base_width=1024.0):
    w_percent = (base_width / float(image.shape[1]))
    hsize = (float(image.shape[0]) * float(w_percent))
    img_resize = cv2.resize(image, (int(base_width), int(hsize)), interpolation=cv2.INTER_LINEAR)
    return img_resize


# Convert raw images into jpg images
def cr2_to_jpg(path, image_name):
    raw_img = rawpy.imread(path + '/' + image_name)
    rgb_post = raw_img.postprocess()
    return rgb_post


def cr2_to_jpg_folder(path):
    image_list = os.listdir(path)
    for num, image_name in enumerate(image_list):
        if '.CR2' in image_name:
            cr2_to_jpg(path, image_name)
            print('Image', str(num), image_name, 'done')


def images_to_jpg_folder(path, base_width=1024.0):
    image_list = os.listdir(path)
    image_list = general_lib.filter_images(image_list)
    for num, image_name in enumerate(image_list):
        print('Image', str(num), image_name)
        if '.CR2' in image_name:
            try:
                img_jpg = cr2_to_jpg(path, image_name)
                print('copied and transformed to .jpg')
            except:
                print('cannot transform to jpg')
                return 0

        else:
            img_jpg = mpimg.imread(path + '/' + image_name)
            print('copied')

        # img_gray = black_white_image(img_jpg)
        img_resize = resize_image(img_jpg, base_width)
        general_lib.save_image(img_resize, image_name[:-4], path + '/output/pictures')

