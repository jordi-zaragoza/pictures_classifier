from matplotlib import image as mpimg

from lib import image_format, general_lib

path = '../data/work'
image_name = '3V9A9066.CR2'

img = mpimg.imread(path + '/' + image_name)
img_gray = image_format.black_white_image(img)
img_resize = image_format.resize_image(img_gray)
general_lib.save_image(img_resize, image_name[:-4], path + '/output/pictures')