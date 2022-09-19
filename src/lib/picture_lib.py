import os
import pandas as pd
from lib import general_lib, image_format


def sort_pictures(directory):
    path = directory
    path_result = directory + 'faces/eyes/results'
    eyes_classification = pd.read_csv(path_result + '/pictures_classification.csv', index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = str(eyes_classification.name[index]) + '.jpg'
        closed_faces = eyes_classification.closed[index]
        if closed_faces > 0:
            general_lib.move_file(image_name, path, path + '/closed_eyes')


def get_pictures_from_folders(path_folders, path_save):
    folders = os.listdir(path_folders)
    # Get images from folder
    for folder in folders:
        retrieve_files(path_folders+'/'+folder, path_save+'/'+folder)


def retrieve_files(directory_pictures, directory_store='./output'):
    # Convert raw images into jpg images
    image_format.images_to_jpg_folder(directory_pictures, directory_store, base_width=2042.0)


# Writes rating value in the xmp file
def xmp_rating(path, stars):
    with open(path) as f:
        lines = f.readlines()

    new_lines = []
    rating_found = False
    for line in lines:
        if 'xmp:Rating' in line:
            print(line)
            l1 = line.split('\"')
            l1[1] = str(stars)
            l2 = '\"'.join(l1)
            new_lines.append(l2)
            rating_found = True

        else:
            new_lines.append(line)

    with open(path, 'w') as f:
        for line in new_lines:
            f.write(line)

            if "xmlns:crs" in line and not rating_found:
                print('not rating found')
                f.write('   xmp:Rating="{}"\n'.format(stars))


def rate_pictures(path_result, path_raw_folder, rating=1):
    eyes_classification = pd.read_csv(path_result, index_col=0)

    for index in range(eyes_classification.shape[0]):
        image_name = eyes_classification.name[index]
        closed_faces = eyes_classification.closed[index]
        if closed_faces > 0:
            try:
                xmp_rating(path_raw_folder+'/'+image_name+'.xmp', rating)
                print('image rated')
            except:
                print('cannot find image', image_name)