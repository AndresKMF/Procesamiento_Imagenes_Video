# Script in charge of add noise to the images in the database

# libraries
import numpy as np
import cv2
import os
import imutils
from imutils import paths

### MAIN
if __name__ == '__main__':
    ## LOAD DATASET
    inputPath = "C:\Proyecto\Data_Base/all_augmented"
    # Path to images
    imagePaths = list(paths.list_images(inputPath))
    print(imagePaths)
    # Names of folders
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    # Add names to list
    names = names.tolist()
    print(names)

    # count
    i = 1
    # Path to images
    ruta = "C:\Proyecto\Data_Base/all_augmented"
    # loop over the image paths
    for imagePath in imagePaths:
        print(imagePath)
        # load the image from disk and extract the name of the person
        # from the subdirectory structure
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        ## ADD DIFFERENT TYPES OF NOISE
        path_file = os.path.join(ruta, name)
        # names of fields
        name_gauss = str(i) + "_" + "gauss.png"
        name_speckle = str(i) + "_" + "speckle.png"

        path_file_1 = os.path.join(path_file, name_gauss)
        path_file_2 = os.path.join(path_file, name_speckle)
        print(path_file_1)
        print(path_file_2)

        # Generate Gaussian noise
        gauss = np.random.normal(0, 1, image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv2.add(image, gauss)

        # Generate Speckle noise
        noise = image + image * gauss

        # write images with noise to the database
        cv2.imwrite(path_file_1, img_gauss)
        cv2.imwrite(path_file_2, noise)

        i = i + 1