### Pontificia Universidad Javeriana
### Procesamiento de Imágenes y Video
### Examen Final
### Luis Miguel Muñoz Flórez

### Import libraries
import cv2
import os
import sys
import numpy as np

### Global variables
points = []

### Definition of functions
## function that calculates the percentage of white pixels in a binary image
def calculate_percentage( image_mask ):
    number_of_pixels = image_mask.size
    grass = 0
    for i in range( image_mask.shape[0] ):
        for j in range ( image_mask.shape[1] ):
            if image_mask[i][j] == 255:
                grass += 1

    percentaje_grass = ( float( grass ) / float( number_of_pixels ) ) * 100

    return percentaje_grass

## function that calculates the percentage of grass pixels in the input image
def percentage_grass_pixels( image ):
    # Hue histogram
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Hue histogram max and location of max
    max_val = hist_hue.max()
    max_pos = int(hist_hue.argmax())

    # Peak mask
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask = cv2.inRange(image_hsv, lim_inf, lim_sup)
    # mask_not = cv2.bitwise_not(mask)

    # Structuring Element
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))

    # Opening
    mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_1)
    # cv2.imshow("mask opening ", mask_opening)

    # Closing
    mask_opening_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel_2)
    # cv2.imshow("mask opening closing", mask_opening_closing)
    # cv2.waitKey(0)

    percentage_grass = calculate_percentage( mask_opening_closing )

    return mask_opening_closing, percentage_grass

## function that find the contours in the image
def contours( image ):

    # Hue histogram
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Hue histogram max and location of max
    max_val = hist_hue.max()
    max_pos = int(hist_hue.argmax())

    # Peak mask
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask = cv2.inRange(image_hsv, lim_inf, lim_sup)
    # mask_not = cv2.bitwise_not(mask)
    # cv2.imshow("mask", mask)

    # Structuring Element
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))

    # Opening
    mask_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_1)
    # cv2.imshow("mask opening ", mask_opening)

    # Closing
    mask_opening_closing = cv2.morphologyEx(mask_opening, cv2.MORPH_CLOSE, kernel_2)
    # cv2.imshow("mask opening closing", mask_opening_closing)
    # cv2.waitKey(0)

    # contours
    contours, hierarchy = cv2.findContours(mask_opening_closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_draw = image.copy()
    aux = 0
    for idx, cont in enumerate(contours):
        if len(contours[idx]) > 70 and len(contours[idx]) < 1000:
            color = (0, 0, 255)
            x, y, width, height = cv2.boundingRect(contours[idx])
            cv2.rectangle(image_draw, (x, y), (x + width, y + height), color, 2)
            aux += 1
    return image_draw, aux

## function that manage the mouse events
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

## function that find the slope
def slope(p1,p2):
    x1,y1=p1
    x2,y2=p2
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'infinite slope'

## function to draw lines in all the image, not only between the two points
def drawLine(image,p1,p2, color):
    x1,y1=p1
    x2,y2=p2
    ### finding slope
    m=slope(p1,p2)
    ### getting image shape
    h,w=image.shape[:2]

    if m!='infinite slope':
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, 2)
    return image

### MAIN
if __name__ == '__main__':

    print("Final Exam")
    print("Every time you want to see another point, you will have to run the program")
    print("Please select the number of the point you want to see: ")
    print("POINT 1: Grass pixels")
    print("POINT 2: Find players and referees")
    print("POINT 3: Lines")
    s = int( input( ) )

    path = "C:\Imagenes_Video\Images"
    image_name = "soccer_game.png"
    path_file = os.path.join( path, image_name )
    image = cv2.imread( path_file )

    ### POINT 1: Grass pixels
    if s == 1:
        mask, percentage_grass = percentage_grass_pixels( image )

        print( "Percentage of grass pixels in the image: ", percentage_grass, " %" )

        cv2.namedWindow("Grass pixels", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Grass pixels", 1280, 720)
        cv2.imshow("Grass pixels", mask)
        cv2.waitKey(0)

    ### POINT 2: Find players and referees
    elif s == 2:
        image_contours, number = contours( image )
        print( "From 21 players and referees in the image, the system found ", number )
        cv2.imshow("Players & Referees", image_contours)
        cv2.waitKey(0)

    ### POINT 3: Lines
    elif s == 3:
        # Copy the image
        image_draw = np.copy( image )
        ## Save selected points
        points1 = []
        points2 = []
        ## Name the window
        cv2.namedWindow("Image")
        ## Mouse event
        cv2.setMouseCallback("Image", click )

        ## Blue points
        point_counter = 0
        while True:
            cv2.imshow("Image", image_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x") or key == ord("X") :
                points1 = points.copy()
                points = []
                break
            if len(points) > point_counter:
                point_counter = len(points)
                cv2.circle(image_draw, (points[-1][0], points[-1][1]), 5, [255, 0, 0], -1)

        N1 = len( points1 )
        pts1 = np.array(points1[:N1])

        image_draw = drawLine( image_draw, (pts1[0][0], pts1[0][1]), (pts1[1][0], pts1[1][1]), (255, 0, 0) )

        ## Yellow point
        point_counter = 0
        while True:
            cv2.imshow("Image", image_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x") or key == ord("X") :
                points2 = points.copy()
                points = []
                break
            if len(points) > point_counter:
                point_counter = len(points)
                cv2.circle(image_draw, (points[-1][0], points[-1][1]), 5, [0, 255, 255], -1)

        N2 = len(points2)
        pts2 = np.array(points2[:N2])

        m = slope( (pts1[0][0], pts1[0][1]), (pts1[1][0], pts1[1][1]) )

        image_draw = drawLine(image_draw, (0, -(pts2[0][0]-0)*m+pts2[0][1] ) , (pts2[0][0], pts2[0][1]), (0, 255, 255))

        cv2.imshow("Image", image_draw)
        cv2.waitKey(0)
    else:
        print("You selected an invalid option.")