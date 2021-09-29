# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 4
# main: Taller_4.py

# Importar librerías
import cv2
import random as rd
import os
import sys
import numpy as np
from enum import Enum
from _7_lines_detection.hough import Hough
from _6_edges_and_orientation.orientation_methods import gradient_map
from quadrilateral import *

### CLASES
class Methods(Enum):
    Standard = 1
    Direct = 2

### DEFINICIÓN DE FUNCIONES
# cargarN: cargar tamaño N
def cargarN():
    val = False
    while val == False:
        x = int( input( "Ingresar tamaño N: " ) )
        val = esPar( x )
        if val == False:
            print( "El número ingresado es impar, ingreselo nuevamente: " )
    return x

# Revisar Par: función encargada de revisar si el número ingresado es par o no
def esPar( N ):
    if N % 2 == 0:
        return True
    else:
        return False
# det: determinandte
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

# interseccion: calcula la interseccion entre dos líneas
def interseccion( line_1, line_2 ):
    x_diff = ( line_1[0][0] - line_1[1][0], line_2[0][0] - line_2[1][0])
    y_diff = ( line_1[0][1] - line_1[1][1], line_2[0][1] - line_2[1][1])

    div = det(x_diff, y_diff)

    d = (det(*line_1), det(*line_2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y

# DetectCorners: función encargada de encontrar las esquinas del polígono y dibujar circunferencias en
#                en las esquinas para su visualización
def DetectCorners( image ):

    lines = []

    method = Methods.Standard
    high_thresh = 300
    bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

    hough = Hough(bw_edges)
    if method == Methods.Standard:
        accumulator = hough.standard_transform()
    elif method == Methods.Direct:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        theta, _ = gradient_map(image_gray)
        accumulator = hough.direct_transform(theta)
    else:
        sys.exit()

    acc_thresh = 50
    N_peaks = 4
    nhood = [25, 9]
    peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

    _, cols = image.shape[:2]
    image_draw = np.copy(image)

    for peak in peaks:
        rho = peak[0]
        theta_ = hough.theta[peak[1]]

        theta_pi = np.pi * theta_ / 180
        theta_ = theta_ - 180
        a = np.cos(theta_pi)
        b = np.sin(theta_pi)
        x0 = a * rho + hough.center_x
        y0 = b * rho + hough.center_y
        c = -rho
        x1 = int(round(x0 + cols * (-b)))
        y1 = int(round(y0 + cols * a))
        x2 = int(round(x0 - cols * (-b)))
        y2 = int(round(y0 - cols * a))

        lines.append([ [x1,y1], [x2, y2] ])

        image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

    # print(lines)

    ### Intersección entre las líneas encontradas del polígono
    esquinas = []

    # Línea 1
    for i in range(1, len(lines)):
        x = interseccion(lines[0], lines[i])
        if x[0] <= image.shape[0] and x[1] <= image.shape[1]:
            esquinas.append( [int(x[0]), int(x[1])] )
    # Línea 2
    for i in range(2, len(lines)):
        x = interseccion(lines[1], lines[i])
        if x[0] <= image.shape[0] and x[1] <= image.shape[1]:
            esquinas.append( [int(x[0]), int(x[1])] )
    # Línea 4
    for i in range(3, len(lines)):
        x = interseccion(lines[2], lines[i])
        if x[0] <= image.shape[0] and x[1] <= image.shape[1]:
            esquinas.append( [int(x[0]), int(x[1])] )


    ### DIBUJAR ESQUINAS
    # print(esquinas)
    radio = 10
    ancho_linea = 2
    amarillo = (0, 255, 255)

    for centro in esquinas:
        coordenadas = ( centro[0], centro[1] )
        image = cv2.circle( image, coordenadas, radio, amarillo, ancho_linea)

    # cv2.imshow("frame", bw_edges)
    # cv2.imshow("lines", image_draw)
    # cv2.imshow("Esquinas", image)
    # cv2.waitKey(0)

    return image, esquinas


if __name__ == '__main__':
    # Tamaño de la imagen
    N = cargarN()
    # Objeto cuadrilatero de la clase quadrilateral
    cuadrilatero = quadrilateral( N )
    # Generar cuadrilatero
    image = cuadrilatero.generate()

    # Mostrar imagen
    # cv2.imshow("Cuadrilatero", image)
    # cv2.waitKey( 0 )

    imagen_esquinas, coordenadas_esquinas = DetectCorners(image)

    # Imprimir coordenas
    print("Coordenas de las esquinas: ")
    print(coordenadas_esquinas)

    # Mostrar imagen
    cv2.imshow("Cuadrilatero esquinas", imagen_esquinas)
    cv2.waitKey( 0 )
