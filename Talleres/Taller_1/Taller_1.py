# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 1
# main: Taller_1.py

# Importar librerías
import cv2
from basicColor import *

# Main
if __name__ == '__main__':
    # Solicitar ruta de la imagen al usuario
    ruta_imagen = input("Por favor, ingrese ruta de la imagen: ")
    # ruta_imagen = "C:\Imagenes_Video\Images\soccer_game.png"
    # Crear objeto color de la clase basicColor
    color = basicColor(ruta_imagen)
    # Mostrar imagen de entrada
    cv2.imshow("Imagen de entrada", color.imagen)
    # cv2.waitKey(0)
    # Visualizar número de píxeles y canales de la imagen
    color.displayProperties()
    # Visualizar imagen binarizada con el método de OTSU
    imagen_binarizada = color.makeBW()
    cv2.imshow("Imagen binazirada", imagen_binarizada)
    # cv2.waitKey(0)
    # Solicitar al usuario la entrada de hue
    hue = input("Por favor, ingrese un valor de Hue entre 0 y 179: ")
    # Visualizar imagen colorizada
    imagen_colorizada = color.colorize(hue)
    cv2.imshow("Imagen colorizada", imagen_colorizada)
    cv2.waitKey(0)


