# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 2
# main: Taller_2.py

# Importar librerías
import cv2
import os
from thetaFilter import *

# Definición de funciones
# promedio_imagen: función que promedia 4 imágenes
def promedio_imagen(imagen_1, imagen_2, imagen_3, imagen_4):
    imagen_promedio = (imagen_1 + imagen_2 + imagen_3 + imagen_4) / 4
    return imagen_promedio

# filtros_direccionales: función que aplica 4 distintos filtros a una imagen de entrada
def filtros_direccionales( imagen_gris, theta_1, theta_2, theta_3, theta_4, delta_theta ):
    # Filtro 1
    filtro_1 = thetaFilter(imagen_gris)
    filtro_1.set_theta(theta_1, delta_theta)
    filtro_1.filtering()
    str1 = "Imagen con F1: " + str(theta_1) + " grados"
    cv2.imshow(str1, filtro_1.imagen_filtrada)
    # cv2.imshow("Frecuencia F1", 255 * filtro_1.mascara)
    # cv2.waitKey(0)

    # Filtro 2
    filtro_2 = thetaFilter(imagen_gris)
    filtro_2.set_theta(theta_2, delta_theta)
    filtro_2.filtering()
    str2 = "Imagen con F2: " + str(theta_2) + " grados"
    cv2.imshow(str2, filtro_2.imagen_filtrada)
    # cv2.imshow("Frecuencia F2", 255 * filtro_2.mascara)
    # cv2.waitKey(0)

    # Filtro 3
    filtro_3 = thetaFilter(imagen_gris)
    filtro_3.set_theta(theta_3, delta_theta)
    filtro_3.filtering()
    str3 = "Imagen con F3: " + str(theta_3) + " grados"
    cv2.imshow(str3, filtro_3.imagen_filtrada)
    # cv2.imshow("Frecuencia F3", 255 * filtro_3.mascara)
    # cv2.waitKey(0)

    # Filtro 4
    filtro_4 = thetaFilter(imagen_gris)
    filtro_4.set_theta(theta_4, delta_theta)
    filtro_4.filtering()
    str4 = "Imagen con F4: " + str(theta_4) + " grados"
    cv2.imshow(str4, filtro_4.imagen_filtrada)
    # cv2.imshow("Frecuencia F4", 255 * filtro_4.mascara)
    # cv2.waitKey(0)

    # Promedio
    imagen_promedio = promedio_imagen(filtro_1.imagen_filtrada, filtro_2.imagen_filtrada, filtro_3.imagen_filtrada, filtro_4.imagen_filtrada)
    cv2.imshow("Imagen Promedio", imagen_promedio)
    cv2.waitKey(0)

# Main
if __name__ == '__main__':
    ruta = "C:\Imagenes_Video\Images\huellas"
    nombre_imagen = "01_4.tif"
    ruta_archivo = os.path.join(ruta, nombre_imagen)
    # Leer imagen
    imagen = cv2.imread(ruta_archivo)
    # Convertir imagen a grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Imagen original
    cv2.imshow("Imagen original", imagen)

    # Aplicar filtros dirreccionales
    filtros_direccionales(imagen_gris, 0, 45, 90, 135, 30)

    ### PRUEBAS
    '''
    # Crear Objeto theta_filter de la Clase thetaFilter
    theta_filter = thetaFilter(imagen_gris)
    # Asignar ángulo y umbral
    theta_filter.set_theta(135, 20)
    # Filtrado
    theta_filter.filtering()

    # Imagen original
    cv2.imshow("Imagen original", imagen)
    # FFT imagen
    cv2.imshow("Transformada de Fourier de la Imagen", theta_filter.fft_imagen)
    # Respuesta en frecuencia del filtro
    cv2.imshow("Respuesta en frecuencia del filtro", 255 * theta_filter.mascara)
    # Imagen original filtrada
    cv2.imshow("Imagen filtrada", theta_filter.imagen_filtrada)
    cv2.waitKey(0)
    '''
