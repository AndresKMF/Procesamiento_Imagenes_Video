# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 2
# Clase thetaFilter

# Importar librerías
import cv2
import numpy as np

# Construcción de la clase thetaFilter
class thetaFilter:

    # Atributos de la clase
    imagen_gris = []
    fft_imagen = []
    mascara = []
    imagen_filtrada = []
    theta = 0
    delta_theta = 0

    # Constructor de la clase
    def __init__(self, imagen_gris):
        self.imagen_gris = imagen_gris

    # Método set_theta: recibir los parámetros theta y delta_theta que definirán
    # la respuesta del filtro
    def set_theta(self, theta, delta_theta):
        self.theta = theta
        self.delta_theta = delta_theta

    # Método filtering: permite solo el paso de componentes de frecuencia orientados en theta - delta_theta
    # y en theta + delta_theta
    def filtering(self):
        # Transformada de fourier (FFT) de la imagen en gris
        fft_imagen_gris = np.fft.fft2(self.imagen_gris)
        # Desplazar la FFT de la imagen
        fft_imagen_gris_desplazada = np.fft.fftshift(fft_imagen_gris)

        # Visualización de la FFT
        # Valor absoluto de la FFT desplazada
        fft_imagen_gris_magnitud = np.absolute(fft_imagen_gris_desplazada)
        # Aplicamos algortimo para poder visualizar
        fft_imagen_vista = np.log(fft_imagen_gris_magnitud + 1)
        # Normalizar la FFT
        fft_imagen_vista = fft_imagen_vista / np.max(fft_imagen_vista)
        # Mostrar FFT de la imagen
        self.fft_imagen = fft_imagen_vista
        # cv2.imshow("FFT Imagen", fft_imagen_vista)

        # Pre procesamiento
        # Extracción del tamaño de la imagen
        num_filas, num_columnas = (self.imagen_gris.shape[0], self.imagen_gris.shape[1])
        # Vectores de filas y columnas
        enum_filas = np.linspace(0, num_filas - 1, num_filas)
        enum_columnas = np.linspace(0, num_columnas - 1, num_columnas)
        # Matriz para iterar en toda la imagen
        iterador_columnas, iterador_filas = np.meshgrid(enum_columnas, enum_filas)
        # Encontrar el centro de la imagen
        mitad_filas = num_filas / 2  # en filas
        mitad_columnas = num_columnas / 2  # en columnas

        # Máscara del filtro
        mascara_1 = np.zeros_like(self.imagen_gris)
        # Límite inferior
        idx_bajo = (180 * np.arctan2((iterador_columnas - mitad_columnas), (iterador_filas - mitad_filas)) / np.pi ) > (self.theta - self.delta_theta)
        # Límite superior
        idx_alto = (180 * np.arctan2((iterador_columnas - mitad_columnas), (iterador_filas - mitad_filas)) / np.pi ) < (self.theta + self.delta_theta)
        # Realizar AND entre los límites
        idx_total = np.bitwise_and(idx_bajo, idx_alto)
        # Traer solamente las componentes en frecuenciaque cumplan el rango
        mascara_1[idx_total] = 1
        # Agregar componente DC
        mascara_1[int(mitad_filas), int(mitad_columnas)] = 1

        # cv2.imshow("Mascara", mascara)
        # Rotar 180 grados la primera máscara
        mascara_2 = cv2.rotate(mascara_1, cv2.ROTATE_180)
        # Juntar las dos máscaras
        mascara = np.bitwise_or(mascara_1, mascara_2)
        # Filtrado
        # Aplicar máscara a la FFT de la imagen desplazada
        fft_filtrada = fft_imagen_gris_desplazada * mascara
        # Guardar máscara

        self.mascara = mascara
        # FFT inversa para obtener la imagen filtrada
        imagen_filtrada = np.fft.ifft2(np.fft.fftshift(fft_filtrada))
        imagen_filtrada = np.absolute(imagen_filtrada)
        imagen_filtrada = imagen_filtrada / np.max(imagen_filtrada)
        # Guardar imagen filtrada
        self.imagen_filtrada = imagen_filtrada

