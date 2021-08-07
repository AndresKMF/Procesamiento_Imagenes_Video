# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 1
# Clase basicColor

# Importar librerías
import cv2

# Construcción de la clase basicColor
class basicColor:

    # Atributos de la clase
    ruta_imagen = ''
    imagen = []
    imagen_gris = []
    imagen_hsv = []

    # Constructor de la clase
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen
        self.imagen = cv2.imread(ruta_imagen)
        self.imagen_gris = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        self.imagen_hsv = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2HSV)

    # Métodos de la clase
    # displayPropesties: mostrar algunas propiedades de la imagen como, número de píxeles y canales
    def displayProperties(self):
        shape = self.imagen.shape
        pixeles = shape[0] * shape[1]
        pixeles_MP = float(pixeles)/float(1000000)
        canales = shape[2]
        print("Número de píxeles de la imagen: ", pixeles_MP, " MP" )
        print("Número de canales de la imagen: ", canales )

    # makeBW: Imagen binarizada a través del método Otsu
    #         Retorna la imagen binarizada
    def makeBW(self):
        # Umbral global Otsu
        ret, ibw_otsu = cv2.threshold(self.imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ibw_otsu

    # colorize: colorizar la imagen basado en un valor de Hue ingresado por el usuario
    #           Recibe un valor de Hue
    #           Retorna la imagen colorizada
    def colorize(self, h):
        # Colorizar imagen
        imagen_hue = self.imagen_hsv
        imagen_hue[..., 0] = h
        # Convertir imagen nuevamente a BGR
        imagen_colorizada = cv2.cvtColor(imagen_hue, cv2.COLOR_HSV2BGR)
        return imagen_colorizada


