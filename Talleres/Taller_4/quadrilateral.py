# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 4
# Clase quadrilateral

# Importar librerías
import cv2
import numpy as np
import random as rd

# Definición de funciones
# random_number: generar numeros aleatorios entre el numero de inicio y final
def random_number( inicio, final ):
    return rd.randint( inicio, final )


# Construcción de la clase quadrilateral
class quadrilateral:

    # Constructor de la clase
    def __init__(self, N):
        self.N = N

    # Métodos de la clase
    # generate: generar un cuadrilatero de color magenta sobre un fondo cian
    def generate(self):
        ### CREAR IMAGEN
        image = np.zeros((self.N, self.N, 3), dtype=np.uint8)
        # Color Cian
        cian = [255, 255, 0] # [ B, G, R ]
        # Poner color de fondo
        image[:] = cian

        ### CUADRANTES
        # Primer cuadrante
        c1_start = [ 0, 0 ]
        c1_last = [ int( self.N/2 ), int( self.N/2 ) ]
        # Segundo cuadrante
        c2_start = [ int( self.N/2 ) + 1, 0 ]
        c2_last = [ self.N, int( self.N/2 ) + 1 ]
        # Tercer cuadrante
        c3_start = [ int( self.N/2 ) + 1, int( self.N/2 ) + 1  ]
        c3_last = [ self.N, self.N ]
        # Cuarto cuadrante
        c4_start = [ 0, int( self.N/2 ) + 1 ]
        c4_last = [ int( self.N/2 ) + 1, self.N ]

        ## Puntos aleatorios en cada cuadrante
        # Primer cuadrante
        x1 = random_number( c1_start[0], c1_last[0] )
        y1 = random_number( c1_start[1], c1_last[1] )
        punto_1 = [ x1, y1 ]
        # Segundo cuadrante
        x2 = random_number( c2_start[0], c2_last[0] )
        y2 = random_number( c2_start[1], c2_last[1] )
        punto_2 = [x2, y2]
        # Tercer cuadrante
        x3 = random_number( c3_start[0], c3_last[0] )
        y3 = random_number( c3_start[1], c3_last[1] )
        punto_3 = [x3, y3]
        # Cuarto cuadrante
        x4 = random_number( c4_start[0], c4_last[0] )
        y4 = random_number( c4_start[1], c4_last[1] )
        punto_4 = [x4, y4]

        ### UNION DE LOS PUNTOS
        # Polígono
        puntos = np.array( [ punto_1, punto_2,
                             punto_3, punto_4 ],
                            np.int32 )
        # puntos = puntos.reshape((-1, 1, 2))
        # Cerrar el polígono
        isClosed = True

        # Color del cuadrilátero
        magenta =(118, 52, 207)

        # Rellenar el cuadrilatero
        image = cv2.fillPoly( image, [puntos], magenta )

        return image
