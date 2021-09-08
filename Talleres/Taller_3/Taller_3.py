# Pontificia Universidad Javeriana
# Procesamiento de Imágenes y Video
# Luis Miguel Muñoz Flórez
# Taller 3
# main: Taller_3.py

# Importar librerías
import cv2
import os
import numpy as np
from _3_Fourier_analysis.fft_show import FFT

# Definición de funciones
# cargar_imagen: función encargada de cargar una imagen
def cargar_imagen( ruta_imagen, nombre_imagen, extension_imagen ):
    imagen_con_formato = nombre_imagen + extension_imagen
    ruta_archivo = os.path.join( ruta_imagen, imagen_con_formato )
    imagen = cv2.imread( ruta_archivo )
    return imagen
# filtro_pasa_bajas_fft: filtrado de la imagen a través de un filtro FFT. Retorna la
#                        mascara del filtro y la imagen filtrada
def filtro_pasa_bajas_fft( objeto_fft, factor ):
    # Imagen en grises
    imagen_gris = objeto_fft.image_gray

    # Pre procesamientos
    num_filas, num_columnas = ( imagen_gris.shape[0], imagen_gris.shape[1] )
    enum_filas = np.linspace( 0, num_filas - 1, num_filas )
    enum_columnas = np.linspace( 0, num_columnas - 1, num_columnas )
    columnas_iter, filas_iter = np.meshgrid( enum_columnas, enum_filas )
    tam_medio_filas = num_filas / 2 - 1
    tam_medio_columnas = num_columnas / 2 - 1

    # Máscara filtro pasa bajas (LP)
    mascara_pasa_bajos = np.zeros_like( imagen_gris )
    frecuencia_corte = 1 / factor
    radio_corte_filas = int( frecuencia_corte * tam_medio_filas )
    radio_corte_columnas = int( frecuencia_corte * tam_medio_columnas )
    idx_pb_filas = np.sqrt( ( columnas_iter - tam_medio_filas ) ** 2 + ( filas_iter - tam_medio_filas ) ** 2 ) < radio_corte_filas
    idx_pb_columnas = np.sqrt( ( columnas_iter - tam_medio_columnas ) ** 2 + ( filas_iter - tam_medio_columnas ) ** 2 ) < radio_corte_columnas
    idx_pb = np.bitwise_and( idx_pb_filas, idx_pb_columnas )
    mascara_pasa_bajos[idx_pb] = 1

    ### Filtrado via FFT
    ### Aplicar pasa bajos a la imagen
    mascara = mascara_pasa_bajos
    fft_filtrada = objeto_fft.image_gray_fft_shift * mascara
    imagen_filtrada = np.fft.ifft2( np.fft.fftshift( fft_filtrada ) )
    imagen_filtrada = np.absolute( imagen_filtrada )
    imagen_filtrada /= np.max( imagen_filtrada)

    return mascara_pasa_bajos, imagen_filtrada

# diezmado_D: Diezamdo de la imagen por un factor D
#             Retorna la imagen diezmada
def diezmado_D( imagen, D ):

    ### FFT de la imagen
    FFT_imagen = FFT( imagen, 512 )
    FFT_imagen.display()

    ### Filtro pasa bajas
    mascara_pasa_bajos, imagen_filtrada = filtro_pasa_bajas_fft( FFT_imagen, D )
    # Respuesta en frecuencia
    # cv2.imshow( "LP", 255 * mascara_pasa_bajos )
    # Imagen filtrada
    # cv2.imshow( "Imagen filtrada", imagen_filtrada )

    ### Diezmado
    imagen_diezmada = imagen_filtrada[::D, ::D]
    return imagen_diezmada

# interpolacion_I: Interpola la imagen por un factor I
#                  Retorna la imagen interpolada
def interpolacion_I( imagen, I ):

    ### Convertir en gris la imagen
    # imagen_gris = cv2.cvtColor( imagen, cv2.COLOR_BGR2GRAY )
    imagen_gris = imagen

    ### Interpolación
    ### Insertar ceros
    filas, columnas = imagen_gris.shape
    num_ceros = I
    imagen_ceros = np.zeros( ( num_ceros * filas, num_ceros * columnas), dtype = imagen_gris.dtype )
    imagen_ceros[::num_ceros, ::num_ceros] = imagen_gris

    ### FFT de la imagen
    FFT_imagen = FFT( imagen_ceros, 512 )
    FFT_imagen.display() # mostrar fft de la imagen

    ### Filtro pasa bajas
    mascara_pasa_bajos, imagen_filtrada = filtro_pasa_bajas_fft( FFT_imagen, I )
    # Respuesta en frecuencia
    # cv2.imshow( "LP", 255 * mascara_pasa_bajos )
    # Imagen filtrada
    # cv2.imshow( "Imagen filtrada", imagen_filtrada )

    imagen_interpolada = imagen_filtrada
    return imagen_interpolada

# convolucion_diezmado: aplicar el banco de filtros sobre la imagen de entrada y diezma las imágenes resultantes
#                       por un factor de 2. Retorna una lista con las imagenes filtradas y diezmadas
def convolucion_diezmado( imagen, filtros ):

    nivel_descomposicion = []
    ### CONVOLUCIONES
    imagen_IH = cv2.filter2D(imagen, -1, filtros[0])
    imagen_IV = cv2.filter2D(imagen, -1, filtros[1])
    imagen_ID = cv2.filter2D(imagen, -1, filtros[2])
    imagen_IL = cv2.filter2D(imagen, -1, filtros[3])

    ### MOSTRAR FILTROS
    # cv2.imshow("IH", imagen_IH)
    # cv2.imshow( "IV", imagen_IV )
    # cv2.imshow( "ID", imagen_ID )
    # cv2.imshow( "IL", imagen_IL )

    ### DIEZMADOS
    ## Factor de interpolación
    D = 2
    # Diezmado
    imagen_IH_diezmada = diezmado_D(imagen_IH, D)
    imagen_IV_diezmada = diezmado_D(imagen_IV, D)
    imagen_ID_diezmada = diezmado_D(imagen_ID, D)
    imagen_IL_diezmada = diezmado_D(imagen_IL, D)

    # cv2.imshow("H Diezmada", imagen_IH_diezmada)

    nivel_descomposicion.append(imagen_IH_diezmada)
    nivel_descomposicion.append(imagen_IV_diezmada)
    nivel_descomposicion.append(imagen_ID_diezmada)
    nivel_descomposicion.append(imagen_IL_diezmada)

    return nivel_descomposicion

# descomposicion: descomposicion de la imagen a travez de 4 filtros, a un nivel N
#                 retorna una lista con todas las imagenes de la descomposición de la misma
def descomposicion( imagen, N ):

    lista_imagenes_resultantes = []
    filtros = []
    ### BANCO DE FILTROS
    ## Filtro Horizontal (H)
    H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtros.append(H)
    ## Filtro Vertical (V)
    V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filtros.append(V)
    ## Filtro Diagonal (D)
    D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
    filtros.append(D)
    ## Filtro pasa bajas, Low-Pass (L)
    L = np.array([[1/2, 1, 1/2], [1, 2, 1], [1/2, 1, 1/2]]) * (1/8)
    filtros.append(L)

    ### IMAGEN A GRIS
    imagen_gris = cv2.cvtColor( imagen, cv2.COLOR_BGR2GRAY )

    ### DESCOMPOSICIÓN DE LA IMAGEN
    for i in range(1, N+1):
        if i == 1:
            imagen_entrada = imagen
        else:
            imagen_entrada = aux[3]
        aux = convolucion_diezmado( imagen_entrada, filtros )
        lista_imagenes_resultantes.append( aux )

    return lista_imagenes_resultantes

########## MAIN ##########
if __name__ == '__main__':

    ########## CARGA DE LA IMAGEN ##########
    # Ruta, nombre y extensión de la imagen a utilizar
    ruta_imagen = 'C:\Imagenes_Video\Images'
    nombre_imagen = 'lena'
    extension_imagen = '.png'
    # cargar imagen
    imagen = cargar_imagen( ruta_imagen, nombre_imagen, extension_imagen )
    # mostrar imagen original
    cv2.imshow("Lena", imagen)

    ### DESCOMPOSICIÓN DE LA IMAGEN
    ## Factor de descomposición N
    N = 2
    imagenes_resultantes = descomposicion( imagen, N )

    ## Ver imágenes resultantes
    for i in range( len( imagenes_resultantes ) ):
        for j in range( len( imagenes_resultantes[i] ) ):
            imagenes = imagenes_resultantes[i]
            cv2.imshow("H" + str( i + 1 ), imagenes[0])
            cv2.imshow("V" + str( i + 1 ), imagenes[1])
            cv2.imshow("D" + str( i + 1 ), imagenes[2])
            cv2.imshow("L" + str( i + 1 ), imagenes[3])
        if i == len( imagenes_resultantes ) - 1:
            ultima_imagen = imagenes_resultantes[i][3]

    imagen_ILL = ultima_imagen

    ### INTERPOLAR
    ## Factor de interpolación
    I = 2 ** N
    imagen_ILL_interpolada = interpolacion_I( imagen_ILL, I )

    ## Mostrar imagen interpolada
    cv2.imshow( "ILL Interpolada x " + str(I) , imagen_ILL_interpolada )

    cv2.waitKey( 0 )

    '''
    
    ############### PRUEBAS ###############    
    
    ############### DIEZMADO ###############
    
    ### Factor de diezmado
    D = 2

    ### Diezmado
    imagen_diezmada = diezmado_D( imagen, D )
    # Imagen diezmada
    cv2.imshow("Imagen diezmada", imagen_diezmada)
    print("DIEZMADO")
    print( "Tamaño imagen original: ", imagen.shape )
    print( "Tamaño imagen diezmada: ", imagen_diezmada.shape )

    ############### INTERPOLACIÓN ###############
    
    ### Factor de interpolación
    I = 2
    
    ### Interpolación
    imagen_interpolada = interpolacion_I( imagen, I )
    # Imagen interpolada
    cv2.imshow("Imagen interpolada", imagen_interpolada)
    print( "INTERPOLACIÓN" )
    print( "Tamaño imagen original: ", imagen.shape )
    print( "Tamaño imagen diezmada: ", imagen_interpolada.shape )
    
    ##########################################
    
    '''







