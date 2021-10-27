# facial_detection_video.py
# USAGE
# python facial_detection_video.py --prototxt deploy.prototxt.txt --modelo res10_300x300_ssd_iter_140000.caffemodel

# importar los paquetes necesarias
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construcción del objeto argumento para la ejecución del código y agregar cada uno de los argumentos al objeto
ap = argparse.ArgumentParser()
# ruta a la arquitectura del modelo de la red neuronal
ap.add_argument("-p", "--prototxt", required=True,
                help="ruta al archivo Caffe prototxt file")
# ruta al modelo pre-entrenado de la red neuronal, en este archivo se encuentran valores para cada nodo de la red
ap.add_argument("-m", "--modelo", required=True,
                help="ruta al modelo Caffe pre-entrenado")
# valor mínimo de probabilidad para que el modelo de la cara sea aceptado, esto para evitar falso positivos
ap.add_argument("-c", "--confianza", type=float, default=0.5,
                help="mínima probabilidad para filtrar detecciones débiles")
args = vars(ap.parse_args())

# cargar modelo de la red neural
print("[INFO] cargando modelo...")
red = cv2.dnn.readNetFromCaffe(args["prototxt"], args["modelo"])

# inicializar el video
print("[INFO] empezando el video...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# ciclo sobre los frames del video
while True:
    # tomar el frame del video y cambiar su tamaño para tener un máximo de 400 píxeles de ancho
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # tomar el frame y encontrar la región de interés (blob)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pasar la región de interés (blob) a través de la red para realizar las detecciones y predicciones
    red.setInput(blob)
    detecciones = red.forward()

    # ciclo sobre las detecciones
    for i in range(0, detecciones.shape[2]):
        # extraer la confianza (probabilidad) asociada a al predicción
        confianza = detecciones[0, 0, i, 2]

        # filtrar detecciones débiles asegurando que la confianza (probabilidad) de la imagen es mayor a la confianza mínima
        if confianza > args["confianza"]:
            # encontrat la pareja ordenada (x,y) compute para la bounding box del objeto
            box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
            (comienzo_X, comienzo_Y, final_X, final_Y) = box.astype("int")

            # dibujar la bounding box de la cara junto con su probabilidad
            texto = "{:.2f}%".format(confianza * 100)
            y = comienzo_Y - 10 if comienzo_Y - 10 > 10 else comienzo_Y + 10
            cv2.rectangle(frame, (comienzo_X, comienzo_Y), (final_X, final_Y),
                          (0, 0, 255), 2)
            cv2.putText(frame, texto, (comienzo_X, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # mostrar el frame de salida
    cv2.imshow("Frame de salida", frame)
    key = cv2.waitKey(1) & 0xFF

    # si se oprime la tecla 'q' se rompe el ciclo
    if key == ord("q"):
        break

# limpiar las ventanas y detener el video
cv2.destroyAllWindows()
vs.stop()