# IMPORTACION DE NUESTRAS LIBrerias
import face_recognition as fr
import numpy as np
import cv2

camara = cv2.VideoCapture(0)

#cargara la imagen
nnielpro_imagen = fr.load_image_file("niel.jpg")
niel_codigo_rostro = fr.face_encodings(nnielpro_imagen)[0]

#decodificaccion del rostro
codcarared = [
    niel_codigo_rostro
]

#colocacion del nombre
nombre = [
    "nnielpro"
]

while True:
    #toma de datos
    ret, frame = camara.read()
    #convercion de colores BGR A RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #ECONTRAR DETALLES DEL ROSTRO
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (arriba, derecha, abajo, izquierda), face_encodings in zip(face_locations, face_encodings):
        noresult = fr.compare_faces(codcarared, face_encodings)
        #si no encuntra el rostro
        dat = "ROSTRO NO DETECTADO"

        #cuando existe un rostro
        distancia = fr.face_distance(codcarared, face_encodings)
        coincidencia = np.argmin(distancia)

        if noresult[coincidencia]:
            nombre2 = nombre[coincidencia]

        #dibujo del cuadro
        cv2.rectangle(frame, (izquierda, arriba), (derecha, abajo), (0,255,0), 2)
        #creacion de la etiqueta
        cv2.rectangle(frame, (izquierda, abajo -35), (derecha, abajo), (0, 255,0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nombre2, (izquierda +6, abajo -6), font, 1.0, (255,255,255), 1)

    #nombre a la cqamra wed
    cv2.imshow('DETECTOR DE PERSONAS POR ROSTRO', frame)

    #apagar la camara
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

#liberacion de la camara
camara.release()
cv2.destroyAllWindows()