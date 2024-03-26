import cv2

#crear la cascada de haar
#patrón para reconocer rostros:
cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vc = cv2.VideoCapture(0)

while True: 
    #captura diapositiva por diapositiva
    ret, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
#Detectar los rostros en la imgOrignialn
    rostros = cc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    #dibuja un rectangulo alrededor de los rostros
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #muestra el resultado
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#libera la memoria de la captura de video
vc.release()
cv2.destroyAllWindows()