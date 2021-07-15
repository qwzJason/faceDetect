import cv2 as cv

# turn on the camera and keep it running
cap = cv.VideoCapture(0)
cascPath = r"C:\Users\Jason\Desktop\FacialRecogn\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)

# face detection using haarcascade_frontalface_default.xml
while True:
    isTrue, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # face detection
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow('frame', frame)

    # turn off the camera if q is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()






