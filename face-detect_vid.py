import cv2
import time

vid = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('face-cascade.xml')

a = 1

while True:
    a = a+1
    check, frame = vid.read()
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    final = face_cascade.detectMultiScale(
        result, scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in final:
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)

    print(frame)

    cv2.imshow("cap", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(a)

vid.release()

cv2.destroyAllWindows()
