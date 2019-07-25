import cv2
import sys

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

facexml = cv2.CascadeClassifier('face-cascade.xml')

result = facexml.detectMultiScale(
    gray, scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(format(len(result)))

for (x, y, w, h) in result:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("faces", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
