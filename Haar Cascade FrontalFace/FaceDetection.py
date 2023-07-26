import cv2

Algorithm = "haarcascade_frontalface_default.xml"


HaarCascade = cv2.CascadeClassifier(Algorithm)

Camera = cv2.VideoCapture(0)

while True:
    _,Image = Camera.read()
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    Face = HaarCascade.detectMultiScale(GrayImage, 1.3, 4)

    for (x,y,w,h) in Face:
        cv2.rectangle(Image, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow("FaceDetection.",Image)
        Key = cv2.waitKey(10)
        if Key == 27:
            break;

cv2.Camera.release()
cv2.destroyAllWindows()


