import cv2
import face_recognition

#With the usual OpenCV procedure, we extract the image, in this case, Messi1.webp, and convert it into RGB color format. Then we do the “face encoding” with the functions of the Face recognition library.

img = cv2.imread("Messi1.webp")
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("images/Messi.webp")
rgb_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

#With a single line, we make a simple face comparison and print the result. If the images are the same it will print True otherwise False.

result = face_recognition.compare_faces([img_encoding],img_encoding2)
print("Result ",result)

cv2.imshow("Img",img)
cv2.imshow("Img2",img2)
cv2.waitKey(0)