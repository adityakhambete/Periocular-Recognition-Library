import numpy as np
import cv2

def getRegion(eye, y1, y2, x1, x2):
    w, h = x2-x1, y2-y1
    ex1, ey1, w1, h1 = eye

    if (x1 < ex1 < x1+w//2) and (x1 < ex1 + w1 < x1+w//2): 
        return 1, w1*h1
    else:
        return 2, w1*h1
    
def getPersonImages(ID, path, imcount):
    eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    cap = cv2.VideoCapture(1)
    i = 0
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        y1, y2 = 150, 300
        x1, x2 = 150, 450
        new_img = img[y1:y2, x1:x2]
        cv2.rectangle(img,(x1,y1),(x2, y2),(255,0,0),1)
        cv2.line(img, ((x1+x2)//2, y1), ((x1+x2)//2, y2), (0, 0, 255), 1)

        eyes = eye_cascade.detectMultiScale(new_img, 1.3, 1)


        if (len(eyes) == 2 ):
            r1, a1 = getRegion(eyes[0], y1, y2, x1, x2)
            r2, a2 = getRegion(eyes[1], y1, y2, x1, x2)

            # for (ex,ey,ew,eh) in eyes:
            #     ex += x1
            #     ey += y1
            #     cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,255,255),1)

            if (r1 != r2) and (a1 >= 6500 and a2 >= 6500):
                cv2.rectangle(img,(x1,y1),(x2, y2),(0,255,0),1)
                if i < imcount:
                    i += 1
                    cv2.imwrite(path + ID + '_L_' +str(i)+'.jpg',new_img[1:,1:(x2-x1)//2])
                    cv2.imwrite(path + ID + '_R_' +str(i)+'.jpg',new_img[1:,(x2-x1)//2 +1:])
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        else:
            cv2.rectangle(img,(x1,y1),(x2, y2),(0,0,255),1)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:        
            break

