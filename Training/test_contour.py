import cv2
im = cv2.imread('/home/captain_jack/Codes/OCR/crop1.jpg')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,40,100)
im2, cnts, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,cnts,-1,(0,255,255),1)
#cnts = sorted(cnts,key=cv2.contourArea,reverse=True)

#cv2.imshow('Edges',edges)
count = 0
##l = []
for i in range(len(cnts)):
    x,y,w,h = cv2.boundingRect(cnts[i])
    ar = w / float(h)
    if ar >=0.5 and ar <=2:
        if (w >= 15 and w <=50) and (h>=15 and h <=80):
            ##l = l + [i]
            cv2.putText(im,str(count),(x,y),cv2.FONT_ITALIC,0.4,(0,255,0),1)
            #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
            count  = count + 1
            
    #roi = im[y:y+h, x:x+w]
    #cv2.imshow('iniviual_contours',roi)
    #if cv2.waitKey(2000):
        #cv2.destroyAllWindows()
cv2.putText(im, 'Contours Found : '+str(count),(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)

##x,y,w,h = cv2.boundingRect(cnts[l[1]])
##roi = im[y:y+h, x:x+w]
#a1 = cv2.resize(roi,(64,64))
#a2 = cv2.resize(roi,(32,32))
##a3 = cv2.resize(roi,(128,128))
##cv2.imshow('resize32',a2)
##cv2.imshow('resize64',a1)
##cv2.imshow('resize128',a3)

cv2.imshow('Contours',im)
if cv2.waitKey(90000):
    cv2.destroyAllWindows()

