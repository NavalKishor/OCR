import cv2

def nothing(x):
    pass




cam = cv2.VideoCapture(0)
cv2.namedWindow('Real-time Contours')
cv2.createTrackbar('min','Real-time Contours',0,255,nothing)
cv2.createTrackbar('max','Real-time Contours',80,255,nothing)
while (1):
    mi = cv2.getTrackbarPos('min','Contours')
    ma = cv2.getTrackbarPos('max','Contours')
    
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,0,80)
    im2,cnts,h = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.putText(frame,'Contours Found : '+str(len(cnts)), (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(frame,'Contours plotted : '+str(90), (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    cv2.drawContours(frame,cnts[:100],-1,(0,255,255),2)
    cv2.imshow('Real-time Contours',frame)
    
    key = cv2.waitKey(10)
    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()


