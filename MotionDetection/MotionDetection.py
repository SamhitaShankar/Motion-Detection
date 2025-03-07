import cv2
import imutils
import time
cam = cv2.VideoCapture(0)  
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
area_threshold = 500
motion_timeout = 5  
buffer_size = 10    


last_motion_time = time.time()
motion_detected = False
frame_buffer = []

while True:
    
    ret, img = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    
    img = imutils.resize(img, width=500)
    
    
    fg_mask = bg_subtractor.apply(img)
    
    
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    fg_mask = cv2.erode(fg_mask, None, iterations=1)
    
    
    cnts = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    
    motion_detected = False
    
    for c in cnts:
        if cv2.contourArea(c) < area_threshold:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True
    
    
    if motion_detected:
        last_motion_time = time.time()

   
    frame_buffer.append(fg_mask)
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    
    if not motion_detected and (time.time() - last_motion_time) > motion_timeout:
        
        
        pass
    
    
    text = "Normal"
    if motion_detected:
        text = "Moving Object detected"
    
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    cv2.imshow("cameraFeed", img)
    
    
    key = cv2.waitKey(10)
    if key == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()
