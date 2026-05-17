import cv2
import numpy as np
import os 
from datetime import datetime

#setup output folders

os.makedirs('output/screenshots' , exist_ok=True)
os.makedirs('output/reports',exist_ok=True)

#session_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
session_log = []
total_cracks_session = 0
frame_count = 0

#Load stream
#cap = cv2.VideoCapture("http://10.86.165.39:8081/video")
cap = cv2.VideoCapture('rtsp://admin:admin@10.20.103.167:8554/live')

if not cap.isOpened():
    print("Cannot connect to camera")
else:
    print("Camera connected sucessfully")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_count +=1
    frame = cv2.resize(frame,(600,400))


    #"Convert to grayscale, create a blurred background, subtract to find dark cracks, amplify the signal, then make a black and white decision" 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    background = cv2.GaussianBlur(gray,(51,51),0)

    diff = cv2.subtract(background,gray)#subtracts the original from the blurred version 
    
    #Stretches the difference values to fill full 0-255 range.
    diff = cv2.normalize(diff,None,0,255,cv2.NORM_MINMAX ) 
    _, thresh = cv2.threshold(diff, np.mean(diff) + 20, 255,cv2.THRESH_BINARY)


    #Morphology-Remove noise dots with opening, then fill crack gaps with closing
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    closed = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel_close)

#Contour Detection-Find all shapes, then filter out anything too small, too square, or invalid — only long thin shapes survive"
    contours ,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    crack_count = 0
    has_critical = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.0001:
            continue
        x,y,w_rect,h_rect = cv2.boundingRect(cnt)
        if h_rect == 0 or w_rect == 0:
            continue
        ratio = max(w_rect,h_rect) / min(w_rect, h_rect)
        if ratio < 2:
            continue
        perimeter = cv2.arcLength(cnt,True)
        if perimeter < 150:
            continue
        if perimeter == 0:
            continue   
        # Severity Rating
        if area > frame_area * 0.005:
            severity = 'CRITICAL'
            color = (0,0,255)#red
            has_critical = True
        elif area > frame_area*0.001:
            severity = 'MODERATE'
            color = (0,165,255)#orange
        else:
            severity = 'MINOR'
            color = (0,255,255)#yellow
        #Draw  contour with severity color
        cv2.drawContours(frame ,[cnt],-1,color, 2)
        #label each crack
        cv2.putText(frame,severity,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
        crack_count +=1
        
        total_cracks_session += crack_count
        #Auto_screenshort
        if crack_count > 0 and frame_count % 30 ==0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = (f'output/screenshots/cracks_{timestamp}.jpg')
            cv2.imwrite(filename,frame)
            session_log.append({
                'time':timestamp,
                'crack_count':crack_count,
                'file':filename
            })

#Critical Alert
    fps = cap.get(cv2.CAP_PROP_FPS)


#Dashboard Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(600,80),(0,0),-1)
   # cv2.addWeight(overlay,0.6,frame,0.4,0,frame)

#Display info
    cv2.putText(frame,f'CRACK DETECTION SYSTEM',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6
    ,(255,255,255),2)

    cv2.putText(frame, f'Cracks:{crack_count}',(10,45),cv2.FONT_HERSHEY_SIMPLEX,
    0.5,(0,255,0),1)

    cv2.putText(frame, f'Session Total: {total_cracks_session}',
    (150, 45), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 0), 1)

    cv2.putText(frame,f'FPS: {fps:.0f}',
    (400,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    cv2.putText(frame,datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    (350,20),cv2.FONT_HERSHEY_SIMPLEX,
    0.5,(200,200,200),1)

    cv2.putText(frame, '● MINOR',(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.4,
    (0,255,255),1)
    cv2.putText(frame, '● MODERATE',
    (100,70),cv2.FONT_HERSHEY_SIMPLEX,
    0.4,(0,165,255),1)
    cv2.putText(frame,'● CRITICAL',
    (220,70),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

    #DISPLAY

    cv2.imshow("UAV Crack Detection System", frame)
    cv2.imshow("Difference",diff)
    cv2.imshow("Threshold",thresh)
    cv2.imshow("Processed",closed)

    # Press S to manually save screenshot
    # Press Q to quit and generate report
    key = cv2.waitKey(25) & 0xFF
    if key == ord('s'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output/screenshots/manual_{timestamp}.jpg'
        cv2.imwrite(filename,frame)
        print(f"Screenshot saved: {filename }")
    if key == ord('q'):
        break


#Generate Inspection Report 

cap.release()
cv2.destroyAllWindows()

report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
report_file = f'output/reports/inspection_{report_time}.txt'
with open(report_file,'w') as f:
    f.write("=" * 50 + "\n")
    f.write("UAV CRACK DETECTION INSPECTION REPORT\n")
    f.write(f"Session started : {session_log}\n")
    f.write(f"Session Ended : {datetime.now().strftime('Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Frames : {frame_count}\n")
    f.write(f"Screenshots: {len(session_log)}\n")
    f.write("=" * 50 +"\n\n")
    f.write("DETECTION LOG :\n\n")
    for log in session_log:
        f.write(f"Time: {log['time']}| Cracks: {log['crack_count']} | File: {log['file']}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("END OF REPORT\n")
print(f"\n Inspection report saved: {report_file}")
print(f" Total cracks detected: {total_cracks_session}")
print(f" Screenshots saved: {len(session_log)}")
