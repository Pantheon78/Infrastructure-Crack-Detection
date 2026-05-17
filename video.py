import cv2
import numpy as np

cap = cv2.VideoCapture('Crack Detection System/videos/crack 2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)#makes video loop 
        continue

    frame = cv2.resize(frame, (600, 400))

    #converting to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Large blur captures background, subtracting reveals only anomalies
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.subtract(background, gray)  # dark cracks become bright

    # Normalize
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Every pixel above 40 → becomes 255 (white) and Every pixel below 40 → becomes 0 (black)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)


    #  MORPHOLOGY - clean up amd  shapes the detected cracks

    #tiny dots disappear but real cracks lines survive
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    # Connect crack segments into lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)

   #A contour is just the outline/boundary of a white shape in your black and white image. This function finds ALL the outlines.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    crack_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Must be meaningful size
        if area < frame_area * 0.0001:
            continue

        # Must be elongated like a crack
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        if h_rect == 0 or w_rect == 0:
            continue

        ratio = max(w_rect, h_rect) / min(w_rect, h_rect)

        # Ratio > 2 means clearly elongated = crack shaped
        if ratio < 2:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        compactness = area / (perimeter * perimeter)

        # Cracks are thin and long = low compactness
        if compactness > 0.05:
            continue

        # Passed all filters — it's a crack!
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        
        # Draw a label on each crack
        cv2.putText(frame, 'CRACK', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        crack_count += 1

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f'Cracks detected: {crack_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.imshow("Crack Detection (Video)", frame)
    cv2.imshow("Difference", diff)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Processed", closed)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()