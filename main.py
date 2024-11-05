# Import libraries
import cv2
import numpy as np

# For signal processing (we'll use a basic moving average)
from collections import deque
import scipy.signal

RESOLUTION = (640, 360)

TALKING_TREAHOLD = 60
TALKING_INTENSITY = 20
MOUTH_TALK_BOOLEAN_BUFFER = 10


# Video capture using WebCam
cap = cv2.VideoCapture(1)

# print a feedback
print('Camera On')

# Initialize a queue to store forehead pixel data over time
forehead_data = deque(maxlen=300)  # Adjust maxlen to control the time window

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Variables for heart rate calculation
bpm = 0
beat_timer = 0
last_beat = 0

# FPS Calculation Variables
start_time = cv2.getTickCount()
frame_count = 0
fps = 0

mouthTalkBuffer = []


while True:
    # Original frame ~ Video frame from camera
    ret, frame = cap.read()

    # Rescale the image
    frame = cv2.resize(frame, RESOLUTION) #(640, 360) (360, 640)

    # Convert original frame to gray
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        print("Assertion error")
        continue

    # Get location of the faces in term of position
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)


    # Detect faces
    for (x, y, w, h) in faces:
        # Draw rectangle in the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 53, 18), 1)

        # Display heart rate (with caution!)
        text_bpm = last_beat if last_beat > 0 else "Processing"
        cv2.putText(frame, f"Heart Rate: {text_bpm} BPM (Unreliable)", (x, y+h+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Forehead detection (ADJUSTED VALUES)
        forehead_height = int(h * 0.15)   # Reduced height (20% of face height)
        forehead_y = y + 10                # Forehead starts at the top of the face
        forehead_x = x + int(w * 0.3)   # Further inwards (20% from face sides)
        forehead_w = int(w * 0.4)   # Less wide (60% of face width)

        # 1. Isolate Mouth ROI
        mouth_y = int(y + h * 0.69) 
        mouth_h = int(h * 0.25)    
        mouth_x = int(x + w * 0.3) 
        mouth_w = int(w * 0.4)

        cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (50, 100, 255), 1)    

        mouth_roi = frame[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]

        
        # 1. Lip Color Segmentation (HSV)
        hsv_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
        lower_lip_color = np.array([150, 100, 120])   # Adjust these values!
        upper_lip_color = np.array([280, 230, 250]) # Adjust these values!
        lip_mask = cv2.inRange(hsv_mouth, lower_lip_color, upper_lip_color)

        # 2. Apply Mask to Mouth ROI
        masked_mouth_roi = cv2.bitwise_and(mouth_roi, mouth_roi, mask=lip_mask)

        # 3. Frame Differencing (on masked ROI)
        gray_mouth_roi = cv2.cvtColor(masked_mouth_roi, cv2.COLOR_BGR2GRAY)
        if 'prev_gray_mouth' in locals():  # Check if it's the first frame
            # Resize prev_gray_mouth to match gray_mouth_roi BEFORE comparison
            prev_gray_mouth = cv2.resize(prev_gray_mouth, (gray_mouth_roi.shape[1], gray_mouth_roi.shape[0])) 

            frame_diff = cv2.absdiff(gray_mouth_roi, prev_gray_mouth)
            _, thresh = cv2.threshold(frame_diff, TALKING_TREAHOLD, 255, cv2.THRESH_BINARY)  # Adjust threshold (30)
            movement_intensity = np.sum(thresh) // 255  # Count white pixels

            # 3. Speech/Silence Classification (Simple thresholding)
            mouthTalkBuffer.append(movement_intensity > TALKING_INTENSITY)
            if len(mouthTalkBuffer) > MOUTH_TALK_BOOLEAN_BUFFER:
                mouthTalkBuffer = mouthTalkBuffer[-MOUTH_TALK_BOOLEAN_BUFFER:]

            if mouthTalkBuffer.count(True) > len(mouthTalkBuffer)/2:  # Adjust threshold (100)
                cv2.putText(frame, "Talking", (x, mouth_y + mouth_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Silent", (x, mouth_y + mouth_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            #print(len(mouthTalkBuffer), mouthTalkBuffer.count(True), mouthTalkBuffer.count(False))

        prev_gray_mouth = gray_mouth_roi.copy()



        # Draw forehead rectangle
        cv2.rectangle(frame, (forehead_x, forehead_y), (forehead_x + forehead_w, forehead_y + forehead_height), (0, 255, 0), 1)

        # Heart Rate Estimation Attempt:

        # 1. Extract forehead region
        forehead_roi = frame[forehead_y:forehead_y + forehead_height, 
                            forehead_x:forehead_x + forehead_w]

        # 2. Convert to green channel (more sensitive to blood flow)
        forehead_green = cv2.cvtColor(forehead_roi, cv2.COLOR_BGR2HSV)[:,:,1]

        # 3. Calculate average green pixel value
        avg_green = np.mean(forehead_green)
        forehead_data.append(avg_green)

        # 4. Simple signal processing (moving average + bandpass filter)
        if len(forehead_data) == forehead_data.maxlen:
            smoothed_data = np.array(forehead_data)

            # Bandpass filter design (adjust frequencies as needed)
            fs = float(fps) if fps > 15 else 30.0  # Sample rate (your video frame rate)
            lowcut = 0.7  # Lower cutoff frequency (Hz)
            highcut = 3.0  # Upper cutoff frequency (Hz)
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered_data = scipy.signal.lfilter(b, a, smoothed_data)

            # 5. Peak detection (apply to filtered_data)
            peaks = np.where((filtered_data[1:-1] > filtered_data[:-2]) & 
                             (filtered_data[1:-1] > filtered_data[2:]))[0]

            # 6. Calculate heart rate if peaks are detected
            if len(peaks) > 2:
                # Estimate heart rate based on peak distances (very rough)
                peak_diffs = np.diff(peaks)
                avg_peak_diff = np.mean(peak_diffs)
                bpm = int(60*fps / avg_peak_diff)

                # Simple beat timer for visualization
                if beat_timer > 0:
                    beat_timer -= 1
                else:
                    beat_timer = 10
                    last_beat = bpm

    # FPS Calculation and Display
    frame_count += 1
    if frame_count >= 30:  # Update FPS every 30 frames
        end_time = cv2.getTickCount()
        fps = int(30 / ((end_time - start_time) / cv2.getTickFrequency()))
        start_time = end_time
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


    # Load video frame
    frame = cv2.resize(frame, (RESOLUTION[0]*2, RESOLUTION[1]*2))
    cv2.imshow('Video Frame', frame)

    # Wait 1 millisecond second until q key is press
    if cv2.waitKey(1) == ord('q'):
        print('Camera Off')
        break

# Close windows
cap.release()
cv2.destroyAllWindows()