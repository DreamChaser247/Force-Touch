import cv2
import time
from evdev import UInput, ecodes as e, AbsInfo
import mediapipe as mp
from mediapipe.tasks.python import vision



# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


#Setting up gesture recognising model

gesturePath = 'gesture_recognizerv2.task'

baseOptions = BaseOptions(model_asset_path=gesturePath)


GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
gestureVisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest hand landmarks
latest_hand_result = None
latestGesture = None
lastX = None
lastY = None
isClicked = 0
isNewTrack = 1
isTapped = 0
quitResolve = 0
# Download from MediaPipe


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_result
    latest_hand_result = result
    #print(f'Hand landmarker result: {len(result.hand_landmarks)} hands detected')
    
# Create a gesture recognizer instance with the live stream mode:
def gesturePrintResult(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latestGesture
    gesture_name = ""  # Initialize an empty string to store the gesture name
    if result.gestures:  # Check if gestures list is not empty
        for gesture in result.gestures:  # Iterate over list of gesture lists
            if gesture:  # Check if the inner gesture list is not empty
                gesture_name = gesture[0].category_name  # Get the first category's name
                break  # Exit loop after getting the first gesture name
    print(f"Gesture name: {gesture_name} ")
    latestGesture = gesture_name
    
    
#Create Hand Gesture options
gestureOptions = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesturePath),
    running_mode=gestureVisionRunningMode.LIVE_STREAM,
    num_hands=1,  
    result_callback=gesturePrintResult
)

# Create hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),  
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=print_result
)


# evdev setup - virtual mouse
capabilities = {
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(value=0, min=0, max=1920, fuzz=0, flat=0, resolution=0)),
        (e.ABS_Y, AbsInfo(value=0, min=0, max=1080, fuzz=0, flat=0, resolution=0)),
    ],
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
    e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL, e.REL_HWHEEL],
}

def moveMousePointer(deltaX, deltaY, threshhold):
    if abs(deltaY) > threshhold or abs(deltaX) > threshhold:
        ui.write(e.EV_REL, e.REL_X, -deltaX)
        ui.write(e.EV_REL, e.REL_Y, -deltaY)
        ui.syn() 
        
def scrollMousePointer(deltaX, deltaY, threshhold):
    if abs(deltaY) > threshhold or abs(deltaX) > threshhold:
        ui.write(e.EV_REL, e.REL_WHEEL, -int(deltaY/50))
        ui.write(e.EV_REL, e.REL_HWHEEL, int(deltaX/50))
        ui.syn()

def tapDetect(thumb_tip, middle_tip):
    # Simple click detection: if thumb and index finger are close
        distance = ((thumb_tip.x - middle_tip.x) ** 2 + 
                    (thumb_tip.y - middle_tip.y) ** 2) ** 0.5
        
        global isTapped
        if distance < 0.05 and isTapped == 0:  # Threshold for "pinch" gesture
            ui.write(e.EV_KEY, e.BTN_LEFT, 1)  # Press
            ui.syn()
            ui.write(e.EV_KEY, e.BTN_LEFT, 0)  # Release
            ui.syn()
            isTapped = 1
            print("Click detected!")
        elif distance > 0.05 and isTapped == 1:
            isTapped = 0
            
    
            
def dragenholdScroll(thumb_tip, middle_tip, scroll, hand_landmarks):
        global lastX
        global lastY
        global isNewTrack
        threshhold = 5
        
        num_hands = len(hand_landmarks)
        # print(f"Number of hands detected: {num_hands}")
        if num_hands == 2:
            if scroll == 0:
                pointigLandmark = hand_landmarks[1][8]  # Second hand, landmark 8
            if scroll == 1:
                pointigLandmark = hand_landmarks[1][20]  # Second hand, landmark 8
            threshhold = 10
        else:
            if scroll == 0:
                pointigLandmark = hand_landmarks[0][8]  # Second hand, landmark 8
            if scroll == 1:
                pointigLandmark = hand_landmarks[0][20]  # Second hand, landmark 8
            threshhold = 5
        
        
        if isNewTrack == 1:
            lastX = pointigLandmark.x
            lastY = pointigLandmark.y
        isNewTrack = 0
        
        deltaX = int(5000*(lastX - pointigLandmark.x))
        deltaY = int(5000*(lastY - pointigLandmark.y))
        
        print(f"lastX = {lastX} ; lastY = {lastY} ; x = {pointigLandmark.x} ; y = {pointigLandmark.y} ; deltaX = {deltaX} ; deltaY = {deltaY} ; time {time.time()}")
        
        if scroll == 0:
            moveMousePointer(deltaX, deltaY,threshhold)
            tapDetect(thumb_tip=thumb_tip, middle_tip=middle_tip)
        if scroll == 1:
            scrollMousePointer(deltaX, deltaY, threshhold)
            
        lastX = pointigLandmark.x
        lastY = pointigLandmark.y
def process_hand_landmarks(hand_landmarks, ui, frame_width, frame_height):
    if not hand_landmarks:
        return
    
    thumb_tip = hand_landmarks[0][4]        # First hand, landmark 4
    middle_tip = hand_landmarks[0][12]      # First hand, landmark 12
    # pinky_tip = hand_landmarks[0][20]       # First hand, landmark 20
    
    if latestGesture == "drag" or latestGesture == "hold":
        dragenholdScroll(thumb_tip, middle_tip, 0, hand_landmarks)   
    elif latestGesture == "scroll":
        dragenholdScroll(thumb_tip, middle_tip, 1, hand_landmarks)    
    else:
        global isNewTrack
        isNewTrack = 1  
        
    # scrolling
    
    
    # drag and dropping       
    global isClicked
    if latestGesture == "hold" and isClicked == 0:
        ui.write(e.EV_KEY, e.BTN_LEFT, 1)
        isClicked = 1
    elif latestGesture != "hold" and isClicked == 1:
        ui.write(e.EV_KEY, e.BTN_LEFT, 0)
        isClicked = 0
            
    # special gesture, for quiting
    if latestGesture == "special":
        global quitResolve
        quitResolve += 0.025
    elif latestGesture != "special" and quitResolve > 0:
        quitResolve -= 0.01
            

# Create virtual mouse device
ui = UInput(capabilities, name='virtual-mouse', version=3)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get frame dimensions
cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Main loop with proper MediaPipe context management
with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp = 0
    with GestureRecognizer.create_from_options(gestureOptions) as recognizer:
        print("aaaa")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process frame with MediaPipe (async)
            landmarker.detect_async(mp_image, frame_timestamp)
            frame_timestamp += 1
            
            recognizer.recognize_async(mp_image, frame_timestamp)
            
            # if latestGesture:
            #     print("")
            
            # Process any detected hand landmarks
            if latest_hand_result and latest_hand_result.hand_landmarks:
                process_hand_landmarks(latest_hand_result.hand_landmarks, ui, frame_width, frame_height)
                
                # Draw landmarks on frame for visualization
                for hand_landmarks in latest_hand_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Manual controls for testing
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w'):
                ui.write(e.EV_REL, e.REL_X, 100)
                ui.write(e.EV_REL, e.REL_Y, 50)
                ui.syn()
            elif key == ord('s'):
                ui.write(e.EV_REL, e.REL_HWHEEL, 30)
                ui.syn()
            elif key == ord('q'):
                break
            
            if quitResolve > 1:
                print("gesture initialized quit")
                break
            
            # Display frame
            cv2.imshow('Hand Gesture Mouse Control', frame)
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)

# Cleanup
cap.release()
cv2.destroyAllWindows()
ui.close()