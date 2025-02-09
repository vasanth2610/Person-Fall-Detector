import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
from twilio.rest import Client
from deep_sort_realtime.deepsort_tracker import DeepSort

# Twilio credentials
account_sid = 'AC35ae97c5cd78e3b51c2b624a08f577d3'
auth_token = '33bd3797651ee606d74e06c7ac4c67a1'
twilio_number = '+12294944491'
recipient_number = '+919345659196'

client = Client(account_sid, auth_token)

# Initialize call counter
call_count = 0  # Track the number of calls made

def send_alert():
    global call_count

    if call_count < 2:  # Allow only two calls
        # Send SMS
        message = client.messages.create(
            body="Fall detected! Please check immediately.",
            from_=twilio_number,
            to=recipient_number
        )
        print(f"SMS sent: {message.sid}")

        # Make a call
        call = client.calls.create(
            twiml='<Response><Say>Fall detected. Please respond immediately.</Say></Response>',
            from_=twilio_number,
            to=recipient_number
        )
        print(f"Call initiated: {call.sid}")

        call_count += 1  # Increment the call counter
    else:
        print("Call limit reached. No further calls will be made.")

# Load YOLO model
model = YOLO("fall_det_1.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Adjust max_age as needed

# Using the default webcam or video
cap = cv2.VideoCapture(0)  # Change 0 if using a different camera

# Load class list
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
fall_detected = False  # Flag to track if a fall has been detected
cooldown_frames = 30  # Cooldown period in frames
cooldown_counter = 0  # Counter for cooldown

while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO detection
    results = model(frame)
    detections = []

    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            x1, y1, x2, y2, confidence, class_id = box
            if confidence > 0.5 and class_list[int(class_id)] == 'person':  # Filter by confidence and class
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, int(class_id)))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, ltrb)

        h = y2 - y1
        w = x2 - x1
        aspect_ratio = h / w  # Use aspect ratio for fall detection

        if aspect_ratio < 0.8 and not fall_detected and cooldown_counter == 0:  # Fall detected
            send_alert()  # Send SMS and make a call
            fall_detected = True  # Set flag to true, alert has been sent
            cooldown_counter = cooldown_frames  # Start cooldown
            cvzone.putTextRect(frame, f'{"person_fall"}', (x1, y1), 1, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif aspect_ratio >= 0.8 and fall_detected:  # Person has likely gotten up
            fall_detected = False  # Reset the flag when fall condition is no longer met
            cvzone.putTextRect(frame, f'person', (x1, y1), 1, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if cooldown_counter > 0:
        cooldown_counter -= 1

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()