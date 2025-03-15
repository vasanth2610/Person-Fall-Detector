import cv2
import time
import pandas as pd
import cvzone
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
from twilio.rest import Client

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO Model
model = YOLO("yolov10s.pt")

# Load class labels
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Twilio API credentials (replace with your Twilio details)
TWILIO_ACCOUNT_SID = 'ACd7b8974f11c71399c2dc45175786a221'
TWILIO_AUTH_TOKEN = '6902e2d94e2b2830cae57c2f6e2f0fa8'
TWILIO_PHONE_NUMBER = '+17755219711'
TO_PHONE_NUMBER = '+919345659196'

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Variables
detection_enabled = True
fall_detected = False
last_alert_time = 0
alert_cooldown = 10  # Min time between alerts in seconds
call_count = 0  # Track calls per fall detection
max_calls = 2  # Limit calls per detection event

def send_alerts():
    """Send Twilio call & message alerts when a fall is detected."""
    global call_count
    if call_count < max_calls:
        message = client.messages.create(
            body="🚨 Alert! A fall has been detected. Please check immediately.",
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER,
        )
        print("Twilio Message Sent:", message.sid)

        call = client.calls.create(
            twiml='<Response><Say>Emergency! A fall has been detected. Please check immediately.</Say></Response>',
            from_=TWILIO_PHONE_NUMBER,
            to=TO_PHONE_NUMBER,
        )
        print("Twilio Call Sent:", call.sid)
        call_count += 1  # Increment call count

def generate_frames():
    """Video streaming with YOLO fall detection."""
    global fall_detected, last_alert_time, detection_enabled, call_count
    cap = cv2.VideoCapture("fall5.mp4")
    frame_skip = 3  # Process every 3rd frame
    frame_count = 0

    while True:
        if not detection_enabled:
            time.sleep(1)  # Pause streaming when detection is disabled
            yield b''  # Send an empty response to pause video feed
            continue

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames for speed
            continue

        frame = cv2.resize(frame, (1020, 600))
        results = model(frame)
        px = pd.DataFrame(results[0].boxes.data).astype("float")

        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            d = int(row[5])
            c = class_list[d]

            if "person" in c:
                h, w = y2 - y1, x2 - x1
                thresh = h - w

                if thresh < 0:  # Fall detected
                    cvzone.putTextRect(frame, "🚨 FALL DETECTED", (x1, y1), 1, 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    current_time = time.time()
                    if not fall_detected and (current_time - last_alert_time > alert_cooldown):
                        send_alerts()
                        socketio.emit("fall_alert", {"fall": True})
                        last_alert_time = current_time
                        fall_detected = True
                        call_count = 0  # Reset call count for next detection
                else:
                    cvzone.putTextRect(frame, f"{c}", (x1, y1), 1, 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Start/Stop fall detection."""
    global detection_enabled
    detection_enabled = not detection_enabled
    socketio.emit("detection_status", {"detection_enabled": detection_enabled})
    return jsonify({"detection_enabled": detection_enabled})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
