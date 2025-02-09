# Person-Fall-Detector
# Fall Detection System using YOLO and DeepSORT

This project is a real-time fall detection system using a YOLO object detection model and DeepSORT tracking. When a fall is detected, the system sends an SMS alert and makes a phone call using Twilio.

## Features

- Real-time fall detection using YOLO
- Object tracking with DeepSORT
- Sends SMS and calls using Twilio when a fall is detected
- Adjustable sensitivity and cooldown mechanism to prevent false alarms

## Technologies Used

- Python
- OpenCV
- YOLO (You Only Look Once)
- DeepSORT (Tracking Algorithm)
- Twilio API (for SMS and Call alerts)

## Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/vasanth2610/Person-Fall-Detector.git
   cd Person-Fall-Detecto
   ```

2. **Install dependencies**

   ```sh
   pip install opencv-python ultralytics pandas cvzone twilio deep_sort_realtime
   ```

3. **Download the YOLO model**

   - Place your YOLO model file (`yolov10s.pt`) in the project directory.

4. **Prepare the class list**

   - Ensure `coco.txt` is present in the directory and contains class labels.

5. **Set up Twilio Credentials**

   - Replace the placeholders in the script with your Twilio credentials:
     ```python
     account_sid = 'your_account_sid'
     auth_token = 'your_auth_token'
     twilio_number = 'your_twilio_number'
     recipient_number = 'your_phone_number'
     ```

## Usage

1. **Run the script**

   ```sh
   python pmain.py
   ```

2. **Exit the program**

   - Press `q` to close the window.

## How It Works

- The YOLO model detects people in the video feed.
- The DeepSORT tracker assigns unique IDs to detected individuals.
- The aspect ratio of a person’s bounding box is analyzed to determine if they have fallen.
- If a fall is detected, an SMS and call alert are sent via Twilio.
- A cooldown mechanism prevents repeated alerts.

## Notes

- Adjust `max_age` in DeepSORT to fine-tune tracking behavior.
- The aspect ratio threshold (0.8) can be tweaked for better accuracy.
- Ensure your webcam is properly configured or replace `cv2.VideoCapture(0)` with a video file path.

## License

This project is open-source and available for modification and distribution.

## Contact

For any questions or contributions, feel free to reach out via GitHub or email.

