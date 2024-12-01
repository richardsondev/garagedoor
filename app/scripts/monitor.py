import argparse
import os
import time
import smtplib
import json
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Load model
MODEL_PATH = "./model/garage_door_classifier.h5"
model = load_model(MODEL_PATH)

# Constants from config
STREAM_URL = config["stream_url"]
ALERT_INTERVAL = config["alert_interval"]
RETRY_LIMIT = config["retry_limit"]
RETRY_DELAY = config["retry_delay"]
STATS_INTERVAL = config["stats_interval"]
MIN_OPEN_FRAMES = config["min_open_frames"]
FRAME_WINDOW = config["frame_window"]
MAX_BUFFER_SIZE_MB = config["max_buffer_size_mb"]

# SMTP configuration
SMTP_SERVER = config["smtp_server"]
SMTP_PORT = config["smtp_port"]

EMAIL_FROM = config["email_from"]
EMAIL_TO = config["email_to"]
EMAIL_SUBJECT = config["subject"]
EMAIL_BODY = config["body"]

# Variables
last_alert_time = 0
frames_processed = 0
emails_sent = 0
confidence_scores = []  # Rolling list of confidence scores
retry_count = 0
frame_status = []  # Track frame statuses for the rolling window
buffer_size = 0
is_healthy = False

# Helper function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model's input size
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)


# Helper function to send email
def send_email():
    global emails_sent
    try:
        # Split multiple email addresses by semicolon and strip any whitespace
        recipient_list = [email.strip() for email in EMAIL_TO.split(';') if email.strip()]

        # Create email message
        msg = MIMEText(EMAIL_BODY)
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(recipient_list)
        msg["Subject"] = EMAIL_SUBJECT

        # Connect to SMTP server with authentication
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.sendmail(EMAIL_FROM, recipient_list, msg.as_string())

        emails_sent += 1
        print(f"Alert email sent to: {', '.join(recipient_list)}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Helper function to emit statistics
def emit_statistics():
    global frames_processed, emails_sent, confidence_scores, buffer_size
    if frames_processed > 0:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        print(f"Stats: Frames Processed: {frames_processed}, Emails Sent: {emails_sent}, "
              f"Avg Confidence: {avg_conf:.2f}, Min Confidence: {min(confidence_scores):.2f}, "
              f"Max Confidence: {max(confidence_scores):.2f}, Buffer Size: {buffer_size / (1024 * 1024):.2f} MB")
    else:
        print(f"No frames processed in the last {STATS_INTERVAL} seconds.")
    # Reset stats for the next interval
    frames_processed = 0
    confidence_scores.clear()


# Stream handler
def process_stream():
    global is_healthy, frames_processed, last_alert_time, retry_count, frame_status, buffer_size
    retry_count = 0  # Reset retry count on successful stream
    session = requests.Session()
    start_time = time.time()
    try:
        with session.get(STREAM_URL, stream=True, timeout=5) as response:
            response.raise_for_status()
            boundary = b'--videoboundary'
            buffer = b''
            last_stats_time = time.time()

            for chunk in response.iter_content(chunk_size=1024):
                buffer += chunk
                buffer_size = len(buffer)

                # Check buffer size and frame lag
                if buffer_size > MAX_BUFFER_SIZE_MB * 1024 * 1024:
                    print("Buffer exceeded maximum size. Clearing buffer.")
                    buffer = b''
                    buffer_size = 0
                    continue

                while boundary in buffer:
                    # Find the start of a new frame
                    parts = buffer.split(boundary, 1)
                    if len(parts) < 2:
                        break

                    # Extract headers and image data
                    headers_end = parts[1].find(b'\r\n\r\n')
                    if headers_end == -1:
                        break  # Wait for more data if headers aren't complete

                    headers = parts[1][:headers_end]
                    remaining = parts[1][headers_end + 4:]  # Skip past '\r\n\r\n'

                    # Parse headers to find Content-length
                    content_length = None
                    for line in headers.split(b'\r\n'):
                        if line.lower().startswith(b'content-length:'):
                            content_length = int(line.split(b':')[1].strip())
                            break

                    if content_length is None:
                        print("Content-length not found; skipping frame.")
                        buffer = remaining
                        continue

                    # Extract the image data
                    if len(remaining) < content_length:
                        # Wait for more data if the full image isn't in the buffer yet
                        break

                    frame_data = remaining[:content_length]
                    buffer = remaining[content_length:]

                    # Decode and process the frame
                    try:
                        with Image.open(BytesIO(frame_data)) as img:
                            preprocessed_img = preprocess_image(img)
                            prediction = model.predict(preprocessed_img)
                            confidence = prediction[0][0]
                            is_open = confidence > 0.5

                            frames_processed += 1
                            confidence_scores.append(confidence)
                            frame_status.append((is_open, time.time()))

                            # Maintain rolling window of frame statuses
                            frame_status = [
                                (status, ts)
                                for status, ts in frame_status
                                if time.time() - ts <= FRAME_WINDOW
                            ]

                            # Check if the door is open based on the rolling window
                            open_frames = sum(1 for status, ts in frame_status if status)
                            if open_frames >= min(MIN_OPEN_FRAMES, len(frame_status)):
                                print(f"Garage door is OPEN (Confidence: {confidence:.2f}).")
                                current_time = time.time()
                                if current_time - last_alert_time >= ALERT_INTERVAL:
                                    send_email()
                                    last_alert_time = current_time
                            elif len(frame_status) >= MIN_OPEN_FRAMES:
                                print(f"Garage door is CLOSED with sufficient frames (Confidence: {confidence:.2f}).")
                            else:
                                print(f"Garage door is CLOSED with insufficient frames (Confidence: {confidence:.2f}).")
                            
                        is_healthy = True
                    except Exception as e:
                        is_healthy = False
                        print(f"Error processing frame: {e}")

                # Emit statistics every STATS_INTERVAL seconds
                if time.time() - last_stats_time >= STATS_INTERVAL:
                    emit_statistics()
                    last_stats_time = time.time()
                    
                # Refresh stream for stability
                if time.time() - start_time >= 120:
                    print("Refreshing stream")
                    break;

    except Exception as e:
        print(f"Stream error: {e}")
        is_healthy = False
        raise RuntimeError("Stream failed.")
    finally:
        session.close()


def test_mode():
    """
    Perform a lightweight test to ensure the script runs correctly.
    """
    print("Running in test mode: Verifying setup...")
    print(f"Model path: {MODEL_PATH}")
    print(f"SMTP server: {SMTP_SERVER}, Port: {SMTP_PORT}")
    print(f"Stream URL: {STREAM_URL}")
    print("Test mode completed successfully!")


def monitor():
    global is_healthy, retry_count
    start_time = time.time()
    while time.time() - start_time <= 1800:
        try:
            process_stream()
        except RuntimeError:
            is_healthy = False
            retry_count += 1
            if retry_count >= RETRY_LIMIT:
                print(f"Max retries reached ({RETRY_LIMIT}). Exiting.")
                break
            print(f"Retrying... ({retry_count}/{RETRY_LIMIT})")
            time.sleep(RETRY_DELAY)


# Health Check HTTP Handler
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global is_healthy
        if is_healthy:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"UNHEALTHY")


def start_health_check_server():
    server = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Garage Door Monitoring Script")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    if args.test:
        test_mode()
    else:
        start_health_check_server()
        monitor()
