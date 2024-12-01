# Garage Door Monitoring Project

This project uses a deep learning model to monitor the state of a garage door (open or closed) via an MJPEG camera stream.
It sends email alerts when the door is open and continues sending reminders at a configurable interval until the door is closed.

## Directory Structure

```
garage-door/
├── data/                       # Dataset folder
│   ├── open/                   # All "open" images
│   └── closed/                 # All "closed" images
├── app/                        # Application folder
│   ├── model/                  # Trained model
│   ├── scripts/                # Monitoring script and config
│   └── requirements.txt        # Dependencies for monitoring
│   └── Dockerfile              # Dockerfile for the container
├── train.py                    # Training script
├── build.py                    # Build script
├── start.sh                    # Start container script
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## Prerequisites

- Python 3.10+
- Docker
- SMTP server for email alerts

---

## Usage

### 1. Train the Model

#### Dataset Setup:
Place your dataset in the `data/` folder with the following structure:
```
data/
├── open/    # Images of the door open
├── closed/  # Images of the door closed
```

#### Train the Model:
Run the training script to preprocess the data and train the model:
```bash
pip install -r requirements.txt
python train.py
```

The trained model will be saved to `app/model/garage_door_classifier.h5`.

---

### 2. Configure `config.json`

Before building the monitoring container, configure the `app/scripts/config.json` file with your desired settings.

Below is an example configuration:

```json
{
    "stream_url": "http://videoserver.local:8080/video",
    "alert_interval": 300,
    "retry_limit": 20,
    "retry_delay": 3,
    "stats_interval": 60,
    "min_open_frames": 3,
    "frame_window": 10,
    "max_buffer_size_mb": 2,
    "smtp_server": "smtpserver.local",
    "smtp_port": 25,
    "email_from": "server@example.com",
    "email_to": "user1@example.com;user2@example.com",
    "subject": "Garage Door Alert",
    "body": "The garage door is OPEN."
}
```

#### Configuration Parameters:
- **`stream_url`**: URL of the live MJPEG camera stream.
- **`alert_interval`**: Interval (in seconds) between repeated alerts while the door is open.
- **`retry_limit`**: Maximum number of retries for fetching the camera stream.
- **`retry_delay`**: Delay (in seconds) between retry attempts.
- **`stats_interval`**: Interval (in seconds) for logging system stats.
- **`min_open_frames`**: Minimum consecutive frames to confirm the door is open.
- **`frame_window`**: Number of frames analyzed per cycle.
- **`max_buffer_size_mb`**: Maximum memory buffer for frames before cleanup.
- **`smtp_server`**: SMTP server address for sending emails.
- **`smtp_port`**: SMTP server port.
- **`email_from`**: Sender's email address.
- **`email_to`**: Recipients' email addresses separated by semicolons.
- **`subject`**: Email subject line.
- **`body`**: Email body text.

---

### 3. Build the Docker Container

Run the build script to ensure the model is trained and the container is created:
```bash
python build.py
```

---

### 4. Run the Monitoring App

Run the monitoring system using the `start.sh` script:
```bash
./start.sh
```

---

### 5. Monitor the Logs

You can check the logs to verify the monitoring system is working:
```bash
docker logs -f garagedoor
```

---

### 6. Stop the Monitoring System

To stop the monitoring system:
```bash
docker stop garagedoor
```

---

## Notes

- The `monitor.py` script streams the live MJPEG stream URL and analyzes frames using the trained model.
- Alerts are sent immediately upon detecting an open door and repeated at configurable intervals while the door remains open.
- System stats, such as retry attempts and buffer usage, are logged periodically.

---

## FAQ

1. **How do I test the system locally?**
   You can directly run the monitoring script:
   ```bash
   python app/scripts/monitor.py
   ```

2. **What happens if the camera stream is unreachable?**
   The system retries fetching the stream according to `retry_limit` and `retry_delay`. If the stream remains unreachable, it logs the issue.

3. **Can I add more email recipients?**
   Yes, update the `email_to` field in `config.json` with additional addresses separated by semicolons.

4. **How do I retrain the model?**
   Delete the existing model file at `app/model/garage_door_classifier.h5` and rerun:
   ```bash
   python train.py
   ```

---

## License
This project is licensed under the MIT License.
