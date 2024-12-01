import os
import subprocess

MODEL_PATH = "./app/model/garage_door_classifier.h5"
DOCKER_IMAGE_NAME = "garagedoor"

def check_model_exists():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}.")
    else:
        print("Model not found. Training the model...")
        subprocess.run(["python", "train.py"], check=True)

def build_docker_image():
    print("Building Docker image...")
    subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, "./app"], check=True)
    print(f"Docker image '{DOCKER_IMAGE_NAME}' built successfully.")

if __name__ == "__main__":
    # Ensure the model is trained
    check_model_exists()
    # Build the Docker image
    build_docker_image()
