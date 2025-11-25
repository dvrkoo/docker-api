import time
import os
import sys

import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from playerModules.mantranet import pre_trained_model, check_forgery
from PIL import Image
import numpy as np
import queue
import threading

# Default to local folders if running natively, Docker paths if in container
folder_to_watch = os.getenv("WATCH_FOLDER", "./input" if not os.path.exists("/data") else "/data/input")
output_folder = os.getenv("OUTPUT_FOLDER", "./output" if not os.path.exists("/data") else "/data/output")

print(f"Starting Face Forensics API")
print(f"Watch folder: {folder_to_watch}")
print(f"Output folder: {output_folder}")

os.makedirs(folder_to_watch, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Force CPU in Docker environments to avoid MPS incompatibility
if os.getenv("FORCE_CPU", "false").lower() == "true":
    device = torch.device("cpu")
else:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
print(f"Using device: {device}")

MantraNetmodel = pre_trained_model(
    weight_path="trained_models/MantraNetv4.pt", device=device
)


def check_image_mantra(img_path):
    image = Image.open(img_path)

    # Convert to RGB if it's RGBA or grayscale
    if image.mode != "RGB":
        print(f"Converting image from {image.mode} to RGB")
        image = image.convert("RGB")

    # Match FaceForensicsTrainer/api_test.py behavior
    figs = check_forgery(MantraNetmodel, img_path=img_path, device=device)
    return figs


def process_image(file_path):
    print("Image detected")
    figs = check_image_mantra(file_path)
    print("Figures Processed")
    print(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.basename(file_path)
    name_without_ext, _ = os.path.splitext(base_name)
    for key, img in figs.items():
        output_file_name = f"{name_without_ext}_{key}.png"
        output_file_path = os.path.join(output_folder, output_file_name)

        try:
            if isinstance(img, np.ndarray):
                if np.issubdtype(img.dtype, np.floating):
                    if np.max(img) != np.min(img):
                        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                    else:
                        img = np.zeros_like(img)
                    img = img.astype(np.uint8)

                pil_img = Image.fromarray(img)
            else:
                pil_img = img

            if pil_img.mode == "F":
                pil_img = pil_img.convert("L")

            pil_img.save(output_file_path)
            print(f"Saved {key} image to: {output_file_path}")
        except Exception as e:
            print(f"Error saving {key} image: {e}")


def process_file(file_path):
    print(f"Processing file: {file_path}")
    if file_path.endswith(".jpg") or file_path.endswith(".png"):
        process_image(file_path)
    else:
        print("The file is not a supported image format. Skipping.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


file_queue = queue.Queue()


class FileCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected and queued: {event.src_path}")
            file_queue.put(event.src_path)


def worker():
    while True:
        file_path = file_queue.get()
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            file_queue.task_done()


def main():
    print(f"Monitoring folder: {folder_to_watch}")

    event_handler = FileCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=folder_to_watch, recursive=False)
    observer.start()

    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping observer...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
