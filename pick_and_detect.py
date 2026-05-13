import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "flower_detection.py"


def load_detector():
    spec = importlib.util.spec_from_file_location("flower_detection", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load detector script: {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return filedialog.askopenfilename(
        title="Choose a flower image to test",
        initialdir=str(BASE_DIR / "test_images"),
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )


def main():
    image_path = pick_image()
    if not image_path:
        print("No image selected.")
        return

    detector = load_detector()
    detector.detect(image_path)


if __name__ == "__main__":
    main()
