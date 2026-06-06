"""
Optical Character Recognition (OCR) Pipeline Enhanced Version
Computer Vision Course Project 7
Uses Tesseract OCR with advanced preprocessing for improved text extraction accuracy.
"""

import os
import sys
import cv2
import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# File Selection 

def browse_for_image():
    """Opens a native OS file dialog to select an image."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("Opening file browser … Please select an image containing text.")
    file_path = filedialog.askopenfilename(
        title="Select Image File for OCR",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("All Files", "*.*"),
        ],
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)
    return file_path


# Preprocessing Strategies 

def preprocess_basic(gray: np.ndarray) -> np.ndarray:
    """Otsu's global thresholding (fast, good for clean documents)."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — handles uneven lighting common in photographed documents.
    Each pixel's threshold is calculated from its local neighbourhood mean.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # neighbourhood size (must be odd)
        C=10,           # constant subtracted from mean
    )
    return adaptive


def preprocess_morph(gray: np.ndarray) -> np.ndarray:
    """
    Morphological cleaning — removes salt-and-pepper noise after thresholding,
    then applies a slight dilation to strengthen thin strokes before OCR.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove tiny noise
    strengthened = cv2.dilate(cleaned, kernel, iterations=1)      # thicken strokes
    return strengthened


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Corrects skew (tilt) in scanned documents by computing the dominant angle
    via the Hough line transform and rotating accordingly.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return gray
    median_angle = np.median(angles)
    # Only correct if the skew is noticeable (> 0.5°) but not extreme
    if abs(median_angle) < 0.5 or abs(median_angle) > 45:
        return gray
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def upscale_if_small(gray: np.ndarray, min_height: int = 800) -> np.ndarray:
    """
    Upscales low-resolution images so that character strokes are thick enough
    for Tesseract to recognise accurately (recommended: ≥ 300 DPI equivalent).
    """
    h, w = gray.shape
    if h < min_height:
        scale = min_height / h
        gray = cv2.resize(gray, (int(w * scale), min_height),
                          interpolation=cv2.INTER_CUBIC)
    return gray


# Core OCR Pipeline

def process_ocr_pipeline(image_path: str, save_output: bool = True):
    """
    Full pipeline:
      1. Load image
      2. Upscale if too small
      3. Deskew
      4. Apply the three preprocessing strategies in parallel
      5. Run Tesseract on each and pick the best result (longest valid text)
      6. Optionally save the binarised output to disk
    Returns (original_bgr, best_binary, extracted_text, strategy_name, elapsed_s).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    original = cv2.imread(image_path)
    if original is None:
        raise IOError(f"OpenCV could not decode: {image_path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = upscale_if_small(gray)
    gray = deskew(gray)

    strategies = {
        "Otsu (global)":     preprocess_basic(gray),
        "Adaptive Gaussian": preprocess_adaptive(gray),
        "Morphological":     preprocess_morph(gray),
    }

    psm_config = r'--psm 3 --oem 3'   # psm 3 = auto page segmentation; oem 3 = best LSTM engine

    print(f"\nProcessing: {os.path.basename(image_path)}")
    print(f"Dimensions: {original.shape[1]}×{original.shape[0]} px  →  "
          f"preprocessed: {gray.shape[1]}×{gray.shape[0]} px")

    best_text, best_binary, best_name = "", None, ""
    t0 = time.perf_counter()

    for name, binary in strategies.items():
        pil_img = Image.fromarray(binary)
        text = pytesseract.image_to_string(pil_img, config=psm_config)
        word_count = len(text.split())
        print(f"  [{name}]  →  {word_count} words extracted")
        if word_count > len(best_text.split()):
            best_text, best_binary, best_name = text, binary, name

    elapsed = time.perf_counter() - t0
    print(f"\nBest strategy: {best_name}  ({elapsed:.2f} s total)")

    if save_output:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_binarized{ext}"
        cv2.imwrite(out_path, best_binary)
        print(f"Binarised image saved → {out_path}")

    return original, best_binary, best_text, best_name, elapsed


# Visualisation

def display_results(bgr_orig, binary, text, strategy_name):
    """
    Shows original vs. best binarised image side-by-side,
    then prints all extracted text to the console.
    """
    rgb_orig = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(rgb_orig)
    axes[0].set_title("Original Image", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f"Best Preprocessing: {strategy_name}", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    print("\n" + "=" * 60)
    print("               EXTRACTED TEXT OUTPUT")
    print("=" * 60)
    if text.strip():
        print(text)
    else:
        print("[No text found in this image.]")
    print("=" * 60 + "\n")

    plt.show()


# OCR Confidence Report

def print_confidence_report(image_path: str, binary: np.ndarray):
    """
    Runs Tesseract in data mode and prints per-word confidence scores
    along with a summary mean confidence value.
    """
    pil_img = Image.fromarray(binary)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data['conf'] if str(c).isdigit() and int(c) >= 0]
    if confidences:
        mean_conf = np.mean(confidences)
        high = sum(1 for c in confidences if c >= 80)
        low  = sum(1 for c in confidences if c < 50)
        print(f"Confidence report → mean: {mean_conf:.1f}%  "
              f"high (≥80%): {high}  low (<50%): {low}  total words: {len(confidences)}")
    else:
        print("Confidence report: no scoreable words found.")


# Entry Point

if __name__ == "__main__":
    print("=" * 60)
    print("  Tesseract OCR Pipeline — Enhanced Edition")
    print("=" * 60)

    try:
        # 1. Select image
        image_file = browse_for_image()

        # 2. Run pipeline
        orig, binary, extracted_text, strategy, elapsed = process_ocr_pipeline(image_file)

        # 3. Confidence breakdown
        print_confidence_report(image_file, binary)

        # 4. Display visual comparison and print text
        display_results(orig, binary, extracted_text, strategy)

    except Exception as err:
        print(f"\n[Error]: {err}")
