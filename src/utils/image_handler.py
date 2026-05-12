"""
Utility functions for image processing operations.
"""
import cv2
import numpy as np
from pathlib import Path


class ImageHandler:
    """Utility class for image loading, saving, and basic operations."""
    
    @staticmethod
    def load_image(image_path, gray=False):
        """Load an image from file."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        if gray:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    @staticmethod
    def load_video(video_path):
        """Load a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found or cannot be opened: {video_path}")
        return cap
    
    @staticmethod
    def save_image(image, output_path):
        """Save an image to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
    
    @staticmethod
    def save_video(output_path, frames, fps=30, is_color=True):
        """Save frames as video file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), is_color)
            for frame in frames:
                out.write(frame)
            out.release()
    
    @staticmethod
    def resize_image(image, width=None, height=None, scale=None):
        """Resize image maintaining aspect ratio."""
        if scale:
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
        elif width is None and height is None:
            return image
        elif width is None:
            scale_factor = height / image.shape[0]
            width = int(image.shape[1] * scale_factor)
        elif height is None:
            scale_factor = width / image.shape[1]
            height = int(image.shape[0] * scale_factor)
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def apply_morphological_ops(image, kernel_size=5, operation='close', iterations=1):
        """Apply morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
        """Draw contours on image."""
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    @staticmethod
    def apply_threshold(image, threshold=127, method='binary'):
        """Apply thresholding to image."""
        if method == 'binary':
            _, result = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'adaptive':
            result = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        return result
