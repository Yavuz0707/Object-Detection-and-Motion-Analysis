"""
Background Subtraction Algorithm
Implements classic frame difference and MOG2 methods for moving object detection.
"""
import cv2
import numpy as np


class BackgroundSubtraction:
    """Classic background subtraction using frame difference."""
    
    def __init__(self, threshold=30):
        """
        Initialize background subtraction.
        
        Args:
            threshold: Pixel intensity difference threshold
        """
        self.threshold = threshold
        self.background = None
    
    def set_background(self, image):
        """Set background image for subtraction."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.background = image.astype(np.float32)
    
    def subtract(self, image):
        """
        Perform background subtraction: P[ζ(t)] = P[I[t]] - P[A]
        
        Args:
            image: Current frame
            
        Returns:
            Foreground mask
        """
        if self.background is None:
            raise ValueError("Background image not set")
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_float = image.astype(np.float32)
        diff = cv2.absdiff(image_float, self.background)
        _, foreground = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        return foreground.astype(np.uint8)
    
    def process_frame(self, image):
        """Process single frame and return foreground mask."""
        return self.subtract(image)
    
    def set_threshold(self, threshold):
        """Update threshold value."""
        self.threshold = threshold


class MOG2BackgroundSubtractor:
    """Gaussian Mixture Model for adaptive background subtraction."""
    
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        """
        Initialize MOG2 background subtractor.
        
        Args:
            history: Number of frames to learn from
            var_threshold: Threshold for the squared Mahalanobis distance
            detect_shadows: Whether to detect shadows
        """
        self.detector = cv2.createBackgroundSubtractorMOG2(
            detectShadows=detect_shadows,
            history=history,
            varThreshold=var_threshold
        )
        self.history = history
        self.var_threshold = var_threshold
    
    def process_frame(self, image):
        """
        Process frame with MOG2.
        
        Args:
            image: Input frame (BGR or Grayscale)
            
        Returns:
            Foreground mask
        """
        mask = self.detector.apply(image)
        return mask
    
    def get_background(self):
        """Get learned background model."""
        return self.detector.getBackgroundImage()
    
    def reinit(self):
        """Reinitialize the detector."""
        self.detector = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            history=self.history,
            varThreshold=self.var_threshold
        )


class FrameDifferenceSubtractor:
    """Frame difference method for detecting motion."""
    
    def __init__(self, threshold=30):
        """
        Initialize frame difference subtractor.
        
        Args:
            threshold: Pixel intensity difference threshold
        """
        self.threshold = threshold
        self.prev_frame = None
    
    def process_frame(self, image):
        """
        Detect motion using frame difference.
        
        Args:
            image: Input frame
            
        Returns:
            Motion mask
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        
        diff = cv2.absdiff(self.prev_frame, gray)
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        self.prev_frame = gray
        return motion_mask
    
    def set_threshold(self, threshold):
        """Update threshold value."""
        self.threshold = threshold
    
    def reset(self):
        """Reset the frame buffer."""
        self.prev_frame = None


def apply_morphological_cleanup(mask, kernel_size=5, iterations=2):
    """
    Apply morphological operations to clean up the mask.
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
        iterations: Number of iterations
        
    Returns:
        Cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def extract_foreground_objects(image, mask, min_area=100):
    """
    Extract foreground objects from image using mask.
    
    Args:
        image: Original image
        mask: Binary foreground mask
        min_area: Minimum contour area
        
    Returns:
        Image with only foreground objects, list of contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    result = image.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)
    
    return result, filtered_contours
