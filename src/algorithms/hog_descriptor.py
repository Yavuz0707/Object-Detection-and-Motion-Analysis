"""
HOG (Histogram of Oriented Gradients) Descriptor
Used for human detection and object classification.
"""
import cv2
import numpy as np
from skimage.feature import hog as skimage_hog


class HOGDescriptor:
    """HOG feature extraction for object detection."""
    
    def __init__(self, cell_size=8, block_size=2, nbins=9):
        """
        Initialize HOG descriptor.
        
        Args:
            cell_size: Size of each cell in pixels (e.g., 8x8)
            block_size: Number of cells in each block (e.g., 2x2)
            nbins: Number of orientation bins
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        self.hog = cv2.HOGDescriptor(
            (64, 128),  # win_size
            (16, 16),   # block_size in pixels
            (8, 8),     # block_stride in pixels
            (8, 8),     # cell_size in pixels
            nbins
        )
    
    def extract_features(self, image):
        """
        Extract HOG features from image.
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            Feature vector
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size for HOG
        image = cv2.resize(image, (64, 128))
        
        features = self.hog.compute(image)
        return features.flatten()
    
    def detect_objects(self, image):
        """
        Detect people in image using HOG + SVM.
        
        Args:
            image: Input image
            
        Returns:
            List of detection boxes (x, y, w, h)
        """
        detections = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(16, 16))
        return detections[0] if len(detections) > 0 else []


class CustomHOGExtractor:
    """Custom HOG implementation for better understanding and control."""
    
    def __init__(self, cell_size=8, nbins=9):
        """
        Initialize custom HOG extractor.
        
        Args:
            cell_size: Size of each cell
            nbins: Number of orientation bins
        """
        self.cell_size = cell_size
        self.nbins = nbins
    
    def compute_hog(self, image):
        """
        Compute HOG features.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            HOG feature vector, visualization image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        
        # Compute magnitude and direction
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Initialize HOG feature vector
        h, w = image.shape
        cells_h = h // self.cell_size
        cells_w = w // self.cell_size
        hog_features = np.zeros((cells_h, cells_w, self.nbins))
        
        # Fill histogram for each cell
        for i in range(cells_h):
            for j in range(cells_w):
                cell_y = slice(i * self.cell_size, (i + 1) * self.cell_size)
                cell_x = slice(j * self.cell_size, (j + 1) * self.cell_size)
                
                cell_angle = angle[cell_y, cell_x]
                cell_magnitude = magnitude[cell_y, cell_x]
                
                # Compute histogram
                hist, _ = np.histogram(cell_angle, bins=self.nbins, 
                                     range=(0, 180), weights=cell_magnitude)
                hog_features[i, j, :] = hist
        
        # Normalize features (L2 normalization)
        hog_features_norm = self._normalize_hog(hog_features)
        
        # Flatten for feature vector
        feature_vector = hog_features_norm.flatten()
        
        # Create visualization
        visualization = self._visualize_hog(hog_features, h, w)
        
        return feature_vector, visualization
    
    def _normalize_hog(self, hog_features):
        """L2 normalization of HOG features."""
        hog_features_flat = hog_features.reshape(-1, self.nbins)
        for i in range(hog_features_flat.shape[0]):
            norm = np.linalg.norm(hog_features_flat[i, :])
            if norm > 0:
                hog_features_flat[i, :] /= norm
        return hog_features_flat.reshape(hog_features.shape)
    
    def _visualize_hog(self, hog_features, h, w):
        """Create visualization of HOG features."""
        visualization = np.zeros((h, w), dtype=np.uint8)
        
        cells_h, cells_w, nbins = hog_features.shape
        
        for i in range(cells_h):
            for j in range(cells_w):
                cell_y = i * self.cell_size
                cell_x = j * self.cell_size
                
                # Find dominant orientation
                hist = hog_features[i, j, :]
                if np.sum(hist) > 0:
                    dominant_bin = np.argmax(hist)
                    angle = dominant_bin * (180 / nbins)
                    magnitude = hist[dominant_bin]
                    
                    # Draw on visualization
                    center_y = cell_y + self.cell_size // 2
                    center_x = cell_x + self.cell_size // 2
                    
                    # Draw line indicating gradient direction
                    scale = min(self.cell_size // 2, int(magnitude))
                    angle_rad = np.radians(angle)
                    end_x = int(center_x + scale * np.cos(angle_rad))
                    end_y = int(center_y + scale * np.sin(angle_rad))
                    
                    cv2.line(visualization, (center_x, center_y), (end_x, end_y), 200, 1)
        
        return visualization


def extract_hog_features_scikit(image, orientations=9, pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features using scikit-image.
    
    Args:
        image: Input image
        orientations: Number of orientation bins
        pixels_per_cell: Size of each cell
        cells_per_block: Number of cells per block
        visualize: Whether to return visualization
        
    Returns:
        Feature vector and optional visualization
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if visualize:
        features, hog_image = skimage_hog(image, orientations=orientations,
                                         pixels_per_cell=pixels_per_cell,
                                         cells_per_block=cells_per_block,
                                         visualize=True)
        return features, hog_image
    else:
        features = skimage_hog(image, orientations=orientations,
                             pixels_per_cell=pixels_per_cell,
                             cells_per_block=cells_per_block)
        return features


def compare_hog_features(feature1, feature2):
    """
    Compare two HOG feature vectors.
    
    Args:
        feature1: First feature vector
        feature2: Second feature vector
        
    Returns:
        Similarity score (lower is more similar)
    """
    # Euclidean distance
    distance = np.linalg.norm(feature1 - feature2)
    
    # Cosine similarity
    dot_product = np.dot(feature1, feature2)
    mag1 = np.linalg.norm(feature1)
    mag2 = np.linalg.norm(feature2)
    if mag1 > 0 and mag2 > 0:
        cosine_sim = dot_product / (mag1 * mag2)
    else:
        cosine_sim = 0
    
    return {'euclidean': distance, 'cosine_similarity': cosine_sim}
