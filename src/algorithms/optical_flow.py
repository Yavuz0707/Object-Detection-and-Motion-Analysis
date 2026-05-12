"""
Optical Flow Analysis
Implements sparse (Lucas-Kanade) and dense (Farneback) optical flow methods.
"""
import cv2
import numpy as np


class OpticalFlowLucasKanade:
    """Sparse optical flow using Lucas-Kanade algorithm with Shi-Tomasi corners."""
    
    def __init__(self, max_corners=100, quality_level=0.01, min_distance=10):
        """
        Initialize Lucas-Kanade optical flow.
        
        Args:
            max_corners: Maximum number of corners to track
            quality_level: Quality threshold for corner detection
            min_distance: Minimum distance between corners
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_gray = None
        self.prev_corners = None
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    def process_frame(self, frame):
        """
        Process frame for optical flow.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Flow visualization image, flow vectors
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_corners = cv2.goodFeaturesToTrack(gray, self.max_corners, 
                                                       self.quality_level, self.min_distance)
            return frame.copy(), None
        
        if self.prev_corners is None or len(self.prev_corners) == 0:
            self.prev_gray = gray
            self.prev_corners = cv2.goodFeaturesToTrack(gray, self.max_corners,
                                                       self.quality_level, self.min_distance)
            return frame.copy(), None
        
        # Calculate optical flow
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, 
                                                              self.prev_corners, None, **self.lk_params)
        
        # Filter good points
        good_prev = self.prev_corners[status == 1]
        good_next = next_corners[status == 1]
        
        # Visualize flow
        vis = frame.copy()
        for prev, next_pt in zip(good_prev, good_next):
            x0, y0 = prev.ravel()
            x1, y1 = next_pt.ravel()
            vis = cv2.circle(vis, (int(x0), int(y0)), 5, (0, 255, 0), -1)
            vis = cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        
        # Update for next frame
        self.prev_gray = gray
        self.prev_corners = good_next.reshape(-1, 1, 2)
        
        flow_vectors = {'prev': good_prev, 'next': good_next}
        
        return vis, flow_vectors
    
    def reset(self):
        """Reset optical flow tracking."""
        self.prev_gray = None
        self.prev_corners = None


class OpticalFlowFarneback:
    """Dense optical flow using Farneback algorithm."""
    
    def __init__(self, num_levels=3, window_size=15, num_iters=3, poly_n=5, poly_sigma=1.2):
        """
        Initialize Farneback optical flow.
        
        Args:
            num_levels: Number of pyramid levels
            window_size: Averaging window size
            num_iters: Number of iterations
            poly_n: Size of polynomial expansion
            poly_sigma: Standard deviation for polynomial expansion
        """
        self.num_levels = num_levels
        self.window_size = window_size
        self.num_iters = num_iters
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.prev_gray = None
    
    def process_frame(self, frame):
        """
        Process frame for dense optical flow.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Flow visualization image, flow matrix
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame.copy(), None
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                          pyr_scale=0.5,
                                          levels=self.num_levels,
                                          winsize=self.window_size,
                                          iterations=self.num_iters,
                                          poly_n=self.poly_n,
                                          poly_sigma=self.poly_sigma,
                                          flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        # Visualize flow
        vis = self._draw_flow(frame, flow)
        
        self.prev_gray = gray
        return vis, flow
    
    def _draw_flow(self, frame, flow, step=16):
        """Draw optical flow on frame."""
        h, w = flow.shape[:2]
        x = np.arange(0, w, step, dtype=np.int32)
        y = np.arange(0, h, step, dtype=np.int32)
        xx, yy = np.meshgrid(x, y)
        
        fx = flow[yy, xx, 0]
        fy = flow[yy, xx, 1]
        
        lines = np.column_stack([xx.ravel(), yy.ravel(), 
                                 (xx + fx).ravel(), (yy + fy).ravel()]).astype(np.int32)
        
        vis = frame.copy()
        for (x0, y0, x1, y1) in lines:
            cv2.line(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(vis, (x0, y0), 1, (0, 255, 0), -1)
        
        return vis
    
    def get_flow_magnitude(self, flow):
        """Get magnitude of optical flow."""
        return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    def get_flow_angle(self, flow):
        """Get angle of optical flow."""
        return np.arctan2(flow[..., 1], flow[..., 0])
    
    def visualize_flow_hsv(self, flow, frame=None):
        """
        Visualize optical flow using HSV color space.
        
        Args:
            flow: Optical flow matrix
            frame: Optional background frame
            
        Returns:
            HSV visualization (BGR)
        """
        mag = self.get_flow_magnitude(flow)
        angle = self.get_flow_angle(flow)
        
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip((mag / mag.max() * 255) if mag.max() > 0 else 0, 0, 255).astype(np.uint8)
        
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if frame is not None:
            # Blend with original frame
            bgr_flow = cv2.addWeighted(frame, 0.5, bgr_flow, 0.5, 0)
        
        return bgr_flow
    
    def reset(self):
        """Reset optical flow tracking."""
        self.prev_gray = None


class OpticalFlowAnalyzer:
    """Comprehensive optical flow analysis and comparison."""
    
    def __init__(self):
        """Initialize optical flow analyzer."""
        self.lk_flow = OpticalFlowLucasKanade()
        self.farneback_flow = OpticalFlowFarneback()
    
    def analyze_frame(self, frame):
        """
        Analyze frame with both sparse and dense optical flow.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with results from both methods
        """
        lk_vis, lk_vectors = self.lk_flow.process_frame(frame)
        farneback_vis, farneback_flow = self.farneback_flow.process_frame(frame)
        
        results = {
            'sparse': {
                'visualization': lk_vis,
                'vectors': lk_vectors
            },
            'dense': {
                'visualization': farneback_vis,
                'flow': farneback_flow
            }
        }
        
        if farneback_flow is not None:
            results['dense']['hsv_visualization'] = self.farneback_flow.visualize_flow_hsv(farneback_flow, frame)
            results['dense']['magnitude'] = self.farneback_flow.get_flow_magnitude(farneback_flow)
        
        return results
    
    def compare_methods(self, video_path, num_frames=30):
        """
        Compare sparse and dense optical flow on video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to process
            
        Returns:
            Statistics comparing both methods
        """
        cap = cv2.VideoCapture(video_path)
        
        stats = {
            'sparse_vectors_count': [],
            'dense_flow_magnitude': [],
            'computation_time': []
        }
        
        frame_count = 0
        while frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.analyze_frame(frame)
            
            if results['sparse']['vectors'] is not None:
                stats['sparse_vectors_count'].append(len(results['sparse']['vectors']['prev']))
            
            if results['dense']['flow'] is not None:
                mag = results['dense']['magnitude']
                stats['dense_flow_magnitude'].append(np.mean(mag))
            
            frame_count += 1
        
        cap.release()
        
        return stats
    
    def reset(self):
        """Reset both flow trackers."""
        self.lk_flow.reset()
        self.farneback_flow.reset()


def estimate_motion_statistics(flow):
    """
    Estimate motion statistics from optical flow.
    
    Args:
        flow: Optical flow matrix (H x W x 2)
        
    Returns:
        Dictionary with motion statistics
    """
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    
    stats = {
        'mean_magnitude': np.mean(magnitude),
        'max_magnitude': np.max(magnitude),
        'median_magnitude': np.median(magnitude),
        'std_magnitude': np.std(magnitude),
        'dominant_angle': np.median(angle),
        'motion_area_ratio': np.sum(magnitude > np.mean(magnitude)) / magnitude.size
    }
    
    return stats
