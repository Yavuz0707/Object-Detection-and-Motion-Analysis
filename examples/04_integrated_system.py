"""
Example 4: Integrated Warehouse Monitoring System
Combines all algorithms for a complete object detection and motion analysis system.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.background_subtraction import MOG2BackgroundSubtractor, apply_morphological_cleanup
from src.algorithms.hog_descriptor import HOGDescriptor
from src.algorithms.optical_flow import OpticalFlowFarneback, estimate_motion_statistics
from src.utils.image_handler import ImageHandler


def create_warehouse_video(output_path, width=1280, height=720, duration=10, fps=30):
    """Create a realistic warehouse monitoring video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(duration * fps):
        # Create warehouse-like background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 180
        
        # Add racks (background)
        for x in range(0, width, 150):
            cv2.rectangle(frame, (x, 0), (x + 50, height), (100, 100, 100), -1)
        
        # Add floor markings
        cv2.line(frame, (0, height//2), (width, height//2), (150, 150, 150), 3)
        
        # Worker 1: Moving left-right
        worker1_x = int(200 + 300 * np.sin(frame_idx * 0.05))
        worker1_y = 300
        # Draw person shape
        cv2.circle(frame, (worker1_x, worker1_y - 40), 20, (0, 100, 255), -1)
        cv2.rectangle(frame, (worker1_x - 15, worker1_y), (worker1_x + 15, worker1_y + 80), (0, 100, 255), -1)
        
        # Worker 2: Moving up-down
        worker2_x = 700
        worker2_y = int(250 + 200 * np.sin(frame_idx * 0.04))
        cv2.circle(frame, (worker2_x, worker2_y - 40), 20, (0, 150, 255), -1)
        cv2.rectangle(frame, (worker2_x - 15, worker2_y), (worker2_x + 15, worker2_y + 80), (0, 150, 255), -1)
        
        # Forklift: Moving diagonally
        forklift_x = int(400 + frame_idx * 2)
        forklift_y = int(500 + frame_idx * 0.5)
        if forklift_x < width:
            cv2.rectangle(frame, (forklift_x, forklift_y), (forklift_x + 100, forklift_y + 60), (255, 0, 0), -1)
            cv2.rectangle(frame, (forklift_x + 20, forklift_y - 50), (forklift_x + 50, forklift_y), (255, 100, 0), -1)
        
        # Add lighting variations
        if frame_idx % 60 < 30:
            frame = np.clip(frame.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        
        # Add noise (camera noise, dust particles)
        noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Warehouse video created: {output_path}")


class WarehouseMonitoringSystem:
    """Integrated warehouse monitoring system."""
    
    def __init__(self):
        """Initialize all components."""
        self.bg_subtractor = MOG2BackgroundSubtractor(history=500)
        self.optical_flow = OpticalFlowFarneback()
        self.hog_detector = HOGDescriptor()
        self.object_count = 0
        self.motion_history = []
    
    def process_frame(self, frame):
        """
        Process frame with complete pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            Annotated frame with detections
        """
        # Step 1: Background Subtraction
        mask = self.bg_subtractor.process_frame(frame)
        mask_clean = apply_morphological_cleanup(mask, kernel_size=5, iterations=2)
        
        # Step 2: Optical Flow Analysis
        flow_vis, flow = self.optical_flow.process_frame(frame)
        
        if flow is not None:
            motion_stats = estimate_motion_statistics(flow)
            self.motion_history.append(motion_stats)
        
        # Step 3: Contour Detection
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: Object Detection and Annotation
        result = frame.copy()
        detected_objects = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum object size
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Object ID
                cv2.putText(result, f"Object {detected_objects}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                detected_objects += 1
        
        self.object_count = detected_objects
        
        # Add statistics overlay
        result = self._add_stats_overlay(result, mask_clean, flow)
        
        return result, mask_clean, flow
    
    def _add_stats_overlay(self, frame, mask, flow):
        """Add statistics overlay to frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add text
        cv2.putText(frame, f"Objects Detected: {self.object_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        foreground_ratio = np.sum(mask > 0) / mask.size
        cv2.putText(frame, f"Motion Area: {foreground_ratio*100:.1f}%", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.motion_history:
            avg_motion = np.mean([s['mean_magnitude'] for s in self.motion_history[-10:]])
            cv2.putText(frame, f"Avg Motion: {avg_motion:.2f}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame


def demo_integrated_system():
    """Demonstrate complete warehouse monitoring system."""
    print("\n" + "="*60)
    print("INTEGRATED WAREHOUSE MONITORING SYSTEM")
    print("="*60)
    
    Path("output").mkdir(exist_ok=True)
    
    video_path = "output/warehouse_video.mp4"
    create_warehouse_video(video_path, duration=10)
    
    system = WarehouseMonitoringSystem()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps_values = []
    
    import time
    last_time = time.time()
    
    print("Processing warehouse monitoring video...")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result, mask, flow = system.process_frame(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time > last_time else 0
        fps_values.append(fps)
        last_time = current_time
        
        # Save visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        vis = np.hstack([result, mask_vis])
        cv2.putText(vis, f"FPS: {fps:.1f}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {system.object_count} objects detected")
            output_path = f"output/warehouse_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
            print(f"Saved: {output_path}")
        
        frame_count += 1
    
    cap.release()
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {np.mean(fps_values):.1f}")
    print(f"  Average objects: {frame_count / len(fps_values) if fps_values else 0:.1f}")


def analyze_system_performance():
    """Analyze system performance metrics."""
    print("\n" + "="*60)
    print("SYSTEM PERFORMANCE ANALYSIS")
    print("="*60)
    
    video_path = "output/warehouse_video.mp4"
    
    system = WarehouseMonitoringSystem()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detection_rates = []
    motion_magnitudes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, mask, flow = system.process_frame(frame)
        
        detection_rates.append(system.object_count)
        
        if flow is not None:
            motion_stats = estimate_motion_statistics(flow)
            motion_magnitudes.append(motion_stats['mean_magnitude'])
        
        frame_count += 1
    
    cap.release()
    
    print(f"Analysis of {frame_count} frames:\n")
    
    print("Object Detection Statistics:")
    print(f"  Average objects per frame: {np.mean(detection_rates):.2f}")
    print(f"  Max objects detected: {np.max(detection_rates)}")
    print(f"  Min objects detected: {np.min(detection_rates)}")
    
    if motion_magnitudes:
        print(f"\nMotion Analysis:")
        print(f"  Average motion magnitude: {np.mean(motion_magnitudes):.4f}")
        print(f"  Max motion magnitude: {np.max(motion_magnitudes):.4f}")
        print(f"  Motion std dev: {np.std(motion_magnitudes):.4f}")


def main():
    """Run integrated warehouse monitoring demonstration."""
    demo_integrated_system()
    analyze_system_performance()
    
    print("\n" + "="*60)
    print("Integrated System Demo Complete")
    print("="*60)


if __name__ == "__main__":
    main()
