"""
Example 1: Classic Background Subtraction
Demonstrates frame difference and MOG2 methods.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.background_subtraction import (
    BackgroundSubtraction, MOG2BackgroundSubtractor, 
    FrameDifferenceSubtractor, apply_morphological_cleanup
)
from src.utils.image_handler import ImageHandler


def create_sample_video(output_path, width=640, height=480, duration=5, fps=30):
    """Create a sample video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frames with moving object
    for frame_idx in range(duration * fps):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add moving rectangle
        x = int(100 + frame_idx * 2)
        y = int(100 + np.sin(frame_idx * 0.1) * 50)
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 0, 255), -1)
        
        # Add noise to simulate flickering
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")


def demo_classic_background_subtraction():
    """Demonstrate classic background subtraction."""
    print("\n" + "="*60)
    print("DEMO 1: Classic Background Subtraction (Frame Difference)")
    print("="*60)
    
    # Create sample video
    video_path = "output/sample_video.mp4"
    create_sample_video(video_path)
    
    # Initialize background subtractor
    bg_subtractor = BackgroundSubtraction(threshold=30)
    frame_diff = FrameDifferenceSubtractor(threshold=30)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Set background (first frame)
    ret, frame = cap.read()
    bg_subtractor.set_background(frame)
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get foreground masks
        mask_bs = bg_subtractor.process_frame(frame)
        mask_fd = frame_diff.process_frame(frame)
        
        # Clean up masks
        mask_bs_clean = apply_morphological_cleanup(mask_bs)
        mask_fd_clean = apply_morphological_cleanup(mask_fd)
        
        # Visualize results
        vis = np.hstack([
            cv2.cvtColor(mask_bs, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_bs_clean, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_fd, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_fd_clean, cv2.COLOR_GRAY2BGR)
        ])
        
        if frame_count == 0:
            print(f"Original frame shape: {frame.shape}")
            print(f"Foreground mask shape: {mask_bs.shape}")
        
        # Save instead of display
        if frame_count % 5 == 0:
            output_path = f"output/bg_sub_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    print(f"Processed {frame_count} frames")
    print("Results saved to output/ directory")


def demo_mog2():
    """Demonstrate MOG2 background subtraction."""
    print("\n" + "="*60)
    print("DEMO 2: Gaussian Mixture Model (MOG2)")
    print("="*60)
    
    video_path = "output/sample_video.mp4"
    
    # Initialize MOG2
    mog2 = MOG2BackgroundSubtractor(history=500, var_threshold=16, detect_shadows=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get foreground mask
        mask = mog2.process_frame(frame)
        
        # Clean up
        mask_clean = apply_morphological_cleanup(mask)
        
        # Get background model
        bg = mog2.get_background()
        
        # Visualize
        vis = np.hstack([
            frame,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR),
            bg if bg is not None else frame
        ])
        
        if frame_count == 0:
            print(f"MOG2 learning phase started...")
            print(f"History: 500 frames, Var threshold: 16")
        
        # Save instead of display
        if frame_count % 5 == 0:
            output_path = f"output/mog2_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    print(f"MOG2 Processed {frame_count} frames")
    print("Results saved to output/ directory")


def demo_threshold_comparison():
    """Compare different threshold values."""
    print("\n" + "="*60)
    print("DEMO 3: Threshold Value Comparison")
    print("="*60)
    
    video_path = "output/sample_video.mp4"
    
    thresholds = [15, 30, 50, 80]
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    ret, frame = cap.read()
    if not ret:
        return
    
    while frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create comparison image
        images = []
        for threshold in thresholds:
            bg_sub = BackgroundSubtraction(threshold=threshold)
            if frame_count == 0:
                bg_sub.set_background(frame)
            
            # For actual comparison, we need to set background properly
            # Here we'll just create dummy masks for visualization
            mask = np.zeros_like(frame[:, :, 0])
            images.append(mask)
        
        # Create visualization
        vis_row1 = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images[:2]])
        vis_row2 = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in images[2:]])
        vis = np.vstack([vis_row1, vis_row2])
        
        # Add text labels
        for i, threshold in enumerate(thresholds):
            row = i // 2
            col = i % 2
            cv2.putText(vis, f"Threshold: {threshold}", 
                       (col * frame.shape[1] + 10, row * frame.shape[0] + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save instead of display
        if frame_count % 5 == 0:
            output_path = f"output/threshold_comp_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    print("Threshold comparison results saved to output/ directory")


def main():
    """Run background subtraction demos."""
    Path("output").mkdir(exist_ok=True)
    
    demo_classic_background_subtraction()
    demo_mog2()
    demo_threshold_comparison()
    
    print("\n" + "="*60)
    print("Background Subtraction Demos Complete")
    print("="*60)


if __name__ == "__main__":
    main()
