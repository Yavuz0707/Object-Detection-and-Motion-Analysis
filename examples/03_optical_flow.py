"""
Example 3: Optical Flow Analysis
Demonstrates Lucas-Kanade (sparse) and Farneback (dense) optical flow methods.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.optical_flow import (
    OpticalFlowLucasKanade, OpticalFlowFarneback,
    OpticalFlowAnalyzer, estimate_motion_statistics
)
from src.utils.image_handler import ImageHandler


def create_motion_video(output_path, width=640, height=480, duration=5, fps=30):
    """Create a video with complex motion patterns."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(duration * fps):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 150
        
        # Multiple moving objects
        # Object 1: Fast linear motion
        x1 = int(50 + frame_idx * 3)
        cv2.rectangle(frame, (x1, 100), (x1 + 60, 160), (0, 0, 255), -1)
        
        # Object 2: Circular motion
        x2 = int(320 + 100 * np.cos(frame_idx * 0.05))
        y2 = int(240 + 80 * np.sin(frame_idx * 0.05))
        cv2.circle(frame, (int(x2), int(y2)), 40, (0, 255, 0), -1)
        
        # Object 3: Oscillating motion
        y3 = int(350 + 50 * np.sin(frame_idx * 0.1))
        cv2.rectangle(frame, (100, y3), (200, y3 + 50), (255, 0, 0), -1)
        
        # Add some background variations
        cv2.circle(frame, (320, 240), 200, (100, 100, 100), 2)
        
        # Add small amount of noise
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Motion video created: {output_path}")


def demo_lucas_kanade():
    """Demonstrate Lucas-Kanade sparse optical flow."""
    print("\n" + "="*60)
    print("DEMO 1: Lucas-Kanade Sparse Optical Flow")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    create_motion_video(video_path)
    
    lk_flow = OpticalFlowLucasKanade(max_corners=100, quality_level=0.01)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    vectors_count = []
    
    while frame_count < 80:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis, flow_vectors = lk_flow.process_frame(frame)
        
        if flow_vectors is not None:
            vectors_count.append(len(flow_vectors['prev']))
        
        if frame_count == 0:
            print(f"Lucas-Kanade initialized")
            print(f"Max corners to track: 100")
        
        # Save instead of display
        if frame_count % 5 == 0:
            output_path = f"output/lk_flow_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Processed {frame_count} frames")
    if vectors_count:
        print(f"Average tracked points: {np.mean(vectors_count):.1f}")
        print(f"Max tracked points: {np.max(vectors_count)}")


def demo_farneback():
    """Demonstrate Farneback dense optical flow."""
    print("\n" + "="*60)
    print("DEMO 2: Farneback Dense Optical Flow")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    
    farneback = OpticalFlowFarneback(num_levels=3, window_size=15, num_iters=3)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    magnitude_values = []
    
    while frame_count < 80:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis, flow = farneback.process_frame(frame)
        
        if flow is not None:
            mag = farneback.get_flow_magnitude(flow)
            magnitude_values.append(np.mean(mag))
        
        if frame_count == 0:
            print(f"Farneback dense optical flow initialized")
            print(f"Parameters: levels=3, window=15, iterations=3")
        
        # Save instead of display
        if frame_count % 5 == 0:
            output_path = f"output/farneback_frame_{frame_count:03d}.png"
            cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Processed {frame_count} frames")
    if magnitude_values:
        print(f"Average flow magnitude: {np.mean(magnitude_values):.2f}")
        print(f"Max flow magnitude: {np.max(magnitude_values):.2f}")


def demo_farneback_hsv_visualization():
    """Demonstrate HSV color-based visualization of dense optical flow."""
    print("\n" + "="*60)
    print("DEMO 3: Dense Optical Flow HSV Visualization")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    
    farneback = OpticalFlowFarneback(num_levels=3, window_size=15, num_iters=3)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < 80:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        _, flow = farneback.process_frame(frame)
        
        if flow is not None:
            # HSV visualization
            hsv_vis = farneback.visualize_flow_hsv(flow, frame)
            
            # Display original and HSV visualization side by side
            vis = np.hstack([frame, hsv_vis])
            
            # Save instead of display
            if frame_count % 5 == 0:
                output_path = f"output/hsv_flow_frame_{frame_count:03d}.png"
                cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    
    print(f"HSV visualization processed {frame_count} frames")
    print("Colors represent: Hue=Direction, Saturation=Fixed, Value=Magnitude")


def demo_method_comparison():
    """Compare sparse and dense optical flow methods."""
    print("\n" + "="*60)
    print("DEMO 4: Sparse vs Dense Optical Flow Comparison")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    
    analyzer = OpticalFlowAnalyzer()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Method Comparison on Motion Video:")
    print("-" * 50)
    
    while frame_count < 80:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = analyzer.analyze_frame(frame)
        
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}:")
            if results['sparse']['vectors'] is not None:
                print(f"  Sparse: {len(results['sparse']['vectors']['prev'])} points")
            if results['dense']['magnitude'] is not None:
                print(f"  Dense: Mean magnitude = {np.mean(results['dense']['magnitude']):.2f}")
        
        # Create comparison visualization
        lk_vis = results['sparse']['visualization']
        dense_hsv_vis = results['dense']['hsv_visualization']
        
        if dense_hsv_vis is not None:
            vis = np.hstack([lk_vis, dense_hsv_vis])
            cv2.putText(vis, "Lucas-Kanade (Sparse)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis, "Farneback (Dense)", (frame.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save instead of display
            if frame_count % 5 == 0:
                output_path = f"output/comparison_frame_{frame_count:03d}.png"
                cv2.imwrite(output_path, vis)
        
        frame_count += 1
    
    cap.release()
    
    print("-" * 50)
    print(f"Comparison completed for {frame_count} frames")


def demo_motion_statistics():
    """Analyze motion statistics from optical flow."""
    print("\n" + "="*60)
    print("DEMO 5: Motion Statistics Analysis")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    
    farneback = OpticalFlowFarneback(num_levels=3, window_size=15, num_iters=3)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_stats = []
    
    while frame_count < 80:
        ret, frame = cap.read()
        if not ret:
            break
        
        _, flow = farneback.process_frame(frame)
        
        if flow is not None:
            stats = estimate_motion_statistics(flow)
            all_stats.append(stats)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Analyzed {frame_count} frames\n")
    
    if all_stats:
        # Calculate averages
        mean_mag_list = [s['mean_magnitude'] for s in all_stats]
        max_mag_list = [s['max_magnitude'] for s in all_stats]
        motion_area_list = [s['motion_area_ratio'] for s in all_stats]
        
        print("Motion Statistics Summary:")
        print(f"  Average flow magnitude: {np.mean(mean_mag_list):.4f}")
        print(f"  Max flow magnitude: {np.mean(max_mag_list):.4f}")
        print(f"  Median flow magnitude: {np.median(mean_mag_list):.4f}")
        print(f"  Motion area ratio: {np.mean(motion_area_list):.4f}")
        
        # Find frames with most motion
        max_motion_idx = np.argmax(mean_mag_list)
        print(f"\nFrame with maximum motion: {max_motion_idx}")
        print(f"  Max magnitude: {max_mag_list[max_motion_idx]:.4f}")


def demo_computation_cost():
    """Compare computational efficiency of methods."""
    print("\n" + "="*60)
    print("DEMO 6: Computational Efficiency Analysis")
    print("="*60)
    
    video_path = "output/motion_video.mp4"
    
    import time
    
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    
    # Read first 30 frames
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame)
    
    cap.release()
    
    if len(frame_list) < 2:
        print("Not enough frames")
        return
    
    # Test Lucas-Kanade
    lk = OpticalFlowLucasKanade()
    start = time.time()
    for frame in frame_list:
        lk.process_frame(frame)
    lk_time = time.time() - start
    
    # Test Farneback
    farneback = OpticalFlowFarneback()
    start = time.time()
    for frame in frame_list:
        farneback.process_frame(frame)
    farneback_time = time.time() - start
    
    print("Computational Efficiency Comparison:")
    print(f"  Frames processed: {len(frame_list)}")
    print(f"\n  Lucas-Kanade (Sparse):")
    print(f"    Total time: {lk_time:.4f}s")
    print(f"    Per frame: {lk_time/len(frame_list):.4f}s")
    
    print(f"\n  Farneback (Dense):")
    print(f"    Total time: {farneback_time:.4f}s")
    print(f"    Per frame: {farneback_time/len(frame_list):.4f}s")
    
    print(f"\n  Speedup (LK/Farneback): {farneback_time/lk_time:.2f}x")
    
    frame_size = frame_list[0].shape[0] * frame_list[0].shape[1]
    print(f"\n  Data density:")
    print(f"    Frame size: {frame_size} pixels")
    print(f"    Dense flow: Full coverage (100%)")
    print(f"    Sparse flow: Selective point tracking (~10-20%)")


def main():
    """Run optical flow demonstrations."""
    Path("output").mkdir(exist_ok=True)
    
    demo_lucas_kanade()
    demo_farneback()
    demo_farneback_hsv_visualization()
    demo_method_comparison()
    demo_motion_statistics()
    demo_computation_cost()
    
    print("\n" + "="*60)
    print("Optical Flow Demos Complete")
    print("="*60)


if __name__ == "__main__":
    main()
