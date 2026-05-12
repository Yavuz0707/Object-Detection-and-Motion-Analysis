"""
Example 2: HOG (Histogram of Oriented Gradients) Descriptor
Demonstrates HOG feature extraction for object detection.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.hog_descriptor import (
    HOGDescriptor, CustomHOGExtractor, 
    extract_hog_features_scikit, compare_hog_features
)
from src.utils.image_handler import ImageHandler


def create_test_image_with_person():
    """Create a test image with a person-like shape."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw person-like shape
    # Head
    cv2.circle(img, (320, 100), 30, (0, 0, 0), -1)
    # Body
    cv2.rectangle(img, (300, 130), (340, 250), (0, 0, 0), -1)
    # Arms
    cv2.line(img, (300, 160), (250, 180), (0, 0, 0), 15)
    cv2.line(img, (340, 160), (390, 180), (0, 0, 0), 15)
    # Legs
    cv2.line(img, (310, 250), (300, 320), (0, 0, 0), 10)
    cv2.line(img, (330, 250), (340, 320), (0, 0, 0), 10)
    
    # Add noise
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def demo_hog_extraction():
    """Demonstrate HOG feature extraction."""
    print("\n" + "="*60)
    print("DEMO 1: HOG Feature Extraction")
    print("="*60)
    
    # Create test image
    img = create_test_image_with_person()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG using custom implementation
    hog_extractor = CustomHOGExtractor(cell_size=8, nbins=9)
    features, hog_vis = hog_extractor.compute_hog(gray)
    
    print(f"HOG Feature Vector Length: {len(features)}")
    print(f"HOG Feature Statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Std Dev: {np.std(features):.4f}")
    print(f"  Min: {np.min(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    
    # Visualize
    hog_vis_display = cv2.resize(hog_vis, (320, 240))
    gray_display = cv2.resize(gray, (320, 240))
    
    vis = np.hstack([
        cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(hog_vis_display, cv2.COLOR_GRAY2BGR)
    ])
    
    # Save visualization
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite("output/hog_features.png", vis)
    print("HOG visualization saved to output/hog_features.png")


def demo_hog_scikit():
    """Demonstrate HOG using scikit-image."""
    print("\n" + "="*60)
    print("DEMO 2: HOG with scikit-image")
    print("="*60)
    
    img = create_test_image_with_person()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract features with visualization
    features, hog_image = extract_hog_features_scikit(gray, visualize=True)
    
    print(f"Scikit-image HOG Feature Vector Length: {len(features)}")
    print(f"Feature Statistics:")
    print(f"  Mean: {np.mean(features):.4f}")
    print(f"  Max: {np.max(features):.4f}")
    
    # Normalize for display
    hog_image = (hog_image * 255).astype(np.uint8)
    hog_image_display = cv2.resize(hog_image, (320, 240))
    gray_display = cv2.resize(gray, (320, 240))
    
    vis = np.hstack([
        cv2.cvtColor(gray_display, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(hog_image_display, cv2.COLOR_GRAY2BGR)
    ])
    
    cv2.imwrite("output/hog_scikit.png", vis)
    print("Scikit-image HOG visualization saved to output/hog_scikit.png")


def demo_hog_cell_sizes():
    """Compare different cell sizes."""
    print("\n" + "="*60)
    print("DEMO 3: HOG with Different Cell Sizes")
    print("="*60)
    
    img = create_test_image_with_person()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cell_sizes = [4, 8, 16, 32]
    visualizations = []
    
    for cell_size in cell_sizes:
        hog_extractor = CustomHOGExtractor(cell_size=cell_size, nbins=9)
        features, hog_vis = hog_extractor.compute_hog(gray)
        
        hog_vis_display = cv2.resize(hog_vis, (200, 150))
        visualizations.append(hog_vis_display)
        
        print(f"Cell Size {cell_size}x{cell_size}: {len(features)} features")
    
    # Create comparison image
    vis_row1 = np.hstack([cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in visualizations[:2]])
    vis_row2 = np.hstack([cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in visualizations[2:]])
    vis = np.vstack([vis_row1, vis_row2])
    
    # Add labels
    labels = [f"Cell: {cs}x{cs}" for cs in cell_sizes]
    for i, label in enumerate(labels):
        row = i // 2
        col = i % 2
        cv2.putText(vis, label, (col * 220 + 10, row * 160 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite("output/hog_cell_sizes.png", vis)
    print("Cell size comparison saved to output/hog_cell_sizes.png")


def demo_hog_feature_comparison():
    """Compare HOG features of similar and different images."""
    print("\n" + "="*60)
    print("DEMO 4: HOG Feature Comparison")
    print("="*60)
    
    # Create multiple test images
    img1 = create_test_image_with_person()
    
    # Create similar image (slightly modified)
    img2 = create_test_image_with_person()
    
    # Create different image
    img3 = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.circle(img3, (320, 240), 80, (0, 0, 0), -1)
    noise = np.random.randint(-30, 30, img3.shape, dtype=np.int16)
    img3 = np.clip(img3.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Extract features
    features1 = extract_hog_features_scikit(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    features2 = extract_hog_features_scikit(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    features3 = extract_hog_features_scikit(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY))
    
    # Compare features
    comp_12 = compare_hog_features(features1, features2)
    comp_13 = compare_hog_features(features1, features3)
    comp_23 = compare_hog_features(features2, features3)
    
    print("Feature Comparison Results:")
    print(f"\nImage 1 vs Image 2 (Similar):")
    print(f"  Euclidean Distance: {comp_12['euclidean']:.4f}")
    print(f"  Cosine Similarity: {comp_12['cosine_similarity']:.4f}")
    
    print(f"\nImage 1 vs Image 3 (Different):")
    print(f"  Euclidean Distance: {comp_13['euclidean']:.4f}")
    print(f"  Cosine Similarity: {comp_13['cosine_similarity']:.4f}")
    
    print(f"\nImage 2 vs Image 3 (Different):")
    print(f"  Euclidean Distance: {comp_23['euclidean']:.4f}")
    print(f"  Cosine Similarity: {comp_23['cosine_similarity']:.4f}")


def demo_hog_normalization_effect():
    """Demonstrate normalization effect on brightness changes."""
    print("\n" + "="*60)
    print("DEMO 5: Normalization Effect on Brightness")
    print("="*60)
    
    img = create_test_image_with_person()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create brightness variations
    bright_img = cv2.convertScaleAbs(gray.astype(np.float32) * 1.5)
    dark_img = cv2.convertScaleAbs(gray.astype(np.float32) * 0.7)
    
    # Extract features
    features_original = extract_hog_features_scikit(gray)
    features_bright = extract_hog_features_scikit(bright_img)
    features_dark = extract_hog_features_scikit(dark_img)
    
    # Compare
    comp_orig_bright = compare_hog_features(features_original, features_bright)
    comp_orig_dark = compare_hog_features(features_original, features_dark)
    
    print("Normalization Effect on Brightness Changes:")
    print(f"\nOriginal vs Bright (+50%):")
    print(f"  Euclidean Distance: {comp_orig_bright['euclidean']:.4f}")
    print(f"  Cosine Similarity: {comp_orig_bright['cosine_similarity']:.4f}")
    
    print(f"\nOriginal vs Dark (-30%):")
    print(f"  Euclidean Distance: {comp_orig_dark['euclidean']:.4f}")
    print(f"  Cosine Similarity: {comp_orig_dark['cosine_similarity']:.4f}")
    
    print("\nNote: Cosine similarity shows better invariance to brightness")
    print("because HOG uses normalization within blocks.")


def main():
    """Run HOG demonstrations."""
    Path("output").mkdir(exist_ok=True)
    
    demo_hog_extraction()
    demo_hog_scikit()
    demo_hog_cell_sizes()
    demo_hog_feature_comparison()
    demo_hog_normalization_effect()
    
    print("\n" + "="*60)
    print("HOG Descriptor Demos Complete")
    print("="*60)


if __name__ == "__main__":
    main()
