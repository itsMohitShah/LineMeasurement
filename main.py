"""
This script processes images to detect lines and measures the length in terms of pixels.
"""

import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt


def get_image_and_resize(image_path):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        aspect_ratio = image_width / image_height
        if aspect_ratio > 1:
            new_width = 1080
            new_height = int(1080 / aspect_ratio)
        else:
            new_height = 720
            new_width = int(720 * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))

        return image

def apply_skeletonization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    skeleton = skeletonize(binary // 255) * 255
    skeleton = skeleton.astype(np.uint8)
    
    # In an actual application, there could be an object detector model that finds the scratch and then apply this process within that box.
    # For this example, I am filtering out the small noises which were not removed by the gaussian blur.
    labeled = label(skeleton)
    filtered = np.zeros_like(skeleton)

    for region in regionprops(labeled):
        if region.area > 200:
            coords = region.coords
            for y,x in coords:
                filtered[y, x] = 255

    
    
    
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    filtered[:, :, 0] = 0  
    filtered[:, :, 1] = 0  
    
    return filtered


def get_skeleton_pixel_count(skeleton):
    return np.count_nonzero(skeleton)

def plot_original_and_skeleton_and_overlay(image_name, image, skeleton, overlay):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(skeleton)
    plt.title('Skeletonized Image')
    plt.axis('off')
    

    plt.subplot(2, 1, 2)
    plt.imshow(overlay)
    plt.title('Overlay Image')
    plt.axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    image_name.replace(".jpg", ".png")
    plt.savefig(f"output/{image_name}", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    images_dir = "images"
    image_names = os.listdir(images_dir)
    for image_name in image_names:
        image_path = os.path.join(images_dir, image_name)
        image = get_image_and_resize(image_path)
        
        filtered_image = apply_skeletonization(image)
        skeleton_pixel_count = get_skeleton_pixel_count(filtered_image) 
        print(f"Skeleton pixel count for {image_name}: {skeleton_pixel_count}")
        image = cv2.imread(image_path)
        image = cv2.resize(image, (filtered_image.shape[1], filtered_image.shape[0]))


        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 0.9, filtered_image, 0.5, 0)    
        cv2.putText(overlay, f"Pixel count: {skeleton_pixel_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        plot_original_and_skeleton_and_overlay(image_name, image, filtered_image, overlay)


    print("Execution complete")