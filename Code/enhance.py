import os
import cv2
import numpy as np

def apply_gabor_filter(image):

    orientations = 8  # 方向数量
    scales = 5  # 尺度数量
    min_wave_length = 4  # 最小波长
    mult = 2.1  # 波长倍数
    sigma_onf = 0.55  # 频率标准差
    gamma = 1.0  # 空间纵横比

    filtered_images = np.zeros((scales, orientations, image.shape[0], image.shape[1]))

    for scale in range(scales):
        wavelength = min_wave_length * mult ** scale
        for orientation in range(orientations):
            angle = orientation * np.pi / orientations
            gabor_kernel = cv2.getGaborKernel((image.shape[1], image.shape[0]), sigma_onf, angle, wavelength, gamma)
            filtered_images[scale, orientation] = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)

    return filtered_images

def enhance_blood_vessels(image):
    # 应用Gabor滤波增强
    filtered_images = apply_gabor_filter(image)
    gabor_result = np.max(filtered_images, axis=(0, 1))
    enhanced_image = cv2.add(image, gabor_result.astype(np.uint8))  # 指定输出图像的数据类型为uint8
    # 进行CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(enhanced_image)
    # 中值滤波
    enhanced_image = cv2.medianBlur(enhanced_image, 1)

    return enhanced_image

def extract_blood_vessels(image):
    ret, thresholded_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresholded_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
    return thresholded_image

# Define input and output directories
input_dir = 'D:/Uni/Project/dataset/Palmvein_ROI_gray_128x128/session1'
output_dir = 'D:/Uni/Project/0415/Tongji_enhance_2'

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'color'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'enhanced'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'vessels'), exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
        # Read the image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Image enhancement
        enhanced_image = enhance_blood_vessels(image)

        # Extract blood vessels
        vessels_image = extract_blood_vessels(enhanced_image)

        # Convert grayscale image to color image
        color_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        # Save images to the output folders
        cv2.imwrite(os.path.join(output_dir, 'color', filename), color_image)
        cv2.imwrite(os.path.join(output_dir, 'enhanced', filename), enhanced_image)
        cv2.imwrite(os.path.join(output_dir, 'vessels', filename), vessels_image)

        print(f"Processed image: {filename}")

print("Batch processing completed.")
