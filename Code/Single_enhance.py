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


# 读取图像
image_path = 'D:/Uni/Project/dataset/Palmvein_ROI_gray_128x128/session1/00001.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 图像增强
enhanced_image = enhance_blood_vessels(image)

# 血管提取
vessels_image = extract_blood_vessels(enhanced_image)

# 显示原始图像、增强图像和血管图像
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.imshow("Vessels Image", vessels_image)

# 等待按下任意键后关闭图像窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
