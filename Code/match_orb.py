import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from skimage.feature import hog  # 用于提取HOG特征

# 图像文件夹路径
image_folder = "D:/Uni/Project/0415/Tongji/Tongji_enhance/verify"

# 提取图像文件名和标签
image_files = sorted(os.listdir(image_folder))
X = []  # 存储图像数据
y = []  # 存储标签

label = 1
count = 0

# 定义图像预处理函数
def preprocess_image(image_path):
    image = cv2.imread(image_path)  # 使用OpenCV读取图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    # 进行其他的图像预处理操作，如调整大小、归一化等
    return image

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    # 读取图像数据并进行相应的预处理
    image = preprocess_image(image_path)
    X.append(image)
    y.append(label)
    count += 1
    if count == 10:
        count = 0
        label += 1

# 转换为NumPy数组
X = np.array(X)
y = np.array(y)

# 提取ORB特征
def extract_orb_features(images):
    orb = cv2.ORB_create()
    features = []
    for image in images:
        _, des = orb.detectAndCompute(image, None)
        features.append(des)
    return features

# 数据预处理和特征提取
X_features = extract_orb_features(X)  # 提取ORB特征

# 找到最大的特征描述子长度
max_length = max(len(des) for des in X_features)

# 对特征描述子进行填充
for i in range(len(X_features)):
    des = X_features[i]
    if len(des) < max_length:
        padded_des = np.pad(des, ((0, max_length - len(des)), (0, 0)), 'constant')
        X_features[i] = padded_des

# 转换为NumPy数组
X_features = np.array(X_features)
X_features = np.array(X_features)  # 转换为NumPy数组
X_features = preprocessing.scale(X_features)  # 特征缩放

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# 创建SVM分类器
svm_classifier = svm.SVC()

# 拟合（训练）模型
svm_classifier.fit(X_train, y_train)

# 对于训练好的SVM分类器，您可以使用以下代码进行预测和评估：

# 预测
y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)