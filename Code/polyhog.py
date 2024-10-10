import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report  # 生产报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from skimage.feature import hog  # 用于提取HOG特征
import seaborn as sns
import json
from PIL import Image

train_folder = '/root/data/vessels'
annotations_file = '/root/annotations.json'  # 替换为包含标注信息的JSON文件路径

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

with open(annotations_file, "r") as file:
    annotations = json.load(file)

X = []
y = []

for annotation in annotations:
    hand_id = annotation["hand_id"]
    images = annotation["images"]
    for image in images:
        file_name = image["file_name"]
        label = image["label"]

        image_path = os.path.join(train_folder, file_name)

        if not os.path.exists(image_path):
            continue

        img = preprocess_image(image_path)
        img = np.array(img)
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# 提取HOG特征
def extract_hog_features(images):
    features = []
    for image in images:
        hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        features.append(hog_features)
    return features

# 数据预处理和特征提取
X_features = extract_hog_features(X)  # 提取HOG特征
X_features = np.array(X_features)  # 转换为NumPy数组
X_features = preprocessing.scale(X_features)  # 特征缩放

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的形状
print("训练集形状:", x_train.shape, y_train.shape)
print("测试集形状:", x_test.shape, y_test.shape)

# rf
start_time = time.time()
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(x_train.reshape(x_train.shape[0], -1), y_train)  # Reshape训练数据以适应模型
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_rf = rf.predict(x_test.reshape(x_test.shape[0], -1))  # Reshape测试数据进行预测
print('predict took %fs!' % (time.time() - start_time))
report_rf = classification_report(y_test, pred_rf)
confusion_mat_rf = confusion_matrix(y_test, pred_rf)
print(report_rf)
print(confusion_mat_rf)
print('随机森林准确率: %0.4lf' % accuracy_score(pred_rf, y_test))
# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_rf, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
save_path = "/root/poly_heatmap_rf.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量

# LR
start_time = time.time()
lr = LogisticRegression()
lr.fit(x_train.reshape(x_train.shape[0], -1), y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_lr = lr.predict(x_test.reshape(x_test.shape[0], -1))
print('predict took %fs!' % (time.time() - start_time))
report_lr = classification_report(y_test, pred_lr)
confusion_mat_lr = confusion_matrix(y_test, pred_lr)
print(report_lr)
print(confusion_mat_lr)
print('LR准确率: %0.4lf' % accuracy_score(pred_lr, y_test))
# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_lr, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
save_path = "/root/poly_heatmap_lr.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量

# SVM
start_time = time.time()
svm = SVC()
svm.fit(x_train.reshape(x_train.shape[0], -1), y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_svm = svm.predict(x_test.reshape(x_test.shape[0], -1))
report_svm = classification_report(y_test, pred_svm)
print('predict took %fs!' % (time.time() - start_time))
confusion_mat_svm = confusion_matrix(y_test, pred_svm)
print(report_svm)
print(confusion_mat_svm)
print('SVC准确率: %0.4lf' % accuracy_score(pred_svm, y_test))
# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_svm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
save_path = "/root/poly_heatmap_svm.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量

