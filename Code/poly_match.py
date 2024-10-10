import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.metrics import classification_report  # 生产报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
import os
import json
from PIL import Image

train_folder = 'D:/Uni/Project/0424/Poly/vessels'
annotations_file = 'D:/Uni/Project/0424/Poly/annotations.json'  # 替换为包含标注信息的JSON文件路径

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

def extract_orb_features(images):
    orb = cv2.ORB_create()
    features = []
    for image in images:
        _, des = orb.detectAndCompute(image, None)
        if des is not None:
            features.append(des)
    return features

X_features = extract_orb_features(X)
X_features = [des for des in X_features if des is not None]  # Remove None elements from the list

max_length = max(len(des) for des in X_features)

for i in range(len(X_features)):
    des = X_features[i]
    if len(des) < max_length:
        padded_des = np.pad(des, ((0, max_length - len(des)), (0, 0)), 'constant')
        X_features[i] = padded_des

X_features = np.array(X_features)

num_samples, num_descriptors, descriptor_length = X_features.shape
X_features = X_features.reshape(num_samples, num_descriptors * descriptor_length)

X_features = preprocessing.scale(X_features)

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
save_path = "D:/Uni/Project/essay/image/chap05/poly_heatmap_rf.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()
'''
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
save_path = "D:/Uni/Project/essay/image/chap05/poly_heatmap_lr.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()

# 决策树
start_time = time.time()
dtc = DecisionTreeClassifier()
dtc.fit(x_train.reshape(x_train.shape[0], -1), y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_dtc = dtc.predict(x_test.reshape(x_test.shape[0], -1))
print('predict took %fs!' % (time.time() - start_time))
report_dtc = classification_report(y_test, pred_dtc)
confusion_mat_dtc = confusion_matrix(y_test, pred_dtc)
print(report_dtc)
print(confusion_mat_dtc)
print('决策树准确率: %0.4lf' % accuracy_score(pred_dtc, y_test))
# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat_dtc, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
save_path = "D:/Uni/Project/essay/image/chap05/poly_heatmap_dtc.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()

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
save_path = "D:/Uni/Project/essay/image/chap005/poly_heatmap_svm.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()
'''
