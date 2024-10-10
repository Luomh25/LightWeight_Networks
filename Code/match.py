import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report  # 生产报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
import os
import json
from PIL import Image

train_folder = 'D:/Uni/Project/0424/vein_ml/train'  # 替换为训练集文件夹路径
val_folder = 'D:/Uni/Project/0424/vein_ml/val'  # 替换为测试集文件夹路径
annotations_file = 'D:/Uni/Project/0424/vein_ml/annotations.json'  # 替换为包含标注信息的JSON文件路径

# 读取注释文件
with open(annotations_file, "r") as file:
    annotations = json.load(file)

# 创建训练集和测试集的图像和标签列表
x_train = []
y_train = []
x_test = []
y_test = []

# 处理训练集
for annotation in annotations:
    hand_id = annotation["hand_id"]
    images = annotation["images"]
    for image in images:
        file_name = image["file_name"]
        label = image["label"]

        # 读取图像
        image_path = os.path.join(train_folder, file_name)

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path)
        img = np.array(img)  # 转换为NumPy数组

        # 添加到训练集
        x_train.append(img)
        y_train.append(label)

# 处理测试集
for annotation in annotations:
    hand_id = annotation["hand_id"]
    images = annotation["images"]
    for image in images:
        file_name = image["file_name"]
        label = image["label"]

        # 读取图像
        image_path = os.path.join(val_folder, file_name)

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path)
        img = np.array(img)  # 转换为NumPy数组

        # 添加到测试集
        x_test.append(img)
        y_test.append(label)

# 转换为NumPy数组
x_train = np.array(x_train)
x_test = np.array(x_test)

# 提取ORB特征
def extract_orb_features(images):
    orb = cv2.ORB_create()
    features = []
    for image in images:
        _, des = orb.detectAndCompute(image, None)
        features.append(des)
    return features

# 特征提取
x_train = extract_orb_features(x_train)  # 提取ORB特征
x_test = extract_orb_features(x_test)

# 找到最大的特征描述子长度
max_length_train = max(len(des) for des in x_train)
max_length_test = max(len(des) for des in x_test)

# 对特征描述子进行填充
for i in range(len(x_train)):
    des = x_train[i]
    if len(des) < max_length_train:
        padded_des = np.pad(des, ((0, max_length_train - len(des)), (0, 0)), 'constant')
        x_train[i] = padded_des

for i in range(len(x_test)):
    des = x_test[i]
    if len(des) < max_length_test:
        padded_des = np.pad(des, ((0, max_length_test - len(des)), (0, 0)), 'constant')
        x_test[i] = padded_des

# 转换为NumPy数组
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# 打印训练集和测试集的形状
print("训练集形状:", x_train.shape, y_train.shape)
print("测试集形状:", x_test.shape, y_test.shape)

'''
print('x_train:\n', x_train[:5])
print('x_test:\n', x_test[:5])
print('y_train:\n', y_train[:5])
print('y_test:\n', y_test[:5])
'''
'''
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
save_path = "D:/Uni/Project/essay/image/chap04/heatmap_rf_2.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()

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
save_path = "D:/Uni/Project/essay/image/chap04/heatmap_lr_2.png"
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
save_path = "D:/Uni/Project/essay/image/chap04/heatmap_dtc_2.png"
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
save_path = "D:/Uni/Project/essay/image/chap04/heatmap_svm_2.png"
plt.savefig(save_path, dpi=300)  # 指定dpi以控制保存的图像质量
plt.show()
'''
