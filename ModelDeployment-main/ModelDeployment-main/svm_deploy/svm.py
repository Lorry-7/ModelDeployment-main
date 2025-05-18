# 加载示例数据
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
import pickle
warnings.filterwarnings('ignore')
# 读取数据
df = pd.read_csv('new.csv')
# 定义事件列（作为分类标签）
event_col = 'Surgery'
# 分离特征和标签
X = df.drop(columns=[event_col])
y = df[event_col]
# # 前4列是分类变量
# categorical_cols = X.columns[:6]
# encoder = OneHotEncoder(drop='first', sparse_output=False)
# X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
# # 剩余列是连续变量
# continuous_cols = X.columns[6:]
# X_continuous = X[continuous_cols]
# # 合并编码后的分类变量和连续变量
# X = pd.concat([X_encoded, X_continuous], axis=1)
# # print(X.columns)
# # 标准化数值特征
# scaler = StandardScaler()
# X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
# 构建SVM模型
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
# 保存模型为 pkl 文件
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
print("模型已保存为 svm_model.pkl")


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
# # 模型预测
# y_pred = svm_model.predict(X_test)
# y_pred_proba = svm_model.predict_proba(X_test)[:, 1]  # 获取正类的概率
# # 计算评估指标
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# # 计算 ROC 曲线和 AUC 值
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)
# # 输出评估指标
# print(f"准确率: {accuracy}")
# print(f"精确率: {precision}")
# print(f"召回率: {recall}")
# print(f"F1 值: {f1}")
# print(f"AUC 值: {roc_auc}")
