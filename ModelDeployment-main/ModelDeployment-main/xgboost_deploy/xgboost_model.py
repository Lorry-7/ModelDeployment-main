# 加载示例数据
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
import pickle

warnings.filterwarnings('ignore')
# 读取数据
df = pd.read_csv('new_up.csv')
# 定义事件列（作为分类标签）
event_col = 'Surgery_Group'
# 分离特征和标签
X = df.drop(columns=[event_col])
y = df[event_col]
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)
# 构建SVM模型
xgboost_model = xgb.XGBClassifier(objective='binary:logistic',
                                  random_state=24,
                                  n_estimators=550,
                                  learning_rate=0.01,
                                  eval_metric='logloss')
# xgboost_model = SVC(kernel='linear', probability=True)
xgboost_model.fit(X_train, y_train)
# 保存模型为 pkl 文件
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(xgboost_model, file)
print("模型已保存为 xgboost_model.pkl")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
# 模型预测
y_pred = xgboost_model.predict(X_test)
y_pred_proba = xgboost_model.predict_proba(X_test)[:, 1]  # 获取正类的概率
# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# 计算 ROC 曲线和 AUC 值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
# 输出评估指标
print(f"准确率: {accuracy}")
print(f"精确率: {precision}")
print(f"召回率: {recall}")
print(f"F1 值: {f1}")
print(f"AUC 值: {roc_auc}")
