from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter

filename = r'chinesetzxzbm.csv'
data = pd.read_csv(filename)

y = data['liver']
X = data.drop(columns=['liver'])

# 初始化 SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=2024)

# 首先应用 SMOTE
X_smote, y_smote = smote.fit_resample(X, y)

# 然后应用 TomekLinks
tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X_smote, y_smote)

# 查看采样后的数据分布
print('Original dataset shape:', Counter(y))
print('After SMOTE shape:', Counter(y_smote))
print('After TomekLinks shape:', Counter(y_resampled))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2024)

# 定义基学习器
base_learners = [
    ('lr', LogisticRegression(max_iter=10000, random_state=2024)),
    ('svc', SVC(probability=True, random_state=2024)),
    ('rfc', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, min_samples_split=10, random_state=2024)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('mlp', MLPClassifier(alpha=0.1, max_iter=1000, random_state=2024)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2024))
]

# 定义元学习器（Logistic Regression 或其他分类器）
meta_learner = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2024)

# 使用堆叠法进行集成
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# 训练堆叠模型
stacking_clf.fit(X_train, y_train)
# print(X_train)

import pickle
# 保存模型为 pkl 文件
with open('stacking_clf.pkl', 'wb') as file:
    pickle.dump(stacking_clf, file)

print("模型已保存为 stacking_clf.pkl")
