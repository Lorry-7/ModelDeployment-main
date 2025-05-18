import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# 加载示例数据
filename = r'Chabuhou.csv'
data = pd.read_csv(filename)

y = data['Age_Group']
X = data.drop(columns=['Age_Group'])

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 GBDT 分类器
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 训练 XGBoost 模型
# model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
gb_clf.fit(X_train, y_train)

# 保存模型为 pkl 文件
with open('gbdt_model.pkl', 'wb') as file:
    pickle.dump(gb_clf, file)

print("模型已保存为 gbdt_model.pkl")
