# 加载示例数据
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lifelines import CoxPHFitter
import warnings
import pickle
warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('/mnt/e/liangdong/Machine_Learning/github/ModelDeployment/predict_app_deploy/filter_data1.csv')

# 删除ID列
df = data.drop(columns=['ID'])  # 确保删除正确的ID列

# 定义时间和事件列
time_col = 'OS.time'
event_col = 'OS'

# 检查缺失值并删除包含缺失值的行
df = df.dropna()

# 分离特征和标签
X = df.drop(columns=[time_col, event_col])
y = df[[time_col, event_col]]

# 将分类变量转换为独热编码
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)

# 标准化数值特征
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 合并X和y训练数据
train_data = X_train.copy()
train_data[time_col] = y_train[time_col]
train_data[event_col] = y_train[event_col]

# 构建Cox回归模型
cph = CoxPHFitter()
cph.fit(train_data, duration_col=time_col, event_col=event_col)

# 保存模型为 pkl 文件
with open('prediction_model.pkl', 'wb') as file:
    pickle.dump(cph, file)

print("模型已保存为 prediction_model.pkl")
