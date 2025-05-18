import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'xgboost_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("Prediction Model for Prolonged Length of Stay Beyond 48 Hours in Ambulatory Total Hip Arthroplasty Patients Based on Machine Learning")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
# st.sidebar.subheader("Picking up parameters")

Age = st.sidebar.slider("Age, year", min_value=18, max_value=90, value=55, step=1)

OP_option = ["Yes", "No"]
OP_map = {"Yes": 1, "No": 0}
OP_sb = st.sidebar.selectbox("OP", OP_option, index=1)

eGFR = st.sidebar.slider("eGFR", min_value=30, max_value=165, value=110, step=1)

ASA_option = ["Yes", "No"]
ASA_map = {"Yes": 3, "No": 2}
ASA_sb = st.sidebar.selectbox("ASA", ASA_option, index=1)

SD = st.sidebar.slider("SD", min_value=35, max_value=470, value=95, step=1)
AD = st.sidebar.slider("AD", min_value=70, max_value=540, value=230, step=1)
IBL = st.sidebar.slider("IBL", min_value=50, max_value=2200, value=300, step=1)
BMI = st.sidebar.slider("BMI", min_value=15.0, max_value=37.0, value=24.4, step=0.1)

Smoking_option = ["Yes", "No"]
Smoking_map = {"Yes": 1, "No": 0}
Smoking_sb = st.sidebar.selectbox("Smoking", Smoking_option, index=1)

ADL_Score = st.sidebar.slider("ADL_Score", min_value=35, max_value=100, value=85, step=1)

input_data = pd.DataFrame({
    'Age': [Age],
    'OP': [OP_map[OP_sb]],
    'eGFR': [eGFR],
    'ASA': [ASA_map[ASA_sb]],
    'SD': [SD],
    'AD': [AD],
    'IBL': [IBL],
    'BMI': [BMI],
    'Smoking': [Smoking_map[Smoking_sb]],
    'ADL_Score': [ADL_Score]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predictive Probability: {final_pred_proba:.2f}%")
