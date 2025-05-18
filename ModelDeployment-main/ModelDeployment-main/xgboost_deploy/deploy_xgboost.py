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
st.title("Same-Day Total Hip Replacement Discharge Time Calculator - Evaluating Delayed Discharge Risk")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
# st.sidebar.subheader("Picking up parameters")

Smoking_option = ["Yes", "No"]
Smoking_map = {"Yes": 1, "No": 0}
Smoking_sb = st.sidebar.selectbox("Smoking", Smoking_option, index=1)

DM_option = ["Yes", "No"]
DM_map = {"Yes": 1, "No": 0}
DM_sb = st.sidebar.selectbox("DM", DM_option, index=1)

OP_option = ["Yes", "No"]
OP_map = {"Yes": 1, "No": 0}
OP_sb = st.sidebar.selectbox("OP", OP_option, index=1)

ASA_option = ["Yes", "No"]
ASA_map = {"Yes": 1, "No": 0}
ASA_sb = st.sidebar.selectbox("ASA", ASA_option, index=0)

Lumbar_option = ["Yes", "No"]
Lumbar_map = {"Yes": 1, "No": 0}
Lumbar_sb = st.sidebar.selectbox("Lumbar_Anesthesia", Lumbar_option, index=1)

PDC_option = ["Yes", "No"]
PDC_map = {"Yes": 1, "No": 0}
PDC_sb = st.sidebar.selectbox("PDC", PDC_option, index=1)

Contralateral_Surgery_option = ["Yes", "No"]
Contralateral_Surgery_map = {"Yes": 1, "No": 0}
Contralateral_Surgery_sb = st.sidebar.selectbox("Contralateral_Surgery", Contralateral_Surgery_option, index=0)

Age = st.sidebar.slider("Age, year", min_value=18, max_value=90, value=64, step=1)
WBC = st.sidebar.slider("WBC, year", min_value=2.30, max_value=24.56, value=5.43, step=0.01)
ANC = st.sidebar.slider("ANC, year", min_value=0.1, max_value=99.0, value=62.5, step=0.1)
eGFR = st.sidebar.slider("eGFR", min_value=30, max_value=165, value=109, step=1)
ALB = st.sidebar.slider("ALB", min_value=26.0, max_value=53.0, value=46.3, step=0.1)
SD = st.sidebar.slider("SD", min_value=35, max_value=470, value=95, step=1)
IBL = st.sidebar.slider("IBL", min_value=50, max_value=2200, value=200, step=1)
BMI = st.sidebar.slider("BMI", min_value=15.0, max_value=37.0, value=31.1, step=0.1)

input_data = pd.DataFrame({
    'Smoking': [Smoking_map[Smoking_sb]],
    'DM': [DM_map[DM_sb]],
    'OP': [OP_map[OP_sb]],
    'ASA': [ASA_map[ASA_sb]],
    'Lumbar_Anesthesia': [Lumbar_map[Lumbar_sb]],
    'PDC': [PDC_map[PDC_sb]],
    'Contralateral_Surgery': [Contralateral_Surgery_map[Contralateral_Surgery_sb]],
    'Age': [Age],
    'WBC': [WBC],
    'ANC': [ANC],
    'eGFR': [eGFR],
    'ALB': [ALB],
    'SD': [SD],
    'IBL': [IBL],
    'BMI': [BMI]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predictive Probability: {final_pred_proba:.2f}%")
