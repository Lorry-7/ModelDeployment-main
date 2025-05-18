import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'svm_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("日间全髋关节置换出院时间计算器 - 评估延迟出院风险")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
# st.sidebar.subheader("Picking up parameters")

ASA_option = ["Yes", "No"]
ASA_map = {"Yes": 1, "No": 0}
ASA_sb = st.sidebar.selectbox("ASA", ASA_option, index=1)

DM_option = ["Yes", "No"]
DM_map = {"Yes": 1, "No": 0}
DM_sb = st.sidebar.selectbox("DM", DM_option, index=1)

PDC_option = ["Yes", "No"]
PDC_map = {"Yes": 1, "No": 0}
PDC_sb = st.sidebar.selectbox("PDC", PDC_option, index=0)

Lumbar_option = ["Yes", "No"]
Lumbar_map = {"Yes": 1, "No": 0}
Lumbar_sb = st.sidebar.selectbox("Lumbar_Anesthesia", Lumbar_option, index=1)

Smoking_option = ["Yes", "No"]
Smoking_map = {"Yes": 1, "No": 0}
Smoking_sb = st.sidebar.selectbox("Smoking", Smoking_option, index=1)

OP_option = ["Yes", "No"]
OP_map = {"Yes": 1, "No": 0}
OP_sb = st.sidebar.selectbox("OP", OP_option, index=1)

Age = st.sidebar.slider("Age, year", min_value=18, max_value=90, value=81, step=1)
HAA = st.sidebar.slider("HAA", min_value=0, max_value=50, value=15, step=1)
HB = st.sidebar.slider("HB", min_value=70, max_value=380, value=116, step=1)
FIB = st.sidebar.slider("FIB", min_value=0.9, max_value=27.0, value=3.01, step=0.01)
eGFR = st.sidebar.slider("eGFR", min_value=16, max_value=570, value=114, step=1)
SD = st.sidebar.slider("SD", min_value=35, max_value=470, value=155, step=1)
AD = st.sidebar.slider("AD", min_value=70, max_value=540, value=230, step=1)
IBL = st.sidebar.slider("IBL", min_value=50, max_value=2200, value=1000, step=1)
BMI = st.sidebar.slider("BMI", min_value=15.0, max_value=37.0, value=20.9, step=0.1)


input_data = pd.DataFrame({
    'ASA': [ASA_map[ASA_sb]],
    'DM': [DM_map[DM_sb]],
    'PDC': [PDC_map[PDC_sb]],
    'Lumbar_Anesthesia': [Lumbar_map[Lumbar_sb]],
    'Smoking': [Smoking_map[Smoking_sb]],
    'OP': [OP_map[OP_sb]],
    'Age': [Age],
    'HAA': [HAA],
    'HB': [HB],
    'FIB': [FIB],
    'eGFR': [eGFR],
    'SD': [SD],
    'AD': [AD],
    'IBL': [IBL],
    'BMI': [BMI]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"预测概率: {final_pred_proba:.2f}%")
