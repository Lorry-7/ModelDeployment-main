import streamlit as st  # 导入 Streamlit 库，用于创建 Web 应用
import pandas as pd  # 导入 Pandas 库，用于数据处理
import pickle  # 导入 pickle 库，用于加载已训练的模型
import os  # 导入 os 库，用于处理文件路径

# 加载模型
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 组合当前目录与模型文件名，生成模型的完整路径
model_path = os.path.join(current_dir, 'gbdt_model.pkl')
# 打开并加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)  # 使用 pickle 加载模型文件

# 设置 Streamlit 应用的标题
st.title("A precise predictive model for centenarians from the China Healthy Longevity Multicenter Study (CHLMS)")

st.sidebar.header("Selection Panel")  # 侧边栏的标题
st.sidebar.subheader("Picking up parameters")

Gender_option = ["Male", "Female"]
Gender_map = {"Male": 0, "Female": 1}
Gender_sb = st.sidebar.selectbox("Gender", Gender_option, index=0)

Nation_option = ["Han", "Hui", "Jing", "Li", "Miao", "Zhuang", "Yao", "Others"]
Nation_map = {"Han": 1, "Hui": 2, "Jing": 3, "Li": 4, "Miao": 6, "Zhuang": 7, "Yao": 8, "Others": 9}
Nation_sb = st.sidebar.selectbox("Nation", Nation_option, index=3)

Marital_Status_option = ["Married", "Unmarried", "Divorced", "Widowed"]
Marital_Status_map = {"Married": 1, "Unmarried": 2, "Divorced": 3, "Widowed": 4}
Marital_Status_sb = st.sidebar.selectbox("Marital status", Marital_Status_option, index=2)

Hypertension_option = ["Yes", "No"]
Hypertension_map = {"Yes": 1, "No": 0}
Hypertension_sb = st.sidebar.selectbox("Hypertension", Hypertension_option, index=1)

Smoking_option = ["Often", "Sometimes", "Never", "Withdrawal"]
Smoking_map = {"Often": 1, "Sometimes": 2, "Never": 3, "Withdrawal": 4}
Smoking_sb = st.sidebar.selectbox("Smoking", Smoking_option, index=2)

Bedtime_option = ["Within an hour", "Within two hours", "Two hours away"]
Bedtime_map = {"Within an hour": 1, "Within two hours": 2, "Two hours away": 3}
Bedtime_sb = st.sidebar.selectbox("Bedtime", Bedtime_option, index=0)

Outdoor_exercise_option = ["Yes", "No"]
Outdoor_exercise_map = {"Yes": 1, "No": 0}
Outdoor_exercise_sb = st.sidebar.selectbox("Outdoor exercise", Outdoor_exercise_option, index=0)

Play_cards_option = ["Yes", "No"]
Play_cards_map = {"Yes": 1, "No": 0}
Play_cards_sb = st.sidebar.selectbox("Play cards", Play_cards_option, index=1)

Watching_TV_option = ["Yes", "No"]
Watching_TV_map = {"Yes": 1, "No": 0}
Watching_TV_sb = st.sidebar.selectbox("Watching TV", Watching_TV_option, index=1)

Feeding_option = ["Yes", "No"]
Feeding_map = {"Yes": 1, "No": 0}
Feeding_sb = st.sidebar.selectbox("Feeding", Feeding_option, index=1)

Rice_option = ["Yes", "No"]
Rice_map = {"Yes": 1, "No": 0}
Rice_sb = st.sidebar.selectbox("Rice intake", Rice_option, index=0)

Hearing_option = ["Good", "Blurred", "Deafness"]
Hearing_map = {"Good": 1, "Blurred": 2, "Deafness": 3}
Hearing_sb = st.sidebar.selectbox("Hearing", Hearing_option, index=1)

SBP = st.sidebar.slider("SBP, mmHg", min_value=38, max_value=230, value=148, step=1)
DBP = st.sidebar.slider("DBP, mmHg", min_value=38, max_value=218, value=78, step=1)
TC = st.sidebar.slider("TC, mmol/L", min_value=0.35, max_value=11.81, value=4.34, step=0.01)

# 创建输入数据框，将输入的特征整理为 DataFrame 格式
input_data = pd.DataFrame({
    'Gender': [Gender_map[Gender_sb]],
    'Nation': [Nation_map[Nation_sb]],
    'Marital_Status': [Marital_Status_map[Marital_Status_sb]],
    'Hypertension': [Hypertension_map[Hypertension_sb]],
    'Smoking': [Smoking_map[Smoking_sb]],
    'Bedtime': [Bedtime_map[Bedtime_sb]],
    'Outdoor_exercise': [Outdoor_exercise_map[Outdoor_exercise_sb]],
    'Play_cards': [Play_cards_map[Play_cards_sb]],
    'Watching_TV': [Watching_TV_map[Watching_TV_sb]],
    'Feeding': [Feeding_map[Feeding_sb]],
    'Rice': [Rice_map[Rice_sb]],
    'Hearing': [Hearing_map[Hearing_sb]],
    'SBP': [SBP],
    'DBP': [DBP],
    'TC': [TC]
})

if st.button("Calculate"):
    y_pred = model.predict(input_data)
    y_pred_proba = model.predict_proba(input_data)[:, 1]
    final_pred_proba = y_pred_proba[0] * 100
    st.write(f"Predicted probability of centenarians: {final_pred_proba:.2f}%")
