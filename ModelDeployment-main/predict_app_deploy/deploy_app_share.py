import streamlit as st  # Import Streamlit library for creating Web application
import pandas as pd  # Import Pandas library for data processing
import numpy as np  # Import numpy library for numerical calculations
import pickle  # Import pickle library for loading trained models
import os  # Import os library for handling file paths
from sklearn.preprocessing import RobustScaler  # Import data standardization tool
import joblib
import shap  # Import SHAP library for model interpretation
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Set Streamlit application title
st.title("HPAHs-Stacking-Model Prediction Platform")

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sidebar title
st.sidebar.header("HPAHs-Stacking-Model")  # Sidebar title

# Load all models
targets = ["LogP", "LogBCF", "LogKOC", "Biowin7"]
models = {}
models_status = {}

for target in targets:
    model_path = os.path.join(current_dir, f'enhanced_stacking_model_{target}.pkl')
    try:
        with open(model_path, 'rb') as file:
            models[target] = joblib.load(file)
        models_status[target] = True
    except Exception as e:
        st.sidebar.error(f"{target} model loading failed: {str(e)}")
        models_status[target] = False

# If all models fail to load, stop the application
if not any(models_status.values()):
    st.error("All models failed to load, please check if model files exist!")
    st.stop()

# Read HPAHs-Data.xlsx file
try:
    hpahs_data_path = os.path.join(current_dir, 'HPAHs-Data.xlsx')
    hpahs_data = pd.read_excel(hpahs_data_path)
    
    # Check if necessary columns exist
    required_columns = ['CAS']
    missing_columns = [col for col in required_columns if col not in hpahs_data.columns]
    
    if missing_columns:
        st.sidebar.error(f"Data file is missing the following columns: {', '.join(missing_columns)}")
        use_cas_feature = False
    else:
        use_cas_feature = True
        
        # Check if there is an English name column
        has_english_name = 'English Name' in hpahs_data.columns
        
        # Create CAS number to feature mapping and CAS number to English name mapping
        cas_features_map = {}
        cas_english_name_map = {}
        
        for _, row in hpahs_data.iterrows():
            cas = str(row['CAS']).strip()
            if cas:  # If CAS number is not empty
                # Store features
                features = {}
                for col in hpahs_data.columns:
                    if col not in ['CAS', 'English Name']:
                        features[col] = row[col]
                cas_features_map[cas] = features
                
                # Store English name
                if has_english_name:
                    english_name = row['English Name'] if not pd.isna(row['English Name']) else "Unknown"
                    cas_english_name_map[cas] = english_name
        
        # Get all available CAS number list
        cas_list = list(cas_features_map.keys())
        
        # Create a display option, including CAS number and English name (if available)
        display_options = []
        # Add empty option as default
        display_options.append("Select a compound...")
        
        for cas in cas_list:
            if has_english_name and cas in cas_english_name_map:
                display_options.append(f"{cas} - {cas_english_name_map[cas]}")
            else:
                display_options.append(cas)
        
        # Add search functionality
        cas_search = st.sidebar.text_input("Search CAS number or English name")
        
        # If a search term is entered, filter option list
        if cas_search:
            filtered_options = [opt for opt in display_options if cas_search.lower() in opt.lower()]
            if not filtered_options:
                st.sidebar.warning(f"No options found containing '{cas_search}'")
                selected_option = None
            else:
                selected_option = st.sidebar.selectbox("Select compound", filtered_options)
        else:
            selected_option = st.sidebar.selectbox("Select compound", display_options) if display_options else None
        
        # Extract CAS number from selected option
        selected_cas = None
        if selected_option and selected_option != "Select a compound...":
            selected_cas = selected_option.split(" - ")[0]
            
        # Display selected compound information
        if selected_cas and has_english_name and selected_cas in cas_english_name_map:
            english_name = cas_english_name_map[selected_cas]
            st.sidebar.write(f"**Current Selection**: {english_name} (CAS: {selected_cas})")
            
            # Create information panel on main interface
            with st.expander("View Compound Details", expanded=True):
                st.write(f"### {english_name}")
                st.write(f"**CAS Number**: {selected_cas}")
                
                # Display molecular features
                if selected_cas in cas_features_map:
                    st.write("#### Molecular Features:")
                    feature_data = cas_features_map[selected_cas]
                    
                    # Create table data
                    feature_table = []
                    for feature, value in feature_data.items():
                        if not pd.isna(value):
                            feature_table.append({
                                "Feature Name": feature,
                                "Value": value
                            })
                    
                    # Display as table
                    if feature_table:
                        st.table(pd.DataFrame(feature_table))
                    else:
                        st.write("No feature data available")
except Exception as e:
    st.sidebar.error(f"Failed to read HPAHs-Data.xlsx file: {str(e)}")
    use_cas_feature = False
    selected_cas = None

# Feature default values
default_values = {
    'Isotropic Average Polarizability': 20.0,
    'Molecular Mass': 200.0,
    'Boiling Point': 100.0,
    'Hardness': 2.0,
    'Magnitude of Molecular Dipole Moment': 2.0,
    'Cubic electrophilicity index': 1.0,
    'Melting Point': 50.0,
    'ESPmax': 0.5,
    'ESPmin': -0.5,
    'LOMO': 0.0
}

# If a CAS number is selected and there is corresponding feature data
if use_cas_feature and selected_cas and selected_cas in cas_features_map:
    features_data = cas_features_map[selected_cas]
    
    # Update default values
    for feature, value in features_data.items():
        if feature in default_values and not pd.isna(value):
            default_values[feature] = value

st.sidebar.subheader("Set Molecular Feature Parameters")
# User can manually adjust feature sliders and input boxes
col1, col2 = st.sidebar.columns([3, 1])
with col1:
    polarizability = st.slider("Isotropic Average Polarizability", 
                              min_value=0.0, max_value=300.0, 
                              value=default_values['Isotropic Average Polarizability'], step=0.00001)
with col2:
    polarizability_input = st.number_input("Precise Value", value=float(polarizability), format="%.5f")
polarizability = polarizability_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    molecular_mass = st.slider("Molecular Mass", 
                              min_value=10.0, max_value=1000.0, 
                              value=default_values['Molecular Mass'], step=0.01)
with col2:
    molecular_mass_input = st.number_input("Precise Value", value=float(molecular_mass), format="%.2f")
molecular_mass = molecular_mass_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    boiling_point = st.slider("Boiling Point [°C]", 
                             min_value=-100.0, max_value=600.0, 
                             value=default_values['Boiling Point'], step=0.01)
with col2:
    boiling_point_input = st.number_input("Precise Value", value=float(boiling_point), format="%.2f")
boiling_point = boiling_point_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    hardness = st.slider("Hardness", 
                        min_value=0.0, max_value=10.0, 
                        value=default_values['Hardness'], step=0.0001)
with col2:
    hardness_input = st.number_input("Precise Value", value=float(hardness), format="%.4f")
hardness = hardness_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    dipole_moment = st.slider("Magnitude of Molecular Dipole Moment", 
                             min_value=0.0, max_value=50.0, 
                             value=default_values['Magnitude of Molecular Dipole Moment'], step=0.00001)
with col2:
    dipole_moment_input = st.number_input("Precise Value", value=float(dipole_moment), format="%.5f")
dipole_moment = dipole_moment_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    electrophilicity = st.slider("Cubic electrophilicity index", 
                                min_value=0.0, max_value=2.0, 
                                value=default_values['Cubic electrophilicity index'], step=0.00001)
with col2:
    electrophilicity_input = st.number_input("Precise Value", value=float(electrophilicity), format="%.5f")
electrophilicity = electrophilicity_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    melting_point = st.slider("Melting Point [°C]", 
                             min_value=-100.0, max_value=600.0, 
                             value=default_values['Melting Point'], step=0.01)
with col2:
    melting_point_input = st.number_input("Precise Value", value=float(melting_point), format="%.2f")
melting_point = melting_point_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    esp_max = st.slider("ESPmax", 
                       min_value=0.0, max_value=1000.0, 
                       value=default_values['ESPmax'], step=0.00001)
with col2:
    esp_max_input = st.number_input("Precise Value", value=float(esp_max), format="%.5f")
esp_max = esp_max_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    esp_min = st.slider("ESPmin", 
                       min_value=0.0, max_value=1000.0, 
                       value=default_values['ESPmin'], step=0.00001)
with col2:
    esp_min_input = st.number_input("Precise Value", value=float(esp_min), format="%.5f")
esp_min = esp_min_input

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    lomo = st.slider("LOMO", 
                    min_value=-10.0, max_value=10.0, 
                    value=default_values['LOMO'], step=0.00001)
with col2:
    lomo_input = st.number_input("Precise Value", value=float(lomo), format="%.5f")
lomo = lomo_input

# Create input data frame, organizing input features into DataFrame format
input_data = pd.DataFrame({
    'Isotropic Average Polarizability': [polarizability],
    'Molecular Mass': [molecular_mass],
    'Boiling Point': [boiling_point],
    'Hardness': [hardness],
    'Magnitude of Molecular Dipole Moment': [dipole_moment],
    'Cubic electrophilicity index': [electrophilicity],
    'Melting Point': [melting_point],
    'ESPmax': [esp_max],
    'ESPmin': [esp_min],
    'LOMO': [lomo]
})

# Use fixed scaling instead of preprocessing function for each fit
def preprocess_input(input_df):
    # These values should be obtained from training data, here using estimated values
    # Provide median and scale for each feature
    feature_stats = {
        'Isotropic Average Polarizability': {'median': 30.0, 'scale': 15.0},
        'Molecular Mass': {'median': 250.0, 'scale': 100.0},
        'Boiling Point': {'median': 200.0, 'scale': 80.0},
        'Hardness': {'median': 3.0, 'scale': 1.0},
        'Magnitude of Molecular Dipole Moment': {'median': 2.5, 'scale': 1.5},
        'Cubic electrophilicity index': {'median': 1.5, 'scale': 0.8},
        'Melting Point': {'median': 100.0, 'scale': 60.0},
        'ESPmax': {'median': 1.0, 'scale': 1.5},
        'ESPmin': {'median': -1.0, 'scale': 1.5},
        'LOMO': {'median': 0.0, 'scale': 2.0}
    }
    
    # Create numpy array to store results
    scaled_values = np.zeros(input_df.shape)
    
    # Manually apply RobustScaler transformation: (X - median) / scale
    for i, col in enumerate(input_df.columns):
        if col in feature_stats:
            scaled_values[0, i] = (input_df[col].values[0] - feature_stats[col]['median']) / feature_stats[col]['scale']
        else:
            scaled_values[0, i] = input_df[col].values[0]  # If no statistical information, don't transform
    
    return scaled_values

# Define explanations for each indicator
explanations = {
    'LogP': {
        'title': 'Octanol-Water Partition Coefficient',
        'description': 'LogP is the logarithm of the octanol-water partition coefficient, indicating the distribution of a compound between octanol and water.',
        'high': 'Higher LogP values indicate more lipophilic compounds (stronger hydrophobicity)',
        'low': 'Lower LogP values indicate more hydrophilic compounds (better water solubility)'
    },
    'LogBCF': {
        'title': 'Bioconcentration Factor',
        'description': 'LogBCF is the logarithm of the bioconcentration factor, indicating the degree of chemical accumulation in organisms.',
        'high': 'Higher LogBCF values indicate substances that are more likely to accumulate in organisms',
        'low': 'Lower LogBCF values indicate substances that accumulate less in organisms'
    },
    'LogKOC': {
        'title': 'Organic Carbon-Water Partition Coefficient',
        'description': 'LogKOC is the logarithm of the organic carbon-water partition coefficient, indicating the distribution of chemicals between soil organic carbon and water.',
        'high': 'Higher LogKOC values indicate substances that are more likely to adsorb to soil organic matter',
        'low': 'Lower LogKOC values indicate substances that are more likely to dissolve in water'
    },
    'Biowin7': {
        'title': 'Biodegradation Probability',
        'description': 'Biowin7 evaluates the probability of biodegradation of chemicals under anaerobic conditions.',
        'high': 'Higher Biowin7 values indicate substances that are more easily biodegraded under anaerobic conditions',
        'low': 'Lower Biowin7 values indicate substances that are difficult to biodegrade under anaerobic conditions'
    }
}

# Predict button
if st.button("Predict All Indicators"):
    # Preprocess input data
    input_scaled = preprocess_input(input_data)
    
    # Display prediction results for each indicator
    st.write("## Prediction Results")
    
    # Display current compound information being predicted
    if use_cas_feature and selected_cas:
        # Determine if English name should be displayed
        display_name = selected_cas
        if has_english_name and selected_cas in cas_english_name_map:
            display_name = f"{cas_english_name_map[selected_cas]} (CAS: {selected_cas})"
        st.write(f"**Current Compound**: {display_name}")
    
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    # Results dictionary, for later export
    results = {}
    
    # Display prediction results in column
    for i, target in enumerate(targets):
        if models_status[target]:
            # Use model for prediction
            prediction = models[target].predict(input_scaled)[0]
            results[target] = prediction
            
            # Determine which column to display based on index
            current_col = col1 if i % 2 == 0 else col2
            
            # Create prediction result display card
            with current_col:
                st.write(f"### {explanations[target]['title']} ({target})")
                st.write(f"**Predicted Value**: {prediction:.4f}")
                
                # Add SHAP analysis
                try:
                    # Create SHAP explainer with simple background data
                    background_data = np.zeros((1, input_scaled.shape[1]))
                    explainer = shap.KernelExplainer(models[target].predict, background_data)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(input_scaled)
                    
                    # Create force plot
                    plt.figure(figsize=(10, 3))
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[0,:],
                        input_data,
                        feature_names=input_data.columns,
                        matplotlib=True,
                        show=False
                    )
                    
                    # Display the plot
                    st.pyplot(plt)
                    plt.clf()
                    
                    # Add SHAP value explanation
                    st.write("#### Feature Importance Analysis")
                    feature_importance = pd.DataFrame({
                        'Feature': input_data.columns,
                        'SHAP Value': np.abs(shap_values[0])
                    })
                    feature_importance = feature_importance.sort_values('SHAP Value', ascending=False)
                    st.write("Top features affecting the prediction:")
                    st.table(feature_importance)
                    
                except Exception as e:
                    st.warning(f"Could not generate SHAP analysis for {target}: {str(e)}")
                
                with st.expander("View Explanation"):
                    st.write(explanations[target]['description'])
                    st.write(f"- {explanations[target]['high']}")
                    st.write(f"- {explanations[target]['low']}")
    
    # Add CAS number and English name to results
    if use_cas_feature and selected_cas:
        results['CAS'] = selected_cas
        if has_english_name and selected_cas in cas_english_name_map:
            results['English Name'] = cas_english_name_map[selected_cas]
    
    # Create results DataFrame for download
    results_df = pd.DataFrame([results])
    
    # Generate CSV download link
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv,
        file_name="molecular_properties_prediction.csv",
        mime="text/csv",
    )