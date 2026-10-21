# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:41:30 2025

@author: Irineu

Predicting FZI using RandomForestRegressor

Libs:
    pip install pandas numpy shap joblib yellowbrick
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import shap
import os
import joblib #To save and load a trained model
from yellowbrick.features import Rank2D

#-------------------------------------------------------
def Manual_mean_abs_error(y_true, y_pred):

    # Calculate absolute errors
    absolute_errors = np.abs(y_true - y_pred)
    
    # Calculate the sum of absolute errors
    sum_absolute_errors = np.sum(absolute_errors)
    
    # Calculate MAE
    mae_manual = sum_absolute_errors / len(y_true)
    
    return mae_manual
#-------------------------------------------------------    

# Change the current working directory to a specific path
path = 'F:\\ROCKPHYS\\RockPhys_IA_tests\\W861'

try:
    os.chdir(path)
    print(f"Current working directory after change: {os.getcwd()}")
except FileNotFoundError:
    print("Error: The specified directory was not found.")
except NotADirectoryError:
    print("Error: The specified path is not a directory.")
except PermissionError:
    print("Error: You do not have the necessary permissions to change to this directory.")
            

# Set global font style to Arial, bold, and specify font size
plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'bold',
    'font.family': 'Arial'
})


# 1. Read the training data file
file_path = 'Auddys_table.xlsx'
#Lendo xlsx com todas as abas padrão a partir da linha de interesse header, retorna com controle key o nome da aba (amostra)
dft = pd.read_excel(file_path) #dicionário de dataframes  

#Target Y
y_fzi_lab = np.round(dft.iloc[:,11].values,1) #Obtém a coluna FZI_lab e arredonda para 1 casa decimal
#Feature X
df = dft.drop(columns=['Depth(m)', 'FZI_lab'])
x_log_data = df.iloc[:].values

# Normalize all data?
flag_norm = 1 #0-without /1-with MinMaxScaler normalization

# 2. Divide data set into training and test sets
features = x_log_data
targets = y_fzi_lab

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=42)

visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(df)
visualizer.show()

# 3. Build Random Forest Regressor model
model = RandomForestRegressor(n_estimators=200, random_state=42)
#rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
if flag_norm:
    scaler = MinMaxScaler()
# columns_to_normalize = df.columns[:]  # Select all columns
# df[columns_to_normalize] = scaler.fit_transform(df)



# 4. Train the model
model.fit(train_features, train_targets)
print('RF Trainning score R² = ' + str(np.round(model.score(train_features, train_targets),3)))
print('RF Test score R² = ' + str(np.round(model.score(test_features, test_targets),3)))
#rf_model.fit(x_log_data, y_fzi_lab)

# 5. Saving the trainned model
filename = 'trained_FZI_RandonForest_model_w861.sav'
joblib.dump(model, filename)
print('Trained model has been saved to: ' + filename)

# 6. Use model to predict test set data
predicted_values = model.predict(test_features)
#predicted_values_2d = predicted_values.reshape(-1, 1)
#Computing R² between test_targets and predicted_values
mae = mean_absolute_error(test_targets, predicted_values)
print(f"Mean Absolute Error between test_features and predicted_values: {mae}")
mmae = Manual_mean_abs_error(test_targets, predicted_values)
print(f"Mean Absolute Error (Manual) between test_features and predicted_values: {mmae}")

mse = mean_squared_error(test_targets, predicted_values)
print(f"Mean Squared Error between test_features and predicted_values: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (R²) between test_features and predicted_values: {rmse}")


plt.figure()
plt.xlabel(r'FZI_predicted', fontsize=11)
plt.ylabel(r'FZI_target', fontsize=11)
plt.title('FZI Randon Forest W#861')
plt.plot(predicted_values, test_targets, 'o', np.arange(-1,11,11), np.arange(-1,11,11), 'k--', linewidth=2)
plt.axis([0, 8, 0, 8])
plt.text(0.5,7,'R² = '+str(np.round(rmse,3)))
plt.grid()
# Ensure compact layout
#plt.tight_layout()

# Save image
plt.savefig('FZI Randon Forest W#861 test.png', format='png', dpi=600, bbox_inches='tight')
plt.show()

#-----------------------------------------------------------------------------------------
# Part 2:
# Continuing with SHAP analysis for predicting porosity using the random forest algorithm
#-----------------------------------------------------------------------------------------
# Create SHAP Explainer object
feature_names = df.columns # Getting column headers from Feature X
explainer = shap.KernelExplainer(model.predict, features)
# Calculate SHAP values
print(f"Iterating SHAP explainer values for array shape: {features.shape}")
shap_values = explainer.shap_values(features)

# Draw and save vector graphics
plt.figure()
plt.xlabel(r'SHAP values', fontsize=11)
shap.summary_plot(shap_values, features, show=False, plot_type="bar", feature_names=feature_names, axis_color='black')

# Set title and X-axis range
plt.title('Shap Bar for all input features to predict FZI_lab')
plt.xlim(0, 1.0)

# Set X-axis tick intervals
ax = plt.gca()
#ax.xaxis.set_major_locator(MultipleLocator(0.01))

# Ensure compact layout
plt.tight_layout()

# Save image
#plt.savefig(os.path.join(output_folder, 'Shap bar_plot-RF_F03-4ϕ_rectify.svg'))  # Save vector graphics
plt.show()
#print("Vector graphics have been saved in the output folder.")

explainer2 = shap.Explainer(model, feature_names=feature_names)
shap_values2 = explainer2(features)

# visualize the first prediction's explanation
plt.figure()
shap.plots.waterfall(shap_values2[0], max_display=feature_names.size, show=False)
plt.title('Shap Waterfall for a unique (first) input, features to predict FZI_lab')
plt.show()

explainer3 = shap.Explainer(model, train_features, feature_names=feature_names)
shap_values3 = explainer3(features)
plt.figure()
shap.plots.beeswarm(shap_values3, max_display=feature_names.size, show=False)
plt.title('Shap Beeswarm for all input features to predict FZI_lab')
# Save image
#plt.savefig(os.path.join(output_folder, 'Shap bar_plot-RF_F03-4ϕ_rectify.svg'))  # Save vector graphics
plt.show()
#---------------------------------------------------------------------------------