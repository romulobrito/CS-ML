# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:15:09 2025

@author: Irineu

Este script foi criado para auxiliar a filtragem de propriedades de logs para as unidades de fluxo hidráulico (HFU) para o poço 861, segundo estabelecido na dissertação da Auddys.
Este serviu para incluir a coluna HFU na própria planilha de entrada, 'Auddys_table.xlsx'.
"""

import pandas as pd
import matplotlib.pyplot as plt

import os


#---------------
def well_log_plot(df):
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 10), sharey=True)

    # Plot Gamma Ray
    axes[0].plot(df['RQI'], df['Depth(m)'], color='green')
    axes[0].set_title('RQI')
    axes[0].set_xlabel('RQI')
    axes[0].grid(True)
    axes[0].set_ylim(df['Depth(m)'].max(), df['Depth(m)'].min())
    # Plot Density
    axes[1].plot(df['FZI_lab'], df['Depth(m)'], color='red')
    axes[1].set_title('FZI_lab')
    axes[1].set_xlabel('FZI_lab')
    axes[1].grid(True)
    axes[1].set_ylim(df['Depth(m)'].max(), df['Depth(m)'].min())
          
    plt.tight_layout()
    plt.show()
#---------------
def create_plot(wellname, dataframe, curves_to_plot, depth_curve, log_curves=[]):
    
    # Count the number of tracks we need
    num_tracks = len(curves_to_plot)
    
    # Setup the figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=num_tracks, figsize=(num_tracks*2, 10))
    
    # Create a super title for the entire plot
    fig.suptitle(wellname, fontsize=20, y=1.05)
    
    # Loop through each curve in curves_to_plot and create a track with that data
    for i, curve in enumerate(curves_to_plot):
        
        
        ax[i].plot(dataframe[curve], depth_curve)
        
        # Setup a few plot cosmetics
        ax[i].set_title(curve, fontsize=14, fontweight='bold')
        ax[i].grid(which='major', color='lightgrey', linestyle='-')
        
        # We want to pass in the deepest depth first, so we are displaying the data 
        # from shallow to deep
        ax[i].set_ylim(depth_curve.max(), depth_curve.min())

        # Only set the y-label for the first track. Hide it for the rest
        if i == 0:
            ax[i].set_ylabel('DEPTH (m)', fontsize=18, fontweight='bold')
        else:
            plt.setp(ax[i].get_yticklabels(), visible = False)
        
        # Check to see if we have any logarithmic scaled curves
        if curve in log_curves:
            ax[i].set_xscale('log')
            ax[i].grid(which='minor', color='lightgrey', linestyle='-')
    
    plt.tight_layout()
    plt.show()
#---------------
    
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
    

file_path = 'Auddys_table.xlsx'
#Lendo xlsx com todas as abas padrão a partir da linha de interesse header, retorna com controle key o nome da aba (amostra)
df = pd.read_excel(file_path) #dicionário de dataframes   

#--- computing HFU --- 
#Filtering depth for HFU1 (poor)
phi_min= 0.05
phi_max= 0.18
k_min= 0.04
k_max= 8
fzi_min= 0
fzi_max= 1

hfu1 = df[(df['Phi_lab (pu)'] >= phi_min) & (df['Phi_lab (pu)'] <= phi_max) &
          (df['k_lab (mD)']>= k_min) & (df['k_lab (mD)'] <= k_max) &
          (df['FZI_lab']>= fzi_min) & (df['FZI_lab'] <= fzi_max)]
        
#Filtering depth for HFU2 (medium)
phi_min= 0.06
phi_max= 0.22 #correto é este
k_min= 0.65
k_max= 39
fzi_min= 1
fzi_max= 2

hfu2 = df[(df['Phi_lab (pu)'].round(2) >= phi_min) & (df['Phi_lab (pu)'].round(2) <= phi_max) &
          (df['k_lab (mD)'].round(2)>= k_min) & (df['k_lab (mD)'].round(2) <= k_max) &
          (df['FZI_lab'].round(1)>= fzi_min) & (df['FZI_lab'].round(1) <= fzi_max)]

#Filtering depth for HFU3 (good)
phi_min= 0.07
phi_max= 0.21
k_min= 2.5
k_max= 213
fzi_min= 2
fzi_max= 4

hfu3 = df[(df['Phi_lab (pu)'] >= phi_min) & (df['Phi_lab (pu)'] <= phi_max) &
          (df['k_lab (mD)']>= k_min) & (df['k_lab (mD)'] <= k_max) &
          (df['FZI_lab']>= fzi_min) & (df['FZI_lab'] <= fzi_max)]

#Filtering depth for HFU4 (excellent)
phi_min= 0.08
phi_max= 0.22
k_min= 17
k_max= 500
fzi_min= 4
fzi_max= 10

hfu4 = df[(df['Phi_lab (pu)'] >= phi_min) & (df['Phi_lab (pu)'] <= phi_max) &
          (df['k_lab (mD)']>= k_min) & (df['k_lab (mD)'] <= k_max) &
          (df['FZI_lab']>= fzi_min) & (df['FZI_lab'] <= fzi_max)]


df_hfus = []
df_hfus = hfu1
df_hfus.merge(hfu2)
df_hfus.merge(hfu3)
df_hfus.merge(hfu4)

#--- test 1 ---    
well_log_plot(df)  
#--- test 2 ---
curves_to_plot = ['GR (API)', 'Res_Deep', 'Res_Shallow', 'Phi_Neutron (pu)', 'Phi_Sonic (pu)', 
                  'Phi_ND (pu)', 'Density (g/cc)', 'Phi_lab (pu)','k_lab (mD)', 'RQI', 'FZI_lab']

logarithmic_curves = ['k_lab (mD)']  
create_plot('Well-861', df, curves_to_plot, df['Depth(m)'], logarithmic_curves)    

#--- test 3 ---
curves_to_plot = ['Density (g/cc)', 'Phi_lab (pu)','k_lab (mD)', 'RQI', 'FZI_lab']

logarithmic_curves = ['k_lab (mD)']  
create_plot('Well-861', df, curves_to_plot, df['Depth(m)'], logarithmic_curves)        




