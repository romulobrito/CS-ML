# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:50:57 2025

@author: Irineu

Este script trabalha 5 regressores mais empregados de Machine Learning em um conjunto de dados fictício.
A idiea é testar e entender os regressores para aplicação em trabalhos de interesse.

"""
# Referencias - prinicipais aplicadas neste estudo, separadas por:
# XGBoost early stopping (Python API): https://xgboost.readthedocs.io/en/latest/python/python_api.html
# MLPRegressor (early_stopping, validation_fraction, n_iter_no_change): https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
# RandomizedSearchCV (API): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# Hyper-parameter tuning guide (RandomizedSearchCV): https://scikit-learn.org/stable/modules/grid_search.html
# Randomized search vs grid search (example): https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
# Common pitfalls / data leakage / Pipeline: https://scikit-learn.org/stable/common_pitfalls.html


# Bibliotecas básicas:
import pandas as pd
import numpy as np

#----------------
# --- Funções gerais, utilitárias (Utils), empregáveis pelos diversos regressores:

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product

# Function to evaluate model performance: useful for ML models: LR (Linear Regression), ML
def evaluate_model(model, X, y, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Store scores for each fold
    rmse_scores = [] #RMSE
    r2_scores = [] #R-squared (R²)
    #scores = [] #Accuracy: R-squared (R²) - test
    # Pandas DataFrame to a NumPy array if necessary - to prevent index failure
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy() # IMPORTANT: assign back, otherwise y may remain a pandas object

    # Guarantee NumPy arrays and a 1D target vector (prevents shape issues in some regressors)
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    print("---------------------------------------------------------")
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        #Evaluate model prediction scoring...
        #test 1: model score - R-squared (R²)
        #score = model.score(X_test, y_test)
        #scores.append(score)
        #test 2: predicting and calculating...
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        #print(f"Fold {fold + 1} - RMSE: {rmse:.8f}, R²: {r2:.8f}, Acc-R²: {score:.8f}")
        print(f"Fold {fold + 1} - RMSE: {rmse:.8f}, R²: {r2:.8f}")

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"\nMean RMSE from total Folds={n_folds}: {mean_rmse:.8f} (±{std_rmse:.8f})")
    print(f"Mean R² from total Folds={n_folds}: {mean_r2:.8f} (±{std_r2:.8f})")
    print("---------------------------------------------------------")
    return mean_rmse, mean_r2

# Main function for training and hyperparameter optimization
def train_optimize_model_RF(X, y, n_folds=5, n_iter=200):
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Initialize Randomized Search Cross-Validation
    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist_RF,
        n_iter=n_iter,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        #error_score=0, # Assign 0 if a fit fails # Necessário verificar, 0 pode ser o maior valor
        error_score='raise', # If a fit fails, raise immediately (avoid silently accepting bad candidates)
        # error_score can be numeric, but using 0 with neg-MSE can look artificially "good"
        n_jobs=-1,  # Use all available threads (parallelism)
        verbose=0  # Display the search process
    )

    # Execute randomized search
    random_search.fit(X, y)

    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score (RMSE):", np.sqrt(-random_search.best_score_))

    return random_search.best_estimator_, random_search.best_params_

# Main function for training and hyperparameter optimization
def train_optimize_model_GB(X, y, n_folds=5, n_iter=200):
    # Initialize K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize Gradient Boosting model
    gb = GradientBoostingRegressor(random_state=42)

    # Initialize Randomized Search Cross-Validation
    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist_GB,
        n_iter=n_iter,
        cv=kf,
        scoring='neg_mean_squared_error',
        random_state=42,
        #error_score=0, # Assign 0 if a fit fails # Necessário verificar, 0 pode ser o maior valor
        error_score='raise',  # Fail fast if a candidate breaks (easier debugging)
        n_jobs=-1,  # Use all available threads (parallelism)
        verbose=0  # Display the search process
    )

    # Execute randomized search
    random_search.fit(X, y)

    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score (RMSE):", np.sqrt(-random_search.best_score_))

    return random_search.best_estimator_, random_search.best_params_
# --- FIM - Funções gerais, utilitárias (Utils), empregáveis pelos diversos regressores
#--------------------------------

# BLOCO DE TESTES - ESCOLHA DE REGRESSORS & SCALERS A SIMULAR...

# Types of Scalers ML techniquees for test:
# 0- Simulate all,
# 1- Randon Forest Regression,
# 2- Gradient Boosting Regression,
# 3- XGBoost Regression,
# 4- Multi-layer Perceptron Regression,
# 5- Ordinary least squares Linear Regression.
flag_MLsim = 0

# ATENÇÃO: A teoria diz que decision tree algorithms não necessitam de scaler. Apenas para testar efeito...
# Types of Scalers available for flag_scaler:
# 0- no scaler (Generally, neither scaler is required for tree-based models (RForest, GradientBoosted, etc.), except combining with other models or specific feature engineering techniques),
# 1- MinMaxScaler (Normalization, this scales features to a specific range, typically between 0 and 1),
# 2- StandardScaler (Z-score normalization, this transforms features to have a mean of 0 and a standard deviation of 1)(indicated for MLPR), 
# 3- RobustScaler (This scales features using the interquartile range, making it robust to outliers)
flag_scaler = 1


# Data set de teste hipotético
# =>Create or load a sample dataset (replace with your actual data)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

#Expected format:
# X = data[['feature1', 'feature2']]
# y = data['target']
# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#--------------------------------

# INICIANDO O PROCESSAMENTO DESEJADO DAQUI EM DIANTE:
    
# 1- Preparar scaler de dados    
#-- Preparing scaler if desirable --
if flag_scaler == 1: #MinMax  
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
elif flag_scaler == 2: #Standard
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
elif flag_scaler == 3: #Robust
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
else: #No scaler
    X_train_scaled = X_train
    X_test_scaled = X_test
#-----------------------------

# 2- Aplicar regressores 
# Evaluating ML Regressors:
#-- 1-Randon Forest Regression --
if flag_MLsim == 1 or flag_MLsim == 0: 
    
    from sklearn.ensemble import RandomForestRegressor
    print("---------------------------------------------------------")
    print("=> Starting Randon Forest Regression:")
    print("---------------------------------------------------------")
    
    ## Test 1: simple regressor parameters:
    # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf_model.fit(X_train_scaled, y_train)
    # y_pred = rf_model.predict(X_test_scaled)
    # print("Randon Forest RegressorRandon Forest Regressor:")
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
    # rmse = np.sqrt(mse)
    # print(f"Root Mean Squared Error (R²): {rmse}")
    
    ## Test 2: advanced test of tuning regressor parameters:    
    # Parameter space for Randomized Search
    param_dist_RF = {
        "n_estimators": np.arange(1000, 1601, 200),  # 1000 to 1600 inclusive, step of 200
        "max_depth": list(np.arange(12, 17, 2)) + [None],
        "min_samples_split": np.arange(3, 7, 1),
        "min_samples_leaf": np.arange(2, 4, 1),
        "max_features": [0.4, 0.5, 0.6, "sqrt"], 
        "bootstrap": [True],
        "max_samples": np.arange(0.90, 1.01, 0.05),
         
        "criterion": ["squared_error"],
        "max_leaf_nodes": [None, 1200, 2000, 3000],
    }

    print("Starting training and hyperparameter optimization for the Randon Forest Regression...")
    # Train and optimize the model
    #rf_model, best_params = train_optimize_model_RF(X, y, n_folds=5, n_iter=200)
    rf_model, best_params = train_optimize_model_RF(X_train_scaled, y_train, n_folds=5, n_iter=200)

    # Evaluate model performance
    print("\nModel performance Randon Forest Regression evaluation (based on the best parameters):")
    #avg_rmse, avg_r2 = evaluate_model(rf_model, X, y, n_folds=5)
    avg_rmse, avg_r2 = evaluate_model(rf_model, X_train_scaled, y_train, n_folds=5)

    # Output the best parameter combination and its performance
    print("\nFinal best parameter combination:")
    print(best_params)
    print("\nCross-validation performance with the best parameters:")
    print(f"Mean RMSE: {avg_rmse:.8f}")
    print(f"Mean R²: {avg_r2:.8f}")
    
    #Predicting
    print("Using the trained Randon Forest Regression Model for test prediction:")
    y_pred = rf_model.predict(X_test_scaled)
    print("Randon Forest Regression results:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (R²): {rmse}")
    
#-----------------------------

#-- 2-Gradient Boosting Regression -- #Falta criar a função do regressor (inspirar em train_optimize_model_RF)
if flag_MLsim == 2 or flag_MLsim == 0: 
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("---------------------------------------------------------")
    print("=> Starting Gradient Boosting Regression:")
    print("---------------------------------------------------------")
    
    
    # # Simple test 1
    # # Define the Gradient Boosting Regressor
    # from sklearn.ensemble import GradientBoostingRegressor
    # gb_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    # # Trainning GB model
    # gb_model.fit(X_train_scaled, y_train)
    # # Using GB model for prediction
    # y_pred = gb_model.predict(X_test_scaled)
    # print("Gradient Boosting Regressor:")
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")
    # rmse = np.sqrt(mse)
    # print(f"Root Mean Squared Error (R²): {rmse}")
    
    # Advanced test 2
    param_dist_GB = {
        'n_estimators': np.arange(3800, 5601, 200),  # 3800..5600
        'learning_rate': [0.05, 0.06, 0.07, 0.08],

        'max_depth': [1, 2, 3],

        'min_samples_split': np.arange(30, 71, 2),   # 30..70
        'min_samples_leaf': [1, 2, 3],

        'subsample': [0.25, 0.3, 0.35, 0.4, 0.45],

        'max_features': [None, 0.7, 1.0]
    }

    # Train and optimize the model
    print("Starting training and hyperparameter optimization (Gradient Boosting Regression)...")
    #best_model, best_params = train_optimize_model_GB(X, y, n_folds=5, n_iter=200)
    best_model, best_params = train_optimize_model_GB(X_train_scaled, y_train, n_folds=5, n_iter=200)

    # Evaluate the performance of the best model
    print("\nModel performance evaluation (based on best parameters):")
    #avg_rmse, avg_r2 = evaluate_model(best_model, X, y, n_folds=5)
    avg_rmse, avg_r2 = evaluate_model(best_model, X_train_scaled, y_train, n_folds=5)
    
    # Output the best parameter combination and corresponding performance
    print("\nFinal best parameter combination:")
    print(best_params)
    print("\nCross-validation performance based on best parameters:")
    print(f"Mean RMSE: {avg_rmse:.8f}")
    print(f"Mean R²: {avg_r2:.8f}")

    #Predicting
    print("Using the trained Gradient Boosting Regression Model for test prediction:")
    y_pred = best_model.predict(X_test_scaled)
    print("Gradient Boosting Regression results:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (R²): {rmse}")
    
#-----------------------------

#-- 3-XGBoost Regression --
#Early stopping on XGBoost - https://xgboost.readthedocs.io/en/latest/python/python_api.html

if flag_MLsim == 3 or flag_MLsim == 0: 
    import xgboost as xgb

    print("---------------------------------------------------------")
    print("=> Starting XGBoost Regression:")
    print("---------------------------------------------------------")

    # Parameter space for Randomized Search (XGBoost) - BROAD SEARCH SPACE
    # (intended to generalize across datasets, not just this one)
    # Note: n_estimators removed -> it will be selected by early stopping
    param_dist_XGB = {
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [1, 2, 3],
        'min_child_weight': [2, 3, 4, 5],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'gamma': [0, 0.01, 0.03, 0.05],
        'reg_alpha': [0, 0.01, 0.03, 0.05],
        'reg_lambda': [3, 5, 7, 10]
    }

    # Hyperparameter optimization function
    def train_optimize_model_XGB(X, y, n_folds=5, n_iter=200):
        # Initialize K-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Initialize XGBoost model
        # Note: we set a fixed n_estimators just to run CV; final n_estimators will be refined later
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            n_estimators=2000
        )

        # Initialize Randomized Search Cross-Validation
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_reg,
            param_distributions=param_dist_XGB,
            n_iter=n_iter,
            cv=kf,
            scoring='neg_mean_squared_error',
            random_state=42,
            error_score='raise',
            n_jobs=-1,
            verbose=0
        )

        # Execute randomized search
        random_search.fit(X, y)

        print("Best parameters:", random_search.best_params_)
        print("Best cross-validation score (RMSE):", np.sqrt(-random_search.best_score_))

        return random_search.best_estimator_, random_search.best_params_


    # Advanced test:
    print("Starting training and hyperparameter optimization for the XGBoost Regression...")
    xgb_model_cv, best_params = train_optimize_model_XGB(X_train_scaled, y_train, n_folds=5, n_iter=200)

    # Evaluate the performance of the best model (CV estimate)
    print("\nModel performance evaluation (based on the best parameters):")
    avg_rmse, avg_r2 = evaluate_model(xgb_model_cv, X_train_scaled, y_train, n_folds=5)

    # Output the best parameter combination and corresponding performance
    print("\nFinal best parameter combination:")
    print(best_params)
    print("\nCross-validation performance based on best parameters:")
    print(f"Mean RMSE: {avg_rmse:.8f}")
    print(f"Mean R²: {avg_r2:.8f}")

    # ---------------------------------------------------------
    # FINAL REFINEMENT: Early stopping para escolher n_estimators
    # ---------------------------------------------------------
    print("\nRefining n_estimators with early stopping (XGBoost)...")

    # Split only inside the training set (keeps X_test untouched -> avoids leakage)
    from sklearn.model_selection import train_test_split
    X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Use best hyperparameters from CV, but allow a large upper bound for n_estimators
    # (early stopping will decide the optimal number of trees)
    es_params = dict(best_params)
    es_params['n_estimators'] = 5000  # large upper bound

    # Model used only to find best_iteration via early stopping
    xgb_es = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=50, # stop if no improvement for N rounds
        **es_params
    )

    xgb_es.fit(
        X_tr_es, y_tr_es,
        eval_set=[(X_val_es, y_val_es)], # validation set required for early stopping
        verbose=False
    )

    # best_iteration is 0-based in XGBoost -> add 1 to get "number of trees"
    best_iter = getattr(xgb_es, "best_iteration", None)
    if best_iter is None:
        best_n_estimators = int(es_params['n_estimators'])
    else:
        best_n_estimators = int(best_iter) + 1

    print("Best n_estimators from early stopping:", best_n_estimators)

    # Train final model on FULL training set with best_n_estimators
    final_params = dict(best_params)
    final_params['n_estimators'] = best_n_estimators

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        **final_params
    )
    xgb_model.fit(X_train_scaled, y_train)

    # Test prediction
    print("Using the trained XGBoost Regression Model for test prediction:")
    y_pred = xgb_model.predict(X_test_scaled)
    print("XGBoost Regressor:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (R²): {rmse}")
#-----------------------------


#--- 4-Multi-layer Perceptron Regression ---
if flag_MLsim == 4 or flag_MLsim == 0: 
    from sklearn.neural_network import MLPRegressor
    # The RandomizedSearchCV is faster than GridSearchCV (similar) - 
    # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    #    https://medium.com/analytics-vidhya/hyper-parameter-tuning-gridsearchcv-vs-randomizedsearchcv-499862e3ca5
    #    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    
    # Define the parameter space for random search with some typical values/settings
    param_dist_MLP = {
        'hidden_layer_sizes': [(200,), (250,), (300,), (250, 125)],
        'activation': ['relu'],
        'solver': ['adam'],

        'alpha': [0.0028, 0.0030, 0.0032, 0.0034, 0.0036],

        'learning_rate_init': [0.05, 0.055, 0.06, 0.065],

        'batch_size': [64, 96, 128],

        'max_iter': [2500],

        'validation_fraction': [0.1, 0.15],
        'n_iter_no_change': [60, 80, 100],
        'tol': [2e-5, 1e-5, 5e-6],

        'learning_rate': ['constant', 'adaptive'],
    }

    # Hyperparameter optimization function
    def train_optimize_model_MLP(X, y, n_folds=5, n_iter=200):
        # Initialize K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Define the MLP neural network
        mlp = MLPRegressor(random_state=42, early_stopping=True)
        
        # Testing RandomizedSearchCV: it is better than GridSearchCV 
        # Set up random search
        from sklearn.model_selection import RandomizedSearchCV
        mlp_search = RandomizedSearchCV(
            estimator=mlp,
            param_distributions=param_dist_MLP,
            n_iter=n_iter,
            scoring='neg_mean_squared_error', #See about available regression scoring in: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            cv=kf, #Cross-validation: evaluating estimator performance, see in: https://scikit-learn.org/stable/modules/cross_validation.html
            random_state=42,
            n_jobs=-1,  # Use all available threads (parallelism)
            verbose=0  # Display the search process
        )
        
        ## Testing GridSearchCV: it is highly expensive than RandomizedSearchCV
        # # Set up grid search
        # from sklearn.model_selection import GridSearchCV
        # mlp_search = GridSearchCV(
        #     estimator=mlp,
        #     param_grid=param_dist,
        #     scoring='neg_mean_squared_error', #See about available regression scoring in: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        #     cv=kf, #Cross-validation: evaluating estimator performance, see in: https://scikit-learn.org/stable/modules/cross_validation.html
        #     #random_state=42,
        #     n_jobs=-1,  # Use all available threads (parallelism)
        #     verbose=2  # Display the search process
        # )
        
        # Perform random search training
        mlp_search.fit(X, y)

        # Print the best parameter combination
        print("\nBest parameter combination:", mlp_search.best_params_)
        print("Best cross-validation score (RMSE):", np.sqrt(-mlp_search.best_score_))

        return mlp_search.best_estimator_, mlp_search.best_params_


    print("---------------------------------------------------------")
    print("=> Starting Multi-layer Perceptron Regression:")
    print("---------------------------------------------------------")

    # #Simple test
    # mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), early_stopping=True, random_state=42, max_iter=500)
    # mlp_model.fit(X_train_scaled, y_train)
    
    #Advanced test: 
    print("Starting training and hyperparameter optimization for the MLP neural network...")
    # Train and optimize the model
    #mlp_model, best_params = train_optimize_model_MLP(X_test_scaled, y_test, n_folds=5, n_iter=200)    
    mlp_model, best_params = train_optimize_model_MLP(X_train_scaled, y_train, n_folds=5, n_iter=200)

    # Evaluate the performance of the trained and optimized model
    print("\nModel Performance Evaluation (Based on the Best Parameters):")
    avg_rmse, avg_r2 = evaluate_model(mlp_model, X_train_scaled, y_train, n_folds=5)

    # Print the best parameter combination and model performance
    print("\nFinal Best Parameter Combination:")
    print(best_params)
    print("\nK-Fold Cross-Validation Performance with the Best Parameters:")
    print(f"Mean RMSE: {avg_rmse:.8f}")
    print(f"Mean R²: {avg_r2:.8f}")
    
    print("Using the trained Multi-layer Perceptron Model for test prediction:")
    y_pred = mlp_model.predict(X_test_scaled)
    print("Multi-layer Perceptron Regressor results:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (R²): {rmse}")

#-----------------------------

#--- 5-Ordinary least squares Linear Regression ---
if flag_MLsim == 5 or flag_MLsim == 0: 
    from sklearn.linear_model import LinearRegression
           
    # Define parameter grid
    param_grid_LR = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [-1, None],
        'positive': [True, False]
    }
    
    def train_evaluate_LR_models(X, y, n_folds=5):
        best_rmse = float('inf')
        best_r2 = float('-inf')
        best_params = None
        param_combinations = list(product(*param_grid_LR.values()))
        for params in param_combinations:
            params_dict = dict(zip(param_grid_LR.keys(), params))
    
            print(f"\nEvaluating parameter combination: {params_dict}")
    
            model = LinearRegression(**params_dict)
    
            avg_rmse, avg_r2 = evaluate_model(model, X, y, n_folds)
    
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_r2 = avg_r2
                best_params = params_dict
        print("---------------------------------------------------------")        
        print("\nFound the best parameter combination:")
        print(best_params)
        print("\nCross-validation performance with the best parameters:")
        print(f"Mean RMSE: {best_rmse:.8f}")
        print(f"Mean R²: {best_r2:.8f}")
        print("---------------------------------------------------------")
    
        # return the best parameter combination for training
        best_model = LinearRegression(**best_params)
        return best_model
    
       
    # Start training and evaluation
    print("---------------------------------------------------------")
    print("=> Starting Ordinary least squares Linear Regression:")
    print("---------------------------------------------------------")
    
    #linear_model = LinearRegression() #Ordinary least squares Linear Regression (versão sem avaliação de parametrização para treinamento)
    print("Starting Linear Regression Model training and hyperparameter evaluation...")
    #linear_model = train_evaluate_LR_models(X, y, n_folds=10) #Evaluating a best model (versão com avaliação de parametrização para treinamento)
    linear_model = train_evaluate_LR_models(X_train_scaled, y_train, n_folds=10)
    linear_model.fit(X_train_scaled, y_train)
    print("Using the trained Linear Regression Model for test prediction:")
    y_pred = linear_model.predict(X_test_scaled)
    print("Linear Regressor results:")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (R²): {rmse}")
#-----------------------------

