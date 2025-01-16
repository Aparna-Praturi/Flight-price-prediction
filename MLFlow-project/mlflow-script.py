import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from sklearn.preprocessing import StandardScaler
import ast
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import logging
import joblib

# adding relative paths to data and preprocessor folders
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_files'))
from flight_price_prediction import DateExtractor, num_scaler, CycEncoder, CatEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

#Function to load data and processor
def load_data(data_path, processor_path):
    # Load the data
    try:
        df_1 = pd.read_csv(data_path)
        processor= joblib.load(processor_path)
        print("Data loaded successfully!")

    except FileNotFoundError:
        print(f"Error: The file(s) at {data_path} were not found. Please check the path.")

    except pd.errors.EmptyDataError:
        print(f"Error: The file(s) at {data_path} is empty. Please check the file content.")

    except pd.errors.ParserError:
        print(f"Error: There was a problem parsing the file(s) at {data_path}. Please check the file(s) format.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return df_1, processor


# function to transform and preprocess data
def data_transform(df_1, processor):

    # drop unnecessary columns
    drop_cols = ['travelCode', 'userCode', 'time']
    df = df_1.drop(drop_cols, axis=1)

    # Define X and y
    X = df.drop(columns = ['price'], axis=1)
    y = df['price']

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

    #Preprocessing
    processed_X_train = processor.fit_transform(X_train)
    processed_X_test = processor.fit_transform(X_test)

    return processed_X_train, y_train,processed_X_test, y_test


# Function to predict prices using linear regression

def predict_price(X_train, y_train, X_test, y_test, models, param_grids):

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):
         
         mlflow.log_param("model", model_name)

         if (model_name in param_grids):
          param_grid = param_grids[model_name]
          grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')# implement grid search
          grid_search.fit(X_train, y_train)

          best_params = grid_search.best_estimator_  # find best tuning
          
          mlflow.sklearn.log_model(best_params, model_name)
          
          y_pred_train = best_params.predict(X_train) # predict using the best tuned model
          y_pred_test = best_params.predict(X_test)

         else:
          model.fit(X_train, y_train)

          mlflow.sklearn.log_model(model, model_name)

          y_pred_train = model.predict(X_train) # predict using the best tuned model
          y_pred_test = model.predict(X_test)

         train_MSE = mean_squared_error(y_train,y_pred_train)
         test_MSE = mean_squared_error(y_test,y_pred_test)

         mlflow.log_metric("train_MSE", train_MSE)
         mlflow.log_metric("test_MSE", test_MSE)

         train_R2 = r2_score(y_train,y_pred_train)
         test_R2 = r2_score(y_test,y_pred_test)

         mlflow.log_metric("train_R2", train_R2)
         mlflow.log_metric("test_R2", test_R2)

         print(f"Model: {model_name}, Accuracy: {test_R2 :.4f}")

    print("Experiment logged in MLflow!")

    return

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'flights.csv')

processor_path= os.path.join(os.path.dirname(__file__), '..', 'models', 'flight_price_preprocessor.joblib')

models = {'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=0.1),
               #'SVR':SVR(),
               #'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(max_depth=15, max_features=20, max_leaf_nodes=80, random_state=42)
               #('Random Forest': RandomForestRegressor(n_estimators=10, random_state=42)
}

param_grids = { 'Random Forest': {'n_estimators': [50, 100, 200],
                      'max_depth': [10, 20, 30, None]},

    'Ridge Regression': {'alpha': [0.1, 1, 10, 100] },

    'SVR': { 'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.2] },

    'KNN': {'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']},

    'Decision Tree':{'max_depth': ( 10, 14, 16),
                    'max_features': (10, 20, 30),
                    'max_leaf_nodes': [30, 50, 100],
                    #'min_samples_leaf': [ 2, 4, 8, 10],
                    'random_state': [42] }
  }



df, processor = load_data(data_path, processor_path)

X_train, y_train,X_test, y_test = data_transform(df, processor)

#int(X_train.head(), y_train.head())

predict_price(X_train, y_train,X_test, y_test, models, param_grids)
