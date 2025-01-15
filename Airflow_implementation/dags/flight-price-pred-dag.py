import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import joblib
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_files'))
from flight_price_prediction import DateExtractor, num_scaler, CycEncoder, CatEncoder


from sklearn.metrics import mean_squared_error, r2_score

# from data_ingestion import DataLoader
# from data_transformation import DataTransformer
# from model_training import RandomForestModel

class DataLoader:
    def __init__(self, data_path):
        self.file_path = data_path
    
    def load_data(self):
        df = pd.read_csv(self.file_path, on_bad_lines='skip')
        
        # Check for missing values
        if df.isnull().sum().sum() == 0:
            print("There are no missing values present")
        else:
            print("There are missing values present and we need to handle them!")

        return df


class DataProcessor:
    def __init__(self, data, processor_path):
        self.data = data
        self.processor_path = processor_path
        
    
    def preprocess(self):
        df_1 = self.data

        # Drop irrelavant columns

        drop_cols = ['travelCode', 'userCode', 'time']

        df = df_1.drop(drop_cols, axis=1)

        # fill missing values

        df['from'].fillna('Not Available', inplace=True)
        df['to'].fillna('Not Available', inplace=True)
        df['agency'].fillna('Not Available', inplace=True)
        df['flightType'].fillna('Not Available', inplace=True)
        df['date'].fillna('Not Available', inplace=True)
        df['price'].fillna('Not Available', inplace=True)

        processor = joblib.load(self.processor_path)

    
        X = df.drop(columns = ['price'], axis=1)
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
        
     
        
        # preprocess train and test separately

        processed_X_train = processor.fit_transform(X_train)
        processed_X_test = processor.fit_transform(X_test)

        y_train_list = y_train.tolist()
        y_test_list = y_test.tolist() 

        
        return processed_X_train,y_train_list, processed_X_test, y_test_list

        

class DecisionTreeModel:
    def __init__(self, X_train,y_train, X_test, y_test ):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_metrics(self, actual, predicted):
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)

        print('MSE is {}'.format(MSE))
        print('R2 score is {}'.format(R2))

        return MSE, R2

    def predict(self):

        param_grid={'max_depth': ( 10, 14),
                    'max_features': (10, 20),
                    'max_leaf_nodes': [ 50, 100],
                    #'min_samples_leaf': [ 2, 4, 8, 10],
                    'random_state': [42] }
    
        model = DecisionTreeRegressor()

        grid = GridSearchCV(estimator=model,
                                     param_grid=param_grid,
                                     cv=5, verbose=2, scoring='r2')
        
        grid.fit(self.X_train, self.y_train)

        best_model = grid.best_estimator_  # find best tuning
        
        y_pred_train = best_model.predict(self.X_train)  # Predict train and test target variable
        y_pred_test = best_model.predict(self.X_test)

        print("Train Set Metrics:")
        print("----------------------------------------------")
        self.evaluate_metrics(self.y_train, y_pred_train)
        print("\n")

        print("Test Set Metrics")
        print("----------------------------------------------")
        metrics = self.evaluate_metrics(self.y_test, y_pred_test)
        print("\n")
        r2_rf = metrics[1]

        pred = pd.DataFrame({'Actual Value': self.y_test,
                              'Predicted Value': y_pred_test})
        print("The top 5 rows of actual vs predicted values for Decision tree\n", pred.head())
        
class LinearRegressionModel:

    def __init__(self, X_train,y_train, X_test, y_test ):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_metrics(self, actual, predicted):
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)

        print('MSE is {}'.format(MSE))
        print('R2 score is {}'.format(R2))

        return MSE, R2

    def predict(self):

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        y_pred_train = model.predict(self.X_train)  # Predict train and test target variable
        y_pred_test = model.predict(self.X_test)

        print("Train Set Metrics:")
        print("----------------------------------------------")
        self.evaluate_metrics(self.y_train, y_pred_train)
        print("\n")

        print("Test Set Metrics")
        print("----------------------------------------------")
        metrics = self.evaluate_metrics(self.y_test, y_pred_test)
        print("\n")
        r2_rf = metrics[1]

        pred = pd.DataFrame({'Actual Value ': self.y_test,
                              'Predicted Value': y_pred_test})
        print("The top 5 rows of actual vs predicted values for linear regression\n", pred.head())
        

class RidgeRegressionModel:
    def __init__(self, X_train,y_train, X_test, y_test ):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_metrics(self, actual, predicted):
        MSE = mean_squared_error(actual, predicted)
        R2 = r2_score(actual, predicted)

        print('MSE is {}'.format(MSE))
        print('R2 score is {}'.format(R2))

        return MSE, R2

    def predict(self):

        param_grid={'alpha': [0.1, 1, 10, 100] }
    
        model = Ridge()

        grid = GridSearchCV(estimator=model,
                                     param_grid=param_grid,
                                     cv=5, verbose=2, scoring='r2')
        
        grid.fit(self.X_train, self.y_train)

        best_model = grid.best_estimator_  # find best tuning
        
        y_pred_train = best_model.predict(self.X_train)  # Predict train and test target variable
        y_pred_test = best_model.predict(self.X_test)

        print("Train Set Metrics:")
        print("----------------------------------------------")
        self.evaluate_metrics(self.y_train, y_pred_train)
        print("\n")

        print("Test Set Metrics")
        print("----------------------------------------------")
        metrics = self.evaluate_metrics(self.y_test, y_pred_test)
        print("\n")
        r2_rf = metrics[1]

        pred = pd.DataFrame({'Actual Value': self.y_test,
                              'Predicted Value': y_pred_test})
        print("The top 5 rows of actual vs predicted values for Ridhe Regression\n", pred.head())
        
        
# Define your file paths here
data_file_path = './data/flights.csv'
preprocessor_file_path = './models/flight_price_preprocessor.joblib'


# Create instances of your classes
data_loader = DataLoader(data_file_path)
df_load = data_loader.load_data()

data_processor = DataProcessor(df_load,preprocessor_file_path)
processed_X_train,y_train, processed_X_test, y_test = data_processor.preprocess()

decisiontree_model = DecisionTreeModel(processed_X_train,y_train, processed_X_test, y_test)
#decisiontree_model.predict()

linearregression_model = LinearRegressionModel(processed_X_train,y_train, processed_X_test, y_test)

ridge_model = RidgeRegressionModel(processed_X_train,y_train, processed_X_test, y_test)

# Define the default arguments for the DAG
default_args = {
    'owner': 'admin',
    'depends_on_past': True,
    'start_date': datetime(2025, 1, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Create the DAG
dag = DAG(
    'travel_price_prediction',
    default_args=default_args,
    description='A DAG for travel price prediction',
    schedule_interval=None,  # Define your desired schedule interval
)

# Task to load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=data_loader.load_data,
    #op_args=[data_file_path],  # Pass any required arguments to the function
    dag=dag,
)

# Task to transform data
process_data_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=data_processor.preprocess,
    dag=dag,
	execution_timeout=timedelta(minutes=30),
)

# Task to run random forest model
decision_tree_task = PythonOperator(
    task_id='decision_tree__task',
    python_callable=decisiontree_model.predict,
    dag=dag,
	execution_timeout=timedelta(minutes=30),
)

linear_regression_task = PythonOperator(
    task_id='random_forest_task',
    python_callable=linearregression_model.predict,
    dag=dag,
	execution_timeout=timedelta(minutes=30),
)

ridge_regression_task = PythonOperator(
    task_id='ridge_regression_task',
    python_callable=ridge_model.predict,
    dag=dag,
	execution_timeout=timedelta(minutes=30),
)

# Define the task dependencies
load_data_task >> process_data_task >> [linear_regression_task, ridge_regression_task, decision_tree_task ]
 