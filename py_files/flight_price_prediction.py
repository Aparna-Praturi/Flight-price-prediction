
# Import Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Define classes for data preprocessing
class DateExtractor(BaseEstimator, TransformerMixin):
  """ This class extracts day, month, year and
   weekday from the date_col of a given df, makes a new feature
   'is_weekend' and returns the modified dataframe """
  def __init__(self, date_col):
        self.date_col = date_col

  def fit(self, X, y=None):
        return self

  def transform(self, X):
        X_copy = X.copy()
        X_copy[self.date_col] = pd.to_datetime(X_copy[self.date_col])
        X_copy['Day'] = X_copy[self.date_col].dt.day
        X_copy['Month'] = X_copy[self.date_col].dt.month
        X_copy['Year'] = X_copy[self.date_col].dt.year
        X_copy['weekday'] = X_copy[self.date_col].dt.weekday
        X_copy['is_weekend'] = np.where(X_copy['weekday'].isin([5, 6]), 1, 0).astype(int)
        X_copy = X_copy.drop(columns=[self.date_col])
        return X_copy
  

class CatEncoder(BaseEstimator, TransformerMixin):
    """ This class encodes given categorical variables in adataframe using one hot encoding
    and returns modified df """
    def __init__(self, cat_features, encoder):
        self.cat_features = cat_features
        self.encoder = encoder

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        transformed_data = self.encoder.transform(X_copy[self.cat_features])
        transformed_df = pd.DataFrame(transformed_data, columns=self.encoder.get_feature_names_out(self.cat_features))
        transformed_df.index = X_copy.index
        X_copy = X_copy.drop(columns=self.cat_features).join(transformed_df)
        return X_copy

class CycEncoder(BaseEstimator, TransformerMixin):
  """ This class performs cyclical encoding over the cyclical features like Day, Month and weekday
      and returns modified df """
  def __init__(self, cyc_features):
      self.cyc_features = cyc_features

  def fit(self, X, y=None):
        return self

  def transform(self, X):
        X_copy = X.copy()
        for feature, max_value in self.cyc_features.items():
            X_copy[feature + '_sin'] = np.sin(2 * np.pi * X_copy[feature] / max_value)
            X_copy[feature + '_cos'] = np.cos(2 * np.pi * X_copy[feature] / max_value)
            X_copy = X_copy.drop(columns=[feature])
        return X_copy


class num_scaler(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, scaler):
        self.num_features = num_features
        self.scaler = scaler

    def fit(self, X, y=None):
       return self

    def transform(self, X):
        X_copy = X.copy()
        X_num = X_copy[self.num_features]
        X_scaled = self.scaler.transform(X_num)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.num_features, index=X_copy.index)
        df_scaled = X_copy.drop(columns=self.num_features).join(X_scaled_df)
        return df_scaled

# function to split into train and test data
def extract_train_test(df, test_size, random_state):
  X = df.drop(columns = ['price'], axis=1)
  Y = df['price']
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
  return X_train, X_test, y_train, y_test

# Function to scale numerical features
def scale_fit(X_train, num_features, scaler):
  # Applying date extractor on X_train to extract num_feature 'Year'
  date_transformer = DateExtractor(date_col='date')
  X_train_transformed = date_transformer.fit_transform(X_train)

  # Applying the standard scaler
  scaler = scaler
  fit = scaler.fit(X=X_train_transformed[num_features])
  return fit



class Model_training:

  """ This class performs model training using given models,
   calculates accuracy scores using given scoring method, plots the residuals,
    hyperparameter tuning for best model using given hyperparameters and returns the best model along with
    evaluation metrics. """

  def __init__(self, models=None, scoring=mean_absolute_error, cv=5, tune_best_model=True, param_grids={}):    ## 1. Initialise the class
    self.models = models
    self.scoring = scoring
    self.cv = cv
    self.tune_best_model = tune_best_model
    self.param_grids = param_grids if param_grids else {}
    self. evaluation_metrics = pd.DataFrame(index =['train_MSE', 'test_MSE', 'train_MAE', 'test_MAE','train_R2', 'test_R2'])

  def plot(self, y_test, y_pred_test):
    # Function to plot residuals
     residuals = y_test - y_pred_test
     fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # plot residuals
     axs[0].scatter(y_pred_test, y_test)
     axs[0].plot([min(y_pred_test), max(y_pred_test)], [min(y_pred_test), max(y_pred_test)], 'k--', lw=2)
     axs[0].set_xlabel('Predicted values')
     axs[0].set_ylabel('Actual values')
     axs[0].set_title('Scatter plot of predicted vs actual values')
     axs[0].grid(True)
     axs[1].scatter(y_pred_test, residuals, color='blue', edgecolors='k', alpha=0.7)
     axs[1].axhline(y=0, color='black', linestyle='--')
     axs[1].set_title(f"Residuals")
     axs[1].set_xlabel('Predicted Values')
     axs[1].set_ylabel('Residuals')

     plt.tight_layout()
     plt.show()
     return
  
  def cal_eval(self,y,y_pred):
     mae = mean_absolute_error(y,y_pred)
     mse = mean_squared_error(y,y_pred)
     r2 = r2_score(y,y_pred)
     return mae, mse, r2


  ## Find best model
  def find_model(self, X_train, y_train, X_test, y_test):   ## 1. Define function to find best model
    # initialise best model parameters
    bestModel_score = float('inf')  
    bestModel = None
    bestModel_name = None

   # loop through all models
    for name, model in self.models:   
     print(f"Training {name} model...")
     model.fit(X_train, y_train)  # Fit the model

     # Predict train and test target variable
     y_pred_train = model.predict(X_train)  
     y_pred_test = model.predict(X_test)

     # evaluate the model
     train_score, train_mse, train_r2 = self.cal_eval(y_train, y_pred_train)
     test_score, test_mse, test_r2 = self.cal_eval(y_test, y_pred_test)
     
     print(f"{name} train score: {train_score}")
     print(f"{name} score: {test_score}")

     # plot residuals
     self.plot(y_test, y_pred_test)

     # update metrics table
     self.evaluation_metrics[name] = [train_score, test_score, train_mse, test_mse, train_r2, test_r2]
     
     # update the best model
     if test_score < bestModel_score:  
      bestModel = model
      bestModel_score = test_score
      bestModel_name = name

    print(f"Best model: {bestModel_name} with Score: {bestModel_score}")  # print the best model
    return bestModel, bestModel_name

 ## Tune hyperparameters for best model
  def tune_bestModel(self, bestModel_name, bestModel, X_train, y_train, X_test, y_test):   ##   Tune hyperparameters for best model
    if (bestModel_name in self.param_grids) and self.tune_best_model:
      print(f"Tuning hyperparameters for {bestModel_name}....")

      param_grid = self.param_grids[bestModel_name]
      grid_search = GridSearchCV(bestModel, param_grid, cv=self.cv, scoring='neg_mean_squared_error')# implement grid search
      grid_search.fit(X_train, y_train)
      best_params = grid_search.best_estimator_  # find best tuning
      
      # predict using the best tuned model
      y_pred_train_bestparam = best_params.predict(X_train) 
      y_pred_test_bestparam = best_params.predict(X_test)

      # evaluate metrics
      train_score_best_param, train_mse, train_r2 = self.cal_eval(y_train, y_pred_train_bestparam)
      test_score_best_param, test_mse, test_r2 = self.cal_eval(y_test, y_pred_test_bestparam)
      print(f"Best Hyperparameters for {bestModel_name}: {best_params} with training eroor of {train_score_best_param} and testing error of{ test_score_best_param}")
      
      # plot
      self.plot(y_test, y_pred_test_bestparam)

      # update evaluation metrics table
      self.evaluation_metrics[f'{bestModel_name}-hyperparameter_tuned '] = [train_score_best_param, test_score_best_param, train_mse, test_mse, train_r2, test_r2]
      return best_params, self.evaluation_metrics

    else:
      print(f"No tuning required for {bestModel_name}")
      return bestModel, self.evaluation_metrics

 ##  Call the class
  def __call__(self, X_train, y_train, X_test, y_test):
     bestModel, bestModel_name = self.find_model(X_train, y_train, X_test, y_test)
     best_params = self.tune_bestModel(bestModel_name, bestModel, X_train, y_train, X_test, y_test)
     return best_params, self.evaluation_metrics
  
##---------------------------------------------------------------------------------------------------------------------------------------------------- 

# Defining the path to data file
path = os.path.join(os.path.dirname(__file__), '..', 'data', 'flights.csv')

# Loading data to df
try:
    df_1 = pd.read_csv(f'{path}')
    print("Data loaded successfully!")

except FileNotFoundError:
     print(f"Error: The file(s) at {path} were not found. Please check the path.")

except pd.errors.EmptyDataError:
      print(f"Error: The file(s) at {path} is empty. Please check the file content.")

except pd.errors.ParserError:
      print(f"Error: There was a problem parsing the file(s) at {path}. Please check the file(s) format.")

except Exception as e:
      print(f"An unexpected error occurred: {e}")

# Decalring parameters
drop_cols = ['travelCode', 'userCode', 'time']
modified_df = df_1.drop(drop_cols, axis=1)
date_col = 'date'
cat_features = ['from', 'to', 'flightType', 'agency']
num_features = ['distance', 'Year']
cyc_features = {'Day': 31, 'Month': 12, 'weekday': 7}
models = [('Linear Regression', LinearRegression()),
                ('Ridge Regression', Ridge(alpha=0.1)),
               #('SVR', SVR()),
               #('KNN', KNeighborsRegressor()),
                ('Decision Tree', DecisionTreeRegressor(max_depth=15, max_features=20, max_leaf_nodes=80, random_state=42)),
               #('Random Forest', RandomForestRegressor(n_estimators=10, random_state=42))
               ]

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

## Data Preprocessing
# Splitting the data 
X_train, X_test, y_train, y_test = extract_train_test(modified_df, test_size=0.25, random_state=42)

# Scaling the train data
fit = scale_fit(X_train, num_features, StandardScaler())

# Encoding the train data
cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoder.fit(modified_df[cat_features])

# Defining preprocessing pipeline
preprocessor = Pipeline([
      ('date_features', DateExtractor(date_col=date_col)),
      ('num_encoding', num_scaler(num_features=num_features, scaler=fit)),
      ('cyclical_encoding', CycEncoder(cyc_features = cyc_features)),
      ('cat_encoding', CatEncoder(cat_features=cat_features, encoder=cat_encoder))
      ])
  
# Save the preprocessor to preprocess data
with open('./models/flight_price_preprocessor.joblib', 'wb') as f:
    joblib.dump(preprocessor, f)


# TRAINING THE MODEL

if __name__ == "__main__":
  # Transforming the train data
  processed_X_train = preprocessor.fit_transform(X_train)
  processed_X_test = preprocessor.fit_transform(X_test)

  # training the model to obtain best model and evaluation metrics
  model_training = Model_training(models=models, scoring=mean_squared_error, cv=5, tune_best_model=True, param_grids=param_grids)
  best_model, evaluation_metrics = model_training(processed_X_train, y_train, processed_X_test, y_test)[0]

  # Evaluation metrics
  print(evaluation_metrics.transpose())

  ## Saving the preprocessor, model and other required data.
  # Save the model to a pickle file
  with open("./models/flight_price_predictor.joblib", "wb") as f:
    joblib.dump(best_model, f)

  # Dictionary to save the distance between cities
  distance_dict = df_1.groupby(['from','to'])['distance'].mean().to_dict()
  with open('./models/distances_dict.joblib', 'wb') as f:
    joblib.dump(distance_dict, f)

  data_dict = {col : df_1[col].unique().tolist() for col in ['from', 'to', 'agency', 'flightType']}
  with open('./models/data_dict.joblib', 'wb') as f:
    joblib.dump(data_dict, f)


  ## Sanity check.
  # Load the model from the pickle file
  with open("./models/flight_price_preprocessor.joblib", "rb") as f:
     loaded_preprocessor = joblib.load(f)
  with open("./models/flight_price_predictor.joblib", "rb") as f:
     loaded_price_predictor = joblib.load(f)

  # preprocessing and predicting the test data
  preprocessed_data = loaded_preprocessor.fit_transform(X_test)
  new_test_preds = loaded_price_predictor.predict(preprocessed_data)

  # Calculate metrics
  mse = mean_squared_error(y_test, new_test_preds)
  mae = mean_absolute_error(y_test, new_test_preds)
  r2 = r2_score(y_test, new_test_preds)

  # Print metrics
  print("MSE:", mse)
  print("MAE:", mae)
  print("R2 Score:", r2)

  