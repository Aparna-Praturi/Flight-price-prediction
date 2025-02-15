
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import bigrams, trigrams
import joblib
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

## Defining classes for preprocessing


class Namesplitter(BaseEstimator, TransformerMixin):
    """ splits the name into first name and last name and extracts firstname"""
    def __init__(self, name_col):
        self.name_col = name_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = pd.DataFrame(X)
        X_copy['Firstname'] = X_copy[self.name_col].apply(lambda x: x.split(" ")[0])

        X_copy.drop(columns = [self.name_col], inplace = True)

        return X_copy[['Firstname']]


class IsVowelEnd(BaseEstimator, TransformerMixin):

    """" Checks whether the firstname starts or ends with a vowel"""

    def __init__(self, col_name):
        self.col_name= col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vowels = 'aeiouAEIOU'
        X['is_vowel_end'] = X[self.col_name].apply(lambda x: 1 if isinstance(x, str) and x[-1] in vowels else 0)
        X['is_vowel_start'] = X[self.col_name].apply(lambda x: 1 if isinstance(x, str) and x[0] in vowels else 0)
        return X
    
    
class TfidfAndDense(BaseEstimator, TransformerMixin):
 
    """ Uses tfidf for firstname n-grams and convert to dense vector"""

    def __init__(self, col_name, ngram_range=(2, 3)):

        self.ngram_range = ngram_range
        self.tfidf_vectorizer = CountVectorizer(analyzer='char', ngram_range=self.ngram_range)
        self.col_name = col_name

    def fit(self, X, y=None):

        z = X[self.col_name].tolist()
        self.tfidf_vectorizer.fit(z)
        return self

    def transform(self, X):

        tfidf_matrix = self.tfidf_vectorizer.transform(X[self.col_name])
        X_dense = tfidf_matrix.toarray()
        X_dense_df = pd.DataFrame(X_dense, columns=self.tfidf_vectorizer.get_feature_names_out(), index=X.index  )

        X = pd.concat([X, X_dense_df], axis=1)
        X = X.drop(columns = [self.col_name])
        return X
    

class Model_training:

  """ This class performs model training using given models,calculates accuracy scores using given scoring method,
      plots the residuals,hyperparameter tuning for best model using given hyperparameters and returns the best model
      along with evaluation metrics. """

  def __init__(self, models=None, cv=5, tune_best_model=True, param_grids={}):
    self.models = models
    self.cv = cv
    self.tune_best_model = tune_best_model
    self.param_grids = param_grids if param_grids else {}
    self. evaluation_metrics = pd.DataFrame(index =['train_accuracy', 'test_accuracy', 'train_precision',
                                                    'test_precision','train_recall', 'test_recall', 'train_f1', 'test_f1'])
  # function for calculating eval parameters
  def eval_cal(self, y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')

    print(f"Accuracy Score: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")

    return accuracy, f1, precision, recall

  # function for plotting confusion matrix
  def plot(self, y, y_pred):
    cm = confusion_matrix(y, y_pred)
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

  ## Find best model
  def find_model(self, X_train, y_train, X_test, y_test):   ## 1. Define function to find best model

    bestModel_score = -float('inf')
    bestModel = None
    bestModel_name = None

    # loop through all models
    for name, model in self.models:

     print(f"Training {name} model...")

     # fit the model
     model.fit(X_train, y_train)

     # Predict train and test
     y_pred_train = model.predict(X_train)
     y_pred_test = model.predict(X_test)

     # calculate evaluation metrics
     train_accuracy, train_f1, train_precision, train_recall = self.eval_cal(y_train, y_pred_train)
     test_accuracy, test_f1, test_precision, test_recall = self.eval_cal(y_test, y_pred_test)

     # update metrics table
     self.evaluation_metrics[name] = [train_accuracy, test_accuracy, train_precision,
                                      test_precision,train_recall, test_recall, train_f1, test_f1]

     # plot confusion_matrix
     self.plot(y_test, y_pred_test)

    # update the best model

     if test_f1 > bestModel_score:

      bestModel = model
      bestModel_score = test_f1
      bestModel_name = name

    if (bestModel_name in self.param_grids) and self.tune_best_model:

      name = bestModel_name
      model= bestModel
      print(f'Tuning hyperparameters for {name}, {model}')
      param_grid = self.param_grids[name]

      # search the grid
      grid_search = RandomizedSearchCV(model, param_grid, cv=self.cv)
      grid_search.fit(X_train, y_train)

      # find the best parameters
      best_params = grid_search.best_estimator_
      best_params.fit(X_train, y_train)

      # predict using the best tuned model
      y_pred_train_bestparam = best_params.predict(X_train)
      y_pred_test_bestparam = best_params.predict(X_test)

      # calculate evaluation metrics
      train_accuracy_bp, train_f1_bp, train_precision_bp, train_recall_bp = self.eval_cal(y_train, y_pred_train_bestparam)
      test_accuracy_bp, test_f1_bp, test_precision_bp, test_recall_bp = self.eval_cal(y_test, y_pred_test_bestparam)

      # plot confusion_matrix
      self.plot(y_test, y_pred_test_bestparam)

      # update metrics table
      self.evaluation_metrics[f'{name}-hyperparameter_tuned '] = [train_accuracy_bp, test_accuracy_bp, train_precision_bp,
                                      test_precision_bp,train_recall_bp, test_recall_bp, train_f1_bp, test_f1_bp]

      print(f"Best Hyperparameters for {bestModel_name}: {best_params} with training eroor of {train_accuracy_bp} and testing error of{ test_accuracy_bp}")

      self.plot(y_test, y_pred_test_bestparam)
      return best_params, bestModel_name,  self.evaluation_metrics

    else:

      print(f"No tuning required for {bestModel_name}")

      return bestModel, bestModel_name, self.evaluation_metrics
    
    
## Define models and tuning parameters
models = [('Logistic Regression', LogisticRegression(class_weight='balanced')),
          ('Multinomial Naive Bayes', MultinomialNB(alpha = 5)),
          ('SVC', SVC( C= 1)),
          ('KNN', KNeighborsClassifier(n_neighbors=3)),
          ('Random Forest', RandomForestClassifier(random_state=42)),
          #('XGB', xgb.XGBClassifier( objective='binary:logistic'))
          ]

param_grids = {

  'Multinomial Naive Bayes' : {'alpha': [0.1, 0.5, 1, 2, 5]},

  'Logistic Regression' :{'C': [0.01, 0.1, 1, 10, 100],
                           'penalty': ['l1', 'l2'],
                           'solver': ['liblinear', 'saga'],
                           'max_iter': [50, 100, 200],},

   'SVC': { 'C': [ 0.1, 1, 10, 100],
             'kernel': ['linear', 'rbf', 'poly'],
             'gamma': ['scale', 'auto', 0.1, 1]},

    'Random Forest' : {'n_estimators': [50, 100, 200, 500],
                      'max_depth': [None, 10, 20, 30],
                      'min_samples_split': [ 10, 15, 20],
                      'min_samples_leaf': [1, 2, 4],
                      'max_features': ['sqrt', 'log2', 0.6],
                      'class_weight': ['balanced', None],
                      'random_state': [42]},


    'XGB' :  {'n_estimators': [ 200, 500, 800],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'max_depth': [3, 5, 7],
                  'min_child_weight': [1, 5, 10],
                  'subsample': [0.8, 0.9, 1.0],
                  'num_class': [2],
                  'eval_metric': ['merror'],
                  'random_state': [42]},

    'KNN' : {'n_neighbors': [3, 5, 7, 10],
             'weights': ['uniform', 'distance'],
             'metric': ['euclidean', 'manhattan', 'cosine'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

}

# Define preprocessing pipeline
preprocessor = Pipeline([
    ('namesplitter', Namesplitter('name')),
    ('is_vowel_end', IsVowelEnd('Firstname')),
    ('vectorizer', TfidfAndDense(col_name='Firstname'))
])


if __name__=='__main__':
   
    # Load Data
    def load_data(data_path):
        # Load the data
        try:
            df_1 = pd.read_csv(data_path)
            print("Data loaded successfully!")

        except FileNotFoundError:
            print(f"Error: The file(s) at {data_path} were not found. Please check the path.")

        except pd.errors.EmptyDataError:
            print(f"Error: The file(s) at {data_path} is empty. Please check the file content.")

        except pd.errors.ParserError:
            print(f"Error: There was a problem parsing the file(s) at {data_path}. Please check the file(s) format.")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return df_1

    data_path = hotel_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.csv')
    df = load_data(data_path)

    # Removing irrelavant columns
    df_mod = df.copy()
    df_mod = df_mod.drop(columns = ['code', 'company', 'age'])

    # replacing 'none' values in gender with male/female based on whether the firstname ends with vowel or consonant
    vowels = 'aeiouAEIOU'
    df_mod['gender'] = df_mod.apply(lambda row: row['gender'].replace('none', 'female')
                    if row['name'].split(' ')[0][-1] in vowels else row['gender'].replace('none', 'male')
                    if row['gender'] == 'none' else row['gender'], axis=1 )


    # splitting into train and test data
    X = df_mod.drop(columns = ['gender'])
    y = df_mod['gender']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)# preprocess the data using the pipeline

    # preprocess the data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # save the preprocessing pipeline
    with open('./models/gender_class_preprocessor.joblib', 'wb') as f:
        joblib.dump(preprocessor , f)


    # Fit and transform on y_train, transform on y_test
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Train the model
    model_training = Model_training(models=models, cv=5, tune_best_model=True, param_grids=param_grids)
    model, model_name, evaluation_metrics = model_training.find_model(X_train_transformed, y_train_encoded, X_test_transformed, y_test_encoded)

    # Print evaluation metrics
    print(evaluation_metrics.transpose())

    # Save the model
    with open('./models/gender_classifier.joblib', 'wb') as file:
        joblib.dump(model, file)