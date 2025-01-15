# Standard libraries
import sys
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from flask import Flask, request, jsonify, render_template_string
# Importing custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py_files'))
from flight_price_prediction import DateExtractor, num_scaler, CycEncoder, CatEncoder



# Load the models from the files 
loaded_lr_model = joblib.load('./models/flight_price_predictor.joblib')
loaded_distance_dict = joblib.load('./models/distances_dict.joblib')
loaded_preprocessor= joblib.load('./models/flight_price_preprocessor.joblib')
loaded_data_dict = joblib.load('./models/data_dict.joblib')

# Defining the predict function
def predict(data, model, preprocessor, dist_dict):

    # Converting input data into a dataframe
    df = pd.DataFrame(data)

    # determining distance between the entered cities using the distance_dict
    from_value = df.loc[0, 'from']
    to_value = df.loc[0, 'to']

    try:
      df['distance'] = dist_dict[from_value,  to_value]
  
    except KeyError:
        # Raise an error if the origin-destination pair is not found in the dictionary
        raise ValueError(f"There are no flights between {from_value} and {to_value}")
    
    except Exception as e:
      print(f"An unexpected error occurred: {e}")

    # Preprocessing and predicting the price using saved preprocessor and model
    preprocessed_df = preprocessor.fit_transform(df)
    predicted_price = model.predict(preprocessed_df)

    return predicted_price

# checking the predict function

# data = {'agency' : ['FlyingDrops'],
#          'from' : ['Recife (PE)'],
#          'to' : ['Florianopolis (SC)'],
#          'flightType': ['firstClass'],
#          'date' : ['12/28/2008']}

# print(predict(data=data, model=loaded_lr_model, preprocessor=loaded_preprocessor,dist_dict= loaded_distance_dict ))

app = Flask(__name__)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flight Price Prediction</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            padding: 20px;
        }
        .container {
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2 class="text-center">Flight Price Prediction</h2>
        
        <!-- Form to input flight details -->
        <form action="/predict" method="POST">
            <div class="form-group">
                <label for="origin">Origin:</label>
                <select class="form-control" id="origin" name="origin" required>
                    <option value="">Select Origin</option>
                    {% for option in loaded_data_dict['to'] %}
                        <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="destination">Destination:</label>
                <select class="form-control" id="destination" name="destination" required>
                    <option value="">Select Destination</option>
                    {% for option in loaded_data_dict['from'] %}
                        <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="agency">Flight Agency:</label>
                <select class="form-control" id="agency" name="agency" required>
                    <option value="">Select Agency</option>
                    {% for option in loaded_data_dict['agency'] %}
                        <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="flight_class">Flight Class:</label>
                <select class="form-control" id="flight_class" name="flight_class" required>
                    <option value="">Select Class</option>
                    {% for option in loaded_data_dict['flightType'] %}
                        <option value="{{ option }}">{{ option }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="form-group">
                <label for="date">Flight Date:</label>
                <input type="date" class="form-control" id="date" name="date" required>
            </div>

            <button type="submit" class="btn btn-primary btn-block">Get Price Prediction</button>
        </form>
        
        {% if prediction_value %}
        <div class="alert alert-info mt-4">
            <h4 class="alert-heading">Predicted Flight Price:</h4>
            <p>The predicted flight price is: <strong>${{ prediction_value }}</strong></p>
        </div>
        {% elif error_message %}
        <div class="alert alert-danger mt-4">
            <h4 class="alert-heading">Error:</h4>
            <p>{{ error_message }}</p>
        </div>
        {% endif %}
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.2/dist/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""

# defining the home page
@app.route('/', methods=['GET', 'POST'])
def home():
     return render_template_string(html_template, loaded_data_dict=loaded_data_dict)


# Defining the predict action
@app.route('/predict', methods=['POST'])
def predict_price():
   
    if request.method == 'POST':

        # Get input data from the form
        origin = request.form.get('origin')
        destination = request.form.get('destination')
        flightType = request.form.get('flight_class')
        agency= request.form.get('agency')
        date = request.form.get('date')
      
        # Make the data into a dictionary
        data = {'agency' : [agency],
         'from' : [origin],
         'to' : [destination],
         'flightType': [flightType],
         'date' : [date]}

        # Perform prediction using the predict function
        try:
            prediction = predict(data=data, model=loaded_lr_model, preprocessor=loaded_preprocessor,dist_dict= loaded_distance_dict )
            prediction_value = round(prediction[0].item(), 2)
            return  jsonify({'Predicted price': prediction_value})
        
        except ValueError as e:
            # If an error occurs, pass the error message to the template
            return  jsonify('There are no flights between these two cities')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)