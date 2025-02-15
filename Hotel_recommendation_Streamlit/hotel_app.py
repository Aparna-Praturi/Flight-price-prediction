## Hotel Recommendation system deployed using Streamlit

# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from datetime import datetime

### Dataset Loading
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

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

# load hotel data
hotel_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hotels.csv')
df_hotels = load_data(hotel_data_path)

# load user data
user_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.csv')
df_users = load_data(user_data_path)

# Renaming columns of hotel data
df_hotels = df_hotels.copy()
df_hotels.rename(columns = {'name':'hotelName'}, inplace = True)

# Retaining only relavant info from users data
df_users1 = df_users[['code','gender', 'age']]

# renaming column names of user data
df_users1 = df_users1.copy()
df_users1.rename(columns = {'code':'userCode'}, inplace = True)

# join df_hotels and df_users1
df = pd.merge(df_hotels, df_users1, on='userCode')
#print(df.head())

# Dataset Rows & Columns count
print(f'df shape:{df.shape}')

# Dataset Information
print(f'df_info:{df.info()}')

# Duplicate Values
print(f'Duplicate values:{df.duplicated().sum()}')

# Missing Values/Null Values
print(f'Missing values:{df.isna().sum()}')

# Check Unique Values for each variable.
print(f'{[df[i].nunique() for i in df.columns]}')

# create user-hotel interaction matrix
user_hotel_matrix = df.groupby(['userCode','hotelName'])['days'].sum().unstack()
user_hotel_matrix.fillna(0, inplace=True)

# Encode gender
cat_encoder = OrdinalEncoder()
gender_column = df['gender'].values.reshape(-1, 1)
df['encoded_gender']=cat_encoder.fit_transform(gender_column)

# split price into quartiles
Q1, Q2, Q3 = df['price'].quantile([0.25, 0.5, 0.75])
df['price_quartile'] = pd.qcut(df['price'], q=4, labels=False)

# create user feature matrix
user_features = ['age', 'encoded_gender','days', 'price_quartile']
df_user_features = df[user_features]
user_feature_matrix = df.groupby('userCode').agg({'days': 'mean','price_quartile': 'mean','age': 'first', 'encoded_gender':'first'}).reset_index('userCode')
user_features = user_feature_matrix.set_index('userCode')


def recommend_hotels(age, gender, days, price, user_features, user_hotel_matrix, destination):
   
   # prepare data
   encoded_gender = cat_encoder.transform(np.array([gender]).reshape(-1, 1))

   # encode_price
   if price <= Q1:
        price_quartile =  1
   elif price <= Q2:
        price_quartile =  2
   elif price <= Q3:
        price_quartile =  3
   else:
        price_quartile =  4

   en_gender =  encoded_gender.flatten()
   target_user= [age, en_gender.item(), days, price_quartile]
  
   reshaped_target_user = np.array(target_user).reshape(1, -1)

   # calculate user similarity
   user_similarity_matrix = cosine_similarity(reshaped_target_user, user_features)
   
   # finding indices of highest values in user similarity 
   indices = np.argsort(user_similarity_matrix[0])[::-1]

   # calculte weighted scores for hotels based on similar users
   weighted_scores = sum([user_hotel_matrix.loc[indices[i]] * (1 - 0.1 * i) for i in range(10)])
   recos = weighted_scores.sort_values(ascending=False).head(20).index.to_list()

   # Select the top hotel which is in the required destination from the list of recommended hotels
   i = 0
   while i < len(recos):
    hotel_places = df[df['hotelName']==recos[i]]['place'].unique()
    if destination in hotel_places:
     best_hotel = recos[i]
     print('found suitable hotel')
     break
    i += 1  # Move to the next hotel
   print(f' The recommended hotel is {best_hotel} at {hotel_places.item()}')
   return best_hotel, hotel_places.item()

# Streamlit UI
st.set_page_config(page_title="Hotel Recommendation", layout="wide")

# Title of the app
st.title("Hotel Recommendation System")

# Dropdown for Destination
destinations = df['place'].unique().tolist()
destination = st.selectbox("Select your Destination", destinations)

#  Date input for Check-in and Check-out
checkin_date = st.date_input("Check-in Date", datetime.today())
checkout_date = st.date_input("Check-out Date", datetime.today())

if checkout_date > checkin_date:
    delta = checkout_date - checkin_date
    days = delta.days
    st.write(f"Number of days between check-in and check-out: {days} days")
else:
    st.warning("Check-out date must be later than check-in date!")

# Age input
age_options = list(range(18, 101))  
age = st.selectbox("Select your Age", age_options)

# Gender Selection Button (Radio button for Gender)
gender = st.radio("Select your Gender", ["male", "female", "other"])
gender =[gender]

# Slider for maximum price
min_price = float(min(df['price']))  
max_price = float(max(df['price']))  
price = st.slider("Select your Price per Night (USD)", min_price, max_price, float(max_price))

# Show selected inputs
st.write(f"Destination: {destination}")
st.write(f"no.of days: {days}")
st.write(f"Age: {age}")
st.write(f"Maximum Price: ${price}")
st.write(f"Gender: {gender}")

# Recommend hotels based on user input
if st.button('Recommend Hotels'):
     recommended_hotel, place = recommend_hotels(age, gender, days, price, user_features, user_hotel_matrix, destination)
     st.write(f"Recommended Hotels:{recommended_hotel} at {place}")
     



