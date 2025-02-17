# Travel Capstone Project

## Overview
The Travel Capstone Project is an end-to-end machine learning solution that includes multiple components for enhancing travel experiences. This project features:

- A **prediction model** deployed with Flask and Docker
- Orchestration using **Apache Airflow** (running from a Docker image)
- **MLflow** for experiment tracking and model management
- A **Streamlit-based hotel recommendation system**
- A **Streamlit-based gender identification app** using names
- Deployment and scaling via **Kubernetes**
- **Continuous Integration (CI) with Jenkins**

## Tech Stack
- **Machine Learning**: Scikit-Learn, MLflow
- **Backend**: Flask, Docker
- **Orchestration**: Apache Airflow
- **Deployment**: Kubernetes, Docker
- **Frontend**: Streamlit
- **Tracking & Monitoring**: MLflow
- **CI/CD**: Jenkins

## Project Structure
```
travel-capstone/
│-- Airflow_Price-prediction/    # Apache Airflow DAGs, configurations and airflow docker image 
|-- data/                        # all the data files            
│-- Flask_price-prediction
│   │-- Dockerfile               # Docker configuration for Flask API
│   │-- requirements.txt         # Dependencies
|   |-- app.py                   # flask app
|   |--service.yaml
|-- Gender_classification-streamlit/
|   |-- gender_app.py            # streamlit app for gender classification
│-- MLFlow_price-prediction/
|   |--mlflow_script.py
|   | ml_runs                     # MLflow experiment tracking setup
│-- Hotel_recommendation_streamlit/
│   │-- hotel_app.py /           # Streamlit app for hotel recommendations
│-- models/                      # Trained models             
│-- requirements.txt             # Dependencies
|-- READMe.txt
```

## Setup & Installation

### Prerequisites
- Docker
- Kubernetes (Minikube or a cloud-managed cluster)
- Apache Airflow
- MLflow
- Streamlit
- Jenkins

### Steps to Run



#### 1. Clone the repository

git clone https://github.com/Aparna-Praturi/Flight-price-prediction.git


#### 2. Build and run Docker container for Flask app

```
cd Flight-price-prediction
docker build -t flight_price_prediction:latest -f  Flask_Price-prediction/Dockerfile .      
docker run  -p 5000:5000 flight_price_prediction:latest

```

#### 3. Deploy on Kubernetes

```
minikube start   
kubectl apply -f Deployment.yaml
kubectl apply -f service.yaml
minikube tunnel         # In another terminal
http://127.0.0.1

```

#### 4. Set up Airflow DAGs from airflow docker image
```
cd Airflow_Price-prediction
docker compose -f 'Airflow_Price-prediction\Docker-compose.yml' up -d --build 
```
#### 5. Start MLflow 
```
python MLFlow_Price-prediction\mlflow-script.py    
```
#### 6. Run Streamlit Apps

For gender identification:
```
streamlit run Gender_classification_Streamlit\gender_app.py
```
For hotel recommendations:
```
streamlit run Hotel_recommendation_Streamlit\hotel_app.py  
```



## Usage
- The **Flask API** serves predictions via a RESTful interface.
- The **hotel recommendation system** suggests hotels based on user preferences.
- The **gender identification app** predicts gender based on a given name.
- **Apache Airflow** manages data pipelines and model retraining.
- **MLflow** tracks experiments, logs models, and manages model lifecycle.
- **Kubernetes** ensures scalable deployment.
- **Jenkins** automates testing, builds, and deployments.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any queries, reach out at: **your.email@example.com**



