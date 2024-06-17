# Adey-Innovation-Fraud-Detection-Project-

## Overview

This project aims to improve the detection of fraud cases for e-commerce transactions and bank credit transactions using advanced machine learning models and data analysis techniques.

## Business Need

Adey Innovations Inc. focuses on solutions for e-commerce and banking. The goal is to create accurate and robust fraud detection models to enhance transaction security, prevent financial losses, and build customer trust.

## Project Tasks

1. **Data Analysis and Preprocessing**
    - Handle Missing Values
    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - Merge Datasets for Geolocation Analysis
    - Feature Engineering
    - Normalization and Scaling
    - Encode Categorical Features

2. **Model Building and Training**
    - Data Preparation
    - Model Selection (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, MLP, CNN, RNN, LSTM)
    - Model Training and Evaluation
    - MLOps Steps (Versioning and Experiment Tracking)

3. **Model Explainability**
    - Using SHAP for Explainability
    - Using LIME for Explainability

4. **Model Deployment and API Development**
    - Setting Up the Flask API
    - API Development
    - Dockerizing the Flask Application

## Datasets

- `Fraud_Data.csv`: E-commerce transaction data.
    - `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`
- `IpAddress_to_Country.csv`: Maps IP addresses to countries.
    - `lower_bound_ip_address`, `upper_bound_ip_address`, `country`
- `creditcard.csv`: Bank transaction data for fraud detection.
    - `Time`, `V1` to `V28`, `Amount`, `Class`

## Learning Outcomes

- **Skills:**
    - Deploying machine learning models using Flask
    - Containerizing applications using Docker
    - Creating REST APIs for machine learning models
    - Testing and validating APIs
    - Developing end-to-end deployment pipelines
    - Implementing scalable and portable ML solutions

- **Knowledge:**
    - Principles of model deployment and serving
    - Best practices for creating REST APIs
    - Understanding containerization benefits
    - Techniques for real-time prediction serving
    - Security considerations in API development
    - Methods for monitoring and maintaining deployed models

## Deliverables

1. **Task 1 - Data Analysis and Preprocessing**
2. **Task 2 - Model Building and Training**
3. **Task 3 - Model Explainability**
4. **Task 4 - Model Deployment and API Development**

## References

- [Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/code)
- [Fraud Detection](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/code)
- [Fraud Detection](https://complyadvantage.com/insights/what-is-fraud-detection/)
- [Fraud Detection](https://www.spiceworks.com/it-security/vulnerability-management/articles/what-is-fraud-detection/)
- [Modeling](https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/)
- [Modeling](https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/)
- [Modeling](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
- [Modeling](https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm)
- [Modeling](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning)
- [Modeling](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns)
- [Modeling](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
- [Modeling](https://www.ibm.com/topics/recurrent-neural-networks)
- [Modeling](https://www.analyticsvidhya.com/blog/2022/03/a-brief-overview-of-recurrent-neural-networks-rnn/)
- [Modeling](https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/)
- [Modeling](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
- [Model Explainability](https://www.larksuite.com/en_us/topics/ai-glossary/model-explainability-in-ai)
- [Model Explainability](https://www.analyticsvidhya.com/blog/2021/11/model-explainability/)
- [Model Explainability](https://www.ibm.com/topics/explainable-ai)
- [Model Explainability](https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)

## Author: Mihret Agegnehu

- [Flask](https://www.geeksforgeeks.org/flask-tutorial/)
