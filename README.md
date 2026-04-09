# HR-Decision-Support-System
Developed a decision support system for HR management using machine learning and Power BI with analyzed employee attrition factors and built predictive models
# Overview
This project develops a Decision Support System (DSS) for Human Resource Management to analyze, predict, and optimize employee retention strategies. The system integrates data analytics, machine learning, survival analysis, and optimization techniques to help HR managers make data-driven decisions.

The main objective is to identify factors influencing employee attrition, estimate the probability of employees leaving the company, and recommend optimal retention strategies under budget constraints.

The system consists of three main components:
- Data analytics and machine learning models (Python)
- Interactive HR analytics dashboard (Power BI)
- Web-based decision support interface (Streamlit)

## Key Features
### 1. Descriptive Analytics (HR Dashboard)
An interactive Power BI dashboard provides an overview of employee trends and attrition patterns.

Main insights include:

- Attrition rate by department, age group, and job level
- Relationship between income, overtime, and employee turnover
- HR KPIs for workforce analysis

The dashboard helps HR managers understand current workforce dynamics and potential risk areas.
### 2. Predictive Analytics (Machine Learning Models)

Several classification models were implemented to predict employee attrition:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Gradient Boosting / XGBoost

These models estimate the probability that an employee will leave the organization.
### 3. Survival Analysis

To estimate how long employees are likely to stay, survival analysis techniques were applied:

- Kaplan-Meier survival curves
- Cox Proportional Hazards model

This analysis helps identify critical periods when employees are most likely to leave the company.
### 4. HR Resource Optimization

The project includes an optimization model based on binary linear programming to determine which employees should receive retention investment under budget constraints.

The optimization answers the question: How should HR allocate limited resources to maximize employee retention?
### 5. Web Decision Support Application

A Streamlit web application allows users to interact with the models. Main functions include:

- Employee attrition risk prediction
- Survival probability visualization
- Risk score interpretation
- Budget optimization recommendations

Users can input employee information and receive data-driven decision support in real time.

## Dataset

The project uses the Employee Attrition Prediction Dataset, which simulates a real corporate HR environment including multiple departments, job roles, and employee attributes.

Dataset source: https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset

## Technology Stack

Programming:

- Python

Libraries:

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Survival Analysis
- Streamlit

Tools:
- VS Code
- Google Colab

Visualization:
- Power BI
## Project Structure
## Project Structure

```
hr-decision-support-system
│
├── data
│   └── employee_attrition_dataset.csv      # Raw dataset used for analysis
│
├── notebooks
│   └── data_analysis.ipynb                 # Data preprocessing and exploratory analysis
│
├── models
│   └── trained_models.pkl                  # Saved machine learning models
│
├── app
│   └── app.py                              # Web application for prediction
│
├── dashboard
│   └── HR_dashboard.pbix                   # Power BI dashboard for HR insights
│
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```
## Example Use Cases

The system can support HR managers in:

- Identifying high-risk employees
- Understanding key drivers of employee attrition
- Planning retention strategies
- Allocating HR budgets effectively
