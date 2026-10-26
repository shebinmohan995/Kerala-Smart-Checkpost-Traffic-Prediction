Kerala Smart Check Post Traffic Density Prediction

🚀 Live Application

Try the Kerala Smart Checkpost Traffic Density Prediction app online:

"Open Live Prediction App" (https://kerala-smart-checkpost-traffic-prediction-lu2rvgvyyevmum46l72c.streamlit.app/)

📌 Project Overview

The Kerala Smart Check Post Traffic Density Prediction project is a Machine Learning application designed to predict traffic density at check posts in Kerala using vehicle, environmental, road, and traffic-related information.

The project uses a Decision Tree Classifier to classify traffic density into four categories: Low, Medium, High, and Very High. A Streamlit web application provides an interface for entering input details and viewing predictions.

🎯 Objectives

Predict traffic density at Kerala check posts using Machine Learning.
Explore the relationship between vehicle, weather, road, and traffic conditions.
Perform data analysis and visualization to understand traffic patterns.
Build an interactive web application for traffic density prediction.
🛠️ Technologies Used

Programming Language: Python
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Data Visualization: Matplotlib
Web Application: Streamlit
Development Environment: Jupyter Notebook, Visual Studio Code
Model Storage: Pickle
📊 Dataset

The project uses a Kerala government smart check post traffic dataset containing 10,000 records.

The dataset includes features related to:

Vehicle type, brand, fuel type, and age
Check post and district
Date, time, and direction
Weather and road conditions
Temperature, humidity, rainfall, and visibility
Vehicle speed, queue length, and waiting time
Target variable: "Traffic_Density"

The target classes are:

Low
Medium
High
Very High
⚙️ Machine Learning Workflow

Load and inspect the dataset.
Perform data cleaning and exploratory data analysis.
Prepare input features and the target variable.
Encode categorical features using OneHotEncoder.
Split the dataset into training and testing sets.
Train a Decision Tree Classifier.
Evaluate the model using accuracy, a classification report, and a confusion matrix.
Save the trained model and encoder using Pickle.
Develop a Streamlit application to generate predictions.
🤖 Model Performance

The Decision Tree Classifier achieved an accuracy of 97.75% on the test dataset in the recorded evaluation.

Evaluation methods include:

Accuracy Score
Classification Report
Confusion Matrix
Note: Model performance depends on the dataset and evaluation procedure. The reported accuracy should be verified against the final saved model and test results.

🚀 Features

Predict traffic density from input features.
Interactive Streamlit user interface.
Data visualizations for exploring traffic patterns.
District-wise traffic analysis.
Vehicle-type and time-of-day analysis.
Four traffic density prediction categories.
📁 Project Structure

Kerala-Smart-Checkpost-Traffic-Prediction/ │ ├── app.py ├── Project_Kerala_Traffic.ipynb ├── Kerala_Government_Smart_CheckPost_Dataset_10000.csv ├── decision_tree_model.pkl ├── OneHot_Encoder.pkl ├── .gitignore └── README.md

💻 Installation and Setup

Clone the repository
git clone https://github.com/shebinmohan995/Kerala-Smart-Checkpost-Traffic-Prediction.git

Navigate to the project folder
cd Kerala-Smart-Checkpost-Traffic-Prediction

Install the required libraries
pip install pandas numpy scikit-learn matplotlib streamlit

Run the Streamlit application
streamlit run app.py

Open the local URL displayed in the terminal to use the application.

🔮 Future Improvements

Compare multiple Machine Learning algorithms.
Improve model generalization and evaluate performance on unseen data.
Add more interactive traffic visualizations.
Deploy the application online.
Explore real-time traffic data integration, if suitable data sources become available.
👨‍💻 Author

Shebin Mohan

GitHub: "@shebinmohan995" (https://github.com/shebinmohan995)

Developed as a Machine Learning project focused on traffic density prediction at check posts in Kerala.
