![image](https://github.com/Vamshitha211/Predicting-Stock-Market/assets/79069337/ae15d911-0ff2-49f9-82d7-82df4d2d309f)# Predicting Future Stock Prices with LSTM Model

This presentation aims to explore the use of LSTM models in predicting future stock prices to aid investors and traders in making better-informed investment decisions.

**Table of Contents**
1. Introduction
2. Problem Statement
3. Objective
4. Data Understanding
5. Data Preparation
6. Modeling
7. Evaluation
8. Deployment
9. Conclusion

**Introduction**

1. The main purpose of the stock market is to provide a marketplace for the buying, selling, and exchange of securities and stock.
2. Applying the correct stock market prediction methods helps minimize losses and improve entry/exit points.
3. The Long Short-Term Memory (LSTM) model is a popular and successful machine learning model used in finance, particularly for processing sequential data.

**Problem Statement**

Predicting future stock prices to aid in investment decision making.
Investors and traders can making better-informed investment decisions, avoid risk, and potentially improve their overall returns

**Objective**

Objective: Develop a machine learning based solution for more accurate predictions compared to traditional analysis techniques.
Benefits: Better-informed investment decisions, risk management, and potentially improved overall returns.

**Data Understanding**

Using the yfinance library to download market data in a threaded and Pythonic manner.
Obtaining stock data from Yahoo Finance website, a valuable source of financial market data.
Provides a simple and convenient interface for downloading historical data from Yahoo Finance.
Retrieves high-quality data including stock prices, dividends, and financial statements.
Offers high customizability in terms of data parameters and format.

 **Data Preparation**

Clean and preprocess the data to remove any discrepancies or missing values.
Normalize or scale the data to ensure that the model can effectively learn from the features without being affected by large variations in value.
Extract relevant features from the data that could help in predicting future stock prices.

**Modeling**

Design an LSTM neural network architecture, which may consist of one or more LSTM layers, followed by Dense layers and a final output layer.
The output layer will have a linear activation function, as we're predicting a continuous value i.e stock price.
Train the LSTM model using the training dataset by minimizing a suitable loss function (e.g., mean squared error) and optimizing the weights with a suitable optimizer.

Support Vector Machine (SVM)
SVM is a powerful machine learning algorithm used for classification and regression.
The SVM model is used for stock price classification, predicting price increase or decrease.
SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data is not linearly separable.
The scikit-learn library's SVM implementation was used, scikit-learn is a widely used open-source ML library in Python.

**Evaluation**

Evaluate the model's performance using appropriate metrics, such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE), to measure how well the model is predicting the stock prices on the validation set.
The accuracy score is currently used to evaluate the SVM model, calculated as the average of the scores obtained on each fold.
Precision, recall, and F1 score are mentioned as alternative evaluation metrics.
These metrics assess the performance of classification models in terms of true positive/negative and false positive/negative predictions.

**Deployment**

Deploy the application on cloud platforms such as AWS.
After finalizing the model, a RESTful API or web service was created using a suitable framework like Flask.
The purpose of this API is to serve as the interface between the users and the LSTM model, allowing users to send requests to the model and receive predictions back.
Once the API was created, it will need to be deployed on a cloud platform such as AWS to ensure accessibility and scalability.

**Conclusion**

Distributed machine learning can help us make better predictions in the stock market by leveraging the power of parallel computing.
Our approach demonstrated how we can use technical indicators and lagged features to improve the prediction performance in a distributed environment.
The results showed that distributed gradient boosting outperformed other models in predicting stock prices for AAPL.
However, the prediction accuracy is subject to various factors, and investors should use caution when making investment decisions based on machine learning predictions.







