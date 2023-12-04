from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

import matplotlib
matplotlib.use('Agg')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime

app = Flask(__name__, template_folder='templates/')

def fig_to_uri(fig):
    """Convert a Matplotlib figure to a URI that can be displayed in a template"""
    # Save the figure to a PNG in memory.
    png = BytesIO()
    fig.savefig(png, format='png')
    # Encoding the PNG as base64
    pngB64String = "data:image/png;base64,"
    pngB64String += base64.b64encode(png.getvalue()).decode('utf8')
    return pngB64String

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        # The tech stocks we'll use for this analysis
        tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NVDA', 'ADBE', 'CRM', 'ACN', 'CSCO', 'MA']

        end = datetime.now()

        # Accept user input for start date
        start_input = request.form['start_date']
        start = datetime.strptime(start_input, "%Y-%m-%d")

        # Accept user input for end date
        end_input = request.form['end_date']
        end = datetime.strptime(end_input, "%Y-%m-%d")

        # Accept user input for company name
        company_name = request.form['company_name'].upper()

        # Validate user input for company name
        while company_name not in ['APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'NVIDIA', 'ADOBE', 'SALESFORCE', 'ACCENTURE', 'CISCO', 'MASTERCARD']:
            error_msg = "Invalid company name. Please choose from the following list: APPLE, GOOGLE, MICROSOFT, AMAZON, NVIDIA, ADOBE, SALESFORCE, ACCENTURE, CISCO, MASTERCARD"
            return render_template('error.html', error_msg=error_msg)

        # Set up End and Start times for data grab
        for stock in tech_list:
            globals()[stock] = yf.download(stock, start, end)

        company_list = [AAPL, GOOG, MSFT, AMZN, NVDA, ADBE, CRM, ACN, CSCO, MA]
        company_names = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "NVIDIA", "ADOBE", "SALESFORCE", "ACCENTURE", "CISCO","MASTERCARD"]

        for company, com_name in zip(company_list, company_names):
            company["company_name"] = com_name

        # Filter the DataFrame based on the selected company name
        selected_company_df = pd.concat(company_list, axis=0)
        selected_company_df = selected_company_df[selected_company_df['company_name'] == company_name]
        #Get the stock quote
        df = selected_company_df

        # #  Create a new dataframe with only the 'Close column 
        data = df.filter(['Close'])
        # # Convert the dataframe to a numpy array
        dataset = data.values
     
        dataset1 = np.array(dataset)
       

        # # # Reshape the data
        # dataset2= np.reshape(dataset1, (dataset1.shape[0], dataset1.shape[1], 1 ))
        # print(dataset2)
        # print(type(dataset2))

        model = pickle.load(open('model.pkl', 'rb'))

        # # Get the models predicted price values 
        predictions = model.predict(dataset1)
        predictions = scaler.inverse_transform(predictions)


       

        
        # # Show the valid and predicted prices
        
        daily_returns = selected_company_df['Close'].pct_change()

        # Plot the daily return on average
        fig1, ax1 = plt.subplots()
        ax1.plot(daily_returns.index, daily_returns)
        ax1.set(title='Daily Return of {} Stock on Average'.format(company_name), xlabel='Date', ylabel='Daily Return')
        # Convert the Matplotlib figure to a URI
        plot_uri1 = fig_to_uri(fig1)

        # Plot the closing price
        fig2, ax2 = plt.subplots()
        ax2.plot(selected_company_df.index, selected_company_df['Close'])
        ax2.set(title='Closing Price of {} from {} to {}'.format(company_name, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), xlabel='Date', ylabel='Closing Price')

        # Convert the Matplotlib figure to a URI
        plot_uri2 = fig_to_uri(fig2)

        return render_template('result.html', plot_url1=plot_uri1, plot_url2=plot_uri2,predictions = predictions)


    
if(__name__)== '__main__':
    app.run(debug=True)
