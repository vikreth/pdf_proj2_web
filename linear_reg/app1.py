from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set up page
st.set_page_config(page_title='Riel Price Prediction by using Linear regression', page_icon=':chart_with_upwards_trend:', layout='wide')

# Load data
data = pd.read_csv('data1.csv', index_col='Date')
target = data['Price']
features = data[['Open', 'High', 'Low']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Define function for prediction
def get_prediction(nday):
    # Use the last nday rows of data as features
    X_predict = data[['Open', 'High', 'Low']].tail(nday)
    # Standardize the features
    X_predict = scaler.transform(X_predict)
    # Predict the next nday days of data
    y_predict = lr.predict(X_predict)
    # Create a pandas dataframe of the predicted stock prices
    dates = []
    for i in range(nday):
        next_date = datetime.strptime(data.index[-1], '%Y-%m-%d') + timedelta(days=i+1)
        dates.append(next_date.strftime('%Y-%m-%d'))
    pred_df = pd.DataFrame({'Date': dates, 'Predicted Price': y_predict})
    return pred_df

# Define Streamlit app
def app():
    # Set page title and icon
    st.title('Riel Price Prediction')
    st.markdown(':chart_with_upwards_trend:')

    # Get user input for number of days to predict
    nday = st.number_input('Enter number of days to predict:', min_value=1, max_value=365, value=30)

    # Predict and display table and plot
    st.write(f'Predicted stock prices for the next {nday} days starting from 2022-06-06:')
    pred_df = get_prediction(nday)
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    pred_df.set_index('Date', inplace=True)
    st.table(pred_df)
    st.write('')

    # Create line plot of predicted stock prices
    plt.figure(figsize=(16,6))
    plt.plot(pred_df, color='blue')
    plt.title('Predicted Tesla Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid(True)
    st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    app()