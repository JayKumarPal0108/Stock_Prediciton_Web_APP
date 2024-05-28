import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
from lightweight_charts import Chart
import pandas_ta as ta
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
import requests
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

st.markdown("<h1 style='text-align: center;'>Stock Dashboard</h1>", unsafe_allow_html=True)

import streamlit as st

# Sidebar for personal information
st.sidebar.title("About Me")

# Personal information
st.sidebar.header("Jaykumar Pal")
st.sidebar.image(r"jay.jpg", caption='Jaykumar Pal', use_column_width=True)

st.sidebar.subheader("Contact Information")
st.sidebar.markdown("📞 Contact No: +91 9768414748")
st.sidebar.write("📧 Email: jaykumarpal10125@gmail.com")

st.sidebar.subheader("Connect with me")
st.sidebar.markdown("[GitHub](https://github.com/JayKumarPal0108)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/jay-kumar-pal-3a7522248/)")

st.sidebar.subheader("About")
st.sidebar.info("""
Hi, I'm Jaykumar Pal. I am an aspiring Data Scientist currently in my final year of studies. 
My goal is to become a successful Data Scientist and continuously learn and grow in this exciting field.
I am passionate about analyzing data, uncovering insights, and applying machine learning techniques to solve real-world problems.
Feel free to reach out to me via the contact information provided.
""")


ticker = st.text_input('Ticker')

start_date = st.date_input('Start Date',value = pd.to_datetime('2024-01-01'))
end_date = st.date_input('End Date', value = pd.to_datetime('today'))

data = yf.download(ticker, start=start_date, end=end_date)
data.index = pd.to_datetime(data.index)

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=False)
    return data

data_load_state = st.text("Load data...")
data = load_data(ticker)
data_load_state.text("Loading data...done!")

tickerData = yf.Ticker(ticker)

Summary, pricing_data, Forecast, Moving_Averages, Model_Prediction, fundamental_data, news = st.tabs(["Summary", "Pricing Data", "Forecast", "Moving_Averages", "Model_Prediction", "Fundamental Data", "Top 10 News"])

# Summary 
with Summary:
    # Initial UI

    buttonClicked = st.button('Set')

# Callbacks
    if buttonClicked:
        try:
            info = tickerData.info

            st.header("Profile")
            st.metric("Sector", info.get("sector", "N/A"))
            st.metric("Industry", info.get("industry", "N/A"))
            st.metric("Website", info.get("website", "N/A"))
            st.metric("Market Cap", info.get("marketCap", "N/A"))

        except Exception as e:
            st.error(f"Error fetching data: {e}")


    # Display company logo
    logo_url = tickerData.info.get('logo_url')
    if logo_url:
        st.markdown(f'<img src="{logo_url}" alt="Company Logo">', unsafe_allow_html=True)
    else:
        st.markdown("No company logo available")
        # Get ticker data from yfinance
        tickerData = yf.Ticker(ticker) 

    # Display company name
    string_name = tickerData.info.get('longName', 'Company Name Unavailable')
    st.header('**%s**' % string_name)

    # Display company summary
    string_summary = tickerData.info.get('longBusinessSummary', 'Business Summary Unavailable')
    st.info(string_summary)

# Price Movement
with pricing_data:
    st.header('Price Movement')

    # Combined Opening and Closing Price Chart
    st.subheader('Opening and Closing Price Chart')
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Closing Price'))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Opening Price'))
    fig_price.update_layout(title=f'{ticker} Opening and Closing Price')
    st.plotly_chart(fig_price)

    # High and Low Price Chart (Merged)
    st.subheader('High and Low Price Chart')
    fig_high_low_price = go.Figure()
    fig_high_low_price.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High'))
    fig_high_low_price.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low'))
    fig_high_low_price.update_layout(title=f'{ticker} High and Low Price')
    st.plotly_chart(fig_high_low_price)

    # All Four Prices Chart
    st.subheader('Prices Chart')
    fig_all_prices = go.Figure()
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High'))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low'))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open'))
    fig_all_prices.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Close'))
    fig_all_prices.update_layout(title=f'{ticker} All Prices', xaxis=dict(showgrid=True))
    st.plotly_chart(fig_all_prices)

    
    # Candlestick Chart
    st.header('Candlestick Chart')
    fig_candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Adj Close'])])
    fig_candlestick.update_layout(title=f'{ticker} Candlestick Chart')
    st.plotly_chart(fig_candlestick)

    # Pricing Data
    st.subheader('Pricing Data')
    st.write(data.sort_values(by='Date', ascending=False))

    # Calculations
    data['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1)
    data.dropna(inplace=True)
    annual_return = data['% Change'].mean() * 252 * 100
    st.write('Annual Return:', annual_return, '%')
    stdev = np.std(data['% Change']) * np.sqrt(252)
    st.write('Standard Deviation:', stdev * 100, '%')
    st.write('Risk Adj. Return:', annual_return / (stdev * 100))

with Forecast:

    starting = "2015-01-01"
    today = date.today().strftime("%Y-%m-%d")

    n_years = 1
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        forecast_data = yf.download(ticker, starting, today)
        forecast_data.reset_index(inplace=True)
        return forecast_data

    data_load_state = st.text("Load data...")
    forecast_data = load_data(ticker)
    data_load_state.text("Loading data...done!")

    st.subheader('Raw data')
    st.write(forecast_data.sort_values(by = ['Date'], ascending = False))

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Open'], name='Stock_Open'))
        fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data['Close'], name='Stock_Close'))
        fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Forecast

    df_train = forecast_data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write('Forecast data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    
with Moving_Averages:

    end = datetime.now()
    start = datetime(end.year-20, end.month, end.day)

    newdata = yf.download(ticker, start= start, end= end)

    def plot_graph(figsize, values, column_name):
        plt.figure()
        values.plot(figsize = figsize)
        plt.xlabel('Years')
        plt.ylabel(column_name)
        plt.title(f"{column_name} of {ticker}")
        plt.show()
        st.pyplot(plt.gcf())  # Use st.pyplot to show the matplotlib figure in Streamlit    

    for column in newdata.columns:
        plot_graph((15,5), newdata[column], column)

    newdata['MA_for_250_days'] = newdata['Adj Close'].rolling(250).mean()
    plot_graph((15,5), newdata[['Adj Close', 'MA_for_250_days']], 'MA_for_250_days')

    newdata['MA_for_100_days'] = newdata['Adj Close'].rolling(100).mean()
    plot_graph((15,5), newdata[['Adj Close', 'MA_for_100_days']], 'MA_for_100_days')

    plot_graph((15,5), newdata[['Adj Close', 'MA_for_100_days', 'MA_for_250_days']], 'MA')

    newdata['percentage_change_cp'] = newdata['Adj Close'].pct_change()
    plot_graph((15,5),newdata['percentage_change_cp'], 'Percentage_Change')
    Adj_close_price = newdata[['Adj Close']]

with Model_Prediction:
    if st.button('Predict'):

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(Adj_close_price)

        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)
        splitting_len = int(len(x_data)*0.8)
        x_train = x_data[:splitting_len]
        y_train = y_data[:splitting_len]

        x_test = x_data[splitting_len:]
        y_test = y_data[splitting_len:]
        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Input
        
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], 1)))  # Input layer with shape
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=2)
        
        predictions = model.predict(x_test)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_test)
        ploting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_predictions.reshape(-1)
        },
            index = newdata.index[splitting_len+100:]
        )
        st.subheader('PLotting Data')
        st.write(ploting_data)

        plot_graph((15,6), ploting_data, 'test data')
        plot_graph((15,6), pd.concat([Adj_close_price[:splitting_len+100], ploting_data], axis=0), 'Whole Data')
        rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions)) 
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
        # Calculate R-squared score
        r2 = r2_score(inv_y_test, inv_predictions)
        st.write(f'R-squared (R2) score: {r2}')

with fundamental_data:
    st.header('Fundamental Data')
    key = 'PJWPPYH1QF8YECF9'
    fd = FundamentalData(key, output_format='pandas')

    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    balance_sheet.columns = balance_sheet.columns.str.replace(' ', '_')
    st.write(balance_sheet)

    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    income_statement.columns = income_statement.columns.str.replace(' ', '_')
    st.write(income_statement)

    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cash_flow.columns = cash_flow.columns.str.replace(' ', '_')
    st.write(cash_flow)

with news:
    st.header('News')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment: {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment: {news_sentiment}')



