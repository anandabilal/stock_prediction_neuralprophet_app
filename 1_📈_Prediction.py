import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from neuralprophet import NeuralProphet
from plotly import graph_objs as go


start_date = '2017-01-01'
end_date = '2023-01-01'
train_end_date = '2022-01-01'
window_size = 1
horizon = 1


@st.cache_data
def create_dataset(ticker):
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    df = pd.DataFrame()
    df['Date'] = dates
    stock_df = yf.download(ticker, start=start_date, end=end_date)
    stock_df = stock_df.reset_index()
    stock_df = stock_df[["Date", "Close"]]
    merged_df = pd.merge(df, stock_df, on='Date', how='left')
    filled_df = merged_df.copy()
    filled_df = merged_df.fillna(method='ffill')
    filled_df = filled_df.rename(columns={'Date': 'ds', 'Close': 'y'})
    return filled_df


@st.cache_resource
def load_model(ticker):
    with open(f'model/{ticker}.pkl', 'rb') as file:
        model = pickle.load(file)
        model.restore_trainer()
    return model


@st.cache_data
def split_data(df, start_date, end_date, train_end_date):
    train_df = df[(df['ds'] >= start_date) & (df['ds'] < train_end_date)]
    test_df = df[(df['ds'] >= train_end_date) & (df['ds'] <= end_date)]
    return train_df, test_df


def predict_period(model, train_df, test_df, p):
    last_x_train_df = train_df.tail(window_size)
    test_df = pd.concat([last_x_train_df, test_df], ignore_index=True)
    test_p_df = test_df[:p + window_size]
    forecast = model.predict(test_p_df)
    forecast = forecast.drop(forecast.index[:window_size])
    dates = np.asarray(forecast['ds']).astype('datetime64')
    y_true = np.asarray(forecast['y']).astype('float32')
    y_pred = np.asarray(forecast['yhat1']).astype('float32')
    return dates, y_true, y_pred


def calculate_metric(y_true, y_pred):
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mape = mape.numpy().item()
    return mape


def plot_graph(dates, y_true, y_pred):
    layout = go.Layout(
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (Rp.)'),
        legend=dict(orientation="h"),
        height=600
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=dates, y=y_true, name='Price', line=dict(color='#146aff')))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name='Pred. Price', line=dict(color='#fc9753')))
    fig.layout.update(
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Prediction - Stock Prediction with NeuralProphet",
        page_icon="📈",
        layout="wide"
    )
    with st.sidebar:
        st.caption("**Stock Prediction with NeuralProphet**")
    st.title("📈Prediction", help=f'The models have been trained and tested by data from the corresponding company. The stock prices data was taken from Yahoo Finance, from {start_date} to {end_date}.')
    st.divider()

    available_companies = {
        'PT. Bank Rakyat Indonesia (Persero) Tbk (BBRI)': 'BBRI.JK',
        'PT. Bank Central Asia Tbk (BBCA)': 'BBCA.JK',
        'PT. Astra Agro Lestari Tbk (AALI)': 'AALI.JK',
        'PT. Bumi Resources Tbk (BUMI)': 'BUMI.JK',
        'PT. Bank Mega Tbk (MEGA)': 'MEGA.JK'
    }
    selected_company = st.selectbox('Select a company:', options=list(available_companies.keys()), )
    selected_ticker = available_companies[selected_company]

    n_period = st.slider("Number of days to predict:", min_value=5, max_value=130, value=20, help="Number of days to predict from testing data's first date in 2022.")

    button_pressed = st.button("Start Prediction", use_container_width=True)

    if button_pressed:
        data = create_dataset(selected_ticker)
        model = load_model(selected_ticker)
        train_df, test_df = split_data(data, start_date, end_date, train_end_date)
        dates, y_true, y_pred = predict_period(model, train_df, test_df, p=n_period)
        st.subheader(f'Accuracy Metric for {selected_ticker.replace(".JK", "")}', help="These metrics are used to assess the accuracy of a prediction or model.")
        mape = calculate_metric(y_true, y_pred)
        st.markdown("**MAPE**", help="Mean Average Percentage Error (MAPE) calculates the absolute percentage difference for each prediction and takes the average of these differences, providing a measure of the average percentage error in the predictions.")
        formatted_mape = "{:.5f}".format(mape)
        st.code(f"{formatted_mape} %")
        st.subheader(f'Stock Price Prediction for {selected_ticker.replace(".JK", "")}')
        plot_graph(dates, y_true, y_pred)


if __name__ == "__main__":
    main()
