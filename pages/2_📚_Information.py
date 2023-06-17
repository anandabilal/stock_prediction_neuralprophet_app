import streamlit as st


def main():
    st.set_page_config(
        page_title="Information - Stock Prediction with NeuralProphet",
        page_icon="📚",
        layout="wide"
    )
    with st.sidebar:
        st.caption("**Stock Prediction with NeuralProphet**")
    st.title("📚Information")

    neuralprophet_tab, background_tab, research_tab = st.tabs([
        "What is NeuralProphet?",
        "Background",
        "Research Method"
    ])

    with neuralprophet_tab:
        # Background info about stocks and investment
        with st.expander(f"💡**Terms & Explanations**", expanded=False):
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("💼Stock")
                st.write("Stocks represent ownership shares in a company. When you buy stocks, you become a shareholder or part-owner of that company. Companies issue stocks to raise capital for various purposes, such as expanding their business or funding new projects. Each stock you own represents a portion of the company's assets and earnings.")

                st.subheader("💰Investment")
                st.write("Investing refers to the act of using your money to purchase assets, such as stocks, with the expectation of generating a profit over time. By investing in stocks, you are essentially betting on the future success of the company. If the company does well and its value increases, the price of the stocks you own can rise, allowing you to sell them at a higher price and make a profit. On the other hand, if the company performs poorly, the stock prices may decrease, resulting in a loss if you sell them at a lower price.")

                st.subheader("🔤Ticker")
                st.write("A ticker is a unique combination of letters (and sometimes numbers) that represents a specific company's stock on the stock market. Tickers are used to identify and track the performance of individual stocks. For example, the ticker symbol for PT. Bank Central Asia Tbk is BBCA.")

            with col_b:
                st.subheader("📅Time Series")
                st.write("Time series is a series of data sorted by time such as seconds, minutes, hours, days, weeks, months and years. Time series shows changes in data over time. Time series analysis can be performed to identify patterns, trends, cycles and fluctuations in data over time.")

                st.subheader("🤖Neural Network")
                st.write("Neural network or is a type of computer program that can learn and recognize patterns in data. This program is designed and created by imitating the structure of the human brain, by having nodes or neurons connected to each other to process information. In a neural network, data is entered into a program and processed through layers (layers) of neurons. Each neuron receives input, performs calculations on that input, and then sends the results to the next layer, until the result last layer, the output layer, is given.")

                st.subheader("🧠Deep Learning")
                st.write("Deep learning is a subset of machine learning that utilizes neural network architecture to learn and extract patterns and representations from data. Deep learning models are inspired by the structure and functioning of the human brain, where each layer of neurons process and learns complex features.")

        st.subheader("💻NeuralProphet", help="Credit: https://github.com/ourownstory/neural_prophet")
        st.write("NeuralProphet is an easy to learn framework for interpretable time series forecasting. NeuralProphet is built on PyTorch and combines Neural Network and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net. This framework is available to be used in Python as an installable library. NeuralProphet consists of several components, which are: autoregression, trend, seasonality, lagged and future regressor, and events.")
        st.image("img/neuralprophet.png", caption="Credit: https://neuralprophet.com/", use_column_width=True)

        st.subheader("❓Why NeuralProphet?")
        st.write("NeuralProphet was used as the model to analyze in this application and research paper because it is:")
        st.markdown("- User-friendly and powerful Python package\n- Automatic selection of training related hyperparameters\n- Simple to build and customize your model\n")

    with background_tab:
        st.subheader("🔍Background")
        st.write("Before machine learning algorithms for artificial intelligence (AI) were developed, humans analyzed the results of the data produced directly, but as technology developed, so did the amount of data produced. Machine learning can help because this method can learn the differences or changes in data, detect patterns created by the data, and create the most suitable solution based on the analysis of the data.")
        st.write("Machine learning is a subset of AI and computer science that focuses on using data and algorithms that aim to imitate the way humans learn, improving their accuracy with each iteration.")
        st.write("With this technology, of course, many companies in many fields want to use machine learning in their products. One example of a field that can take advantage of this machine learning method is the financial sector, more specifically in predicting company stock prices.")
        st.write("AIs that are trained using machine learning methods that can predict stock prices is valuable for companies investing in other companies' stocks. This machine learning method can train an AI in providing predictions about if certain stock prices will fall so that companies can stop investing in stocks there, or make new investments in other stocks because machine learning predicts that these stocks will experience price increases. This makes the investment process less risky because decisions are taken objectively. With these predictions, companies can increase their profits and reduce their losses.")

        st.subheader("🎯Objectives")
        st.write("This application was made as one of the objective of a research paper, the full objective list of the paper are as follow:")
        st.markdown("- Create a system of web application that can predict stock prices of a company using NeuralProphet\n- Evaluate the performance of that NeuralProphet model in predicting stock prices using accuracy metric: mean average percentage error (MAPE)\n- Evaluate the interface of the web application through user feedback")
    
    with research_tab:
        st.subheader("🔬Research Method")
        st.write("The research method that will be used is the design of an application system that can use the NeuralProphet model that has been trained by being given a stock price data set, and using this model to predict stock prices over several periods to determine its effectiveness in predicting each of these periods.")
        st.write("More specifically, a Python-based web application system where users can choose from available companies and make predictions using the NeuralProphet model for each period.")
        st.write("After obtaining stock price predictions for the selected period, the author will compare them with actual stock prices, and assess how accurate this model is in predicting stock prices using the mean absolute percentage error (MAPE) accuracy metric. After obtaining these metrics, the NeuralProphet model will be compared with the performance of the Naïve method and the Artificial Neural Network (ANN) model as benchmarks. And finally, draw conclusions based on the results of the comparison of these metrics.")


if __name__ == "__main__":
    main()