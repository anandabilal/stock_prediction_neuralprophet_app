import streamlit as st


def main():
    st.set_page_config(
        page_title="Contact - Stock Prediction with NeuralProphet",
        page_icon="☎️",
        layout="wide"
    )
    with st.sidebar:
        st.caption("**Stock Prediction with NeuralProphet**")
    st.title("☎️Contact")
    st.divider()

    col_a, col_b, col_c, = st.columns(3)
    with col_a:
        st.subheader("🌐Social")
        st.markdown("""
            - <a href="mailto:ananda.bilal@binus.ac.id">Email</a>
            - <a href="https://github.com/anandabilal">GitHub</a>
            - <a href="https://www.linkedin.com/in/ananda-bilal-5a5a77192/">LinkedIn</a>
        """,
        unsafe_allow_html=True)

    with col_b:
        st.subheader("🙋‍♂️Ananda Bilal")
        st.caption("_2301883725_ - _ananda.bilal@binus.ac.id_")
        st.write("Currently studying at BINUS University, majoring in Computer Science, specifically in Software Development. I am a passionate developer interested in problem-solving, creating projects that test my skills, and working together to reach a common goal.")

    with col_c:
        st.subheader("❓Why was this application made?")
        paper_title = "Development of Stock Price Prediction System with NeuralProphet's Combination of Deep Learning and Statistical Approach"
        st.write(f"This application was made by as part of the thesis paper titled '_{paper_title}_' needed to graduate from BINUS University. The paper discussed the effectiveness of a model that uses a combination of statistical approach and neural network: NeuralProphet, on predicting stock prices. Other method such as Naive, and basic neural network model like Artificial Neural Network (ANN) were used as a comparison against NeuralProphet.")


if __name__ == "__main__":
    main()