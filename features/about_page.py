from features.base_page import BasePage
import streamlit as st

class AboutPage(BasePage):
    def __init__(self):
        super().__init__("About")

    def render(self):
        st.title("About")
        st.markdown("""
        ### Project Statement
        The project aims to leverage artificial intelligence to diagnose diseases in rice and pulse crops. 
        By analyzing images of leaves, the system predicts the type of disease affecting the plant based on a trained machine-learning model.
        
        ### Developed By
        - [Your Name/Team Name]
        
        ### Tech Stack
        - **Python**
        - **Streamlit**
        - **TensorFlow/Keras**
        """)
