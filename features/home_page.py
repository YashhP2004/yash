from features.base_page import BasePage
import streamlit as st

class HomePage(BasePage):
    def __init__(self):
        super().__init__("Home")

    def render(self):
        st.title("🌿 AI-Driven Disease Detection in Rice and Pulse Crops")
        st.markdown("""
        Welcome to the Automated Disease Detection System!
        
        This application uses Artificial Intelligence to diagnose diseases in rice and pulse crops.
        Simply upload an image of a leaf, and our model will predict the disease type.
        
        ### Features:
        - **Instant Diagnosis**: Get results in seconds.
        - **Easy to Use**: Simple interface for farmers and experts.
        - **Accurate**: Powered by deep learning.
        
        👈 **Select 'Disease Detection' from the sidebar to start!**
        """)
        
        st.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ff?ixlib=rb-1.2.1&auto=format&fit=crop&w=1352&q=80", use_container_width=True)
