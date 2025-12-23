import streamlit as st
import os
from auth_manager import AuthManager
from features.home_page import HomePage
from features.detection_page import DetectionPage
from features.about_page import AboutPage
from features.page_registry import PageRegistry

# Constants configuration
MODEL_PATH = 'rice_leaf_disease_model.h5'
# Define class names EXACTLY as they appeared in training
CLASS_NAMES = [
    'Bacterial_Leaf_Blight',
    'Brown_Spot',
    'Healthy',
    'Leaf_Blast',
    'Leaf_Scald',
    'Sheath_Blight'
]

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Rice & Pulse Disease Detection",
        page_icon="🌿",
        layout="wide"
    )

    # Initialize Managers
    auth_manager = AuthManager()
    page_registry = PageRegistry()

    # Register Pages (Composition Root Configuration)
    page_registry.register("Home", HomePage())
    page_registry.register("Disease Detection", DetectionPage(MODEL_PATH, CLASS_NAMES))
    page_registry.register("About", AboutPage())

    # Authentication Check
    if not auth_manager.is_logged_in():
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if auth_manager.login(username, password):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Logout"):
        auth_manager.logout()
        st.rerun()
        
    app_mode = st.sidebar.selectbox("Choose the app mode", page_registry.get_titles())

    # Render Active Page
    current_page = page_registry.get_page(app_mode)
    if current_page:
        current_page.render()

if __name__ == "__main__":
    main()
