import streamlit as st

class AuthManager:
    def __init__(self):
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False

    def is_logged_in(self):
        return st.session_state['logged_in']

    def login(self, username, password):
        # Simple hardcoded credentials as per original code
        if username == "admin" and password == "admin":
            st.session_state['logged_in'] = True
            return True
        return False

    def logout(self):
        st.session_state['logged_in'] = False
