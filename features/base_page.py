from abc import ABC, abstractmethod
import streamlit as st

class BasePage(ABC):
    def __init__(self, title):
        self.title = title

    @abstractmethod
    def render(self):
        """Renders the page content."""
        pass
