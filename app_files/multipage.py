"""
Generates a helper class to assist with the generation of multiple Streamlit apps

This file also allows users to set the app config for the app
"""

# IMPORT STREAMLIT
import streamlit as st
from app_files.pages import data_page, model_training_page, model_optimization_page, model_prediction_page


# DEFINE THE MULTIPAGE CLASS TO MANAGE THE APPS
class MultiPage:
    """
    Combines and manages the different modules within the streamlit application
    """

    def __init__(self) -> None:
        """Constructor to generate a list which will store all our applications as an instance variable."""

        # SAVE FUNCTIONS TO SESSION STATE TO PRESERVE FUNCTIONS ACROSS RERUNS
        if 'pages' not in st.session_state:
            st.session_state.pages = [{'title': 'Data Processing', 'function': data_page.page},
                                      {'title': 'Model Builder and Trainer', 'function': model_training_page.page},
                                      {'title': 'Model Optimizer', 'function': model_optimization_page.page},
                                      {'title': 'Model Prediction', 'function': model_prediction_page.page}]

        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model' not in st.session_state:
            st.session_state.model = None

    @staticmethod
    def run():
        """
        Dropdown menu to select the page to run
        """

        with st.sidebar.container():
            st.markdown('# Model Builder\n'
                        'App interface for processing data and building ML models.')

            # DEFINE THE TITLE AND CONTENTS OF THE MAIN PAGE
            st.markdown('## App Modules\n'
                        'Select the following available modules:')

        # PAGE SELECTOR
        page = st.sidebar.selectbox('Functions',
                                    st.session_state.pages,
                                    format_func=lambda page: page['title'])

        # RUN THE APP
        try:
            page['function']()
        except ValueError:
            page['function']()
