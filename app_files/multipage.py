"""
Generates a helper class to assist with the generation of multiple Streamlit apps

This file also allows users to set the app config for the app
"""

# IMPORT STREAMLIT
import streamlit as st
import app_files.pages.model_page as model_page
import app_files.pages.data_page as data_page


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
                                      {'title': 'Model Builder', 'function': model_page.page}]

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
