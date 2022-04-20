import streamlit as st


def page():
    st.title('Model Builder')
    st.markdown('This module allows you to build and train models using data that is processed or loaded in '
                'the Data Processing page.')
    if st.session_state.data is None:
        st.warning('No Model Data is detected. Model Training cannot occur!')
    elif isinstance(st.session_state.data, int):
        # TODO LINK WITH MODELDATA CLASS
        st.info('Model Data is detected!')
