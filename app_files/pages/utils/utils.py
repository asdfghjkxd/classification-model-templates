import streamlit as st


def init_session_state():
    """Initialises the session state of the app"""

    if 'model_parameters' not in st.session_state:
        st.session_state.model_parameters = {
            'on_error': 'raise',
            'loaded': False,
            'applied': False,
            'submitted': False,
            'ensemble_count': 1,
            'hidden': 1,
            'min_neuron': 1,
            'max_neuron': 1,
            'step': 1,
            'learning_rate': 1e-5,
            'neurons': 1,
            'dropout': 0.800,
            'validation': 0.800,
            'epochs': 1,
            'batch_size': 1,
            'shuffle': True,
            'verbose': 1,
            'patience': 10,
            'persist': True,
            'initial_lr': 1e-5,
            'args': (),
            'kwargs': {}
        }

    if 'data_parameters' not in st.session_state:
        st.session_state.data_parameters = {
            'model_name': None,
            'input': None,
            'loaded': False,
            'applied': False
        }
