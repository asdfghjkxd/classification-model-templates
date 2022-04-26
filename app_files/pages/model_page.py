import pandas as pd
import streamlit as st

from streamlit_tags import st_tags
from ..data_classes.model_validator import *


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
def page():
    # add flags to session state
    if 'input' not in st.session_state:
        st.session_state.input = None
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    if 'applied' not in st.session_state:
        st.session_state.applied = False

    st.title('Model Builder')
    st.markdown('This module allows you to build and train models using data that is processed or loaded in '
                'the Data Processing page.')
    if st.session_state.input is None:
        st.warning('No Model Data is detected. Model Training cannot occur!')
    elif isinstance(st.session_state.input, pd.DataFrame) and st.session_state.input.is_processed():
        st.info('Model Data is detected!')

    st.subheader('Model Tasks')
    st.markdown('Choose one of the following options to perform one of 4 tasks: \n\n'
                '* **Build:** Build a model using specification from the user\n'
                '* **Optimize:** Optimize a model using specification from the user\n'
                '* **Train:** Train the built model using processed data\n'
                '* **Predict:** Conduct predictions on the trained model')
    # TODO: fix duplicate widget id error
    st.session_state.task = st.multiselect('Choose a task to perform',
                                           options=['Build', 'Optimize', 'Train', 'Predict'], key=f'thing')

    if 'Build' in st.session_state.task:
        st.markdown('### Model Builder')
        st.markdown('Build a string classification model using parameters you define.')
        st.markdown('#### Model Parameters')

        with st.form('Input'):
            hidden = st.number_input('Number of hidden layers',
                                     min_value=0,
                                     max_value=99999,
                                     step=1,
                                     value=1,
                                     key='hidden')
            neurons = st_tags(label='Number of neurons per layer')
            dropout = st.number_input('Level of dropout',
                                      min_value=0.,
                                      max_value=1.,
                                      step=1e-3,
                                      format='%.3f',
                                      key='dropout')
            validation = st.number_input('Train-test-validation split',
                                         min_value=0.,
                                         max_value=1.,
                                         step=1e-3,
                                         format='%.3f',
                                         key='validation')
            model_type = st.selectbox('Model Type', options=['Simple', 'RNN', 'BiLSTM'], key='model_type')
            epochs = st.number_input('Number of epochs to train',
                                     min_value=1,
                                     max_value=99999,
                                     step=1,
                                     value=1,
                                     key='epochs')
            batch_size = st.number_input('Number of batch size',
                                         min_value=1,
                                         max_value=99999,
                                         step=1,
                                         value=1,
                                         key='batch_size')
            shuffle = st.checkbox('Shuffle dataset per training cycle?', key='shuffle')
            verbose = st.selectbox('Select level of debugging for training', options=[1, 2, 3], key='verbose')
            patience = st.number_input('Number of epochs to continue training when there are no improvements '
                                       'to measured statistics',
                                       min_value=1,
                                       max_value=99999,
                                       step=1,
                                       value=1,
                                       key='patience')
            if st.form_submit_button('Submit'):
                st.info('Data')
                st.session_state.model = ModelTrainer()
                st.session_state.model.instantiate(hidden_layers=hidden,
                                                   neurons_per_layer=neurons,
                                                   dropout_threshold=dropout,
                                                   validation_split=validation,
                                                   model_type=model_type,
                                                   epochs=epochs,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   verbose=verbose,
                                                   patience=patience)
    elif 'Optimize' in st.session_state.task:
        st.markdown('### Model Optimizer')
    elif 'Train' in st.session_state.task:
        st.markdown('### Model Trainer')
    elif 'Predict' in st.session_state.task:
        st.markdown('### Model Predictor')
