import random
import pandas as pd
import streamlit as st
import json

from streamlit_ace import st_ace
from streamlit_tags import st_tags
from ..data_classes.model_validator import *


def page():
    # add flags to session state
    if 'input' not in st.session_state:
        st.session_state.input = None
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    if 'applied' not in st.session_state:
        st.session_state.applied = False
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    st.title('Model Builder')
    st.markdown('This module allows you to build and train models using data that is processed or loaded in '
                'the Data Processing page.')
    if st.session_state.input is None or not st.session_state.input.is_processed():
        st.warning('No Model Data is detected. Model Training cannot occur!')
    elif isinstance(st.session_state.input, pd.DataFrame) and st.session_state.input.is_processed():
        st.info('Model Data is detected!')

    st.subheader('Model Tasks')
    st.markdown('Choose one of the following options to perform one of 4 tasks on the sidebar: \n\n'
                '* **Build:** Build a model using specification from the user\n'
                '* **Optimize:** Optimize a model using specification from the user\n'
                '* **Train:** Train the built model using processed data\n'
                '* **Predict:** Conduct predictions on the trained model')
    task = st.selectbox('Choose task to do', options=['Build', 'Optimize', 'Train', 'Predict'], key='task_in')

    if task == 'Build':
        st.markdown('### Model Builder')
        st.markdown('Build a string classification model using parameters you define. Make sure that ')
        st.markdown('#### Model Parameters')
        hidden = st.number_input('Number of hidden layers',
                                 min_value=0,
                                 max_value=99999,
                                 step=1,
                                 value=1,
                                 key=f'hidden{random.random()}')
        neurons = st.number_input('Number of neurons per layers',
                                  min_value=0,
                                  max_value=999999,
                                  step=1,
                                  value=1,
                                  key=f'neurons{random.random()}')
        dropout = st.number_input('Level of dropout',
                                  min_value=0.,
                                  max_value=1.,
                                  step=1e-3,
                                  format='%.3f',
                                  key=f'dropout{random.random()}')
        validation = st.number_input('Train-test-validation split',
                                     min_value=0.,
                                     max_value=1.,
                                     step=1e-3,
                                     format='%.3f',
                                     key=f'validation{random.random()}')
        model_type = st.selectbox('Model Type', options=['Simple', 'RNN', 'BiLSTM'], key=f'model_type{random.random()}')
        epochs = st.number_input('Number of epochs to train',
                                 min_value=1,
                                 max_value=99999,
                                 step=1,
                                 value=1,
                                 key=f'epochs{random.random()}')
        batch_size = st.number_input('Number of batch size',
                                     min_value=1,
                                     max_value=99999,
                                     step=1,
                                     value=1,
                                     key=f'batch_size{random.random()}')
        shuffle = st.checkbox('Shuffle dataset per training cycle?', key=f'shuffle{random.random()}')
        verbose = st.selectbox('Select level of debugging for training', options=[1, 2, 3],
                               key=f'verbose{random.random()}')
        patience = st.number_input('Number of epochs to continue training when there are no improvements '
                                   'to measured statistics',
                                   min_value=1,
                                   max_value=99999,
                                   step=1,
                                   value=1,
                                   key=f'patience{random.random()}')
    elif task == 'Optimize':
        pass
    elif task == 'Train':
        pass
    elif task == 'Predict':
        pass

    st.subheader('Other Parameters')
    st.markdown('The following dropdown box contains some of the other optional parameters you can define '
                'for the reading and parsing of the input data.')
    with st.expander('Define exception handling behaviour, optional positional arguments and keyword arguments'):
        on_error = st.selectbox(label='Choose behaviour on exception', options=['raise', 'ignore', 'default'],
                                help='Specifies the behaviour of the app should exceptions occur')
        args = st_tags(label='Enter in positional arguments to pass into file reader function', key='args')
        st.info(f'**Positional arguments accepted:** {args}')

        kwargs = st_ace(value='{}', language='python', auto_update=False, key='kwarg')
        if kwargs:
            try:
                kwargs_parsed = json.loads(kwargs)
            except Exception as ex:
                st.error(ex)
            else:
                st.info(f'**Keyword arguments accepted:** {kwargs_parsed}')

    # duplicatewidgetID exception resolved by checking errors below
    if st.button('Submit', key='continue'):
        if task == 'Build':
            try:
                st.session_state.model = ModelTrainer(ensemble_count=1,
                                                      model_data=st.session_state.input)
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
            except Exception as ex:
                raise ex
            else:
                st.success('Model build successfully')

        elif task == 'Optimize':
            st.markdown('### Model Optimizer')
            try:
                pass
            except Exception as ex:
                raise ex
            else:
                st.success('Optimization completed')

        elif task == 'Train':
            st.markdown('### Model Trainer')
            try:
                pass
            except Exception as ex:
                raise ex
            else:
                st.success('Model Trained successfully')

        elif task == 'Predict':
            st.markdown('### Model Predictor')
            try:
                pass
            except Exception as ex:
                raise ex
            else:
                st.success('Predictions complete')
