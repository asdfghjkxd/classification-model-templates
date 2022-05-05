import random
import pandas as pd
import streamlit as st
import json

from .utils.utils import init_session_state
from streamlit_ace import st_ace
from streamlit_tags import st_tags
# from ..data_classes.model_validator import *


def page():
    """Model Builder and Training page"""

    init_session_state()

    st.title('Model Builder')
    st.markdown('This module allows you to build and train models using data that is processed or loaded in '
                'the Data Processing page.')
    if st.session_state.data_parameters['input'] is None or \
            not st.session_state.data_parameters['input'].is_processed():
        st.warning('No Model Data is detected. Model Training cannot occur!')
    elif isinstance(st.session_state.data_parameters['input'], pd.DataFrame) and \
            st.session_state.data_parameters['input'].is_processed():
        st.info('Model Data is detected!')

    st.markdown('### Model Builder\n'
                'Build a string classification model using parameters you define. Ensure that all parameters '
                'are properly defined before proceeding.\n\n'
                '* The **Model Structure** section contains parameters that define the overall structure of '
                'model, such as the number of ensemble models to instantiate, the number of neurons per '
                'Dense layer, and the overall type of model used\n'
                '* The **Training Parameters** section contains parameters that define the behaviour of training '
                'for the model\n'
                '* The **Model Persistence** section contains parameters that allows you to specify whether to '
                'save the model to your disk or to store it in memory only')

    st.markdown('#### Model Structure')
    if st.session_state.data_parameters['input'] is None:
        st.info('No Model Data detected')
    else:
        st.info(f'Model Data detected: **{st.session_state.data_parameters["model_name"]}**')
    st.session_state.model_parameters['ensemble_count'] = st.number_input('Number of models to create',
                                                                          min_value=0,
                                                                          max_value=99999,
                                                                          step=1,
                                                                          value=1,
                                                                          key=f'ensemble_counts')
    st.session_state.model_parameters['hidden'] = st.number_input('Number of hidden layers',
                                                                  min_value=0,
                                                                  max_value=99999,
                                                                  step=1,
                                                                  value=1,
                                                                  key=f'hidden')
    neuron_temp = st_tags(label='Enter in a series of neuronal count per layer; it can be a single integer or a '
                                'Sequence of integers',
                          text='Press Enter to add more neuronal counts...')
    try:
        if len(neuron_temp) == 1:
            st.session_state.model_parameters['neurons'] = int(neuron_temp[0])
            st.info(f'**Number of neurons per layer:** {st.session_state.model_parameters["neurons"]}')
        elif len(neuron_temp) > 1:
            if len(neuron_temp) != st.session_state.model_parameters['hidden']:
                st.error('Number of neurons per layer in Sequence must be equal to number of hidden layers')
            else:
                st.session_state.model_parameters['neurons'] = [int(n) for n in neuron_temp]
                st.info(f'**Number of neurons per layer:** {st.session_state.model_parameters["neurons"]}')
        else:
            st.warning('Number of neurons per layer cannot be blank!')
    except Exception as ex:
        st.error(ex)

    st.session_state.model_parameters['dropout'] = st.number_input('Level of dropout',
                                                                   min_value=0.,
                                                                   max_value=1.,
                                                                   step=1e-3,
                                                                   format='%.3f',
                                                                   value=0.500,
                                                                   key=f'dropout')

    st.markdown('#### Training Parameters')
    st.session_state.model_parameters['shuffle'] = st.checkbox('Shuffle dataset per training cycle?',
                                                               key=f'shuffle')
    st.session_state.model_parameters['validation'] = st.number_input('Train-test-validation split',
                                                                      min_value=0.,
                                                                      max_value=1.,
                                                                      step=1e-3,
                                                                      value=0.800,
                                                                      format='%.3f',
                                                                      key=f'validation')
    st.session_state.model_parameters['epochs'] = st.number_input('Number of epochs to train',
                                                                  min_value=1,
                                                                  max_value=99999,
                                                                  step=1,
                                                                  value=10,
                                                                  key=f'epochs')
    st.session_state.model_parameters['batch_size'] = st.number_input('Number of batch size',
                                                                      min_value=1,
                                                                      max_value=99999,
                                                                      step=1,
                                                                      value=1,
                                                                      key=f'batch_size')
    st.session_state.model_parameters['verbose'] = st.selectbox('Select level of debugging for training',
                                                                options=[1, 2, 3],
                                                                key=f'verbose')
    st.session_state.model_parameters['patience'] = st.number_input('Number of epochs to continue training when '
                                                                    'there are no improvements to measured '
                                                                    'statistics',
                                                                    min_value=1,
                                                                    max_value=99999,
                                                                    step=1,
                                                                    value=10,
                                                                    key=f'patience')
    if st.session_state.data_parameters['model_name'] in ['bert-base-uncased', 'bert-base-cased',
                                                          'bert-large-uncased', 'bert-large-cased']:
        st.session_state.model_parameters['initial_lr'] = st.number_input('Initial Learning Rate for model',
                                                                          step=1e-6,
                                                                          format='%.6f',
                                                                          min_value=1e-7,
                                                                          max_value=1.,
                                                                          value=1e-5)

    st.markdown('#### Model Persistence')
    st.session_state.model_parameters['persist'] = st.checkbox('Persist Model to Disk?', value=True, key='persist')

    st.subheader('Other Parameters')
    st.markdown('The following dropdown box contains some of the other optional parameters you can define '
                'for the reading and parsing of the input data.')
    with st.expander('Define exception handling behaviour, optional positional arguments and keyword arguments'):
        st.session_state.model_parameters['on_error'] = st.selectbox(label='Choose behaviour on exception',
                                                                     options=['raise', 'ignore', 'default'],
                                                                     help='Specifies the behaviour of the app should '
                                                                          'exceptions occur')
        st.session_state.model_parameters['args'] = st_tags(label='Enter in positional arguments to pass into file '
                                                                  'reader function',
                                                            key='args')
        st.info(f'**Positional arguments accepted:** {st.session_state.model_parameters["args"]}')

        st.session_state.model_parameters['kwargs'] = st_ace(value='{}', language='python', auto_update=False,
                                                             key='kwarg')
        if st.session_state.model_parameters['kwargs']:
            try:
                st.session_state.model_parameters['kwargs'] = json.loads(st.session_state.model_parameters['kwargs'])
            except Exception as ex:
                st.error(ex)
            else:
                st.info(f'**Keyword arguments accepted:** {st.session_state.model_parameters["kwargs"]}')

    st.markdown('---\n'
                '## Build and Train Model\n'
                'Click on the button to begin the construction and building of your model.')
    if st.button('Continue', key='continue'):
        try:
            if st.session_state.data_parameters['model_name'] == 'Simple':
                st.session_state.model = SimpleModel(
                    ensemble_count=st.session_state.model_parameters['ensemble_count'],
                    model_data=st.session_state.data_parameters['input'])
                st.session_state.model.instantiate(hidden_layers=st.session_state.model_parameters['hidden'],
                                                   neurons_per_layer=st.session_state.model_parameters['neurons'],
                                                   dropout_threshold=st.session_state.model_parameters['dropout'],
                                                   validation_split=st.session_state.model_parameters['validation'],
                                                   epochs=st.session_state.model_parameters['epochs'],
                                                   batch_size=st.session_state.model_parameters['batch_size'],
                                                   shuffle=st.session_state.model_parameters['shuffle'],
                                                   patience=st.session_state.model_parameters['patience'],
                                                   verbose=st.session_state.model_parameters['verbose'])
                st.session_state.model.fit(persist=st.session_state.model_parameters['persist'])
            elif st.session_state.data_parameters['model_name'] == 'RNN':
                st.session_state.model = RNNModel(
                    ensemble_count=st.session_state.model_parameters['ensemble_count'],
                    model_data=st.session_state.data_parameters['input'])
                st.session_state.model.instantiate(hidden_layers=st.session_state.model_parameters['hidden'],
                                                   neurons_per_layer=st.session_state.model_parameters['neurons'],
                                                   dropout_threshold=st.session_state.model_parameters['dropout'],
                                                   validation_split=st.session_state.model_parameters['validation'],
                                                   epochs=st.session_state.model_parameters['epochs'],
                                                   batch_size=st.session_state.model_parameters['batch_size'],
                                                   shuffle=st.session_state.model_parameters['shuffle'],
                                                   patience=st.session_state.model_parameters['patience'],
                                                   verbose=st.session_state.model_parameters['verbose'])
                st.session_state.model.fit(persist=st.session_state.model_parameters['persist'])
            elif st.session_state.data_parameters['model_name'] == 'BiLSTM':
                st.session_state.model = BidirectionalRNNModel(
                    ensemble_count=st.session_state.model_parameters['ensemble_count'],
                    model_data=st.session_state.data_parameters['input'])
                st.session_state.model.instantiate(hidden_layers=st.session_state.model_parameters['hidden'],
                                                   neurons_per_layer=st.session_state.model_parameters['neurons'],
                                                   dropout_threshold=st.session_state.model_parameters['dropout'],
                                                   validation_split=st.session_state.model_parameters['validation'],
                                                   epochs=st.session_state.model_parameters['epochs'],
                                                   batch_size=st.session_state.model_parameters['batch_size'],
                                                   shuffle=st.session_state.model_parameters['shuffle'],
                                                   patience=st.session_state.model_parameters['patience'],
                                                   verbose=st.session_state.model_parameters['verbose'])
                st.session_state.model.fit(persist=st.session_state.model_parameters['persist'])

            elif st.session_state.data_parameters['model_name'] == 'bert-base-uncased':
                st.session_state.model = BaseUncasedBERTModel(model_data=st.session_state.data_parameters['input'])
            elif st.session_state.data_parameters['model_name'] == 'bert-base-cased':
                st.session_state.model = BaseCasedBERTModel(model_data=st.session_state.data_parameters['input'])
            elif st.session_state.data_parameters['model_name'] == 'bert-large-uncased':
                st.session_state.model = LargeUncasedBERTModel(model_data=st.session_state.data_parameters['input'])
            elif st.session_state.data_parameters['model_name'] == 'bert-large-cased':
                st.session_state.model = LargeUncasedBERTModel(model_data=st.session_state.data_parameters['input'])

            st.session_state.model.instantiate(
                epochs=st.session_state.model_parameters['epochs'],
                batch_size=st.session_state.model_parameters['batch_size'],
                initial_learning_rate=st.session_state.model_parameters['initial_lr'],
            )
            # model historical records can be accessed by referencing the .history property of self.session_state.model
            st.session_state.model.fit(persist=st.session_state.model_parameters['persist'])
        except Exception as ex:
            raise ex
        else:
            st.success('Model built successfully')
