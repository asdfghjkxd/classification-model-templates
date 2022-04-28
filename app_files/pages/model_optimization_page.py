import streamlit as st
import json

from .utils.utils import init_session_state
from streamlit_tags import st_tags
from streamlit_ace import st_ace
from ..data_classes.model_validator import *


def page():
    """Model Optimization Page"""

    init_session_state()

    st.title('Model Optimizer')
    st.markdown('This module allows you to optimize your model by optimizing the hyperparameters of the model to '
                'create.\n\n'
                '* **Can be Optimized:** Defines parameters that can be varied and be optimized\n'
                '* **Cannot be Optimized:** Defines parameters that cannot be varied and hence cannot be '
                'optimized\n'
                '* **Model Persistence:** Parameters that allows you to specify whether to save the model '
                'to your disk or to store it in memory only')
    if st.session_state.data_parameters['input'] is None or \
            not st.session_state.data_parameters['input'].is_processed():
        st.warning('No Model Data is detected. Model Optimization cannot occur!')
    elif isinstance(st.session_state.data_parameters['input'], pd.DataFrame) and \
            st.session_state.data_parameters['input'].is_processed():
        st.info('Model Data is detected!')

    st.markdown('### Optimization Parameters')
    st.markdown('#### Can be Optimized')
    st.session_state.model_parameters['min_neuron'] = st.number_input('Minimum number of neurons to optimize',
                                                                      min_value=0,
                                                                      max_value=99999,
                                                                      step=1,
                                                                      value=8,
                                                                      key=f'ensemble_counts')
    st.session_state.model_parameters['max_neuron'] = st.number_input('Maximum number of neurons to optimize',
                                                                      min_value=0,
                                                                      max_value=99999,
                                                                      step=1,
                                                                      value=64,
                                                                      key=f'hidden')
    st.session_state.model_parameters['step'] = st.number_input('Number of neurons increase per optimization cycle',
                                                                min_value=0,
                                                                max_value=999999,
                                                                step=1,
                                                                value=8,
                                                                key=f'neurons')
    st.session_state.model_parameters['learning_rate'] = st.number_input('Rate of Model Learning',
                                                                         min_value=0.,
                                                                         max_value=1.,
                                                                         step=1e-6,
                                                                         format='%.6f',
                                                                         value=1e-5,
                                                                         key=f'lrate')
    st.session_state.model_parameters['dropout'] = st.number_input('Level of dropout',
                                                                   min_value=0.,
                                                                   max_value=1.,
                                                                   step=1e-3,
                                                                   value=0.800,
                                                                   format='%.3f',
                                                                   key=f'dropout')

    st.markdown('#### Cannot be Optimized')
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

    st.markdown('#### Model Persistence')
    st.session_state.model_parameters['persist'] = st.checkbox('Persist files', value=True, key='persist')

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

    st.markdown('## Optimize Model\n'
                'Click on the button above to begin the optimization of the model.')
    if st.button('Continue', key='continue'):

        try:
            st.session_state.model = ModelTrainer(
                ensemble_count=st.session_state.model_parameters['ensemble_count'],
                model_data=st.session_state.model_parameters['input'])
            st.session_state.model.optimise(min_neuron=st.session_state.model_parameters['min_neuron'],
                                            max_neuron=st.session_state.model_parameters['max_neuron'],
                                            step=st.session_state.model_parameters['step'],
                                            dropout=st.session_state.model_parameters['dropout'],
                                            learning_rate=st.session_state.model_parameters['learning_rate'],
                                            validation_split=st.session_state.model_parameters['validation'],
                                            model_type=st.session_state.model_parameters['model_type'],
                                            epochs=st.session_state.model_parameters['epochs'],
                                            batch_size=st.session_state.model_parameters['batch_size'],
                                            shuffle=st.session_state.model_parameters['shuffle'],
                                            patience=st.session_state.model_parameters['patience'],
                                            verbose=st.session_state.model_parameters['verbose'],
                                            persist=st.session_state.model_parameters['persist'])
        except Exception as ex:
            raise ex
        else:
            st.success('Optimization completed')
