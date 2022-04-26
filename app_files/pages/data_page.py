import json
import streamlit as st

from ..data_classes.data_validator import *
from streamlit_tags import st_tags
from streamlit_ace import st_ace


def page():
    """Function which is called to render the page"""

    # add flags to session state
    if 'input' not in st.session_state:
        st.session_state.input = None
    if 'loaded' not in st.session_state:
        st.session_state.loaded = False
    if 'applied' not in st.session_state:
        st.session_state.applied = False

    # main page
    st.title('Data Processing')
    st.markdown('This module allows you to read and process the data to use for the Model Builder module.')

    st.header('Upload Data')
    st.markdown('Upload your data files first.\n\nEnsure that your data files are of the **same format**, '
                'and that they are uploaded in the order you want your data expressed (if you are '
                'uploading more than one data files).')
    ftype = st.selectbox(label='Select File Format of Data Files', options=['CSV', 'XLSX', 'JSON'],
                         key='ftype', help='Input File Type')
    paths = st.file_uploader(label='Upload Files', type=ftype, accept_multiple_files=True,
                             key='paths', help='Input files for processing')
    if len(paths) < 1:
        st.warning('No files have been uploaded')
    else:
        st.info(f'**{len(paths)}** file(s) uploaded!')

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

    if not st.session_state.loaded:
        if st.button('Load and parse data'):
            try:
                _in = ModelInput(path=paths, format=ftype.lower(), on_error=on_error,
                                 args=args if len(args) > 0 else None,
                                 kwargs=kwargs_parsed if len(kwargs_parsed) > 0 else None)
            except Exception as ex:
                st.error(ex)
            else:
                if _in.data is not None:
                    st.session_state.input = _in
                    st.success('Data loaded and parsed!')
                    st.session_state.loaded = True
                else:
                    # TODO: DEBUGGING PASSTHROUGH: TO DELETE ONCE DONE
                    st.session_state.loaded = True
                    st.error('Data not loaded or parsed')

    if st.session_state.loaded:
        st.markdown('---')
        st.markdown('## Data Cleaning Options')
        st.markdown('Clean up the data using the three supported functions below. You may: split up the dataset '
                    'into a train-test configuration through *Preprocess*, apply lambdas to the dataset to do '
                    'quick and simple calculations on the fly using *Apply*, or remove unwanted columns of data '
                    'using *Drop*.')
        st.info('**Data Status:** '
                f'{st.session_state.input.is_processed() if st.session_state.input is not None else False}\n\n'
                f'**Data Shape:** '
                f'{st.session_state.input.is_processed() if st.session_state.input is not None else ()}')

        things_to_do = st.multiselect('Data Processes', options=['Preprocess', 'Apply', 'Drop'])

        if 'Preprocess' in things_to_do:
            with st.form('Preprocessing Data'):
                st.markdown('### Preprocessing')
                word = st.selectbox('Select column where text is located at',
                                    options=[col for col in st.session_state.input.data.columns]
                                    if st.session_state.input is not None else [])
                label = st.selectbox('Select column where the labels are located at',
                                     options=[col for col in st.session_state.input.data.columns]
                                     if st.session_state.input is not None else [])
                train_test_split = st.number_input('Ratio of train-test split',
                                                   step=1e-3,
                                                   format="%.3f",
                                                   min_value=0.001,
                                                   max_value=0.999,
                                                   value=0.800)

                if st.form_submit_button("Preprocess"):
                    if word != label:
                        try:
                            st.session_state.input.preprocess(word_col=word, label_col=label,
                                                              train_test_split=train_test_split)
                        except Exception as ex:
                            st.error(ex)
                        else:
                            st.success('Successfully preprocessed')
                    else:
                        st.error('Text column cannot be the same as label column')

        if 'Apply' in things_to_do:
            with st.form('Application of Functions to Dataset'):
                st.markdown('### Application')
                funcs = st_ace(value='[]', language='python', key='apply', auto_update=False)
                st.warning('**WARNING:** *eval()* is used for converting your Sequence type input to the correct '
                           'format used for parsing later on. You are warned that this method allows arbitrary code '
                           'execution and SHOULD NOT be misused to avoid errors.')
                tgt = st_tags(label='Target Columns', key='targets')
                dest = st_tags(label='Destination Columns', key='destinations')

                if funcs:
                    try:
                        funcs_parsed = eval(funcs)
                    except Exception as ex:
                        st.error(ex)
                    else:
                        if isinstance(funcs_parsed, Sequence):
                            st.info(f'**Function map accepted:** {funcs_parsed}')
                            st.session_state.applied = True
                        else:
                            st.error('Inputs is not a valid Sequence')
                            st.session_state.applied = False

                if st.session_state.applied:
                    if st.form_submit_button("Apply"):
                        try:
                            st.session_state.input.apply(func_map=funcs, tgt_map=tgt, dest_map=dest)
                        except Exception as ex:
                            st.error(ex)
                        else:
                            st.info('Functions applied successfully!')

        if 'Drop' in things_to_do:
            with st.form('Dropping Data Columns'):
                st.markdown('### Dropping')
                cols = st_tags(label='Input column names to drop from data source')
                axis = st.radio('Select Axis to drop data from', (0, 1),
                                format_func=lambda x: 'Columns' if x == 0 else 'Rows')

                if st.form_submit_button("Drop"):
                    try:
                        st.session_state.input.drop(tgt_map=cols, axis=axis)
                    except Exception as ex:
                        st.error(ex)
                    else:
                        st.success(f'Data columns: **{cols}** dropped.')
