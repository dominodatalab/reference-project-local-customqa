import os
import streamlit as st

from streamlit.web.server import websocket_headers
from streamlit_chat import message

import requests


# Initialise session state variables.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


st.set_page_config(initial_sidebar_state='collapsed')
clear_button = st.sidebar.button("Clear Conversation", key="clear")
output_tokens = st.sidebar.number_input('Number of output tokens (50-500): ', min_value=50, max_value=500, value=200)

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    if submit_button and user_input and qa_chain:
        answer = None
        with st.spinner("Searching for the answer..."):
            result = requests.post("https://se-demo.domino.tech:443/models/65b2846db2e5737d566de52e/latest/model",
                auth=(
                    "dWyCVvxpastxkWhGf0TIXMXpsWWGnSrfGzFAV7yr3O33f4Hs3qmeQB5sWxbfrLy7",
                    "dWyCVvxpastxkWhGf0TIXMXpsWWGnSrfGzFAV7yr3O33f4Hs3qmeQB5sWxbfrLy7"
                ),
                json={
                    "data": {
                        "prompt": user_input
                    }
                }
            )
        if result:
            answer = result["result"]["text_from_llm"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(answer)
        
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, logo='https://freesvg.org/img/1367934593.png', key=str(i) + '_user')
                message(st.session_state["generated"][i], logo='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6e8aarUy37BOHMTSk-TUcs4AyAy3pfAHL-F2K49KHNEbI0QUlqWJFEqXYQvlBdYMMJA&usqp=CAU', key=str(i))