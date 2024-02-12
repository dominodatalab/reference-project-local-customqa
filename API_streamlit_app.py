import os
import streamlit as st

from streamlit.web.server import websocket_headers
from streamlit_chat import message

import requests

################################################################################
# Application State
################################################################################

# Initialise session state variables.
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

################################################################################
# Application UI
################################################################################

# Configure the side bar. This will allow a user to clear the chat and to set the number of tokens to return in the responses.
st.set_page_config(initial_sidebar_state='collapsed')
clear_button = st.sidebar.button("Clear Conversation", key="clear")
# We will give the end users the option to tailor how many characters are returned by the model as some questions require more detail than others.
output_tokens = st.sidebar.number_input('Number of output characters (50-500): ', min_value=50, max_value=500, value=200)

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Header Image - You can change this to suit your application
st.image("images/domino_banner.png")
# container for chat history
response_container = st.container()
# container for text box
container = st.container()

################################################################################
# User Input and Model API call
################################################################################

# Container for the chat interface
with container:
    with st.form(key='my_form', clear_on_submit=True):
        # Capture the user input
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    
    # When the user enters a question and clicks the submit button we will call out to the Deployed Llama2 model
    # to search on our documents.
    if submit_button and user_input:
        answer = None
        with st.spinner("Searching for the answer..."):
            # You will need to update the "post" and "auth" details for the model you have deployed
            response = requests.post("https://se-demo.domino.tech:443/models/65b2846db2e5737d566de52e/latest/model",
                auth=(
                    "dWyCVvxpastxkWhGf0TIXMXpsWWGnSrfGzFAV7yr3O33f4Hs3qmeQB5sWxbfrLy7",
                    "dWyCVvxpastxkWhGf0TIXMXpsWWGnSrfGzFAV7yr3O33f4Hs3qmeQB5sWxbfrLy7"
                ),
                json={
                    # This is the data payload for the API
                    # It contains the question and the number of output characters from the UI
                    "data": {
                        "input_text": user_input,
                        "max_new_tokens": output_tokens
                    }
                }
            )
        if response:
            # The response from the API is returned as JSON and we want to get the text response from the model
            answer = response.json()["result"]["text_from_llm"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(answer)
    
    # update the state of the Chatbot and display the message to the user with a logo.
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, logo='https://freesvg.org/img/1367934593.png', key=str(i) + '_user')
                # You can change this chat image to the logo of your organisation
                message(st.session_state["generated"][i], logo='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQk6e8aarUy37BOHMTSk-TUcs4AyAy3pfAHL-F2K49KHNEbI0QUlqWJFEqXYQvlBdYMMJA&usqp=CAU', key=str(i))
