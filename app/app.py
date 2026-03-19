import sys
import os

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.assistant import ShoppingAssistant

st.set_page_config(page_title="Retail AI Chat", layout="centered")

st.title("🛒 Retail AI Chat Assistant")

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize assistant once
@st.cache_resource
def load_assistant():
    return ShoppingAssistant()

assistant = load_assistant()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything about products, demand, or recommendations")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Get assistant response
    response = assistant.answer(user_input)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant message
    with st.chat_message("assistant"):
        st.write(response)