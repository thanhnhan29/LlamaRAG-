import streamlit as st
from Llama3 import *
st.set_page_config(layout="centered")
col1 = st.columns([3,1])

if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.message = [{"role":"ai","content":"Hello! how can I help you?"}]

with col1[0]:
    st.title("Llama 3")
with col1[1]:
    if st.button("Reset history chat"):
        st.session_state.message = [{"role":"ai","content":"Hello! how can I help you?"}]

    

prompt = st.chat_input("Say something")
if prompt:
    st.session_state.message.append({"role": "human", "content": prompt})
for mess in st.session_state.message:
    with st.chat_message(mess["role"]):
        st.write(mess["content"])
if st.session_state.message[-1]["role"] == "human":

    with st.status("Thinking.."):
        reply = generate(prompt)
        st.session_state.message.append({"role": "ai", "content": reply})
        st.rerun()
