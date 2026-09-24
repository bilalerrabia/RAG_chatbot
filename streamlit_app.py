import os
import time
import streamlit as st
from src.__main__ import RAG

st.header("Chat with vllm dataset")

@st.cache_resource
def get_rag_agent():
    return RAG()

agent = get_rag_agent()

if not os.path.exists("data/processed/chunks.json"):
    with st.spinner("Indexing repository... this might take a few minutes."):
        agent.index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("ur query")

if query:

    print(f"\n[USER QUERY]: {query}")

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()

            answer = agent.answer(query=query)
            st.markdown(answer)

            end_time = time.time()
            print(f"[ASSISTANT ANSWER]: {answer}")
            print(f"[GENERATION TIME]: {end_time - start_time:.2f} seconds\n")

    st.session_state.messages.append({"role": "assistant", "content": answer})
