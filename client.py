import requests
import streamlit as st

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

def get_openai_response(input_text):
    response=requests.post(
        "http://localhost:8000/openai/invoke",
        json={'input': {'input': input_text}} 
    )
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post(
        "http://localhost:8000/ollama/invoke",
        json={'input': {'input': input_text}} 
    )
    return response.json()['output']

## streamlit framework
st.title('Langchain RAG App')

# Dropdown or radio button for model selection
model_choice = st.selectbox(
    "Choose a model:",
    ("openai", "ollama"),
    index=0  # Default to OpenAI
)

input_text = st.text_input("Enter your query:")

# Button to submit the query
if st.button("Get Response"):
    if input_text:
        if model_choice == "openai":
           response = get_openai_response(input_text)
        else:
           response = get_ollama_response(input_text)
           
        if response:
            st.write(f"**Response from {model_choice.capitalize()}:**")
            st.write(response)
    else:
        st.warning("Please enter a query.")