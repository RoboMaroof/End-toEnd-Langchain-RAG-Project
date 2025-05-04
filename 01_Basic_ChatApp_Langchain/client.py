import requests
import streamlit as st

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

# OpenAI
def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/openai/invoke",
        json={'input': {'input': input_text}} 
    )
    result = response.json()
    return result["output"]["output"]["content"], result["output"]["retrieved_sections"]

# Ollama
def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/ollama/invoke",
        json={'input': {'input': input_text}} 
    )
    result = response.json()
    return result["output"]["output"], result["output"]["retrieved_sections"]


# Streamlit framework
st.title('Langchain RAG App')

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    ("openai", "ollama"),
    index=1  # Default to ollama
)

input_text = st.text_input("Enter your query:")

if st.button("Get Response"):
    if input_text:
        if model_choice == "openai":
            response, retrieved_sections = get_openai_response(input_text)
        else:
            response, retrieved_sections = get_ollama_response(input_text)
        
        if response:
            st.write(f"**Response from {model_choice.capitalize()}:**")
            st.write(response)

            st.write("### Top Retrieved Sections:")
            for i, section in enumerate(retrieved_sections, 1):
                with st.expander(f"Section {i}"):
                    st.write(section)
    else:
        st.warning("Please enter a query.")