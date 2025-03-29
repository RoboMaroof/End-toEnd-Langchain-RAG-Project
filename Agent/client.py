import requests
import streamlit as st

BASE_URL = "http://localhost:8000" 

st.title('Langchain RAG & Agent App')

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False


# Optional URL input
st.markdown("### 🔗 (Optional) Provide a URL to load documents for retrieval:")
custom_url = st.text_input("Enter a document URL (or leave blank to use model/tools only):")

if st.button("🧠 Ingest and Create Vector Store"):
    if custom_url:
        with st.spinner("Ingesting and creating vector store..."):
            response = requests.post(f"{BASE_URL}/vectordb/create", json={"url": custom_url})
            if response.ok and "message" in response.json():
                st.success("✅ Vector store created successfully!")
                st.session_state.vector_store_ready = True
            else:
                st.error("Failed to create vector store.")
    else:
        st.warning("Please enter a valid URL.")


# Mode and Model Selection
mode_choice = st.radio("Choose mode:", ("RAG", "Agent"), index=0, key="mode_choice")
model_choice = st.selectbox("Choose a model:", ("openai", "ollama"), index=1, key="model_choice")
input_text = st.text_input("Enter your query:")


# Get Response
if st.button("🔍 Get Response"):
    if input_text.strip():
        if model_choice == "ollama" and mode_choice == "Agent":
            st.warning("Agent mode is currently only supported with OpenAI.")
        else:
            with st.spinner("Thinking..."):
                if mode_choice == "RAG":
                    endpoint = f"/{model_choice}/invoke"
                else:
                    endpoint = f"/{model_choice}/agent/invoke"

                response = requests.post(f"{BASE_URL}{endpoint}", json={"input": {"input": input_text}})
                if response.ok:
                    data = response.json()["output"]
                    st.markdown(f"**Response from `{model_choice}` ({mode_choice}):**")
                    st.write(data["output"])

                    if mode_choice == "RAG" and "retrieved_sections" in data:
                        st.markdown("### 📚 Retrieved Sections:")
                        for i, section in enumerate(data.get("retrieved_sections", []), start=1):
                            with st.expander(f"Section {i}"):
                                st.write(section)
                    elif mode_choice == "Agent":
                        st.markdown("### 🛠️ Tools Used:")
                        st.write(data.get("tool_used", "N/A"))
                else:
                    st.error("Error retrieving response.")
    else:
        st.warning("Please enter a query.")