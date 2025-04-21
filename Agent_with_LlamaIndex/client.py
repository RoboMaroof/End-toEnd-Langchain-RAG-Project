import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Langchain RAG Agent", layout="centered")
st.title('Langchain RAG Chat App')

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

st.markdown("## Ingest Data into Vector Store")

source_type = st.selectbox("Select data source type:", ["website", "docs", "sql"])
if source_type  == "website":
    source_path = st.text_input("Enter URL path:")

    if st.button("Ingest and Update Vector Store"):
        if source_path:
            with st.spinner("Ingesting and updating vector store..."):
                response = requests.post(f"{BASE_URL}/vectordb/create", json={
                    "source_type": source_type,
                    "source_path": source_path
                })
                if response.ok and "message" in response.json():
                    st.success("✅ Vector store updated successfully!")
                    st.session_state.vector_store_ready = True
                else:
                    st.error(f"❌ Failed to update vector store. {response.json().get('error', '')}")
        else:
            st.warning("⚠️ Please enter a valid source path.")

else:
    uploaded_file = st.file_uploader("Upload a file (PDF, TXT, DOCX, DB, etc.):", type=["pdf", "txt", "docx", "db"])
    if st.button("Upload and Ingest File"):
        if uploaded_file:
            with st.spinner("Uploading and indexing file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{BASE_URL}/vectordb/upload", files=files)
                if response.ok and "message" in response.json():
                    st.success(f"✅ {response.json()['message']}")
                    st.session_state.vector_store_ready = True
                else:
                    st.error(f"❌ Upload failed. {response.json().get('error', '')}")
        else:
            st.warning("⚠️ Please upload a valid file.")

st.markdown("---")
st.markdown("## 🤖 Ask a Question")

model_choice = st.selectbox("Choose a model:", ("openai",), index=0)
input_text = st.text_input("Enter your question:")

if st.button("🔍 Get Response"):
    if input_text.strip():
        with st.spinner("Thinking..."):
            endpoint = f"/{model_choice}/agent/invoke"
            response = requests.post(f"{BASE_URL}{endpoint}", json={"input": {"input": input_text}})
            if response.ok:
                data = response.json()
                st.markdown(f"**🧠 Response:**\n\n{data['output']}")
                tools_used = data.get("tool_used", [])
                if tools_used:
                    st.markdown(f"### 🛠 Tools Used:")
                    st.write(tools_used)
            else:
                st.error(f"❌ Error retrieving response. {response.text}")
    else:
        st.warning("⚠️ Please enter a question.")
