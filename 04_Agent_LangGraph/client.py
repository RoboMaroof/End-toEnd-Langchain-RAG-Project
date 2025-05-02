import requests
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Langchain RAG Agent", layout="centered")
st.title('Langchain RAG Chat App')

if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

with st.expander("📥 Ingest Custom Data into Vector Store (if required)", expanded=False):
    st.markdown("##### Select a Data Source")

    source_type = st.selectbox("Select data source type:", ["website", "docs", "sql"])

    if source_type == "website":
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

model_label_map = {
    "OpenAI (gpt-4o-mini)": "openai:gpt-4o-mini",
    "Groq (qwen-qwq-32b)": "groq:qwen-qwq-32b"
}
model_choice_label = st.selectbox("Choose a model:", list(model_label_map.keys()))
model_choice = model_label_map[model_choice_label]

input_text = st.text_input("Enter your question:")
if st.button("🔍 Get Response"):
    if input_text.strip():
        with st.spinner("Thinking..."):
            endpoint = f"{BASE_URL}/agent/invoke"
            response = requests.post(
                endpoint, 
                json={
                    "input": {"input": input_text},
                    "model": model_choice
                },               
            )
            response.raise_for_status()
            data = response.json()

            # Final Output
            if "final_output" in data:
                st.markdown(f"**🧠 Response:**\n\n{data['final_output']}")

            # Tools Used
            if "tools_used" in data and data["tools_used"]:
                with st.expander("🛠 Tools Used"):
                    for tool in data["tools_used"]:
                        st.markdown(f"- {tool}")

            # Retrieved Data
            if "retrieved_chunks" in data and data["retrieved_chunks"]:
                with st.expander("📄 Retrieved Data"):
                    for i, chunk in enumerate(data["retrieved_chunks"], start=1):
                        tool = chunk.get("tool", "Unknown")
                        ctype = chunk.get("type", "text")
                        content = chunk.get("data", "")
                        st.markdown(f"**Tool**: `{tool}` | **Type**: `{ctype}`")
                        st.code(content if isinstance(content, str) else str(content))

            # Intermediate Steps
            if "intermediate_steps" in data and data["intermediate_steps"]:
                with st.expander("🔎 Intermediate Steps"):
                    for idx, step in enumerate(data["intermediate_steps"], 1):
                        step_type = step.get("type", "unknown")
                        st.markdown(f"**Step {idx}**: `{step_type}`")
                        if step_type == "ai_tool_call":
                            st.markdown(f"- Tool: `{step.get('tool')}`")
                            st.markdown(f"- Arguments: `{step.get('args')}`")
                        elif step_type == "tool_response":
                            st.markdown(f"- Tool: `{step.get('tool')}`")
                            st.markdown("```")
                            st.markdown(step.get("content", ""))
                            st.markdown("```")
                        elif step_type == "ai_final_response":
                            st.markdown("**Final Response:**")
                            st.markdown(step.get("content", ""))
                        else:
                            st.markdown(f"- Content: `{step.get('content', '')}`")

    else:
        st.warning("⚠️ Please enter a question.")
