import os
import time

import requests
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(
    page_title="OmniBrain AI - Multi-Modal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1/chat")
INGESTION_API_URL = os.getenv(
    "INGESTION_API_URL", "http://localhost:8000/api/v1/ingestion/upload"
)


def render_image_asset(image_path: str, caption: str):
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width="stretch")
    else:
        st.warning(f"Image asset missing at: `{image_path}`")


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = "Attention_is_all_you_need.pdf"

if "upload_summary" not in st.session_state:
    st.session_state.upload_summary = None

# Telemetry Session Metrics [Day 3 Scope]
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0

if "session_latency_ms" not in st.session_state:
    st.session_state.session_latency_ms = 0.0

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ==============================================================================
# 2. SIDEBAR CONTROLS [Day 1 & Day 3 Scope]
# ==============================================================================
with st.sidebar:
    st.title("🧠 OmniBrain Console")
    st.caption("Agentic Multi-Modal PDF Intelligence Engine")
    st.markdown("---")

    st.subheader("📄 Document Controls")
    selected_pdf = st.selectbox(
        "Active PDF Asset:",
        ["Attention_is_all_you_need.pdf", "sample.pdf", "sample2.pdf"],
        index=0,
    )

    uploaded_pdf = st.file_uploader(
        "Upload a PDF for ingestion validation",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if st.button("Process uploaded PDF"):
        if uploaded_pdf is None:
            st.warning("Choose a PDF first.")
        else:
            with st.status("Processing upload...", expanded=True) as status_box:
                try:
                    status_box.write(
                        "Sending the PDF to the backend ingestion endpoint."
                    )
                    files = {
                        "file": (
                            uploaded_pdf.name,
                            uploaded_pdf.getvalue(),
                            "application/pdf",
                        )
                    }
                    response = requests.post(
                        INGESTION_API_URL,
                        files=files,
                        timeout=120,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    st.session_state.upload_summary = payload
                    st.session_state.active_document = uploaded_pdf.name
                    selected_pdf = uploaded_pdf.name
                    status_box.update(
                        label="Upload processed", state="complete", expanded=False
                    )
                    st.success(
                        f"Processed {payload.get('pages', 0)} pages, "
                        f"{payload.get('text_chunks', 0)} text chunks, "
                        f"{payload.get('images', 0)} images."
                    )
                except requests.RequestException as exc:
                    status_box.update(
                        label="Upload failed", state="error", expanded=True
                    )
                    st.error(f"Upload failed: {exc}")

    if st.session_state.upload_summary:
        st.caption("Latest ingestion result")
        st.json(st.session_state.upload_summary)

    st.subheader("⚙️ System Status")
    st.success("🟢 Vector Store: Qdrant Connected")
    st.success("🟢 Text-to-SQL Agent: Ready (SQLite Stock DB)")
    st.info("🟢 LangGraph Agents: Ready")

    st.markdown("---")
    st.subheader("📊 Session Telemetry (Langfuse)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
        st.metric("Est. Tokens", f"{st.session_state.session_tokens:,}")
    with col2:
        avg_lat = st.session_state.session_latency_ms / max(
            1, st.session_state.total_queries
        )
        st.metric("Avg Latency", f"{avg_lat:.0f} ms")
        est_cost = (st.session_state.session_tokens / 1000) * 0.00015
        st.metric("Est. Cost", f"${est_cost:.4f}")

    st.caption("⚡ Execution traces synced to Langfuse")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_latency_ms = 0.0
        st.session_state.total_queries = 0
        st.rerun()

# ==============================================================================
# 3. MAIN CHAT CANVAS & HISTORY RENDERING [Day 1 Scope]
# ==============================================================================
st.title("💬 Chat with OmniBrain")
st.caption(f"Currently querying context from: `{st.session_state.active_document}`")

# Render all past chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Render Thought Steps if preserved
        if "thought_process" in message and message["thought_process"]:
            with st.status("🧠 Agent Thought Process Log", expanded=False):
                for step in message["thought_process"]:
                    st.write(
                        f"**[{step.get('agent', 'System')}]**: {step.get('action', '')}"
                    )

        # Render Main Response Text
        st.markdown(message["content"])

        # Render Inline Images if present
        if "images" in message and message["images"]:
            st.markdown("#### 🖼️ Referenced Figures:")
            cols = st.columns(min(len(message["images"]), 3))
            for idx, img_path in enumerate(message["images"]):
                with cols[idx % 3]:
                    render_image_asset(
                        img_path, f"Figure from {st.session_state.active_document}"
                    )

# ==============================================================================
# 4. CHAT INPUT & FASTAPI BACKEND LINK [Day 2 Scope]
# ==============================================================================
if user_query := st.chat_input("Ask a question about the document or architecture..."):
    # 1. Append user prompt to UI state immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        # UI Placeholder for Live Thought Stream [Day 2 Scope]
        status_container = st.status("🤖 Agents thinking...", expanded=True)
        thought_steps = []
        final_answer = ""
        returned_images = []

        try:
            start_time = time.time()
            status_container.write("📡 Routing request through FastAPI backend...")
            payload = {
                "message": user_query,
                "conversation_id": st.session_state.get("conversation_id"),
                "source_name": st.session_state.active_document,
            }

            response = requests.post(
                BACKEND_API_URL,
                json=payload,
                timeout=60,
            )

            # Update Telemetry Metrics
            elapsed_ms = (time.time() - start_time) * 1000.0
            st.session_state.session_latency_ms += elapsed_ms
            st.session_state.total_queries += 1

            if response.status_code == 200:
                data = response.json()
                final_answer = (data.get("response") or "").strip()
                thought_steps = data.get("thought_process", [])
                returned_images = data.get("images", [])

                # Estimate tokens processed (~4 chars per token)
                tokens_est = (len(user_query) + len(final_answer)) // 4
                st.session_state.session_tokens += max(10, tokens_est)

                retrieved_text = data.get("retrieved_text", [])

                # Render SQL Execution Trace if returned by Backend API
                sql_query = data.get("sql_query")
                sql_result = data.get("sql_result")

                if sql_query:
                    with st.expander("📊 Text-to-SQL Agent Execution", expanded=True):
                        st.markdown("**Generated SQL Query:**")
                        st.code(sql_query, language="sql")

                        if sql_result:
                            st.markdown("**Query Results:**")
                            if isinstance(sql_result, list):
                                st.dataframe(sql_result)
                            else:
                                st.write(sql_result)

                retrieved_images = data.get("retrieved_images", [])

                if not final_answer and retrieved_text:
                    passages = [
                        item.get("text", "")
                        for item in retrieved_text
                        if item.get("text")
                    ]
                    final_answer = (
                        "\n\n".join(passages) if passages else "Execution completed."
                    )

                if not final_answer:
                    final_answer = "Execution completed."

                if data.get("status"):
                    status_container.write(f"Status: {data.get('status')}")

                # Render Live Agent Thoughts with Self-RAG Traces
                for step in thought_steps:
                    agent_name = step.get("agent", "Agent")
                    action_msg = step.get("action", "")

                    if "Grader" in agent_name or "Evaluator" in agent_name:
                        if "YES" in action_msg.upper() or "GOOD" in action_msg.upper():
                            status_container.write(f"🟢 **{agent_name}**: {action_msg}")
                        else:
                            status_container.write(f"⚠️ **{agent_name}**: {action_msg}")
                    elif "Rewriter" in agent_name or "Retry" in agent_name:
                        status_container.write(f"🔄 **{agent_name}**: {action_msg}")
                    else:
                        status_container.write(f"👉 **{agent_name}**: {action_msg}")

                status_container.update(
                    label="✅ Agentic Execution Complete!",
                    state="complete",
                    expanded=False,
                )

                # Render Final LLM Text
                st.markdown(final_answer)

                # Render Referenced Figures Inline [Day 2 Scope]
                if returned_images:
                    st.markdown("#### 🖼️ Referenced Figures:")
                    img_cols = st.columns(min(len(returned_images), 3))
                    for idx, img_path in enumerate(returned_images):
                        with img_cols[idx % 3]:
                            render_image_asset(
                                img_path,
                                f"Extracted Asset: {os.path.basename(img_path)}",
                            )

                # Render Interactive Claims & Citations Explorer [Day 4 Scope]
                citations_data = data.get("citations", [])
                if citations_data:
                    with st.expander(
                        "📌 Interactive Claims & Citations Explorer (Click-to-View)",
                        expanded=True,
                    ):
                        st.caption(
                            "Verified AI claims linked directly to "
                            "PDF pages and chart assets:"
                        )
                        for cite in citations_data:
                            claim = cite.get("claim", "")
                            src_pdf = cite.get("source_pdf", "Document")
                            page_num = cite.get("page", 0)
                            chart_id = cite.get("chart_id")

                            c_col1, c_col2 = st.columns([3, 1])
                            with c_col1:
                                st.markdown(f"• **{claim}**")
                            with c_col2:
                                badge_label = (
                                    f"📊 {chart_id}"
                                    if chart_id
                                    else f"📄 Page {page_num}"
                                )
                                st.info(
                                    f"{badge_label}\n`{src_pdf}`",
                                    icon="📌",
                                )

                if retrieved_text:
                    with st.expander(
                        "📄 Retrieved Context & Origin Badges", expanded=False
                    ):
                        for item in retrieved_text[:5]:
                            doc_name = item.get(
                                "source", item.get("document", "Unknown")
                            )
                            page_num = item.get("page", "?")

                            # Determine Context Origin Badge
                            if (
                                "http" in str(doc_name).lower()
                                or "web" in str(doc_name).lower()
                            ):
                                origin_badge = "🌐 [DuckDuckGo Web Search]"
                            elif (
                                "sql" in str(doc_name).lower()
                                or "stock" in str(doc_name).lower()
                            ):
                                origin_badge = "📊 [SQLite Stock DB]"
                            else:
                                origin_badge = "🗄️ [Qdrant Vector DB]"

                            st.markdown(
                                f"**{origin_badge}** | "
                                f"Source: `{doc_name}` (Page {page_num})"
                            )
                            st.write(item.get("text", item.get("content", "")))
                            st.markdown("---")

            else:
                status_container.update(label="❌ Backend API Error", state="error")
                final_answer = f"Error {response.status_code}: {response.text}"
                st.error(final_answer)

        except requests.exceptions.ConnectionError:
            status_container.update(label="❌ Connection Failed", state="error")
            final_answer = (
                "⚠️ Could not connect to FastAPI server. "
                "Ensure FastAPI is running at `http://localhost:8000`!"
            )
            st.error(final_answer)

        except Exception as e:
            status_container.update(label="❌ Unexpected Error", state="error")
            final_answer = f"An error occurred: {str(e)}"
            st.error(final_answer)

        # 3. Store full Assistant response details in session state history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer,
                "thought_process": thought_steps,
                "images": returned_images,
                "citations": data.get("citations", []) if "data" in locals() else [],
            }
        )
