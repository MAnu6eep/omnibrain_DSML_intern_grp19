import os
import time

import requests
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & DARK NAVY THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="OmniBrain",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1/chat")
INGESTION_API_URL = os.getenv(
    "INGESTION_API_URL", "http://localhost:8000/api/v1/ingestion/upload"
)

# ------------------------------------------------------------------------------
# Tunable layout constants (adjust these if your panels visually drift)
# ------------------------------------------------------------------------------
SIDEBAR_WIDTH_PX = 336  # Streamlit's default expanded sidebar width (~21rem)
RETRIEVAL_PANEL_WIDTH_PX = 380  # Approx width of the right retrieval panel when open
RETRIEVAL_TOGGLE_WIDTH_PX = (
    24  # Narrow chevron column between center and retrieval panel
)
COLUMN_GAP_PX = 16  # Streamlit's `gap="small"` spacing
MAIN_CONTENT_INSET_PX = 80  # Streamlit's wide-layout horizontal content inset
MAIN_CONTENT_RIGHT_INSET_PX = 64  # Right edge of the center column content
TOP_BAR_HEIGHT_PX = 64  # Height reserved for the "Active Context" bar
INPUT_ISLAND_MARGIN_PX = 24  # Floating gap around the chat input island

# Custom Dark Navy Theme CSS Injection
st.markdown(
    """
    <style>
    /* Dark Navy Base Theme */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main Layout Padding - extra bottom room so the floating chat input
       never overlaps the last message */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%%, #0B0F19 100%%);
        border-right: 1px solid #1E293B;
    }

    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sidebar-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 1rem;
    }

    /* ---------------------------------------------------------------------
       FIXED / FOREGROUND ELEMENT #1: Active Context bar (top of center panel)
       Sticky keeps it pinned to the top of the scrolling column while the
       messages behind it scroll underneath.
    --------------------------------------------------------------------- */
    .chat-top-bar {
        position: sticky;
        top: 0;
        z-index: 50;
        display: flex;
        align-items: center;
        min-height: %(top_bar_h)spx;
        background: linear-gradient(90deg, #0F172A 0%%, #111C33 100%%);
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(6px);
    }
    .chat-top-doc {
        font-size: 0.9rem;
        font-weight: 600;
        color: #38BDF8;
    }

    /* Retrieval Panel Styling */
    .retrieval-panel-box {
        background: linear-gradient(180deg, #101B32 0%%, #0B1220 100%%);
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 16px;
        height: 100%%;
    }

    .retrieved-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .retrieved-card-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #38BDF8;
        margin-bottom: 6px;
    }
    .retrieved-card-text {
        font-size: 0.85rem;
        color: #CBD5E1;
        line-height: 1.45;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
        margin-bottom: 6px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }

    div.stButton > button {
        background-color: #1E3A8A;
        color: #F8FAFC;
        border: 1px solid #3B82F6;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
        border-color: #60A5FA;
        color: #FFFFFF;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------------------------------------------------------------------
       FIXED / FOREGROUND ELEMENT #2: the right-panel collapse arrow.
       Styled to visually match Streamlit's own native sidebar chevron -
       small, circular, sticky so it stays put while the panel scrolls.
       This is the ONLY control that opens/closes the retrieval panel.
    --------------------------------------------------------------------- */
    div[data-testid="column"]:has(.retrieval-toggle-marker) {
        position: relative;
        z-index: 60;
        margin-left: -10px;
        margin-right: -10px;
    }
    div[data-testid="column"]:has(.retrieval-toggle-marker) div.stButton > button {
        position: sticky;
        top: 45vh;
        width: 24px;
        height: 34px;
        min-width: 24px;
        padding: 0;
        margin: 0;
        border-radius: 0;
        background: transparent !important;
        border: 0 !important;
        color: #38BDF8;
        font-weight: 700;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: none !important;
        transition: color 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="column"]:has(.retrieval-toggle-marker) div.stButton > button:hover {
        background: transparent;
        color: #7DD3FC;
        transform: scale(1.08);
    }

     /* ---------------------------------------------------------------------
         FIXED / FOREGROUND ELEMENT #3: the chat input, turned into a
         floating "island" instead of a full-width bar. Streamlit always
         renders st.chat_input in a fixed container pinned to the bottom of
         the *viewport* (outside the normal column flow), so we target it
         here via its `key` class rather than trying to nest it in a column.
     --------------------------------------------------------------------- */
    div[data-testid="stBottom"] {
        background: transparent !important;
    }
    .st-key-main_chat_input {
        position: fixed !important;
        left: %(left_offset)spx;
        right: %(right_offset)spx;
        bottom: %(island_margin)spx;
        width: calc(100vw - %(left_offset)spx - %(right_offset)spx) !important;
        max-width: calc(100vw - %(left_offset)spx - %(right_offset)spx) !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .st-key-main_chat_input > div,
    .st-key-main_chat_input form,
    .st-key-main_chat_input [data-testid="stChatInput"] {
        width: 100%% !important;
        max-width: none !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .st-key-main_chat_input textarea {
        box-sizing: border-box !important;
        width: 100%% !important;
        min-height: 44px !important;
        border-radius: 22px !important;
        border: 1px solid #1E3A5F !important;
        background-color: #101B32 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.45) !important;
        padding: 10px 44px 10px 14px !important;
    }
    </style>
    """
    % {
        "top_bar_h": TOP_BAR_HEIGHT_PX,
        "left_offset": SIDEBAR_WIDTH_PX + MAIN_CONTENT_INSET_PX,
        "right_offset": (
            RETRIEVAL_PANEL_WIDTH_PX
            + RETRIEVAL_TOGGLE_WIDTH_PX
            + COLUMN_GAP_PX
            + MAIN_CONTENT_RIGHT_INSET_PX
        ),
        "island_margin": INPUT_ISLAND_MARGIN_PX,
    },
    unsafe_allow_html=True,
)


def render_image_asset(image_path: str, caption: str):
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image missing: {image_path}")


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = "Attention_is_all_you_need.pdf"

if "upload_summary" not in st.session_state:
    st.session_state.upload_summary = None

# Right panel: OPEN by default, user collapses it with the single arrow control.
if "is_retrieval_panel_open" not in st.session_state:
    st.session_state.is_retrieval_panel_open = True

if "last_retrieval_data" not in st.session_state:
    st.session_state.last_retrieval_data = {
        "text": [],
        "images": [],
        "citations": [],
        "sql_query": None,
        "sql_result": None,
    }

if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0

if "session_latency_ms" not in st.session_state:
    st.session_state.session_latency_ms = 0.0

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


def fetch_indexed_documents() -> list[str]:
    try:
        docs_url = INGESTION_API_URL.replace("/upload", "/documents")
        resp = requests.get(docs_url, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


dynamic_docs = fetch_indexed_documents()
if dynamic_docs:
    st.session_state.available_pdfs = dynamic_docs
    if st.session_state.active_document not in dynamic_docs:
        st.session_state.active_document = dynamic_docs[0]
else:
    st.session_state.available_pdfs = ["No PDFs Indexed (Upload a PDF)"]
    st.session_state.active_document = "No PDFs Indexed (Upload a PDF)"


# ==============================================================================
# 2. LEFT SIDEBAR CONTROLS
#    (Uses Streamlit's own native collapse arrow - no custom code needed.
#     This is the reference behavior the right panel now mirrors.)
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">OmniBrain</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Multi-Modal Intelligence Console</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("Document Controls")
    curr_idx = (
        st.session_state.available_pdfs.index(st.session_state.active_document)
        if st.session_state.active_document in st.session_state.available_pdfs
        else 0
    )
    selected_pdf = st.selectbox(
        "Active PDF Asset:",
        options=st.session_state.available_pdfs,
        index=curr_idx,
    )
    st.session_state.active_document = selected_pdf

    uploaded_pdf = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if st.button("Process Uploaded PDF", use_container_width=True):
        if uploaded_pdf is None:
            st.warning("Select a PDF file to process.")
        else:
            with st.status("Processing PDF document...", expanded=True) as status_box:
                try:
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
                    if uploaded_pdf.name not in st.session_state.available_pdfs:
                        st.session_state.available_pdfs.insert(0, uploaded_pdf.name)
                    st.session_state.active_document = uploaded_pdf.name
                    status_box.update(
                        label="Processing Complete", state="complete", expanded=False
                    )
                    st.success(
                        f"Parsed {payload.get('pages', 0)} pages, "
                        f"{payload.get('text_chunks', 0)} text chunks, "
                        f"{payload.get('images', 0)} figures."
                    )
                    st.rerun()
                except requests.RequestException as exc:
                    status_box.update(
                        label="Processing Failed", state="error", expanded=True
                    )
                    st.error(f"Error: {exc}")

    if st.session_state.upload_summary:
        st.caption("Latest Ingestion Metrics")
        st.json(st.session_state.upload_summary)

    st.markdown("---")
    st.subheader("System Status")
    st.markdown(
        '<div class="status-badge">Vector Database: Qdrant Connected</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-badge">SQL Agent: SQLite Registry Ready</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-badge">Orchestrator: LangGraph Active</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Session Telemetry")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.metric("Queries", st.session_state.total_queries)
        st.metric("Est. Tokens", f"{st.session_state.session_tokens:,}")
    with t_col2:
        avg_lat = st.session_state.session_latency_ms / max(
            1, st.session_state.total_queries
        )
        st.metric("Avg Latency", f"{avg_lat:.0f} ms")
        est_cost = (st.session_state.session_tokens / 1000) * 0.00015
        st.metric("Est. Cost", f"${est_cost:.4f}")

    st.markdown("---")
    st.subheader("Maintenance & Data Purge")
    confirm_purge = st.checkbox(
        "Confirm permanent deletion of Qdrant vector index",
        value=False,
    )
    if st.button(
        "Purge Vector Database", disabled=not confirm_purge, use_container_width=True
    ):
        with st.status("Purging vector database...", expanded=True) as status_box:
            try:
                purge_url = INGESTION_API_URL.replace("/upload", "/purge")
                resp = requests.post(purge_url, timeout=30)
                if resp.status_code == 200:
                    status_box.update(
                        label="Database Purged", state="complete", expanded=False
                    )
                    st.success("All indexed text and image vectors deleted.")
                    st.session_state.messages = []
                    st.session_state.last_retrieval_data = {
                        "text": [],
                        "images": [],
                        "citations": [],
                        "sql_query": None,
                        "sql_result": None,
                    }
                    st.rerun()
                else:
                    status_box.update(
                        label="Purge Failed", state="error", expanded=True
                    )
                    st.error(f"Failed: {resp.text}")
            except Exception as exc:
                status_box.update(label="Purge Failed", state="error", expanded=True)
                st.error(f"Error: {exc}")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_latency_ms = 0.0
        st.session_state.total_queries = 0
        st.session_state.last_retrieval_data = {
            "text": [],
            "images": [],
            "citations": [],
            "sql_query": None,
            "sql_result": None,
        }
        st.rerun()


# ==============================================================================
# 3. MAIN CHAT + COLLAPSIBLE RETRIEVAL PANEL
# ==============================================================================

if st.session_state.is_retrieval_panel_open:
    main_chat_col, retrieval_toggle_col, retrieval_panel_col = st.columns(
        [7, 0.35, 3],
        gap="small",
    )
else:
    main_chat_col, retrieval_toggle_col = st.columns(
        [9.65, 0.35],
        gap="small",
    )
    retrieval_panel_col = None


# ------------------------------------------------------------------------------
# SINGLE ARROW CONTROL for the retrieval panel (the only control that toggles
# it — no duplicate buttons anywhere else in the app).
# ------------------------------------------------------------------------------
with retrieval_toggle_col:
    # Marker div lets the CSS `:has()` selector above find and style this
    # specific button as a small floating chevron.
    st.markdown('<span class="retrieval-toggle-marker"></span>', unsafe_allow_html=True)
    arrow_symbol = (
        "\u2039" if st.session_state.is_retrieval_panel_open else "\u203a"
    )  # ‹ / ›
    if st.button(arrow_symbol, key="retrieval_panel_toggle_btn"):
        st.session_state.is_retrieval_panel_open = (
            not st.session_state.is_retrieval_panel_open
        )
        st.rerun()


# ------------------------------------------------------------------------------
# A. MAIN CHAT AREA
#    Foreground: sticky Active Context bar (top)
#    Background: scrollable message history (only this part scrolls)
#    Foreground: floating chat input island (bottom, rendered in section 4)
# ------------------------------------------------------------------------------
with main_chat_col:
    st.markdown(
        f"""
        <div class="chat-top-bar">
            <span class="chat-top-doc">
                Active Context: {st.session_state.active_document}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # `key=` gives this container a stable CSS hook (.st-key-chat_scroll_area)
    # so its height can track the viewport instead of a hardcoded 600px.
    chat_container = st.container(height=600, key="chat_scroll_area")
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "thought_process" in message and message["thought_process"]:
                    with st.expander("Execution Thought Process Log", expanded=False):
                        for step in message["thought_process"]:
                            st.write(
                                f"**[{step.get('agent', 'System')}]**: {step.get('action', '')}"
                            )
                st.markdown(message["content"])

    # Keep the input owned by the same center column as the Active Context bar.
    # CSS pins this element to the bottom while preserving the column width.
    user_query = st.chat_input(
        "Ask a question about the active document or system...",
        key="main_chat_input",
    )


# ------------------------------------------------------------------------------
# B. RIGHT-SIDE RETRIEVAL / CONTEXT PANEL (~30% WIDTH WHEN OPEN)
# ------------------------------------------------------------------------------
if st.session_state.is_retrieval_panel_open and retrieval_panel_col is not None:
    with retrieval_panel_col:
        st.markdown(
            '<div class="sidebar-title">Retrieved Context</div>',
            unsafe_allow_html=True,
        )
        st.caption("Extracted Evidence & Source Data")
        st.markdown("---")

        r_data = st.session_state.last_retrieval_data

        ret_imgs = r_data.get("images", [])
        if ret_imgs:
            st.subheader("Retrieved Figures & Images")
            for img_path in ret_imgs:
                st.markdown(
                    f'<div class="retrieved-card-header">Source Page Asset: {os.path.basename(img_path)}</div>',
                    unsafe_allow_html=True,
                )
                render_image_asset(img_path, os.path.basename(img_path))
                st.markdown("---")

        sql_q = r_data.get("sql_query")
        sql_res = r_data.get("sql_result")
        if sql_q:
            st.subheader("Text-to-SQL Registry Execution")
            st.code(sql_q, language="sql")
            if sql_res:
                if isinstance(sql_res, list):
                    st.dataframe(sql_res, use_container_width=True)
                else:
                    st.write(sql_res)
            st.markdown("---")

        cites = r_data.get("citations", [])
        if cites:
            st.subheader("Verified Claims & Citations")
            for cite in cites:
                claim_text = cite.get("claim") or "Extracted Claim"
                src_pdf = cite.get("source_pdf", "Document")
                pg = cite.get("page", 0)
                cid = cite.get("chart_id")
                badge = f"Chart {cid}" if cid else f"Page {pg}"
                st.markdown(
                    f'<div class="retrieved-card">'
                    f'<div class="retrieved-card-header">[{badge}] {src_pdf}</div>'
                    f'<div class="retrieved-card-text">{claim_text}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")

        ret_texts = r_data.get("text", [])
        if ret_texts:
            st.subheader("Retrieved Text Chunks")
            for idx, item in enumerate(ret_texts[:5], start=1):
                doc_name = item.get("source", item.get("document", "Document"))
                pg_num = item.get("page", "?")
                sc = item.get("score", 0.0)
                txt_content = item.get("text", item.get("content", ""))

                st.markdown(
                    f'<div class="retrieved-card">'
                    f'<div class="retrieved-card-header">Chunk #{idx} | {doc_name} (Page {pg_num}) | Score: {sc:.2f}</div>'
                    f'<div class="retrieved-card-text">{txt_content}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif not ret_imgs and not sql_q and not cites:
            st.info(
                "No active retrieval context for this step. Execute a query to inspect source evidence."
            )


if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    status_container = st.status("Processing query...", expanded=True)
    thought_steps = []
    final_answer = ""
    returned_images = []

    try:
        start_time = time.time()
        status_container.write("Routing query through Agentic Graph...")
        payload = {
            "message": user_query,
            "conversation_id": st.session_state.get("conversation_id"),
            "source_name": st.session_state.active_document,
        }

        response = requests.post(
            BACKEND_API_URL,
            json=payload,
            timeout=360,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0
        st.session_state.session_latency_ms += elapsed_ms
        st.session_state.total_queries += 1

        if response.status_code == 200:
            data = response.json()
            final_answer = (data.get("response") or "").strip()
            thought_steps = data.get("thought_process", [])
            returned_images = data.get("images", [])
            retrieved_text = data.get("retrieved_text", [])
            retrieved_images_full = data.get("retrieved_images", [])
            citations_data = data.get("citations", [])
            sql_query = data.get("sql_query")
            sql_result = data.get("sql_result")

            st.session_state.last_retrieval_data = {
                "text": retrieved_text,
                "images": returned_images,
                "retrieved_images_full": retrieved_images_full,
                "citations": citations_data,
                "sql_query": sql_query,
                "sql_result": sql_result,
            }

            query_lower = user_query.lower()
            is_image_req = any(
                w in query_lower
                for w in (
                    "image",
                    "figure",
                    "diagram",
                    "chart",
                    "visual",
                    "display",
                )
            )
            if is_image_req or returned_images:
                st.session_state.is_retrieval_panel_open = True

            tokens_est = (len(user_query) + len(final_answer)) // 4
            st.session_state.session_tokens += max(10, tokens_est)

            if not final_answer:
                final_answer = "Execution completed."

            for step in thought_steps:
                agent_name = step.get("agent", "Agent")
                action_msg = step.get("action", "")
                status_container.write(f"[{agent_name}]: {action_msg}")

            status_container.update(
                label="Execution Complete",
                state="complete",
                expanded=False,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "thought_process": thought_steps,
                    "images": returned_images,
                    "citations": citations_data,
                }
            )
            st.rerun()

        else:
            status_container.update(label="Backend Error", state="error", expanded=True)
            st.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        status_container.update(label="Connection Failed", state="error", expanded=True)
        st.error("Could not connect to FastAPI server at http://localhost:8000")

    except Exception as e:
        status_container.update(label="Execution Error", state="error", expanded=True)
        st.error(f"An error occurred: {str(e)}")
