import streamlit as st
import os
import re
import pandas as pd
from datetime import datetime
from os.path import join
from src import (
    preprocess_and_load_df,
    get_from_user,
    ask_question,
    models,
)
from dotenv import load_dotenv
from huggingface_hub import HfApi
import plotly.graph_objects as go
import uuid

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VayuChat",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Single, minimal CSS ───────────────────────────────────────────────────
st.markdown("""<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Clean base */
.stApp { background: #fff; }
.main .block-container {
    padding-top: 0.5rem;
    padding-bottom: 90px;
    max-width: 960px;
    margin: 0 auto;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #eee; }
[data-testid="stSidebar"] .element-container { margin-bottom: 0.2rem !important; }
[data-testid="stSidebar"] button[kind="secondary"] {
    text-align: left !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.7rem !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    background: #fff !important;
    color: #374151 !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #f3f4f6 !important;
    border-color: #d1d5db !important;
}

/* Chat message tweaks */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
}

/* Chat input — leave it mostly native, just clean up */
[data-testid="stChatInput"] {
    border-top: 1px solid #f0f0f0 !important;
}
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    border-color: #e5e7eb !important;
    border-radius: 16px !important;
    background: #fafafa !important;
}
[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
    border-color: #bbb !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] button {
    border-radius: 50% !important;
}

/* Remove extra element spacing */
.element-container { margin-bottom: 0.4rem !important; }

/* Feedback thumb buttons — ghost style inside chat messages */
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    padding: 2px 6px !important;
    opacity: 0.35;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    min-height: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] button:hover {
    opacity: 1 !important;
    background: #f3f4f6 !important;
    border-color: #e5e7eb !important;
}
</style>""", unsafe_allow_html=True)

# ── Init ───────────────────────────────────────────────────────────────────
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")
gemini_token = os.getenv("GOOGLE_API_KEY")
self_path = os.path.dirname(os.path.abspath(__file__))

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "responses" not in st.session_state:
    st.session_state.responses = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Feedback upload (unchanged) ────────────────────────────────────────────
def upload_feedback(feedback, error, output, last_prompt, code, status):
    try:
        if not hf_token or hf_token.strip() == "":
            return False
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "feedback_score": feedback.get("score", ""),
            "feedback_comment": feedback.get("text", ""),
            "user_prompt": last_prompt,
            "ai_output": str(output),
            "generated_code": code or "",
            "error_message": error or "",
            "is_image_output": status.get("is_image", False),
            "success": not bool(error),
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rid = str(uuid.uuid4())[:8]
        fname = f"feedback_{ts}_{rid}.md"
        local = f"/tmp/{fname}"
        with open(local, "w", encoding="utf-8") as f:
            f.write(f"# VayuChat Feedback\n\n"
                    f"- Timestamp: {feedback_data['timestamp']}\n"
                    f"- Prompt: {feedback_data['user_prompt']}\n"
                    f"- Score: {feedback_data['feedback_score']}\n"
                    f"- Output: {feedback_data['ai_output']}\n"
                    f"- Code: {feedback_data['generated_code']}\n"
                    f"- Error: {feedback_data['error_message']}\n")
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f"data/{fname}",
            repo_id="SustainabilityLabIITGN/VayuChat_Feedback",
            repo_type="dataset",
        )
        if status.get("is_image") and isinstance(output, str) and os.path.exists(output):
            api.upload_file(
                path_or_fileobj=output,
                path_in_repo=f"data/feedback_{ts}_{rid}_plot.png",
                repo_id="SustainabilityLabIITGN/VayuChat_Feedback",
                repo_type="dataset",
            )
        os.remove(local)
        return True
    except Exception:
        return False


# ── Models ─────────────────────────────────────────────────────────────────
available_models = list(models.keys()) if gemini_token and gemini_token.strip() else []
if not available_models:
    st.error("No API keys available. Set GOOGLE_API_KEY in .env")
    st.stop()

default_index = available_models.index("gemini-3-flash") if "gemini-3-flash" in available_models else 0

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:center; gap:10px; padding:0.8rem 0 0.5rem; border-bottom:1px solid #f0f0f0; margin-bottom:0.5rem;">
    <svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M8 18h20a6 6 0 1 0-6-6" stroke="#111" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M6 26h28a5 5 0 1 1-5 5" stroke="#111" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M10 34h16a4 4 0 1 1-4 4" stroke="#111" stroke-width="2.5" stroke-linecap="round"/>
    </svg>
    <span style="font-size:1.4rem; font-weight:700; color:#111; letter-spacing:-0.5px;">VayuChat</span>
    <span style="color:#999; font-size:0.85rem;">AI Air Quality Analysis</span>
</div>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return preprocess_and_load_df(join(self_path, "Data.csv"))

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────
selected_prompt = None

with st.sidebar:
    model_name = st.selectbox("Model", available_models, index=default_index)
    st.markdown("---")
    st.markdown("**Quick queries**")

    @st.cache_data
    def load_questions():
        qf = join(self_path, "questions.txt")
        if os.path.exists(qf):
            with open(qf, "r") as f:
                return [q.strip() for q in f if q.strip()]
        return []

    questions = load_questions()

    _cat_kw = {
        "map": ["map", "geographic", "spatial"],
        "ncap": ["ncap", "funding"],
        "met": ["wind", "temperature", "humidity", "rainfall"],
        "multi": ["ozone", "no2", "correlation"],
    }

    def _cat(q):
        ql = q.lower()
        for c, kws in _cat_kw.items():
            if any(k in ql for k in kws):
                return c
        return "other"

    if questions:
        starters = [q for q in questions if _cat(q) != "map"][:8]
        with st.expander("Getting Started", expanded=True):
            for i, q in enumerate(starters):
                if st.button(q, key=f"s_{i}", use_container_width=True):
                    selected_prompt = q

        starter_set = set(starters)
        sections = [
            ("Maps & Geography", "map"),
            ("NCAP & Policy", "ncap"),
            ("Weather & Meteorology", "met"),
            ("Correlations", "multi"),
            ("More", "other"),
        ]
        for title, cat in sections:
            qs = [q for q in questions if _cat(q) == cat and q not in starter_set]
            if qs:
                with st.expander(title, expanded=False):
                    for i, q in enumerate(qs):
                        if st.button(q, key=f"{cat}_{i}", use_container_width=True):
                            selected_prompt = q

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.responses = []
        st.session_state.processing = False
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ── Display helpers ────────────────────────────────────────────────────────

def _resolve_image(content):
    """Return a valid image path or None."""
    if not isinstance(content, str):
        return None
    if content.endswith((".png", ".jpg", ".jpeg")) and os.path.exists(content):
        return content
    m = re.search(r"([^/\\]+\.(?:png|jpg|jpeg))", content)
    if m and os.path.exists(m.group(1)):
        return m.group(1)
    return None


def show_response(response, rid=0):
    """Render one chat message using Streamlit's native chat UI."""
    role = response.get("role", "assistant")
    content = response.get("content", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
        return

    # Assistant
    with st.chat_message("assistant"):
        error = response.get("error")
        is_plotly = isinstance(content, go.Figure)
        is_df = isinstance(content, pd.DataFrame)
        img_path = _resolve_image(content)

        if error:
            st.error(f"{error}")
            st.caption("Try rephrasing your question or being more specific.")
        elif is_plotly:
            st.plotly_chart(content, use_container_width=True, key=f"plotly_{rid}")
        elif is_df:
            st.dataframe(content, use_container_width=True, hide_index=True)
        elif img_path:
            st.image(img_path, use_container_width=True)
        else:
            st.markdown(content if isinstance(content, str) else str(content))

        if response.get("gen_code"):
            with st.expander("View generated code"):
                st.code(response["gen_code"], language="python")

        # Inline feedback — subtle thumbs
        if not error:
            fkey = f"fb_{rid}"
            if "feedback" in response:
                score = response["feedback"].get("score", "")
                icon = "\U0001f44d" if score == "Good" else "\U0001f44e"
                st.caption(f"{icon} Thanks!")
            else:
                c1, c2, _ = st.columns([0.06, 0.06, 0.88])
                with c1:
                    if st.button("\U0001f44d", key=f"{fkey}_up"):
                        st.session_state.responses[rid]["feedback"] = {"score": "Good", "text": ""}
                        st.rerun()
                with c2:
                    if st.button("\U0001f44e", key=f"{fkey}_dn"):
                        st.session_state.responses[rid]["feedback"] = {"score": "Bad", "text": ""}
                        st.rerun()


# ── Welcome screen ─────────────────────────────────────────────────────────
if not st.session_state.responses and not st.session_state.get("processing"):
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem 2rem; color:#666;">
        <p style="font-size:1.05rem; max-width:600px; margin:0 auto; line-height:1.6;">
            Ask anything about air quality across <strong style="color:#333;">288 Indian cities</strong>
            from 2017 to 2024. Get instant answers, charts, tables, and interactive maps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    caps = [
        ("Interactive Maps", "Show PM2.5 across India on a map"),
        ("Trend Charts", "Plot yearly PM2.5 trends from 2020 to 2024"),
        ("Rankings", "Rank top 10 most polluted cities"),
        ("Comparisons", "Compare north vs south India PM2.5 in a plot"),
    ]
    for col, (label, query) in zip(cols, caps):
        with col:
            if st.button(label, key=f"w_{label}", use_container_width=True, help=query):
                selected_prompt = query

# ── Chat history ───────────────────────────────────────────────────────────
for rid, resp in enumerate(st.session_state.responses):
    show_response(resp, rid=rid)

# ── Chat input ─────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about air quality...", key="main_chat")

if selected_prompt:
    prompt = selected_prompt

if st.session_state.get("follow_up_prompt") and not st.session_state.get("processing"):
    prompt = st.session_state.follow_up_prompt
    st.session_state.follow_up_prompt = None

# ── Handle new query ───────────────────────────────────────────────────────
if prompt and not st.session_state.get("processing"):
    if "last_prompt" in st.session_state:
        if prompt == st.session_state["last_prompt"] and model_name == st.session_state.get("last_model_name"):
            prompt = None

    if prompt:
        st.session_state.responses.append(get_from_user(prompt))
        st.session_state.processing = True
        st.session_state.current_model = model_name
        st.session_state.current_question = prompt
        st.rerun()

# ── Process ────────────────────────────────────────────────────────────────
if st.session_state.get("processing"):
    model_name = st.session_state.get("current_model")
    prompt = st.session_state.get("current_question")

    with st.chat_message("assistant"):
        with st.spinner(f"Processing with {model_name}..."):
            try:
                response = ask_question(model_name=model_name, question=prompt)
                if not isinstance(response, dict):
                    response = {"content": "Error: Invalid response", "error": "Invalid response format"}
                response.setdefault("role", "assistant")
                response.setdefault("content", "No content generated")
                response.setdefault("gen_code", "")
                response.setdefault("last_prompt", prompt)
                response.setdefault("error", None)
                response.setdefault("timestamp", datetime.now().strftime("%H:%M"))
            except Exception as e:
                response = {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {e}",
                    "gen_code": "",
                    "last_prompt": prompt,
                    "error": str(e),
                    "timestamp": datetime.now().strftime("%H:%M"),
                }

    st.session_state.responses.append(response)
    st.session_state["last_prompt"] = prompt
    st.session_state["last_model_name"] = model_name
    st.session_state.processing = False
    for k in ("current_model", "current_question"):
        st.session_state.pop(k, None)
    st.rerun()
