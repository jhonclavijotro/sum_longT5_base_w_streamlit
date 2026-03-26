import streamlit as st
import torch
import tempfile
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSummarizer",
    page_icon="📄",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Root palette */
:root {
    --ink:      #3e3e6d;
    --paper:    #f5f0e8;
    --accent:   #c84b31;
    --muted:    #7a7060;
    --border:   #d6cfc0;
    --surface:  #eee8da;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--paper);
}

[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent; }

/* ── Hero title ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    color: var(--paper);
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.hero h1 span { color: var(--accent); font-style: italic; }
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}

/* ── Card wrapper ── */
.card {
    background: rgba(245,240,232,0.04);
    border: 1px solid rgba(214,207,192,0.15);
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: var(--paper) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(245,240,232,0.07) !important;
    border-color: var(--border) !important;
    color: var(--paper) !important;
    border-radius: 8px !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    color: var(--paper) !important;
    font-size: 0.88rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background: rgba(245,240,232,0.04) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    color: var(--paper) !important;
}

/* ── Generate button ── */
div[data-testid="stButton"] > button {
    background: var(--accent) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 2.2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: opacity 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
}

/* ── Summary output box ── */
.summary-box {
    background: #ffffff;
    color: #111111;
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    margin-top: 0.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.97rem;
    line-height: 1.75;
    border: 1px solid #dddddd;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    white-space: pre-wrap;
    word-break: break-word;
}
.summary-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
}

/* ── Divider ── */
hr { border-color: rgba(214,207,192,0.12) !important; }

/* ── Status / info messages ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    color: var(--paper) !important;
    background: rgba(245,240,232,0.07) !important;
}

/* Expander */
details summary {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper: cached model loader ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Función que retorna el tokenizer, el modelo seleccionado y el enlace al device
    Inputs: model_name de tipo str
    """
    from transformers import AutoTokenizer, LongT5ForConditionalGeneration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Doc<span>Summarizer</span></h1>
    <p>long-t5 · extractive summarization · pdf &amp; text</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Model selection ───────────────────────────────────────────────────────────
st.markdown('<div class="card-label">01 — Model</div>', unsafe_allow_html=True)
# Se incluye el modelo tuneado pero no se carga por peso
MODEL_OPTIONS = {
    "Base  (tglobal-base)  · fastest":  "google/long-t5-tglobal-base",
    "Base (Fine-tuned) · balanced": "./final_model_ready",
}

model_label = st.selectbox(
    "Select a LongT5 model variant",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)
model_name = MODEL_OPTIONS[model_label]

st.markdown("---")

# ── Document type & upload ────────────────────────────────────────────────────
st.markdown('<div class="card-label">02 — Document</div>', unsafe_allow_html=True)

doc_type = st.radio(
    "Document format",
    options=["PDF (.pdf)", "Plain text (.tex / .txt)"],
    horizontal=True,
)
is_pdf = doc_type.startswith("PDF")

uploaded_file = st.file_uploader(
    label="Upload your document",
    type=["pdf"] if is_pdf else ["tex", "txt"],
    help="PDF pages are selected below; text files are read in full.",
)

if is_pdf and uploaded_file:
    with st.expander("PDF page range (optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            page_start = st.number_input("First page (0-indexed)", min_value=0, value=0, step=1)
        with col2:
            page_end = st.number_input("Last page (inclusive)", min_value=0, value=4, step=1)

st.markdown("---")

# ── Generation parameters ─────────────────────────────────────────────────────
with st.expander("⚙  Generation settings", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_new_tokens = st.number_input("Max new tokens", min_value=50, max_value=1024, value=300, step=50)
    with col_b:
        num_beams = st.number_input("Num beams", min_value=1, max_value=10, value=4, step=1)
    with col_c:
        length_penalty = st.number_input("Length penalty", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────
run = st.button("✦  Generate Summary")

if run:
    if uploaded_file is None:
        st.warning("Please upload a document first.")
    else:
        # ── Extract text ──────────────────────────────────────────────────────
        raw_text = ""
        try:
            if is_pdf:
                import pypdf
                pages_to_read = list(range(int(page_start), int(page_end) + 1))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    with open(tmp_path, "rb") as f:
                        reader = pypdf.PdfReader(f)
                        total = len(reader.pages)
                        pages_to_read = [p for p in pages_to_read if p < total]
                        if not pages_to_read:
                            st.error(f"The PDF only has {total} page(s). Adjust the page range.")
                            st.stop()
                        for p in pages_to_read:
                            raw_text += reader.pages[p].extract_text() or ""
                finally:
                    os.unlink(tmp_path)
            else:
                raw_text = uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error reading document: {e}")
            st.stop()

        if not raw_text.strip():
            st.error("Could not extract any text from the document.")
            st.stop()

        # ── Load model ────────────────────────────────────────────────────────
        with st.spinner(f"Loading **{model_name}** — this may take a moment on first run…"):
            try:
                tokenizer, model, device = load_model(model_name)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        # ── Tokenize & generate ───────────────────────────────────────────────
        with st.spinner("Generating summary…"):
            try:
                prompt = "summarize: " + raw_text
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=16384,
                )
                inputs = inputs.to(device)

                out_tokens = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    num_beams=int(num_beams),
                    length_penalty=float(length_penalty),
                    early_stopping=True,
                )
                summary = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
            except Exception as e:
                st.error(f"Error during generation: {e}")
                st.stop()

        # ── Display result ────────────────────────────────────────────────────
        st.markdown('<div class="summary-header">Generated Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        st.download_button(
            label="↓  Download summary as .txt",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
        )