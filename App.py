import streamlit as st
import helper
import pickle
import pandas as pd
import json
from io import StringIO
import os

# Page configuration
st.set_page_config(page_title="Duplicate Question Finder",
                   layout="wide",
                   page_icon="🔎",
                   initial_sidebar_state='collapsed')

# ---------- Modern animated styling (replace previous styles) ----------
st.markdown(
    """
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #0b1220, #112240, #0b2a3a);
        background-size: 400% 400%;
        animation: gradientBG 16s ease infinite;
        color: #e6eef8;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Frosted glass card */
    .card {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255,255,255,0.03);
        color: #e6eef8;
    }

    /* Header */
    .header-title { font-size: 42px; font-weight: 800; margin: 6px 0 0 0; color: #ffffff; }
    .header-sub { color: #cfe6ff; margin-top: 6px; }

    /* Inputs */
    textarea, input[type=text] {
        background: rgba(8,12,20,0.6) !important;
        color: #e6eef8 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        padding: 12px !important;
    }
    ::placeholder { color: #9fb6d4 !important; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#7b61ff,#ff6b9f) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 10px 18px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(123,97,255,0.14) !important;
        transition: transform .12s ease-in-out !important;
    }
    .stButton>button:hover { transform: translateY(-3px) scale(1.02) !important; }

    /* Small muted text */
    .muted { color: #9fb6d4; }
    .big { font-size: 18px; font-weight: 600; }
    .score { font-size: 28px; font-weight: 700; }

    /* Improve selectbox and slider visibility */
    div[role="listbox"], .stSlider { color: #e6eef8 !important; }

    /* Floating icon near header */
    .hero-icon {
        width:64px; height:64px; display:inline-block; margin-right: 10px; vertical-align: middle;
        background: radial-gradient(circle at 30% 30%, #9be7ff 0%, rgba(155,231,255,0.15) 40%, rgba(255,255,255,0.02) 100%);
        border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float { 0% { transform: translateY(0px);} 50% { transform: translateY(-8px);} 100% { transform: translateY(0px);} }

    /* Make dataframes slightly transparent */
    .stDataFrame table { background: rgba(255,255,255,0.02); color: #e6eef8; }

    /* Footer */
    .footer { color: #99bfe0; margin-top: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model ----------
@st.cache_resource
def load_model(path="model.pkl"):
    try:
        m = pickle.load(open(path, "rb"))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        m = None
    return m

model = load_model('model.pkl')

# ---------- Utilities ----------
def predict_duplicate(m, q1, q2):
    """Wrap prediction and return (is_dup (bool), score (float|None))"""
    query = helper.query_point_creator(q1, q2)
    try:
        if hasattr(m, 'predict_proba'):
            prob = m.predict_proba(query)
            score = float(prob[0][1])
            is_dup = score >= 0.5
            return is_dup, score
        else:
            pred = m.predict(query)[0]
            return bool(pred), None
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

# ---------- Session state for history ----------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------- Header ----------
col1, col2 = st.columns([8,1])
with col1:
    st.markdown('<div style="display:flex; align-items:center">\n  <div class="hero-icon"></div>\n  <div>\n    <div class="header-title">🔎 Duplicate Question Finder</div>\n    <div class="header-sub">Find if two questions are duplicates — enhanced UI with explanations and history.</div>\n  </div>\n</div>', unsafe_allow_html=True)
with col2:
    if st.button('🎈'):
        st.balloons()

# Main content
with st.container():
    left, right = st.columns([2.2,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form(key='dup_form'):
            q1 = st.text_area('Enter Question 1', placeholder='e.g. How to reverse a linked list in Python?', height=120)
            q2 = st.text_area('Enter Question 2', placeholder='e.g. What is the method to reverse a singly linked list?', height=120)

            ex_col1, ex_col2 = st.columns(2)
            with ex_col1:
                example = st.selectbox('Try examples', ['-- select an example --',
                                                      'Same meaning, different words',
                                                      'Different meaning, similar words',
                                                      'Technical vs non-technical'])
            with ex_col2:
                threshold = st.slider('Decision threshold (when prob available)', 0.0, 1.0, 0.50)

            submit = st.form_submit_button('Find Duplicate')
        st.markdown('</div>', unsafe_allow_html=True)

        # Examples quick-fill (simple hard-coded examples)
        if example != '-- select an example --' and not submit:
            if example == 'Same meaning, different words':
                q1 = "How can I convert a list to a set in Python?"
                q2 = "What is the way to transform a Python list into a set?"
            elif example == 'Different meaning, similar words':
                q1 = "How to make a cake?"
                q2 = "How to make a code?"
            elif example == 'Technical vs non-technical':
                q1 = "What is gradient descent?"
                q2 = "How does gradient work?"

        # Trigger prediction when form submitted
        if submit:
            if not model:
                st.error('Model is not loaded. Check model.pkl path and try again.')
            elif not q1.strip() or not q2.strip():
                st.warning('Please enter both questions.')
            else:
                with st.spinner('Analyzing similarity...'):
                    is_dup, score = predict_duplicate(model, q1, q2)

                # Display results with styling
                if is_dup is None:
                    st.error('Could not compute result.')
                else:
                    if score is None:
                        if is_dup:
                            st.success('✅ Duplicate — model predicted positive label')
                        else:
                            st.info('❌ Not Duplicate — model predicted negative label')
                    else:
                        # use user threshold
                        label = '✅ Duplicate' if score >= threshold else '❌ Not Duplicate'
                        score_percent = f"{score*100:.2f}%"

                        st.markdown(f"<div class='card'>\n  <div class='big muted'>Result</div>\n  <div class='score'>{label}</div>\n  <div class='muted'>Confidence: <strong>{score_percent}</strong></div>\n</div>", unsafe_allow_html=True)

                        # helpful suggestions
                        if score >= 0.9:
                            st.success('Very confident — these look like duplicates.')
                        elif score >= 0.7:
                            st.info('Likely duplicates but double-check for domain-specific tokens (names, dates).')
                        elif score >= 0.4:
                            st.warning('Low confidence — consider manual review or more context.')
                        else:
                            st.info('Probably not duplicates.')

                    # Save to history
                    st.session_state.history.insert(0, {
                        'q1': q1,
                        'q2': q2,
                        'is_duplicate': bool(is_dup),
                        'score': score
                    })

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('Tips')
        st.markdown('- Keep questions short and focused.')
        st.markdown('- Remove stopwords and normalize numbers if you pre-process.')
        st.markdown('- For high precision, raise the decision threshold.')
        st.markdown('- Use domain-specific tokenizers when questions contain code or math.')

        st.markdown('---')
        st.subheader('Quick controls')
        if st.button('Clear history'):
            st.session_state.history = []
            st.success('History cleared')

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- History and downloads ----------
if st.session_state.history:
    st.markdown('---')
    st.subheader('Recent checks')
    df = pd.DataFrame(st.session_state.history)
    # show readable columns
    df_display = df.copy()
    df_display['score'] = df_display['score'].apply(lambda x: f"{x:.3f}" if x is not None else 'N/A')
    df_display['is_duplicate'] = df_display['is_duplicate'].apply(lambda x: 'Yes' if x else 'No')

    st.dataframe(df_display, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button('Download history (CSV)', csv, file_name='dup_history.csv')

# ---------- Footer ----------
st.markdown('<div class="footer">Built with ❤️ using Streamlit — feel free to customize the layout, colors, and model integration.</div>', unsafe_allow_html=True)
