import streamlit as st
import time
from engine import processor, summarizer
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Legal Document Summarizer", page_icon="⚖️")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #111111 !important; border-right: 1px solid #333; }
    .sidebar-title { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 5px; }
    .sidebar-sub { font-size: 1.2rem; font-weight: 700; color: #6366f1; margin-bottom: 30px; }

    /* Welcome Hero Section */
    .hero-container { text-align: center; padding: 80px 20px; }
    .hero-title { font-size: 3rem; font-weight: 800; color: #ffffff; margin-bottom: 20px; }
    .hero-subtitle { color: #94a3b8; font-size: 1rem; max-width: 800px; margin: 0 auto 50px auto; line-height: 1.6; }

    /* Feature Highlight Cards */
    .feature-grid { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
    .feature-card {
        background: #111111;
        border: 1px solid #222;
        padding: 40px 30px;
        border-radius: 12px;
        width: 280px;
        text-align: center;
    }
    .feature-label { font-size: 0.75rem; font-weight: 800; color: #6366f1; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; }
    .feature-text { font-size: 0.95rem; color: #94a3b8; line-height: 1.5; }

    /* Professional Analysis Status */
    .status-container { max-width: 600px; margin: 0 auto; padding: 20px; background: #111111; border-radius: 10px; border: 1px solid #222; }
    .status-step { padding: 10px 0; border-bottom: 1px solid #222; color: #6366f1; font-family: monospace; font-size: 0.9rem; }
    
    /* Result Cards (Light theme for readability) */
    .data-card { background: #f8fafc; padding: 25px; border-radius: 12px; color: #1e293b; margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .card-label { font-size: 0.7rem; font-weight: 800; color: #64748b; text-transform: uppercase; margin-bottom: 8px; }

    /* --- CUSTOM TAB STYLING --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #111111; 
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        padding: 10px 20px;
        border: 1px solid #333;
    }

    /* Selected tab styling - matches data-card colors */
    .stTabs [aria-selected="true"] {
        background-color: #f8fafc !important; 
        color: #1e293b !important; 
        font-weight: 800;
        border: 1px solid #e2e8f0 !important;
    }

    /* Underline bar color */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #6366f1;
    }

    /* Source Traceability Boxes */
    .source-card { background: #111111; border: 1px solid #222; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6366f1; }
    .source-index { font-family: monospace; font-size: 0.75rem; color: #6366f1; font-weight: 800; margin-bottom: 8px; display: block; }
    .source-text { color: #94a3b8; font-style: italic; font-size: 0.95rem; line-height: 1.6; }
    
    /* Highlight */
    .highlight { background-color: #ffff00; color: black; font-weight: bold; padding: 2px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Upload The Document </div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Here</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload legal document for synthesis", type="pdf", label_visibility="collapsed")

# --- MAIN PAGE LOGIC ---
if not uploaded_file:
    # LANDING PAGE UI
    st.markdown(f"""
        <div class="hero-container">
            <h1 class="hero-title">Legal Document Summarizer and Analysis System</h1>
            <p class="hero-subtitle">Automated synthesis of judicial records and legal instruments with verbatim source traceability. Upload a document in the sidebar to begin structured analysis.</p>
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-label">Extraction</div>
                    <div class="feature-text">Deterministic detection of Court, Case ID, and Jurisdiction markers.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-label">Synthesis</div>
                    <div class="feature-text">Abstractive consolidation of factual history and judicial observations.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-label">Traceability</div>
                    <div class="feature-text">Verification enabled via direct cross-referencing with original source documentation.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    # DATA PROCESSING & SESSION PERSISTENCE
    if "final_data" not in st.session_state:
        status_box = st.empty()
        with status_box.container():
            st.markdown('<div class="status-container">', unsafe_allow_html=True)
            st.write("### Structured Analysis in Progress")
            steps = [
                "Verifying document integrity...",
                "Mapping jurisdictional markers...",
                "Identifying Petitioner vs Respondent...",
                "Synthesizing factual background...",
                "Finalizing verbatim traceability log..."
            ]
            prog = st.progress(0)
            for i, s in enumerate(steps):
                st.markdown(f'<div class="status-step">> {s}</div>', unsafe_allow_html=True)
                time.sleep(0.6)
                prog.progress((i + 1) / len(steps))
            
            raw_text = processor.get_text(uploaded_file)
            st.session_state.full_text = raw_text
            st.session_state.final_data = summarizer.get_summarized_data(raw_text)
            status_box.empty()

    data = st.session_state.final_data
    full_text = st.session_state.full_text

    # RESULTS DISPLAY
    t1, t2, t3, t4 = st.tabs(["CASE BRIEF", "FACTS & ISSUES", "SOURCE SUMMARY", "SEARCH"])

    with t1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="data-card"><div class="card-label">Court Name</div>{data["court"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="data-card"><div class="card-label">Parties</div><b style="color:#4f46e5; font-size:1.1rem;">{data["parties"]}</b></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="data-card"><div class="card-label">Case Number</div>{data["case_no"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="data-card"><div class="card-label">Jurisdiction</div>{data["jurisdiction"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data-card"><div class="card-label">Executive Summary</div>{data["exec_summary"]}</div>', unsafe_allow_html=True)

    with t2:
        st.markdown(f'<div class="data-card"><div class="card-label">Detailed Case Background</div>{data["background"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data-card"><div class="card-label">Issues for Determination</div>{data["issues"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data-card"><div class="card-label">Court Observations</div>{data["observations"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data-card"><div class="card-label">Final Decision</div>{data["decision"]}</div>', unsafe_allow_html=True)

    with t3:
        st.markdown("### Source Traceability Log")
        st.markdown('<p style="color:#94a3b8;">Direct quotes from the original judgment and supporting the analysis.</p>', unsafe_allow_html=True)
        
        for section, sentences in data["source_log"].items():
            st.markdown(f'<div style="color:white; font-weight:bold; margin: 20px 0 10px 0; border-bottom: 1px solid #333; padding-bottom:5px;">{section}</div>', unsafe_allow_html=True)
            for sent in sentences:
                if "]" in sent:
                    ref, content = sent.split("]", 1)
                    ref = ref + "]"
                else:
                    ref, content = "Reference Found", sent
                    
                st.markdown(f"""
                    <div class="source-card">
                        <span class="source-index">{ref}</span>
                        <div class="source-text">"{content.strip()}"</div>
                    </div>
                """, unsafe_allow_html=True)

    with t4:
        st.markdown("### Search Repository")
        keyword = st.text_input("Enter term to locate in original text:")
        if keyword:
            highlighted = re.sub(f"({re.escape(keyword)})", r'<span class="highlight">\1</span>', full_text, flags=re.I)
            st.markdown(f'<div style="background:white; color:black; padding:30px; line-height:1.8; white-space: pre-wrap; border-radius:10px;">{highlighted}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:white; color:black; padding:30px; white-space: pre-wrap; border-radius:10px;">{full_text}</div>', unsafe_allow_html=True)