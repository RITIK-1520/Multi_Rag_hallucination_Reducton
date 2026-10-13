"""
app.py – Streamlit UI for Multi-Agent RAG · BNS Legal Intelligence System.

Run with:
    streamlit run app.py
"""

import sys
import os
import time
import json

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    is_api_available, DEFAULT_TOP_K, MAX_TOP_K,
    DEFAULT_HALLUCINATION_THRESHOLD, DEFAULT_EVIDENCE_THRESHOLD,
    DEFAULT_MAX_REGENERATION_ATTEMPTS, DEFAULT_TEMPERATURE,
    MAX_TEMPERATURE, OUTPUTS_DIR,
)
from src.data_loader import load_documents
from src.chunking import chunk_all_documents
from src.embeddings import generate_embeddings
from src.vector_store import VectorStore
from src.orchestrator import Orchestrator
from src.evaluation import load_evaluation_questions, run_evaluation, save_results_csv, save_results_json
from src.utils import ensure_dir


# ══════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BNS Legal Intelligence · Multi-Agent RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# Premium Dark Theme CSS with 3D Effects
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Reset & Theme ── */
    :root {
        --bg-primary: #0a0a1a;
        --bg-secondary: #0f0f2a;
        --bg-card: rgba(15, 15, 42, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --accent-cyan: #22d3ee;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-subtle: rgba(99, 102, 241, 0.15);
        --border-glow: rgba(99, 102, 241, 0.4);
        --shadow-3d: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(99, 102, 241, 0.1);
        --shadow-card: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(99, 102, 241, 0.05);
    }

    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f0f2a 0%, #1a1a3e 30%, #0f0f2a 70%, #1a0a2e 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-3d);
        transform: perspective(1000px) rotateX(1deg);
        transition: transform 0.4s ease;
    }
    .hero-banner:hover {
        transform: perspective(1000px) rotateX(0deg) translateY(-2px);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        animation: heroFloat 8s ease-in-out infinite;
    }
    @keyframes heroFloat {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(2%, -1%) rotate(0.5deg); }
        66% { transform: translate(-1%, 1%) rotate(-0.5deg); }
    }
    .hero-banner .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    .hero-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f1f5f9, #6366f1, #8b5cf6, #22d3ee);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .hero-banner .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 999px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    .hero-badge .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── 3D Card System ── */
    .card-3d {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-card);
        transform: perspective(800px) rotateX(0deg) rotateY(0deg);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    .card-3d:hover {
        transform: perspective(800px) rotateX(-1deg) rotateY(1deg) translateY(-4px);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.4), 0 0 30px rgba(99, 102, 241, 0.15);
        border-color: var(--border-glow);
    }
    .card-3d::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .card-3d:hover::after { opacity: 1; }

    /* ── Metric Cards with 3D Depth ── */
    .metric-3d {
        background: linear-gradient(145deg, rgba(15, 15, 42, 0.9), rgba(20, 20, 50, 0.8));
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transform: perspective(600px) rotateX(2deg);
        transition: all 0.35s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-3d:hover {
        transform: perspective(600px) rotateX(0deg) translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 20px var(--accent-glow);
    }
    .metric-3d .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    .metric-3d .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-3d .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .metric-3d.accent-indigo { border-top: 3px solid #6366f1; }
    .metric-3d.accent-cyan { border-top: 3px solid #22d3ee; }
    .metric-3d.accent-emerald { border-top: 3px solid #10b981; }
    .metric-3d.accent-rose { border-top: 3px solid #f43f5e; }
    .metric-3d.accent-amber { border-top: 3px solid #f59e0b; }

    /* ── Streaming Answer Box ── */
    .answer-container {
        background: linear-gradient(135deg, rgba(15, 15, 42, 0.95), rgba(20, 20, 60, 0.9));
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-3d);
        position: relative;
        overflow: hidden;
    }
    .answer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #22d3ee, #10b981);
        background-size: 300% 100%;
        animation: borderGlow 3s ease infinite;
    }
    @keyframes borderGlow {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 0%; }
    }
    .answer-container .answer-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 1rem;
    }
    .answer-container .answer-text {
        color: var(--text-primary);
        font-size: 1.05rem;
        line-height: 1.8;
        font-weight: 400;
    }
    .verified-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: #fb7185;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .warning-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Claim Cards ── */
    .claim-card {
        background: rgba(15, 15, 42, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        transform: perspective(400px) rotateX(0deg);
    }
    .claim-card:hover {
        transform: perspective(400px) rotateX(-1deg) translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .claim-card.supported {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(15, 15, 42, 0.6));
    }
    .claim-card.partial {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(15, 15, 42, 0.6));
    }
    .claim-card.unsupported {
        border-left: 4px solid #f43f5e;
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.05), rgba(15, 15, 42, 0.6));
    }
    .claim-card .claim-status {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .claim-card .claim-text {
        color: var(--text-primary);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .claim-card .claim-reason {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.3rem;
        font-style: italic;
    }

    /* ── Agent Trace Timeline ── */
    .timeline-step {
        background: rgba(15, 15, 42, 0.5);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s ease;
        transform: perspective(400px) rotateY(0deg);
    }
    .timeline-step:hover {
        transform: perspective(400px) rotateY(1deg) translateX(4px);
        border-color: var(--border-glow);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.1);
    }
    .timeline-step .step-icon {
        font-size: 1.5rem;
        flex-shrink: 0;
    }
    .timeline-step .step-content {
        flex: 1;
    }
    .timeline-step .step-agent {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    .timeline-step .step-detail {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .timeline-step .step-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-cyan);
        flex-shrink: 0;
    }

    /* ── Evidence Chunk Cards ── */
    .evidence-card {
        background: linear-gradient(135deg, rgba(15, 15, 42, 0.8), rgba(20, 20, 55, 0.6));
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.35s ease;
        transform: perspective(500px) rotateX(0deg);
    }
    .evidence-card:hover {
        transform: perspective(500px) rotateX(-1deg) translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        border-color: var(--border-glow);
    }
    .evidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .evidence-source {
        font-weight: 600;
        color: #a5b4fc;
        font-size: 0.85rem;
    }
    .evidence-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
    }
    .score-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .score-mid {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .score-low {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .evidence-text {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.6;
        max-height: 120px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0f0f2a 50%, #0a0a1a 100%) !important;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }

    /* ── Query Input ── */
    .query-section {
        background: linear-gradient(135deg, rgba(15, 15, 42, 0.9), rgba(20, 20, 55, 0.8));
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-card);
        transform: perspective(800px) rotateX(1deg);
        transition: all 0.3s ease;
    }
    .query-section:hover {
        transform: perspective(800px) rotateX(0deg);
    }

    /* ── Floating 3D Orbs (background decoration) ── */
    .orb-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.15;
        animation: orbFloat 12s ease-in-out infinite;
    }
    .orb-1 {
        width: 300px;
        height: 300px;
        background: #6366f1;
        top: 10%;
        left: 5%;
        animation-delay: 0s;
    }
    .orb-2 {
        width: 250px;
        height: 250px;
        background: #8b5cf6;
        top: 60%;
        right: 10%;
        animation-delay: 4s;
    }
    .orb-3 {
        width: 200px;
        height: 200px;
        background: #22d3ee;
        bottom: 20%;
        left: 30%;
        animation-delay: 8s;
    }
    @keyframes orbFloat {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -20px) scale(1.05); }
        50% { transform: translate(-20px, 30px) scale(0.95); }
        75% { transform: translate(15px, 15px) scale(1.02); }
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 15, 42, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .section-header .icon {
        font-size: 1.3rem;
    }
    .section-header .title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .section-header .line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-subtle), transparent);
    }

    /* ── Button Overrides ── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        transform: perspective(200px) rotateX(0deg);
    }
    .stButton > button:hover {
        transform: perspective(200px) rotateX(-2deg) translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# Floating 3D background orbs
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="orb-container">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# Initialise session state & resources
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Building knowledge base from BNS documents...")
def init_system():
    """Load documents (including Final_BNS.pdf), build/load FAISS index."""
    vs = VectorStore()
    if vs.load():
        return vs

    docs = load_documents()
    if not docs:
        st.error("No documents found! Please place your BNS PDF in the project root or data/ directory.")
        st.stop()
    chunks = chunk_all_documents(docs)
    texts = [c["text"] for c in chunks]
    embs = generate_embeddings(texts)
    vs.build_index(chunks, embs)
    vs.save()
    return vs


vs = init_system()
orchestrator = Orchestrator(vs)


# ══════════════════════════════════════════════════════════
# Helper: Fast Word-by-Word Stream Generator & 3D Visualizer
# ══════════════════════════════════════════════════════════
def stream_word_by_word(text: str, delay: float = 0.012):
    """Yields text word-by-word at high speed for a rapid, fluid typing effect."""
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(delay)


def render_3d_legal_constellation():
    """Render an interactive 3D WebGL knowledge graph of BNS laws and agent pipeline."""
    three_js_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                overflow: hidden;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            #canvas-container {
                width: 100%;
                height: 250px;
                position: relative;
                border-radius: 20px;
                overflow: hidden;
                background: radial-gradient(circle at 50% 50%, #121236 0%, #0a0a1a 100%);
                border: 1px solid rgba(99, 102, 241, 0.3);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6), inset 0 0 30px rgba(99, 102, 241, 0.15);
                cursor: grab;
            }
            #canvas-container:active { cursor: grabbing; }
            #hud-overlay {
                position: absolute;
                top: 14px;
                left: 16px;
                color: #a5b4fc;
                font-size: 11px;
                font-family: monospace;
                letter-spacing: 0.06em;
                pointer-events: none;
                z-index: 10;
            }
            .hud-badge {
                display: inline-block;
                padding: 3px 8px;
                background: rgba(99, 102, 241, 0.25);
                border: 1px solid rgba(99, 102, 241, 0.45);
                border-radius: 4px;
                margin-right: 6px;
                color: #22d3ee;
                font-weight: 700;
            }
            #hint {
                position: absolute;
                bottom: 10px;
                right: 16px;
                color: rgba(148, 163, 184, 0.65);
                font-size: 10px;
                font-family: monospace;
                pointer-events: none;
            }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
        <div id="canvas-container">
            <div id="hud-overlay">
                <span class="hud-badge">3D BNS MESH</span>
                <span>1,388 Chunks · 6 AI Agents</span>
            </div>
            <div id="hint">✦ Drag / Hover to rotate 3D mesh</div>
        </div>

        <script>
            const container = document.getElementById('canvas-container');
            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0a0a1a, 0.002);

            const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.z = 175;

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            // Lighting
            const ambient = new THREE.AmbientLight(0xffffff, 0.9);
            scene.add(ambient);

            const light1 = new THREE.PointLight(0x6366f1, 2.5, 300);
            light1.position.set(100, 100, 120);
            scene.add(light1);

            const light2 = new THREE.PointLight(0x22d3ee, 2.5, 300);
            light2.position.set(-100, -80, 100);
            scene.add(light2);

            // 3D Master Group
            const masterGroup = new THREE.Group();
            scene.add(masterGroup);

            // 1. Central Wireframe Sphere (BNS Codex Mesh)
            const globeGeo = new THREE.IcosahedronGeometry(46, 2);
            const globeMat = new THREE.MeshStandardMaterial({
                color: 0x4f46e5,
                wireframe: true,
                transparent: true,
                opacity: 0.38,
                roughness: 0.1
            });
            const globeMesh = new THREE.Mesh(globeGeo, globeMat);
            masterGroup.add(globeMesh);

            // 2. Inner Glowing Core
            const coreGeo = new THREE.IcosahedronGeometry(20, 1);
            const coreMat = new THREE.MeshBasicMaterial({
                color: 0x22d3ee,
                wireframe: true,
                transparent: true,
                opacity: 0.65
            });
            const coreMesh = new THREE.Mesh(coreGeo, coreMat);
            masterGroup.add(coreMesh);

            // 3. Orbiting Knowledge Nodes (Sections + Verification Agents)
            const nodeCount = 36;
            const nodes = [];
            const nodeGroup = new THREE.Group();
            masterGroup.add(nodeGroup);

            const nodeGeo = new THREE.SphereGeometry(2.0, 12, 12);
            const palette = [0x6366f1, 0x22d3ee, 0x10b981, 0xf43f5e, 0x8b5cf6, 0xfbbf24];

            for (let i = 0; i < nodeCount; i++) {
                const phi = Math.acos(-1 + (2 * i) / nodeCount);
                const theta = Math.sqrt(nodeCount * Math.PI) * phi;
                const r = 68 + (i % 4) * 10;

                const x = r * Math.cos(theta) * Math.sin(phi);
                const y = r * Math.sin(theta) * Math.sin(phi);
                const z = r * Math.cos(phi);

                const col = palette[i % palette.length];
                const mat = new THREE.MeshBasicMaterial({ color: col });
                const mesh = new THREE.Mesh(nodeGeo, mat);
                mesh.position.set(x, y, z);
                nodeGroup.add(mesh);
                nodes.push({ mesh: mesh, basePos: new THREE.Vector3(x, y, z), phase: i * 0.4 });
            }

            // 4. Neural Connective Lines
            const linePositions = [];
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dist = nodes[i].basePos.distanceTo(nodes[j].basePos);
                    if (dist < 42) {
                        linePositions.push(
                            nodes[i].basePos.x, nodes[i].basePos.y, nodes[i].basePos.z,
                            nodes[j].basePos.x, nodes[j].basePos.y, nodes[j].basePos.z
                        );
                    }
                }
            }
            const lineGeo = new THREE.BufferGeometry();
            lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
            const lineMat = new THREE.LineBasicMaterial({
                color: 0x6366f1,
                transparent: true,
                opacity: 0.28
            });
            const linesMesh = new THREE.LineSegments(lineGeo, lineMat);
            masterGroup.add(linesMesh);

            // 5. Star/Vector Particles
            const particleCount = 180;
            const pPositions = new Float32Array(particleCount * 3);
            for (let i = 0; i < particleCount * 3; i += 3) {
                pPositions[i] = (Math.random() - 0.5) * 300;
                pPositions[i + 1] = (Math.random() - 0.5) * 200;
                pPositions[i + 2] = (Math.random() - 0.5) * 250;
            }
            const pGeo = new THREE.BufferGeometry();
            pGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
            const pMat = new THREE.PointsMaterial({
                color: 0x94a3b8,
                size: 1.4,
                transparent: true,
                opacity: 0.45
            });
            const particles = new THREE.Points(pGeo, pMat);
            scene.add(particles);

            // Interactivity: Drag & Mouse Tracking
            let isDragging = false;
            let prevMouseX = 0, prevMouseY = 0;
            let targetRotX = 0, targetRotY = 0;

            container.addEventListener('mousedown', (e) => {
                isDragging = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            });
            window.addEventListener('mouseup', () => { isDragging = false; });
            window.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const dx = e.clientX - prevMouseX;
                    const dy = e.clientY - prevMouseY;
                    masterGroup.rotation.y += dx * 0.008;
                    masterGroup.rotation.x += dy * 0.008;
                    prevMouseX = e.clientX;
                    prevMouseY = e.clientY;
                }
            });

            container.addEventListener('mousemove', (e) => {
                if (!isDragging) {
                    const rect = container.getBoundingClientRect();
                    const mx = ((e.clientX - rect.left) / container.clientWidth) * 2 - 1;
                    const my = -((e.clientY - rect.top) / container.clientHeight) * 2 + 1;
                    targetRotY = mx * 0.4;
                    targetRotX = -my * 0.3;
                }
            });

            window.addEventListener('resize', () => {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            });

            // Animation Loop
            let clock = new THREE.Clock();
            function animate() {
                requestAnimationFrame(animate);
                const elapsed = clock.getElapsedTime();

                if (!isDragging) {
                    masterGroup.rotation.y += 0.004;
                    masterGroup.rotation.x += (targetRotX - masterGroup.rotation.x) * 0.04;
                }
                globeMesh.rotation.x += 0.002;
                globeMesh.rotation.z += 0.001;
                coreMesh.rotation.y -= 0.008;
                particles.rotation.y += 0.0006;

                // Pulsate nodes
                for (let i = 0; i < nodes.length; i++) {
                    const s = 1 + 0.3 * Math.sin(elapsed * 3.5 + nodes[i].phase);
                    nodes[i].mesh.scale.set(s, s, s);
                }

                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_html, height=255)


# ══════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">⚖️</div>
        <div style="font-size: 1.1rem; font-weight: 800; 
             background: linear-gradient(135deg, #6366f1, #22d3ee);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            BNS Legal RAG
        </div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 4px; letter-spacing: 0.1em; text-transform: uppercase;">
            Powered by Multi-Agent AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Mode
    if is_api_available():
        st.markdown("""
        <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); 
             border-radius: 10px; padding: 8px 12px; text-align: center; margin-bottom: 1rem;">
            <span style="color: #34d399; font-weight: 600; font-size: 0.85rem;">🟢 Live Gemini Mode</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); 
             border-radius: 10px; padding: 8px 12px; text-align: center; margin-bottom: 1rem;">
            <span style="color: #fbbf24; font-weight: 600; font-size: 0.85rem;">⚠️ Offline Mode</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ Pipeline Settings")
    top_k = st.slider("Top-K Retrieval", 1, MAX_TOP_K, DEFAULT_TOP_K)
    hallucination_threshold = st.slider(
        "Hallucination Threshold", 0.1, 1.0, DEFAULT_HALLUCINATION_THRESHOLD, 0.05,
        help="Risk above this triggers rejection/refinement",
    )
    max_attempts = st.slider("Max Regeneration Attempts", 0, 5,
                             DEFAULT_MAX_REGENERATION_ATTEMPTS)
    temperature = st.slider("Temperature", 0.0, MAX_TEMPERATURE,
                            DEFAULT_TEMPERATURE, 0.05)
    evidence_threshold = st.slider(
        "Evidence Similarity", 0.1, 0.9, DEFAULT_EVIDENCE_THRESHOLD, 0.05,
        help="Minimum similarity score to consider a chunk as evidence",
    )

    st.divider()

    # Knowledge Base Stats
    doc_names = set(c["document_name"] for c in vs.chunks)
    st.markdown("##### 📚 Knowledge Base")
    kb_cols = st.columns(2)
    kb_cols[0].metric("Documents", len(doc_names))
    kb_cols[1].metric("Chunks", vs.num_chunks)
    st.metric("Vector Index", f"{'✅ Active' if vs.num_vectors > 0 else '❌ Missing'}")

    if doc_names:
        with st.expander("📄 Loaded Documents", expanded=False):
            for dn in sorted(doc_names):
                st.markdown(f"• `{dn}`")


# ══════════════════════════════════════════════════════════
# Hero Banner & Interactive 3D Legal Constellation
# ══════════════════════════════════════════════════════════
hero_left, hero_right = st.columns([1.25, 0.95], gap="large")

with hero_left:
    st.markdown("""
    <div class="hero-banner" style="margin-bottom: 0.5rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
        <div class="hero-content" style="text-align: left;">
            <div class="hero-badge">
                <span class="pulse-dot"></span>
                Multi-Agent Legal RAG · BNS Codex
            </div>
            <h1 style="text-align: left; font-size: 2.1rem; margin-top: 0.2rem;">⚖️ BNS Legal Intelligence</h1>
            <p class="hero-subtitle" style="text-align: left; font-size: 0.95rem; margin-top: 0.5rem;">
                Explore India's new criminal legal system — <strong>Bharatiya Nyaya Sanhita (BNS)</strong>.
                Trained & indexed over <strong>1,388 knowledge chunks</strong>, orchestrated by a 6-agent verification pipeline with zero-hallucination guard.
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 1.2rem;">
                <span style="background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: #a5b4fc; font-weight: 600;">📚 1,388 Chunks</span>
                <span style="background: rgba(34,211,238,0.15); border: 1px solid rgba(34,211,238,0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: #67e8f9; font-weight: 600;">🤖 6 AI Agents</span>
                <span style="background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: #6ee7b7; font-weight: 600;">🛡️ Hallucination Guard</span>
                <span style="background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.75rem; color: #c4b5fd; font-weight: 600;">⚡ Rapid Stream</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    render_3d_legal_constellation()


# ══════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask a Question",
    "🔗 Agent Pipeline",
    "📄 Evidence Chunks",
    "🔍 Hallucination Report",
    "📊 Comparison",
])

# Session state for results
if "result" not in st.session_state:
    st.session_state.result = None
if "basic_result" not in st.session_state:
    st.session_state.basic_result = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""
if "stream_done" not in st.session_state:
    st.session_state.stream_done = False


# ──────────────────────────────────────────────────────────
# TAB 1 – Ask a Question
# ──────────────────────────────────────────────────────────
with tab1:
    # Query input
    st.markdown("""
    <div class="section-header">
        <span class="icon">💡</span>
        <span class="title">Ask About BNS Laws</span>
        <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

    # Quick suggestion chips
    st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 6px; font-weight: 500;'>⚡ Quick BNS Questions:</div>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    if q_col1.button("⚖️ Section 103 (Murder)", width="stretch"):
        st.session_state.query_text = "What constitutes murder under Section 103 of BNS and what are the prescribed punishments?"
        st.session_state.trigger_run = True
    if q_col2.button("💰 Section 303 (Theft)", width="stretch"):
        st.session_state.query_text = "What are the definitions and punishments for theft under Section 303 of BNS?"
        st.session_state.trigger_run = True
    if q_col3.button("🛡️ Right of Self-Defence", width="stretch"):
        st.session_state.query_text = "What are the legal rights of private defence (self-defence) under Bharatiya Nyaya Sanhita?"
        st.session_state.trigger_run = True
    if q_col4.button("🌐 Cyber & Organized Crime", width="stretch"):
        st.session_state.query_text = "How does BNS address organized crime and electronic offences under Section 111?"
        st.session_state.trigger_run = True

    default_val = st.session_state.get("query_text", "")
    query = st.text_area(
        "Your question:",
        value=default_val,
        height=95,
        placeholder="e.g., What is the punishment for murder under BNS? What are the sections related to theft?",
        label_visibility="collapsed",
    )

    col_btn1, col_btn2 = st.columns(2)
    run_multi = col_btn1.button("🟢 Ask Multi-Agent RAG", width="stretch", type="primary")
    run_basic = col_btn2.button("🔵 Ask Basic RAG", width="stretch")

    should_run_multi = run_multi or st.session_state.pop("trigger_run", False)

    # ── Run Multi-Agent RAG ──
    if should_run_multi:
        active_q = query.strip() if query.strip() else default_val.strip()
        if not active_q:
            st.warning("⚠️ Please enter a question first.")
        else:
            st.session_state.query_text = active_q
            st.session_state.stream_done = False
            with st.spinner("🔄 Running multi-agent pipeline..."):
                st.session_state.result = orchestrator.run_multi_agent_rag(
                    active_q, top_k, temperature,
                    hallucination_threshold, evidence_threshold, max_attempts)
                st.session_state.basic_result = None

    # ── Run Basic RAG ──
    if run_basic:
        active_q = query.strip() if query.strip() else default_val.strip()
        if not active_q:
            st.warning("⚠️ Please enter a question first.")
        else:
            st.session_state.query_text = active_q
            st.session_state.stream_done = False
            with st.spinner("🔄 Running basic RAG..."):
                st.session_state.basic_result = orchestrator.run_basic_rag(
                    active_q, top_k, temperature)
                st.session_state.result = None

    # ── Display Result ──
    res = st.session_state.result or st.session_state.basic_result
    if res:
        answer_text = res.get("answer", "")
        vs_status = res.get("verification_status", "")
        pipeline = res.get("pipeline", "")

        # Status badge
        if vs_status == "VERIFIED":
            st.markdown('<span class="verified-badge">✅ Verified Legal Answer</span>', unsafe_allow_html=True)
        elif vs_status in ("HIGH_RISK", "REJECTED"):
            st.markdown('<span class="risk-badge">⚠️ High Risk</span>', unsafe_allow_html=True)
        elif vs_status in ("MEDIUM_RISK", "INSUFFICIENT_EVIDENCE"):
            st.markdown('<span class="warning-badge">⚡ ' + vs_status.replace("_", " ").title() + '</span>', unsafe_allow_html=True)

        # Response container & high-speed word-by-word streaming
        st.markdown("""
        <div class="answer-container" style="margin-bottom: 0.5rem; padding-bottom: 0.5rem;">
            <div class="answer-label">⚖️ BNS Legal AI Response</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.stream_done:
            st.write_stream(stream_word_by_word(answer_text, delay=0.012))
            st.session_state.stream_done = True
        else:
            st.markdown(
                f'<div class="answer-container" style="margin-top: -0.5rem;">'
                f'<div class="answer-text">{answer_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Metrics row
        st.markdown("""
        <div class="section-header">
            <span class="icon">📊</span>
            <span class="title">Response Metrics</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ev_conf = res.get("evidence_confidence", 0)
            st.markdown(f"""
            <div class="metric-3d accent-indigo">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{ev_conf:.1%}</div>
                <div class="metric-label">Evidence Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            risk = res.get("hallucination_risk")
            risk_str = f"{risk:.1%}" if risk is not None else "N/A"
            st.markdown(f"""
            <div class="metric-3d accent-rose">
                <div class="metric-icon">🛡️</div>
                <div class="metric-value">{risk_str}</div>
                <div class="metric-label">Hallucination Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            sp = res.get("support_percentage")
            sp_str = f"{sp:.0f}%" if sp is not None else "N/A"
            st.markdown(f"""
            <div class="metric-3d accent-emerald">
                <div class="metric-icon">✅</div>
                <div class="metric-value">{sp_str}</div>
                <div class="metric-label">Claim Support</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            pt = res.get("processing_time", 0)
            st.markdown(f"""
            <div class="metric-3d accent-cyan">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">{pt:.2f}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)

        # Sources
        sources = res.get("sources", [])
        if sources:
            st.markdown(f"**📚 Sources:** {', '.join(sources)}")


# ──────────────────────────────────────────────────────────
# TAB 2 – Agent Pipeline
# ──────────────────────────────────────────────────────────
with tab2:
    res = st.session_state.result
    if not res:
        st.markdown("""
        <div class="card-3d" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔗</div>
            <div style="color: var(--text-secondary); font-size: 1rem;">
                Run a <strong>Multi-Agent RAG</strong> query to see the agent execution pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-header">
            <span class="icon">🔗</span>
            <span class="title">Agent Execution Trace</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        for step in res.get("agent_trace", []):
            status = step.get("status", "")
            icon_map = {"completed": "✅", "error": "❌", "abstained": "⚠️",
                        "regenerating": "🔄", "VERIFIED": "🏆", "MEDIUM_RISK": "⚡",
                        "HIGH_RISK": "🔴", "REJECTED": "🚫", "INSUFFICIENT_EVIDENCE": "📭"}
            icon = icon_map.get(status, "ℹ️")
            agent_name = step.get("agent", "Unknown")
            detail = step.get("output_summary", "")
            t = step.get("time", 0)

            st.markdown(f"""
            <div class="timeline-step">
                <div class="step-icon">{icon}</div>
                <div class="step-content">
                    <div class="step-agent">{agent_name}</div>
                    <div class="step-detail">{detail}</div>
                </div>
                <div class="step-time">{t:.3f}s</div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# TAB 3 – Evidence Chunks
# ──────────────────────────────────────────────────────────
with tab3:
    res = st.session_state.result or st.session_state.basic_result
    if not res:
        st.markdown("""
        <div class="card-3d" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <div style="color: var(--text-secondary); font-size: 1rem;">
                Run a query to see the retrieved evidence chunks.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        chunks_list = res.get("retrieved_chunks", [])
        st.markdown(f"""
        <div class="section-header">
            <span class="icon">📄</span>
            <span class="title">Retrieved Evidence — {len(chunks_list)} chunks</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        for i, chunk in enumerate(chunks_list):
            score = chunk.get("similarity_score", 0)
            if score >= 0.5:
                score_class = "score-high"
            elif score >= 0.3:
                score_class = "score-mid"
            else:
                score_class = "score-low"

            text_preview = chunk.get("text", "")[:300].replace("\n", " ")
            doc_name = chunk.get("document_name", "Unknown")

            st.markdown(f"""
            <div class="evidence-card">
                <div class="evidence-header">
                    <span class="evidence-source">📁 {doc_name} — Chunk {i+1}</span>
                    <span class="evidence-score {score_class}">{score:.4f}</span>
                </div>
                <div class="evidence-text">{text_preview}{'...' if len(chunk.get("text","")) > 300 else ''}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📖 Full text of Chunk {i+1}", expanded=False):
                st.text(chunk.get("text", ""))


# ──────────────────────────────────────────────────────────
# TAB 4 – Hallucination Report
# ──────────────────────────────────────────────────────────
with tab4:
    res = st.session_state.result
    if not res:
        st.markdown("""
        <div class="card-3d" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <div style="color: var(--text-secondary); font-size: 1rem;">
                Run a <strong>Multi-Agent RAG</strong> query to see hallucination analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-header">
            <span class="icon">🔍</span>
            <span class="title">Hallucination Analysis Report</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        score_details = res.get("score_details", {})

        # Metrics row
        hm1, hm2, hm3, hm4 = st.columns(4)
        with hm1:
            st.markdown(f"""
            <div class="metric-3d accent-indigo">
                <div class="metric-icon">📋</div>
                <div class="metric-value">{score_details.get("total_claims", 0)}</div>
                <div class="metric-label">Total Claims</div>
            </div>
            """, unsafe_allow_html=True)
        with hm2:
            st.markdown(f"""
            <div class="metric-3d accent-emerald">
                <div class="metric-icon">✅</div>
                <div class="metric-value">{score_details.get("supported_count", 0)}</div>
                <div class="metric-label">Supported</div>
            </div>
            """, unsafe_allow_html=True)
        with hm3:
            st.markdown(f"""
            <div class="metric-3d accent-rose">
                <div class="metric-icon">❌</div>
                <div class="metric-value">{score_details.get("unsupported_count", 0)}</div>
                <div class="metric-label">Unsupported</div>
            </div>
            """, unsafe_allow_html=True)
        with hm4:
            st.markdown(f"""
            <div class="metric-3d accent-amber">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">{score_details.get("contradiction_count", 0)}</div>
                <div class="metric-label">Contradictions</div>
            </div>
            """, unsafe_allow_html=True)

        # Explanation
        explanation = score_details.get("explanation", res.get("critic_explanation", ""))
        if explanation:
            st.markdown(f"""
            <div class="card-3d">
                <div style="color: var(--text-muted); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                    📝 Analysis Summary
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Claim-by-claim
        claims = res.get("claims", [])
        if claims:
            st.markdown("""
            <div class="section-header">
                <span class="icon">📌</span>
                <span class="title">Claim-by-Claim Verification</span>
                <span class="line"></span>
            </div>
            """, unsafe_allow_html=True)

            for i, claim in enumerate(claims):
                status = claim.get("status", "UNKNOWN")
                icon = {"SUPPORTED": "✅", "PARTIALLY_SUPPORTED": "🟡",
                        "UNSUPPORTED": "❌", "CONTRADICTED": "⛔"}.get(status, "❓")
                css_class = {"SUPPORTED": "supported",
                             "PARTIALLY_SUPPORTED": "partial"}.get(status, "unsupported")
                color = {"SUPPORTED": "#34d399", "PARTIALLY_SUPPORTED": "#fbbf24",
                         "UNSUPPORTED": "#fb7185", "CONTRADICTED": "#f43f5e"}.get(status, "#94a3b8")

                st.markdown(f"""
                <div class="claim-card {css_class}">
                    <div class="claim-status" style="color: {color};">{icon} Claim {i+1} — {status}</div>
                    <div class="claim-text">{claim.get("claim", "")}</div>
                    <div class="claim-reason">{claim.get("reason", "")}</div>
                </div>
                """, unsafe_allow_html=True)

        # Pie chart
        if score_details.get("total_claims", 0) > 0:
            labels = ["Supported", "Partially Supported", "Unsupported", "Contradicted"]
            values = [
                score_details.get("supported_count", 0),
                score_details.get("partially_supported_count", 0),
                score_details.get("unsupported_count", 0),
                score_details.get("contradiction_count", 0),
            ]
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values,
                marker_colors=["#10b981", "#f59e0b", "#f43f5e", "#8b5cf6"],
                hole=0.5,
                textfont=dict(color="#f1f5f9", size=12),
            )])
            fig.update_layout(
                title=dict(text="Claim Verification Distribution", font=dict(color="#f1f5f9", size=16)),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────────────────
# TAB 5 – Comparison
# ──────────────────────────────────────────────────────────
with tab5:
    mr = st.session_state.result
    br = st.session_state.basic_result

    if not mr and not br:
        st.markdown("""
        <div class="card-3d" style="text-align:center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <div style="color: var(--text-secondary); font-size: 1rem;">
                Run queries with both pipelines to compare Basic RAG vs Multi-Agent RAG.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick comparison run
        st.markdown("""
        <div class="section-header">
            <span class="icon">⚡</span>
            <span class="title">Quick Comparison</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        cmp_query = st.text_input("Enter a question to compare both pipelines:",
                                  placeholder="e.g., What is the punishment for theft under BNS?")
        if st.button("🔄 Run Both Pipelines", type="primary", width="stretch"):
            if cmp_query.strip():
                with st.spinner("Running both pipelines..."):
                    st.session_state.basic_result = orchestrator.run_basic_rag(
                        cmp_query, top_k, temperature)
                    st.session_state.result = orchestrator.run_multi_agent_rag(
                        cmp_query, top_k, temperature,
                        hallucination_threshold, evidence_threshold, max_attempts)
                    st.session_state.stream_done = True
                st.rerun()
    else:
        st.markdown("""
        <div class="section-header">
            <span class="icon">📊</span>
            <span class="title">Pipeline Comparison</span>
            <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        comp_data = {
            "Metric": [
                "Verification Status",
                "Evidence Confidence",
                "Support %",
                "Hallucination Risk",
                "Processing Time",
            ],
            "Basic RAG": [
                br.get("verification_status", "N/A") if br else "—",
                f"{br['evidence_confidence']:.4f}" if br else "—",
                "N/A (no critic)",
                "N/A (no critic)",
                f"{br['processing_time']:.3f}s" if br else "—",
            ],
            "Multi-Agent RAG": [
                mr.get("verification_status", "N/A") if mr else "—",
                f"{mr.get('evidence_confidence', 0):.4f}" if mr else "—",
                f"{mr.get('support_percentage', 0):.1f}%" if mr else "—",
                f"{mr.get('hallucination_risk', 0):.2%}" if mr else "—",
                f"{mr['processing_time']:.3f}s" if mr else "—",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

        # Bar chart
        if br and mr:
            fig = go.Figure()
            metrics_names = ["Evidence Confidence", "Processing Time (s)"]
            basic_vals = [br["evidence_confidence"], br["processing_time"]]
            multi_vals = [mr.get("evidence_confidence", 0), mr["processing_time"]]

            fig.add_trace(go.Bar(name="Basic RAG", x=metrics_names, y=basic_vals,
                                 marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="Multi-Agent RAG", x=metrics_names, y=multi_vals,
                                 marker_color="#10b981"))
            fig.update_layout(
                barmode="group",
                title=dict(text="Pipeline Performance", font=dict(color="#f1f5f9", size=16)),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                legend=dict(font=dict(color="#94a3b8")),
                xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                yaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
            )
            st.plotly_chart(fig, width="stretch")

        # Side-by-side answers
        if br and mr:
            st.markdown("""
            <div class="section-header">
                <span class="icon">💬</span>
                <span class="title">Answer Comparison</span>
                <span class="line"></span>
            </div>
            """, unsafe_allow_html=True)

            ans_col1, ans_col2 = st.columns(2)
            with ans_col1:
                st.markdown(f"""
                <div class="card-3d">
                    <div style="color: #60a5fa; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">
                        🔵 Basic RAG
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.7;">
                        {br['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with ans_col2:
                st.markdown(f"""
                <div class="card-3d">
                    <div style="color: #34d399; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">
                        🟢 Multi-Agent RAG
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.7;">
                        {mr['answer']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
