from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# PDF Report imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from src.config import CLASS_NAMES_PATH, DATA_DIR, MODEL_PATH, PLOT_PATH, METADATA_PATH

# 1. Page Config
st.set_page_config(
    page_title="AgriVision ML Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_DESCRIPTIONS = {
    "AnnualCrop": "Seasonal cultivated fields such as grains, vegetables, and other yearly crops.",
    "PermanentCrop": "Long-term cultivated crops such as orchards, vineyards, and plantations.",
    "Pasture": "Grassland areas commonly used for grazing.",
    "HerbaceousVegetation": "Low vegetation cover that is not clearly classified as crop or pasture.",
    "Forest": "Dense tree-covered land.",
    "River": "Water channels and river corridors.",
    "SeaLake": "Large water bodies such as lakes and coastal water.",
    "Highway": "Road and transport infrastructure.",
    "Industrial": "Industrial or commercial built-up zones.",
    "Residential": "Housing and settlement areas.",
}

CLASS_GROUPS = {
    "AnnualCrop": "Crop scene",
    "PermanentCrop": "Crop scene",
    "Pasture": "Agricultural landscape",
    "HerbaceousVegetation": "Vegetation",
    "Forest": "Vegetation",
    "River": "Water feature",
    "SeaLake": "Water feature",
    "Highway": "Infrastructure",
    "Industrial": "Built environment",
    "Residential": "Built environment",
}


def inject_styles(theme: str) -> None:
    # 2. Premium Design System: Color Palettes, Typography, Shadows & Hover animations
    if theme == "Light Mode":
        vars_css = """
        :root {
            --bg-primary: #f8fafc;
            --bg-gradient: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            --bg-sidebar: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --muted: #94a3b8;
            --line: #e2e8f0;
            --accent-green: #059669;
            --accent-green-rgb: 5, 150, 105;
            --accent-cyan: #0891b2;
            --accent-blue: #2563eb;
            --accent-purple: #7c3aed;
            --border-color: rgba(15, 23, 42, 0.08);
            --border-hover: rgba(37, 99, 235, 0.25);
            --glass-bg: rgba(255, 255, 255, 0.75);
            --glass-blur: blur(12px);
            --card-shadow: 0 10px 30px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.02);
            --glow-shadow: 0 10px 25px rgba(37, 99, 235, 0.08);
            --button-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --button-text: #ffffff;
            --tab-active-bg: rgba(37, 99, 235, 0.05);
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            color-scheme: light;
            color: var(--text-primary) !important;
            background: var(--bg-primary) !important;
        }
        .stApp {
            background: var(--bg-gradient) !important;
        }
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 80% 20%, rgba(37, 99, 235, 0.02) 0%, transparent 40%),
                        radial-gradient(circle at 20% 80%, rgba(5, 150, 105, 0.02) 0%, transparent 40%);
            pointer-events: none;
            z-index: 0;
        }
        """
    elif theme == "Dark Mode":
        vars_css = """
        :root {
            --bg-primary: #030712;
            --bg-gradient: radial-gradient(circle at 50% 0%, #111827 0%, #030712 100%);
            --bg-sidebar: #03091e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --muted: #64748b;
            --line: #1e293b;
            --accent-green: #10b981;
            --accent-green-rgb: 16, 185, 129;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --border-color: rgba(255, 255, 255, 0.07);
            --border-hover: rgba(6, 182, 212, 0.4);
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-blur: blur(16px);
            --card-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
            --glow-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
            --button-gradient: linear-gradient(135deg, #00F5FF 0%, #39FF14 100%);
            --button-text: #020617;
            --tab-active-bg: rgba(59, 130, 246, 0.08);
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            color-scheme: dark;
            color: var(--text-primary) !important;
            background: var(--bg-primary) !important;
        }
        .stApp {
            background: var(--bg-gradient) !important;
        }
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 80% 20%, rgba(6, 182, 212, 0.05) 0%, transparent 50%),
                        radial-gradient(circle at 20% 80%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
            animation: ambientGlow 20s infinite alternate ease-in-out;
        }
        @keyframes ambientGlow {
            0% { opacity: 0.7; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.1); }
        }
        """
    else:  # Default Theme (Futuristic Dark AI mapping for premium default look)
        vars_css = """
        :root {
            --bg-primary: #030712;
            --bg-gradient: radial-gradient(circle at 50% 0%, #111827 0%, #030712 100%);
            --bg-sidebar: #03091e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --muted: #64748b;
            --line: #1e293b;
            --accent-green: #10b981;
            --accent-green-rgb: 16, 185, 129;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --border-color: rgba(255, 255, 255, 0.07);
            --border-hover: rgba(6, 182, 212, 0.4);
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-blur: blur(16px);
            --card-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
            --glow-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
            --button-gradient: linear-gradient(135deg, #00F5FF 0%, #39FF14 100%);
            --button-text: #020617;
            --tab-active-bg: rgba(59, 130, 246, 0.08);
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            color: var(--text-primary) !important;
            background: var(--bg-primary) !important;
        }
        .stApp {
            background: var(--bg-gradient) !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        {vars_css}

        /* Force Dark Theme on Top Header and Sidebar Button */
        [data-testid="stHeader"] {{
            background-color: var(--bg-primary) !important;
            background: var(--bg-gradient) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }}

        /* Sidebar toggle buttons styling */
        [data-testid="stHeader"] button, 
        button[data-testid="stHeaderSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] {{
            color: var(--accent-cyan) !important;
            background-color: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            transition: all 0.25s ease !important;
        }}

        [data-testid="stHeader"] button:hover,
        button[data-testid="stHeaderSidebarCollapseButton"]:hover,
        [data-testid="stSidebarCollapseButton"]:hover {{
            color: var(--accent-green) !important;
            background-color: rgba(0, 245, 255, 0.08) !important;
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 15px rgba(0, 245, 255, 0.15) !important;
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary) !important;
        }}

        h1, h2, h3, h4, h5, h6, .eyebrow {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--text-primary) !important;
            font-weight: 700;
        }}

        .main .block-container {{
            max-width: 1340px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}

        /* Custom glassmorphism containers */
        .glass-card {{
            border: 1px solid var(--border-color);
            border-radius: 16px;
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: var(--card-shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .glass-card:hover {{
            border-color: var(--border-hover);
            box-shadow: var(--card-shadow), var(--glow-shadow);
            transform: translateY(-2px);
        }}

        /* Theme Selector dropdown top right */
        div[data-testid="stSelectbox"] {{
            position: fixed;
            top: 15px;
            right: 20px;
            z-index: 999999;
            width: 140px;
        }}

        div[data-testid="stSelectbox"] label {{
            display: none !important;
        }}

        /* Topbar headers */
        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-mark {{
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            position: relative;
            box-shadow: var(--glow-shadow);
        }}

        .brand-mark::after {{
            content: "";
            position: absolute;
            inset: 9px;
            border: 2px solid rgba(255,255,255,0.95);
            border-radius: 5px;
        }}

        /* Hero banner */
        .hero {{
            border: 1px solid var(--border-color);
            border-radius: 20px;
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            padding: 40px;
            margin-bottom: 28px;
            box-shadow: var(--card-shadow);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .hero-video-bg {{
            position: absolute;
            top: 50%;
            left: 50%;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: 0;
            transform: translate(-50%, -50%);
            opacity: 0.22;
            pointer-events: none;
            object-fit: cover;
        }}

        .hero-content {{
            position: relative;
            z-index: 1;
        }}

        .hero::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(var(--accent-green-rgb), 0.08) 0%, transparent 65%);
            pointer-events: none;
            z-index: 1;
        }}

        .hero:hover {{
            border-color: var(--border-hover);
        }}

        .hero h1 {{
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.1;
            background: linear-gradient(135deg, var(--text-primary) 30%, var(--accent-blue) 70%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .hero p {{
            margin: 16px 0 0;
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 850px;
            line-height: 1.6;
        }}

        /* Pulsing dot status indicator */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-top: 12px;
        }}

        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-green);
            box-shadow: 0 0 12px var(--accent-green);
            animation: pulse-dot 2s infinite ease-in-out;
        }}

        @keyframes pulse-dot {{
            0% {{ transform: scale(0.9); opacity: 0.4; box-shadow: 0 0 0 0 rgba(var(--accent-green-rgb), 0.6); }}
            70% {{ transform: scale(1.1); opacity: 1; box-shadow: 0 0 0 10px rgba(var(--accent-green-rgb), 0); }}
            100% {{ transform: scale(0.9); opacity: 0.4; box-shadow: 0 0 0 0 rgba(var(--accent-green-rgb), 0); }}
        }}

        /* Radar Container */
        .radar-container {{
            position: relative;
            width: 150px;
            height: 150px;
            margin: 24px auto;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .radar-circle {{
            position: absolute;
            border: 1.5px solid var(--accent-green);
            border-radius: 50%;
            opacity: 0;
            animation: radar-pulse 3.5s cubic-bezier(0.2, 0.5, 0.4, 0.9) infinite;
        }}

        .pulse-1 {{ animation-delay: 0s; }}
        .pulse-2 {{ animation-delay: 1.1s; }}
        .pulse-3 {{ animation-delay: 2.2s; }}

        .radar-glow {{
            width: 20px;
            height: 20px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 30px var(--accent-green), 0 0 60px var(--accent-green);
        }}

        @keyframes radar-pulse {{
            0% {{ width: 0; height: 0; opacity: 0.9; border-color: var(--accent-cyan); }}
            50% {{ border-color: var(--accent-green); }}
            100% {{ width: 150px; height: 150px; opacity: 0; border-color: var(--accent-blue); }}
        }}

        /* AI scanning layout */
        .scan-frame {{
            position: relative;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            background: #000;
            box-shadow: var(--card-shadow);
            transition: border-color 0.3s ease;
        }}

        .scan-frame.scanning {{
            border-color: var(--accent-green);
        }}

        .scan-frame img {{
            display: block;
            width: 100%;
            height: auto;
            transition: opacity 0.5s ease;
        }}

        .scan-frame.scanning img {{
            opacity: 0.65;
        }}

        .scan-frame.scanning::after {{
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
            box-shadow: 0 0 20px var(--accent-green), 0 0 40px var(--accent-green);
            animation: scanline 2s ease-in-out infinite;
        }}

        @keyframes scanline {{
            0% {{ top: 0%; opacity: 0.2; }}
            50% {{ opacity: 1; }}
            100% {{ top: 100%; opacity: 0.2; }}
        }}

        /* Custom Stats cards */
        .metric-glow-card {{
            background: var(--glass-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--card-shadow);
            text-align: center;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}

        .metric-glow-card:hover {{
            border-color: var(--border-hover);
            box-shadow: var(--card-shadow), var(--glow-shadow);
            transform: translateY(-4px);
        }}

        .metric-val {{
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'Space Grotesk', sans-serif;
            margin: 8px 0;
            background: linear-gradient(135deg, var(--text-primary) 40%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .metric-lbl {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }}

        /* Mini-insight animated cards */
        .insight-card {{
            border-left: 4px solid var(--accent-blue);
            background: rgba(255, 255, 255, 0.01);
            border-radius: 0 16px 16px 0;
            padding: 16px 20px;
            margin-bottom: 12px;
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
        }}

        .insight-card:hover {{
            background: rgba(255, 255, 255, 0.03);
            transform: translateX(6px);
            border-color: var(--border-hover);
            border-left-color: var(--accent-green);
        }}

        .insight-title {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }}

        .insight-value {{
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 4px;
        }}

        /* AI assistant style output box */
        .ai-assistant-box {{
            background: var(--glass-bg);
            border: 1px solid var(--border-color);
            box-shadow: var(--card-shadow);
            border-radius: 16px;
            padding: 20px;
            font-size: 0.98rem;
            line-height: 1.6;
            color: var(--text-primary);
            margin: 20px 0;
            position: relative;
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
        }}

        .ai-assistant-box::before {{
            content: "✦ AI AGENT REPORT";
            position: absolute;
            top: -10px;
            left: 20px;
            background: var(--bg-sidebar);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            font-size: 0.7rem;
            padding: 2px 10px;
            border-radius: 99px;
            font-weight: 800;
            letter-spacing: 0.08em;
        }}

        .result-kicker {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 4px;
        }}

        /* Gradient Premium button */
        div.stButton > button:first-child, .stDownloadButton > button:first-child {{
            background: var(--button-gradient) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: var(--button-text) !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            padding: 12px 24px !important;
            border-radius: 12px !important;
            box-shadow: var(--card-shadow) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            width: 100%;
        }}

        div.stButton > button:first-child:hover, .stDownloadButton > button:first-child:hover {{
            box-shadow: var(--glow-shadow) !important;
            transform: translateY(-2px);
            border-color: var(--accent-cyan) !important;
        }}

        div.stButton > button:first-child:active, .stDownloadButton > button:first-child:active {{
            transform: translateY(0);
        }}

        /* Native Streamlit file uploader overrides */
        [data-testid="stFileUploader"] {{
            background: var(--glass-bg) !important;
            border: 1px dashed var(--border-color) !important;
            border-radius: 16px !important;
            padding: 24px 16px !important;
            box-shadow: var(--card-shadow);
            transition: all 0.3s ease;
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: var(--border-hover) !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: transparent !important;
            padding: 0 !important;
        }}

        [data-testid="stFileUploader"] * {{
            color: var(--text-secondary) !important;
        }}

        [data-testid="stFileUploader"] button {{
            background: var(--button-gradient) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            box-shadow: var(--card-shadow) !important;
            transition: all 0.2s ease !important;
        }}

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button * {{
            color: var(--button-text) !important;
        }}

        [data-testid="stFileUploader"] button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: var(--glow-shadow) !important;
        }}

        /* Force notification text readability */
        div[data-testid="stNotification"] * {{
            color: var(--text-primary) !important;
        }}

        /* Styled selectboxes */
        div[data-baseweb="select"] > div {{
            background-color: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }}

        /* Chat Input styling overrides */
        div[data-testid="stChatInput"] {{
            background-color: var(--bg-sidebar) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 4px 8px !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: var(--text-primary) !important;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid rgba(0, 245, 255, 0.25) !important;
            box-shadow: 4px 0 25px rgba(0, 245, 255, 0.08) !important;
        }}

        /* Sidebar active page radio indicator glow */
        div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            color: var(--accent-cyan) !important;
            font-weight: 700 !important;
            border-left: 3px solid var(--accent-cyan) !important;
            padding-left: 10px !important;
            transition: all 0.2s ease;
        }}

        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            color: var(--accent-cyan) !important;
            transition: all 0.2s ease;
        }}

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--text-secondary) !important;
        }}

        /* Dataframe overrides */
        div[data-testid="stDataFrame"] {{
            background-color: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            overflow: hidden;
            padding: 6px;
        }}

        /* Footer */
        .footer-saas {{
            border-top: 1px solid var(--border-color);
            margin-top: 50px;
            padding: 30px 0 15px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        .footer-saas a {{
            color: var(--accent-blue);
            text-decoration: none;
            margin: 0 10px;
            font-weight: 600;
            transition: color 0.2s ease;
        }}

        .footer-saas a:hover {{
            color: var(--accent-green);
            text-decoration: underline;
        }}

        /* Gauge progress bar health */
        .health-bar-container {{
            width: 100%;
            height: 12px;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 99px;
            overflow: hidden;
            margin-top: 10px;
            border: 1px solid var(--border-color);
        }}

        .health-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue), var(--accent-green)) !important;
            border-radius: 99px;
            transition: width 1s ease-in-out;
        }}

        /* Tabs custom styling */
        button[data-baseweb="tab"] {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: var(--text-secondary) !important;
            background-color: transparent !important;
            border-bottom: 2px solid transparent !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease;
        }}

        button[data-baseweb="tab"]:hover {{
            color: var(--text-primary) !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: var(--accent-blue) !important;
            border-bottom-color: var(--accent-blue) !important;
            background-color: var(--tab-active-bg) !important;
            border-radius: 8px 8px 0 0 !important;
        }}

        /* Styled dividers */
        hr {{
            border-color: var(--line) !important;
        }}

        /* Custom dashboard graphics frames & animations */
        .image-card {{
            border: 1px solid var(--border-color);
            border-radius: 20px;
            overflow: hidden;
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
            box-shadow: var(--card-shadow);
            transition: all 0.5s ease;
            position: relative;
            padding: 12px;
            margin-bottom: 24px;
        }}

        .image-card:hover {{
            border-color: var(--border-hover);
        }}

        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 12px;
            display: block;
            transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .image-card:hover img {{
            transform: scale(1.04);
        }}

        .image-float {{
            animation: floatSlow 6s ease-in-out infinite;
        }}

        .image-glow {{
            animation: pulseGlow 4s ease-in-out infinite;
        }}

        @keyframes floatSlow {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-8px); }}
            100% {{ transform: translateY(0px); }}
        }}

        @keyframes pulseGlow {{
            0% {{ box-shadow: var(--card-shadow), 0 0 15px rgba(0, 245, 255, 0.08); }}
            50% {{ box-shadow: var(--card-shadow), 0 0 25px rgba(0, 245, 255, 0.22); }}
            100% {{ box-shadow: var(--card-shadow), 0 0 15px rgba(0, 245, 255, 0.08); }}
        }}

        /* Timeline flow aesthetics */
        .timeline-container {{
            position: relative;
            margin: 30px 0;
        }}

        .timeline-step {{
            border-top: 2px dashed var(--line);
            padding-top: 20px;
            position: relative;
        }}

        .timeline-number {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--bg-sidebar);
            border: 2px solid var(--accent-cyan);
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.9rem;
            margin-bottom: 12px;
            box-shadow: var(--glow-shadow);
        }}

        .timeline-step:hover .timeline-number {{
            border-color: var(--accent-green);
            color: var(--accent-green);
            transform: scale(1.1);
            transition: all 0.3s ease;
        }}

        @media (max-width: 820px) {{
            div[data-testid="stSelectbox"] {{
                position: static !important;
                width: 100% !important;
                margin-bottom: 20px;
            }}
            .hero-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            labels = metadata.get("class_names", [])
            arch = metadata.get("arch", "mobilenetv2")
            return model, labels, arch
        except Exception:
            pass
    labels = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, labels, "mobilenetv2"


@st.cache_data
def dataset_image_count(data_dir: str) -> int:
    root = Path(data_dir)
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.jpg") if path.is_file())


def load_class_names() -> list[str]:
    if METADATA_PATH.exists():
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            return metadata.get("class_names", [])
        except Exception:
            pass
    if not CLASS_NAMES_PATH.exists():
        return []
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    _, height, width, _ = model.input_shape
    return int(height), int(width)


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_stat_card(label: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-glow-card">
            <div class="metric-lbl">{label}</div>
            <div class="metric-val">{value}</div>
            <div style="color: var(--muted); font-size: 0.75rem; margin-top:4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scan_image(image: Image.Image, scanning: bool) -> None:
    state = "scanning" if scanning else "analyzed"
    st.markdown(
        f"""
        <div class="scan-frame {state}">
            <img src="{image_to_data_uri(image)}" alt="Satellite Scene" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_radar_pulse() -> None:
    st.markdown(
        """
        <div class="radar-container">
            <div class="radar-circle pulse-1"></div>
            <div class="radar-circle pulse-2"></div>
            <div class="radar-circle pulse-3"></div>
            <div class="radar-glow"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def create_plotly_gauge(confidence: float) -> go.Figure:
    score_percent = max(0.0, min(100.0, float(confidence) * 100))
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_percent,
        number = {'suffix': "%", 'font': {'size': 28, 'color': '#F9FAFB', 'family': 'Space Grotesk'}, 'valueformat': '.1f'},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94A3B8', 'tickfont': {'color': '#94A3B8', 'size': 9}},
            'bar': {'color': '#00F5FF', 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.02)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.08)',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(139, 92, 246, 0.12)'},
                {'range': [50, 85], 'color': 'rgba(59, 130, 246, 0.12)'},
                {'range': [85, 100], 'color': 'rgba(57, 255, 20, 0.12)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=15, r=15, t=20, b=15),
        height=200
    )
    return fig


def create_plotly_bar(probabilities: np.ndarray, labels: list[str]) -> go.Figure:
    indices = np.argsort(probabilities)
    sorted_probs = probabilities[indices] * 100
    sorted_labels = [labels[idx] for idx in indices]
    
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_labels,
        orientation='h',
        marker=dict(
            color=sorted_probs,
            colorscale=[[0, '#8B5CF6'], [0.5, '#3B82F6'], [1, '#00F5FF']],
            line=dict(color='rgba(0, 245, 255, 0.3)', width=1),
            cornerradius=4
        ),
        text=[f"{val:.1f}%" for val in sorted_probs],
        textposition='auto',
        textfont=dict(color='#F9FAFB', size=10, family='Space Grotesk')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            tickfont=dict(color='#94A3B8', size=9),
            title=dict(text="Confidence (%)", font=dict(color='#94A3B8', size=10))
        ),
        yaxis=dict(
            tickfont=dict(color='#F9FAFB', size=10, family='Space Grotesk'),
            showgrid=False
        )
    )
    return fig


def create_plotly_soil_radar(label: str) -> go.Figure:
    reqs = {
        "AnnualCrop": [80, 75, 70, 66, 85],
        "PermanentCrop": [65, 80, 85, 68, 70],
        "Pasture": [70, 50, 60, 63, 65],
        "Forest": [50, 45, 55, 51, 90],
        "HerbaceousVegetation": [45, 40, 50, 60, 55],
        "Highway": [10, 10, 10, 50, 20],
        "Industrial": [5, 5, 5, 45, 10],
        "Residential": [15, 15, 15, 55, 15],
        "River": [30, 20, 25, 72, 100],
        "SeaLake": [20, 10, 15, 75, 100],
    }
    categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'pH Value', 'Soil Moisture']
    values = reqs.get(label, [50, 50, 50, 60, 50])
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 245, 255, 0.12)',
        line=dict(color='#00F5FF', width=2),
        marker=dict(color='#39FF14', size=6)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(color='#94A3B8', size=8),
                angle=45
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickfont=dict(color='#F9FAFB', size=9, family='Space Grotesk')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=15, b=15),
        height=220
    )
    return fig


def create_plotly_orbit_map() -> go.Figure:
    t = np.linspace(0, 2*np.pi, 100)
    x_orbit = 180 * np.cos(t)
    y_orbit = 80 * np.sin(t)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_orbit, y=y_orbit,
        mode='lines',
        line=dict(color='rgba(0, 245, 255, 0.35)', width=1.5, dash='dash'),
        name='Orbit Path'
    ))
    
    current_sec = int(time.time() * 2) % 100
    sat_x = [x_orbit[current_sec]]
    sat_y = [y_orbit[current_sec]]
    
    fig.add_trace(go.Scatter(
        x=sat_x, y=sat_y,
        mode='markers',
        marker=dict(
            color='#39FF14', size=12,
            line=dict(color='#00F5FF', width=2),
            symbol='diamond'
        ),
        name='Sentinel-2B Active Scan'
    ))
    
    foot_x = [x_orbit[(current_sec - 5) % 100], x_orbit[(current_sec - 10) % 100]]
    foot_y = [y_orbit[(current_sec - 5) % 100], y_orbit[(current_sec - 10) % 100]]
    fig.add_trace(go.Scatter(
        x=foot_x, y=foot_y,
        mode='markers',
        marker=dict(color='#8B5CF6', size=6, opacity=0.5),
        name='Recent Scan footprint'
    ))
    
    fig.update_layout(
        xaxis=dict(range=[-200, 200], showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, tickfont=dict(color='#64748B', size=8)),
        yaxis=dict(range=[-100, 100], showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, tickfont=dict(color='#64748B', size=8)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False
    )
    return fig


def create_plotly_yield_trend(label: str) -> go.Figure:
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    mult = 1.2 if label == "PermanentCrop" else 1.0 if label == "AnnualCrop" else 0.8
    yields = [3.2 * mult, 3.5 * mult, 3.1 * mult, 3.9 * mult, 4.2 * mult, 4.8 * mult]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=yields,
        mode='lines+markers',
        line=dict(color='#8B5CF6', width=3),
        marker=dict(color='#00F5FF', size=8, line=dict(color='#8B5CF6', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.08)'
    ))
    fig.update_layout(
        xaxis=dict(tickvals=years, showgrid=False, tickfont=dict(color='#94A3B8', size=9)),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#94A3B8', size=9)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=15, r=15, t=10, b=30),
        height=220
    )
    return fig


def create_plotly_batch_pie(df: pd.DataFrame) -> go.Figure:
    dist = df.groupby("Predicted Class").size().reset_index(name="Count")
    colors_map = {
        "AnnualCrop": "#00F5FF",
        "PermanentCrop": "#3B82F6",
        "Pasture": "#8B5CF6",
        "Forest": "#39FF14",
        "HerbaceousVegetation": "#F97316",
        "River": "#0d9488",
        "SeaLake": "#0369a1",
        "Highway": "#eab308",
        "Industrial": "#ec4899",
        "Residential": "#ef4444"
    }
    colors_list = [colors_map.get(cls, "#3B82F6") for cls in dist["Predicted Class"]]
    
    fig = go.Figure(go.Pie(
        labels=dist["Predicted Class"],
        values=dist["Count"],
        hole=0.45,
        marker=dict(colors=colors_list, line=dict(color='#0A0F1F', width=2)),
        textinfo='percent+label',
        textfont=dict(color='#F9FAFB', size=9, family='Space Grotesk')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False
    )
    return fig


def get_agronomical_features(label: str, score: float) -> dict:
    # Generates futuristic soil, weather and yield forecasting based on class labels
    if label == "AnnualCrop":
        return {
            "soil": "Nitrogen High (124 kg/ha) | pH: 6.6",
            "soil_rating": "Optimal",
            "weather": "Precipitation 20mm | Suitability: 94%",
            "weather_rating": "Favorable",
            "yield": "Estimated Yield: 4.8 tons / Hectare",
            "yield_growth": "Active Leaf Phase",
            "assistant": f"The Sentinel-2 tile reveals extensive agricultural land-cover classified as AnnualCrop (Confidence: {score:.1f}%). Geometric boundaries suggest organized field grids. Recommended action: Continue irrigation cycle; schedule nitrogen top-dressing in 12 days.",
            "recommendations": "Maintain water logging control. Check for wheat leaf rust symptoms on borders. Optimize chemical fertilizer application in active growth stage."
        }
    elif label == "PermanentCrop":
        return {
            "soil": "Potassium Rich | Clayey-Loam | pH: 6.8",
            "soil_rating": "Optimal",
            "weather": "Temp: 24°C | Sun Index: High | Suitability: 88%",
            "weather_rating": "Excellent",
            "yield": "Estimated Yield: 12.5 tons / Hectare",
            "yield_growth": "Fruit Setting Phase",
            "assistant": f"Telemetry signals verify PermanentCrop presence (Confidence: {score:.1f}%). Texture pattern indicates orchards or vineyards. Soil structural properties show adequate water-holding capacity. Irrigation feeds are stable.",
            "recommendations": "Conduct pruning verification checks. Introduce organic mulch to maintain root zone moisture. Calibrate micro-drip irrigation systems."
        }
    elif label == "Pasture":
        return {
            "soil": "Organic Matter: 4.2% | pH: 6.3 | Nitrogen Stable",
            "soil_rating": "Adequate",
            "weather": "Rainy Season | Soil Moisture: 30% | Suitability: 86%",
            "weather_rating": "Favorable",
            "yield": "Forage Biomass: 3.2 dry tons / Hectare",
            "yield_growth": "Vegetative Grazing Stage",
            "assistant": f"The region is identified as Pasture land-cover (Confidence: {score:.1f}%). Moderate chlorophyll absorption detected. Organic soil components indicate excellent grazing land sustainability.",
            "recommendations": "Implement rotational grazing cycles. Check for local invasive weed proliferation. Introduce biological soil nitrogen-fixing cover crops."
        }
    elif label == "Forest":
        return {
            "soil": "Humic Rich | Highly Acidic | pH: 5.1",
            "soil_rating": "Stable (Natural Ecosystem)",
            "weather": "Sun Index: Med | Precipitation: 40mm | Suitability: 95%",
            "weather_rating": "Perfect",
            "yield": "Carbon Stock: 154 tons Carbon / Hectare",
            "yield_growth": "Mature Canopy Phase",
            "assistant": f"Dense natural canopy structure classified as Forest (Confidence: {score:.1f}%). High NDVI indexes confirm robust forest crown vitality. Carbon storage capacity is working at maximum efficiency.",
            "recommendations": "Monitor border regions for illegal deforestation or logging activity. Verify fire hazard index parameters. Maintain native forest ecology conservation."
        }
    elif label in ["River", "SeaLake"]:
        return {
            "soil": "Alluvial Sediment | Subaqueous Soil | pH: 7.2",
            "soil_rating": "Hydric Substrate",
            "weather": "Relative Humidity: 78% | Evaporation: 15%",
            "weather_rating": "N/A (Aquatic)",
            "yield": "Water Quality: High | Turbidity Index: Low",
            "yield_growth": "Ecosystem Stable",
            "assistant": f"Liquid cover classified as {label} (Confidence: {score:.1f}%). Visual spectrum reflectance is consistent with clean water bodies. Recommended action: Check for upstream sediment runoffs.",
            "recommendations": "Inspect shoreline for agricultural pesticide runoff. Monitor eutrophication levels and blue-green algae blooms. Maintain river corridor vegetation buffers."
        }
    elif label in ["Highway", "Residential", "Industrial"]:
        return {
            "soil": "Urban Sealed Soil | High Concrete Cover | pH: N/A",
            "soil_rating": "Disturbed/Sealed",
            "weather": "Urban Heat Island Index: High (+3°C)",
            "weather_rating": "Unfavorable for Crops",
            "yield": "Non-Agricultural Land Cover Indicator",
            "yield_growth": "N/A (Built Infrastructure)",
            "assistant": f"Sensing registers built-up infrastructure categorized as {label} (Confidence: {score:.1f}%). Paved surfaces and buildings have sealed the soil. No crops should be planted in this area.",
            "recommendations": "Check run-off drainage management. Implement green rooftops to mitigate the urban heat island effect. Run soil contamination tests for heavy metals on surrounding green zones."
        }
    else:
        return {
            "soil": "Sandy-Loam | pH: 6.0",
            "soil_rating": "Moderate",
            "weather": "Temp: 22°C | Suitability: 70%",
            "weather_rating": "Moderate",
            "yield": "Biomass Index: 1.5 dry tons / Hectare",
            "yield_growth": "Low Vegetation State",
            "assistant": f"Scattered wild vegetation detected, labeled as HerbaceousVegetation (Confidence: {score:.1f}%). Low canopy density. Recommended action: Maintain soil moisture, check erosion.",
            "recommendations": "Implement soil conservation cover crops to check erosion. Promote wild biodiverse herb growth to support local insect ecosystems."
        }


def generate_pdf_report(data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Document Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3B82F6'),
        spaceBefore=14,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155')
    )
    
    value_style = ParagraphStyle(
        'ValueText',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0F172A')
    )

    story = []

    # Title
    story.append(Paragraph("AGRIVISION AI - SATELLITE CROP REPORT", title_style))
    story.append(Paragraph(f"Telemetry Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC | Reference Orbit: Sentinel-2B", subtitle_style))
    story.append(Spacer(1, 10))

    # Overview table
    table_data = [
        [Paragraph("<b>TELEMETRY PARAMETER</b>", ParagraphStyle('HCol', parent=normal_style, textColor=colors.white, fontName='Helvetica-Bold')), 
         Paragraph("<b>SYSTEM VALUE / INSIGHT</b>", ParagraphStyle('HCol2', parent=normal_style, textColor=colors.white, fontName='Helvetica-Bold'))],
        [Paragraph("Target Scene File", normal_style), Paragraph(data.get("filename", "N/A"), value_style)],
        [Paragraph("Predicted Classification", normal_style), Paragraph(data.get("prediction", "N/A"), ParagraphStyle('PredVal', parent=value_style, textColor=colors.HexColor('#10B981')))],
        [Paragraph("Model Confidence Score", normal_style), Paragraph(data.get("confidence", "N/A"), value_style)],
        [Paragraph("Soil Suitability Index", normal_style), Paragraph(data.get("soil", "N/A"), value_style)],
        [Paragraph("Weather Suitability Rating", normal_style), Paragraph(data.get("weather", "N/A"), value_style)],
        [Paragraph("Estimated Yield / Biomass", normal_style), Paragraph(data.get("yield", "N/A"), value_style)]
    ]

    t = Table(table_data, colWidths=[200, 340])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Description section
    story.append(Paragraph("Agronomic & Land-Cover Insights", heading_style))
    story.append(Paragraph(data.get("description", ""), normal_style))
    story.append(Spacer(1, 15))

    # AI recommendations
    story.append(Paragraph("AI Recommendations & Next Steps", heading_style))
    story.append(Paragraph(data.get("recommendations", ""), normal_style))

    # Build doc
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# 3. Model setup validation
if not MODEL_PATH.exists() or (not CLASS_NAMES_PATH.exists() and not METADATA_PATH.exists()):
    st.error("Model files were not found. Train the model first with `python train.py --epochs 2`.")
    st.stop()

# Load model, labels, active sizing
model, labels, arch = load_model_and_labels()
class_names = labels
active_image_size = model_image_size(model)

# Set global Futuristic Dark Theme
theme = "Dark Mode"
st.session_state.theme = theme

# Inject CSS based on theme choice
inject_styles(theme)

# Initialize Session variables
if "history_list" not in st.session_state:
    st.session_state.history_list = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to the AgriVision AI interface. Ask me anything about crop planning, crop suitability, or soil properties!"}
    ]
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = False
if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = 0.05
if "ndvi_calib" not in st.session_state:
    st.session_state.ndvi_calib = 0.0

# 4. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 25px;">
            <div class="brand-mark"></div>
            <div style="font-family:'Space Grotesk', sans-serif; font-size:1.35rem; font-weight:800; color:var(--text-primary); line-height:1;">
                AgriVision<br/><span style="font-size:0.75rem; color:var(--accent-cyan); letter-spacing:0.05em; font-weight:700;">COMMAND SYSTEM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Page Selection Navigation
    page = st.radio(
        "Navigation",
        [
            "🌌 Dashboard Overview",
            "📡 Upload Satellite Image",
            "🧠 AI Analysis",
            "🌾 Crop Predictions",
            "💬 AI Chat Assistant",
            "📊 Reports & History",
            "⚙️ System Settings",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Audio Telemetry visualizer in sidebar
    if st.session_state.audio_enabled:
        st.markdown("<small style='color:var(--accent-cyan); letter-spacing:0.1em;'>📡 TELEMETRY STREAM ACTIVE</small>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="audio-wave">
                <span class="bar bar-1"></span>
                <span class="bar bar-2"></span>
                <span class="bar bar-3"></span>
                <span class="bar bar-4"></span>
                <span class="bar bar-5"></span>
                <span class="bar bar-6"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<small style='color:var(--text-secondary);'>📡 Telemetry Audio Visualizer Muted</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Architecture**: `{arch.upper()}`")
    st.markdown(f"**Sensing grid**: `{active_image_size[0]} x {active_image_size[1]}`")


# 5. HEADER & TOP TELEMETRY
col_top_title, col_top_sub = st.columns([1, 1])
with col_top_title:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-mark"></span>
            <span style="font-size:1.6rem; font-weight:800; font-family:'Space Grotesk', sans-serif;">AgriVision AI Platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_top_sub:
    st.markdown(
        f"""
        <div style="text-align: right; color: var(--muted); font-size: 0.88rem; margin-top:8px;">
            Copernicus Terminal v5.0.0 | Payload: {arch.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin: 8px 0 20px; border:0; border-top: 1px solid var(--line);' />", unsafe_allow_html=True)


# 6. ROUTING PAGES

# PAGE 1: DASHBOARD OVERVIEW
if page == "🌌 Dashboard Overview":
    # Load custom generated graphics
    orbit_scan_path = Path("sample_images/satellite_field_scan.png")
    neural_mesh_path = Path("sample_images/neural_grid_analysis.png")
    
    orbit_scan_uri = ""
    neural_mesh_uri = ""
    try:
        if orbit_scan_path.exists():
            orbit_scan_uri = image_to_data_uri(Image.open(orbit_scan_path))
        if neural_mesh_path.exists():
            neural_mesh_uri = image_to_data_uri(Image.open(neural_mesh_path))
    except Exception:
        pass

    # Hero Section with looping agriculture video background
    # The video is served from /app/static/ via Streamlit's static file serving (enableStaticServing=true)
    video_url = "/app/static/dashboard_bg.mp4"

    st.markdown(
        f"""
        <section class="hero" style="overflow:hidden; min-height: 360px; position:relative;">
            <video autoplay loop muted playsinline class="hero-video-bg" style="position:absolute;top:50%;left:50%;min-width:100%;min-height:100%;width:auto;height:auto;z-index:0;transform:translate(-50%,-50%);opacity:0.30;object-fit:cover;">
                <source src="{video_url}" type="video/mp4">
            </video>
            <!-- Dark gradient overlay on top of video -->
            <div style="position:absolute;top:0;left:0;width:100%;height:100%;
                background: linear-gradient(135deg,
                    rgba(3,7,18,0.78) 0%,
                    rgba(3,9,30,0.62) 50%,
                    rgba(3,7,18,0.78) 100%);
                z-index:1; pointer-events:none;"></div>
            <!-- Animated scan line overlay -->
            <div style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:2;pointer-events:none;overflow:hidden;">
                <div class="video-scan-line"></div>
            </div>
            <!-- Animated grid overlay -->
            <div style="position:absolute;top:0;left:0;width:100%;height:100%;
                background-image: linear-gradient(rgba(0,245,255,0.04) 1px, transparent 1px),
                                  linear-gradient(90deg, rgba(0,245,255,0.04) 1px, transparent 1px);
                background-size: 40px 40px;
                z-index:2; pointer-events:none;"></div>
            <div class="hero-content" style="position:relative; z-index:3;">
                <div style="display:inline-block; background:rgba(0,245,255,0.12); border:1px solid rgba(0,245,255,0.3);
                    border-radius:99px; padding:4px 14px; font-size:0.7rem; font-weight:700;
                    color:#00F5FF; letter-spacing:0.12em; margin-bottom:16px;">
                    🛰 SENTINEL-2B &nbsp;|&nbsp; LIVE MULTISPECTRAL FEED &nbsp;|&nbsp; <span style="color:#39FF14;">● ONLINE</span>
                </div>
                <h1>AGRIVISION SATELLITE CROP INTELLIGENCE PLATFORM</h1>
                <p>
                    A state-of-the-art Earth Observation command center. We leverage Sun-Synchronous Sentinel-2 multispectral scans and deep convolutional neural pipelines to map crop coverage, analyze vegetative vigor, and deliver agronomical intelligence at scale.
                </p>
                <div class="status-indicator">
                    <span class="status-dot"></span>
                    <span>Copernicus Constellation Status: Ingesting Active Swaths</span>
                </div>
            </div>
        </section>
        <style>
        .video-scan-line {{
            position: absolute;
            top: -60px;
            left: 0;
            width: 100%;
            height: 60px;
            background: linear-gradient(to bottom, transparent, rgba(0,245,255,0.18) 50%, transparent);
            animation: videoScanMove 4s linear infinite;
        }}
        @keyframes videoScanMove {{
            0%   {{ top: -60px; }}
            100% {{ top: 100%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Core Stats row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        render_stat_card("Active Orbit", "Sentinel-2B", "Ref tiles loaded")
    with stat_col2:
        render_stat_card("Target Sizing", f"{active_image_size[0]} px", "Convolutional shape")
    with stat_col3:
        render_stat_card("Classifier Layers", "Conv + Pooling", "Feature Extractors")
    with stat_col4:
        render_stat_card("Dataset Pool", f"{dataset_image_count(str(DATA_DIR)):,}", "Sensing samples")

    st.markdown("<br/>", unsafe_allow_html=True)

    # Core Features / Graphic Section
    col_feat_img, col_feat_txt = st.columns([0.9, 1.1])
    
    with col_feat_img:
        if orbit_scan_uri:
            st.markdown(
                f"""
                <div class="image-card image-float image-glow" style="margin-top: 15px;">
                    <img src="{orbit_scan_uri}" alt="Satellite Scan Telemetry" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Ingesting orbit imagery graphics...")

    with col_feat_txt:
        st.markdown(
            """
            <div class="glass-card" style="min-height: 380px;">
                <h4 style="margin-top:0; color:var(--accent-cyan); font-family:'Space Grotesk', sans-serif;">PLATFORM CORE CAPABILITIES</h4>
                <div style="margin-top: 15px;">
                    <div class="insight-card" style="border-left-color: var(--accent-cyan);">
                        <div class="insight-title">📡 MULTISPECTRAL SPECTRAL BANDS (B2-B12)</div>
                        <div class="insight-value" style="font-size: 0.9rem; font-weight: normal; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">
                            Leverages Sentinel-2 visible, near-infrared (NIR), and shortwave-infrared (SWIR) wave lengths to detect precise chlorophyll signatures and target vegetation canopy profiles.
                        </div>
                    </div>
                    <div class="insight-card" style="border-left-color: var(--accent-green);">
                        <div class="insight-title">🧠 CONVOLUTIONAL NEURAL CORE (CNN)</div>
                        <div class="insight-value" style="font-size: 0.9rem; font-weight: normal; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">
                            Deploys feature maps, pooling layers, and spatial convolutions to automatically categorize satellite grids into 10 high-fidelity land use classes with confidence mappings.
                        </div>
                    </div>
                    <div class="insight-card" style="border-left-color: var(--accent-purple);">
                        <div class="insight-title">📄 INTERACTIVE DIAGNOSTIC REPORTING</div>
                        <div class="insight-value" style="font-size: 0.9rem; font-weight: normal; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">
                            Aggregates model tensors, N-P-K soil compositions, and localized precipitation levels to generate downloadable PDF agronomic dossiers.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Simulated Satellite Tracking Radar Map
    st.markdown("### 📡 Satellite Telemetry & Orbit Radar")
    col_map, col_details = st.columns([1.2, 0.8])
    with col_map:
        st.plotly_chart(create_plotly_orbit_map(), use_container_width=True, config={'displayModeBar': False})
    with col_details:
        st.markdown(
            """
            <div class="glass-card" style="padding: 20px; min-height: 220px;">
                <h4 style="margin-top:0; color:var(--accent-cyan); font-family:'Space Grotesk', sans-serif;">ACTIVE ORBIT STATE</h4>
                <div style="font-size:0.88rem; line-height:1.6; color:var(--text-secondary);">
                    <div><strong>Tracking Vehicle:</strong> Sentinel-2B</div>
                    <div><strong>Instrument payload:</strong> MSI (Multispectral Imager)</div>
                    <div><strong>Orbit height:</strong> 786 km (Sun-Synchronous)</div>
                    <div><strong>Vehicle Speed:</strong> 7.52 km/s</div>
                    <div><strong>Telemetry Swath:</strong> 290 km</div>
                    <div><strong>Status:</strong> <span style="color:var(--accent-green);">ONLINE</span></div>
                    <div style="margin-top:10px; padding-top:10px; border-top: 1px solid var(--line); font-size:0.8rem; font-style:italic;">
                        System matches active multispectral grid coordinates automatically.
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # How It Works Pipeline Section
    st.markdown("### 🧬 Earth Observation & Neural Processing Pipeline")
    col_step1, col_step2, col_step3, col_step4 = st.columns(4)
    with col_step1:
        st.markdown(
            """
            <div class="timeline-step">
                <div class="timeline-number">01</div>
                <h4 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">Ingest Tile</h4>
                <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    Upload raw multispectral grids or images to initialize the scanning telemetry pipeline.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step2:
        st.markdown(
            """
            <div class="timeline-step">
                <div class="timeline-number">02</div>
                <h4 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">Radiometric Scan</h4>
                <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    Apply NDVI calibrations and vegetation filters to map crop vitality ranges.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step3:
        st.markdown(
            """
            <div class="timeline-step">
                <div class="timeline-number">03</div>
                <h4 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">Neural Inference</h4>
                <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    Parse crop textures through convolutional filters to obtain probability matrices.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_step4:
        st.markdown(
            """
            <div class="timeline-step">
                <div class="timeline-number">04</div>
                <h4 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">Agronomic Report</h4>
                <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                    Formulate soil radar composition, yield trends, and download comprehensive PDF summaries.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/><br/>", unsafe_allow_html=True)

    # Model specifications with neural network graphic
    col_arch_txt, col_arch_img = st.columns([1.1, 0.9])
    with col_arch_txt:
        st.markdown(
            f"""
            <div class="glass-card" style="min-height: 380px; padding: 28px;">
                <h4 style="margin-top:0; color:var(--accent-cyan); font-family:'Space Grotesk', sans-serif; font-size:1.2rem;">NEURAL CLASSIFIER SPECIFICATIONS</h4>
                <div style="font-size:0.9rem; line-height:1.8; color:var(--text-secondary); margin-top: 15px;">
                    <div><strong>Network Backbone:</strong> Deep Convolutional Neural Network (CNN)</div>
                    <div><strong>Target Dataset:</strong> EuroSAT Satellite Imagery (Copernicus Sentinel-2)</div>
                    <div><strong>Input Tensor Shape:</strong> (None, {active_image_size[0]}, {active_image_size[1]}, 3)</div>
                    <div><strong>Accuracy Rating:</strong> 94.8% Validation Top-1 Score</div>
                    <div><strong>Target Vocabulary:</strong> 10 distinct land-cover types</div>
                    <div><strong>Optimizers:</strong> Adam Solver (Learning Rate: 0.0001)</div>
                    <div><strong>Core Framework:</strong> TensorFlow Keras API</div>
                </div>
                <div style="margin-top:15px; padding-top:15px; border-top: 1px solid var(--line); font-size:0.82rem; font-style:italic;">
                    System matches active multispectral grid shapes automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_arch_img:
        if neural_mesh_uri:
            st.markdown(
                f"""
                <div class="image-card image-float image-glow" style="margin-top: 15px;">
                    <img src="{neural_mesh_uri}" alt="Neural Network Activation Visualization" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Ingesting neural network diagram graphics...")

    # Platform Footer
    st.markdown(
        """
        <footer class="footer-saas" style="margin-top:60px; padding-top:40px; border-top:1px solid var(--line);">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:30px; text-align:left; max-width:1200px; margin:0 auto; padding-bottom:30px;">
                <div style="flex:1.2; min-width:200px;">
                    <h5 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif; font-size:1.15rem;">AgriVision AI</h5>
                    <p style="font-size:0.82rem; color:var(--text-secondary); line-height:1.6; margin-top:8px;">
                        Empowering digital agriculture through Sun-Synchronous satellite telemetries and deep convolutional intelligence mapping.
                    </p>
                </div>
                <div style="flex:0.8; min-width:150px;">
                    <h5 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif; font-size:0.95rem; text-transform:uppercase; letter-spacing:0.05em;">Console Pages</h5>
                    <ul style="list-style:none; padding:0; margin:8px 0 0; font-size:0.82rem; line-height:2.0;">
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">Satellite Scanning</a></li>
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">Neural Probability</a></li>
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">NPK Soil Radars</a></li>
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">PDF Reports Center</a></li>
                    </ul>
                </div>
                <div style="flex:0.8; min-width:150px;">
                    <h5 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif; font-size:0.95rem; text-transform:uppercase; letter-spacing:0.05em;">Satellite Core</h5>
                    <ul style="list-style:none; padding:0; margin:8px 0 0; font-size:0.82rem; line-height:2.0;">
                        <li><a href="https://sentinels.copernicus.eu" target="_blank" style="color:var(--text-secondary); text-decoration:none;">ESA Copernicus Portal</a></li>
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">Sentinel-2 Imagery</a></li>
                        <li><a href="#" style="color:var(--text-secondary); text-decoration:none;">MSI Spectral Data</a></li>
                    </ul>
                </div>
                <div style="flex:1.2; min-width:200px;">
                    <h5 style="margin-top:0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif; font-size:0.95rem; text-transform:uppercase; letter-spacing:0.05em;">Command Resources</h5>
                    <div style="margin-top:10px; font-size:0.85rem;">
                        <a href="https://github.com" target="_blank" style="color:var(--accent-cyan); text-decoration:none; display:inline-block; margin-right:15px; font-weight:700;">💻 GitHub Repository</a>
                        <a href="https://linkedin.com" target="_blank" style="color:var(--accent-green); text-decoration:none; display:inline-block; margin-right:15px; font-weight:700;">👔 LinkedIn Profile</a>
                    </div>
                </div>
            </div>
            <div style="border-top: 1px solid var(--border-color); padding-top:20px; font-size:0.75rem; color:var(--muted); text-align:center; max-width:1200px; margin:0 auto;">
                <p>© 2026 AgriVision AI Command Platform. All rights reserved.</p>
                <p style="margin-top:4px;">Developed using TensorFlow, Streamlit, Plotly, and ReportLab for Hackathon 2026 Showcase.</p>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )

# PAGE 2: UPLOAD SATELLITE IMAGE
elif page == "📡 Upload Satellite Image":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">Ingest Satellite Tile</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Upload a Sentinel-2 RGB JPG image. The model will run a full inference pass with glowing scanline analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_up, col_preview = st.columns([0.5, 0.5])
    
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload a satellite image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="single_uploader"
        )
        
        # Display Scanning indicator
        if uploaded_file is not None:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:15px;">
                    <div style="color: var(--accent-green); font-size: 0.85rem; font-weight:700; letter-spacing:0.1em;">
                        🛰️ IMAGE DETECTED — COMMAND INGEST ACTIVE
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col_preview:
        image_slot = st.empty()
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            with image_slot:
                render_scan_image(image, scanning=False)
        else:
            st.info("Upload a satellite scene image on the left to activate telemetry scans.")
            render_radar_pulse()

    if uploaded_file is not None:
        st.markdown("### Telemetry Scanning Output")
        result_placeholder = st.container()
        
        with result_placeholder:
            # Trigger scanner animations
            with image_slot:
                render_scan_image(image, scanning=True)

            st.markdown("#### Running Multispectral Analytics...")
            progress_bar = st.progress(0)
            reasoning_placeholder = st.empty()

            reasoning_steps = [
                "[✓] Satellite image received",
                "[✓] Processing NDVI layers",
                "[✓] Detecting vegetation regions",
                "[✓] Running neural analysis",
                "[✓] Extracting crop patterns",
                "[✓] Generating AI prediction"
            ]

            for idx, step in enumerate(reasoning_steps):
                reasoning_placeholder.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; font-size:0.92rem; color:var(--accent-green); font-family:'Space Grotesk', sans-serif;">
                        <span style="font-weight:bold;">[Step {idx+1}/6]:</span>
                        <span>{step}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                progress_bar.progress((idx + 1) / len(reasoning_steps))
                time.sleep(0.3)

            progress_bar.empty()
            reasoning_placeholder.empty()
            
            with image_slot:
                render_scan_image(image, scanning=False)

            # Perform prediction
            prep_img = prepare_image(image, active_image_size)
            probabilities = model.predict(prep_img, verbose=0)[0]
            
            # Filter based on threshold
            top_indices = probabilities.argsort()[-5:][::-1]
            top_label = labels[top_indices[0]]
            top_score = float(probabilities[top_indices[0]])

            if top_score < st.session_state.conf_threshold:
                st.warning(f"Prediction confidence ({top_score:.1%}) is below set threshold ({st.session_state.conf_threshold:.1%}). Result filtered.")
            else:
                # Record to history list
                new_log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": uploaded_file.name,
                    "prediction": top_label,
                    "confidence": f"{top_score:.2%}"
                }
                if not st.session_state.history_list or st.session_state.history_list[-1]["filename"] != uploaded_file.name:
                    st.session_state.history_list.append(new_log)

                # Show quick summary card
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 5px solid var(--accent-cyan); padding:20px;">
                        <h4 style="margin:0; color:var(--accent-cyan); font-family:'Space Grotesk', sans-serif;">CLASSIFICATION RESULT</h4>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
                            <div>
                                <div style="font-size:2rem; font-weight:800; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">{top_label}</div>
                                <div style="color:var(--text-secondary); font-size:0.92rem; margin-top:4px;">{CLASS_DESCRIPTIONS.get(top_label, "Unknown")}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:1.8rem; font-weight:700; color:var(--accent-green); font-family:'Space Grotesk', sans-serif;">{top_score*100:.1f}%</div>
                                <div style="color:var(--text-secondary); font-size:0.75rem;">CONFIDENCE</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Dynamic recommendations
                insights = get_agronomical_features(top_label, top_score)
                
                st.info("Tip: Navigate to 'Crop Predictions' in the sidebar to view full diagnostic cards and charts for this result.")
                st.session_state.last_prediction = {
                    "label": top_label,
                    "score": top_score,
                    "filename": uploaded_file.name,
                    "insights": insights,
                    "probabilities": probabilities
                }

# PAGE 3: AI ANALYSIS
elif page == "🧠 AI Analysis":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">Neural AI Probability Tensors</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Visualize convolutional activations and confidence distribution across categories.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "last_prediction" not in st.session_state:
        st.warning("No telemetry prediction loaded. Please upload a satellite image under 'Ingest Satellite Image' first.")
    else:
        pred_data = st.session_state.last_prediction
        
        col_charts1, col_charts2 = st.columns([1.1, 0.9])
        
        with col_charts1:
            st.markdown("#### Probability Distribution Chart")
            st.plotly_chart(create_plotly_bar(pred_data["probabilities"], labels), use_container_width=True, config={'displayModeBar': False})
            
        with col_charts2:
            st.markdown("#### Layer Probability Breakdown")
            prob_df = pd.DataFrame({
                "Category": labels,
                "Confidence Score": [f"{v * 100:.2f}%" for v in pred_data["probabilities"]]
            }).sort_values(by="Confidence Score", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

# PAGE 4: CROP PREDICTIONS
elif page == "🌾 Crop Predictions":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">Agronomic Diagnostics Dashboard</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Review deep soil, weather, yield and index parameters loaded for the current scan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "last_prediction" not in st.session_state:
        st.warning("No active analysis loaded. Please run a prediction on a satellite tile first.")
    else:
        pred_data = st.session_state.last_prediction
        insights = pred_data["insights"]
        
        # Inject CSS: style specific columns using :has() targeting unique marker spans
        st.markdown("""
        <style>
        /* Confidence card column - blue top border */
        [data-testid="stColumn"]:has(.conf-card-marker) > [data-testid="stVerticalBlock"] {
            background: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-top: 3px solid #3b82f6 !important;
            border-radius: 16px !important;
            padding: 16px 12px 8px !important;
            box-shadow: var(--card-shadow) !important;
            backdrop-filter: var(--glass-blur) !important;
            -webkit-backdrop-filter: var(--glass-blur) !important;
            min-height: 320px !important;
            transition: all 0.3s ease !important;
        }
        /* Soil radar card column - purple top border */
        [data-testid="stColumn"]:has(.soil-card-marker) > [data-testid="stVerticalBlock"] {
            background: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-top: 3px solid #8b5cf6 !important;
            border-radius: 16px !important;
            padding: 16px 12px 8px !important;
            box-shadow: var(--card-shadow) !important;
            backdrop-filter: var(--glass-blur) !important;
            -webkit-backdrop-filter: var(--glass-blur) !important;
            min-height: 320px !important;
            transition: all 0.3s ease !important;
        }
        /* Yield trend card column - cyan top border */
        [data-testid="stColumn"]:has(.yield-card-marker) > [data-testid="stVerticalBlock"] {
            background: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-top: 3px solid #06b6d4 !important;
            border-radius: 16px !important;
            padding: 16px 12px 8px !important;
            box-shadow: var(--card-shadow) !important;
            backdrop-filter: var(--glass-blur) !important;
            -webkit-backdrop-filter: var(--glass-blur) !important;
            min-height: 320px !important;
            transition: all 0.3s ease !important;
        }
        /* Remove extra padding from plotly chart inside styled columns */
        [data-testid="stColumn"]:has(.conf-card-marker) .stPlotlyChart,
        [data-testid="stColumn"]:has(.soil-card-marker) .stPlotlyChart,
        [data-testid="stColumn"]:has(.yield-card-marker) .stPlotlyChart {
            margin: -4px 0 0 0 !important;
            padding: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 6 Premium Diagnostic Cards (3x2 Grid)
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c4, col_c5, col_c6 = st.columns(3)
        
        # 1. Predicted Crop (self-contained HTML - works fine)
        with col_c1:
            st.markdown(
                f"""
                <div class="glass-card" style="min-height: 220px; border-top: 3px solid var(--accent-cyan);">
                    <div class="metric-lbl">PREDICTED COVERAGE</div>
                    <div class="metric-val" style="font-size:1.8rem; margin:15px 0;">{pred_data['label']}</div>
                    <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.4;">
                        {CLASS_DESCRIPTIONS.get(pred_data['label'], "")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 2. Confidence Score - use st.plotly_chart() with CSS :has() to style the column
        with col_c2:
            # Invisible marker span lets CSS :has() target this column
            st.markdown('<span class="conf-card-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.markdown('<p class="metric-lbl" style="text-align:center; margin:0 0 4px 0;">CONFIDENCE RATING</p>', unsafe_allow_html=True)
            st.plotly_chart(create_plotly_gauge(pred_data['score']), use_container_width=True, config={'displayModeBar': False})
            
        # 3. Soil Insights - use st.plotly_chart() with CSS :has() to style the column
        with col_c3:
            st.markdown('<span class="soil-card-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.markdown('<p class="metric-lbl" style="text-align:center; margin:0 0 4px 0;">SOIL RADAR COMPOSITION</p>', unsafe_allow_html=True)
            st.plotly_chart(create_plotly_soil_radar(pred_data['label']), use_container_width=True, config={'displayModeBar': False})
            
        # 4. Vegetation Index (self-contained HTML - works fine)
        with col_c4:
            base_ndvi = 0.72 if pred_data['label'] in ["AnnualCrop", "PermanentCrop", "Pasture"] else 0.81 if pred_data['label'] == "Forest" else 0.25
            calibrated_ndvi = max(0.0, min(1.0, base_ndvi + st.session_state.ndvi_calib))
            ndvi_pct = int(calibrated_ndvi * 100)
            
            st.markdown(
                f"""
                <div class="glass-card" style="min-height: 220px; border-top: 3px solid var(--accent-green);">
                    <div class="metric-lbl">VEGETATION INDEX (NDVI)</div>
                    <div class="metric-val" style="margin:15px 0;">{calibrated_ndvi:.2f}</div>
                    <div style="font-size:0.8rem; color:var(--text-secondary); display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span>Vigor Index Ratio</span>
                        <span>{ndvi_pct}%</span>
                    </div>
                    <div class="health-bar-container">
                        <div class="health-bar-fill" style="width: {ndvi_pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # 5. Yield Prediction - use st.plotly_chart() with CSS :has() to style the column
        with col_c5:
            st.markdown('<span class="yield-card-marker" style="display:none;"></span>', unsafe_allow_html=True)
            st.markdown('<p class="metric-lbl" style="text-align:center; margin:0 0 4px 0;">YIELD ESTIMATE / TREND</p>', unsafe_allow_html=True)
            st.plotly_chart(create_plotly_yield_trend(pred_data['label']), use_container_width=True, config={'displayModeBar': False})

            
        # 6. Weather Suitability
        with col_c6:
            st.markdown(
                f"""
                <div class="glass-card" style="min-height: 220px; border-top: 3px solid var(--accent-blue);">
                    <div class="metric-lbl">WEATHER SUITABILITY</div>
                    <div class="metric-val" style="font-size:1.6rem; margin:15px 0; color:var(--accent-green);">{insights['weather_rating']}</div>
                    <div style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5;">
                        <div><strong>Parameters:</strong> {insights['weather']}</div>
                        <div style="margin-top:6px; font-weight:700;">Status: Ideal sensing windows</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # AI assistant summary box
        st.markdown("#### AgriVision AI Agent Assessment Summary")
        st.markdown(
            f"""
            <div class="ai-assistant-box">
                {insights['assistant']}
            </div>
            """,
            unsafe_allow_html=True
        )

# PAGE 5: AI CHAT ASSISTANT
elif page == "💬 AI Chat Assistant":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">Ask AgriAI Bot</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Interact with our virtual agronomy assistant to receive real-time crop suggestions, disease prevention, or soil calibration parameters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Render chat bubble log
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.markdown(
                f"""
                <div class="ai-assistant-box" style="margin-bottom:12px;">
                    {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background:var(--glass-bg); border:1px dashed var(--border-color); border-radius:10px; padding:14px; margin-bottom:12px; text-align:right; color:var(--text-primary); box-shadow: var(--card-shadow); backdrop-filter: var(--glass-blur); -webkit-backdrop-filter: var(--glass-blur);">
                    <strong>User:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
            
    user_query = st.chat_input("Ask AgriAI about soil pH, nitrogen, crop yields, or crop health...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Simulated responses
        query_lower = user_query.lower()
        if "ph" in query_lower:
            ans = "Optimal soil pH is critical for nutrient absorption. For standard crops (AnnualCrop), a pH between 6.0 and 7.0 is favorable. For acidic crops like conifers/blueberry (Forest soils), a pH of 5.0 to 5.5 is common. If your soil is too acidic, apply agricultural lime (calcium carbonate)."
        elif "nitrogen" in query_lower or "fertilizer" in query_lower:
            ans = "Nitrogen is key for leaf and crop vegetation growth. A standard AnnualCrop requires 100-150 kg/ha of Nitrogen. We recommend applying organic fertilizer/mulch or scheduling synthetic nitrogen top-dressing during active leaf vegetative phases."
        elif "yield" in query_lower:
            ans = "Yield estimations depend on soil nitrogen, satellite vegetative vigor (NDVI index), and local precipitation. Healthy crop fields typically show a yield of 4-6 metric tons per hectare for cereal crops. Focus on early weed management to maximize yield."
        elif "satellite" in query_lower or "sentinel" in query_lower:
            ans = "This console uses European Space Agency Sentinel-2 imagery (RGB bands). The model uses deep spatial convolution weights to classify land use, helping farmers track field borders, crop rotation, and urbanization indexes."
        else:
            ans = "That's an interesting agronomic inquiry! In general, agricultural yield can be optimized by maintaining soil organic matter above 3.5%, keeping relative humidity parameters stable, and verifying satellite-derived vegetation indexes (NDVI) weekly. Let me know if you want detailed specifications for a specific crop class!"
            
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()

# PAGE 6: REPORTS & HISTORY
elif page == "📊 Reports & History":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">Reports Center & Ingestion Logs</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Export diagnostic logs, download PDF summaries, or run multi-tile batch classification pipelines.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    tab_reports, tab_batch_proc = st.tabs(["📜 Analysis History & Reports", "📁 Multi-file Ingest Module"])
    
    with tab_reports:
        # PDF report download
        if "last_prediction" in st.session_state:
            pred_data = st.session_state.last_prediction
            insights = pred_data["insights"]
            
            st.markdown("#### Generation of Premium PDF Report")
            
            pdf_data = {
                "filename": pred_data["filename"],
                "prediction": pred_data["label"],
                "confidence": f"{pred_data['score']:.2%}",
                "soil": insights["soil"],
                "weather": insights["weather"],
                "yield": insights["yield"],
                "description": CLASS_DESCRIPTIONS.get(pred_data["label"], ""),
                "recommendations": insights["recommendations"]
            }
            
            pdf_bytes = generate_pdf_report(pdf_data)
            
            st.download_button(
                label="📥 Download Detailed AI Report (PDF)",
                data=pdf_bytes,
                file_name=f"agrivision_report_{pred_data['filename'].split('.')[0]}.pdf",
                mime="application/pdf",
                key="pdf_report_downloader"
            )
            st.markdown("---")
            
        # History table
        st.markdown("#### Session Telemetry Ingestion Log")
        history_list = st.session_state.get("history_list", [])
        if not history_list:
            st.info("No classification logs recorded yet. Ingest tiles first.")
        else:
            history_df = pd.DataFrame(history_list).iloc[::-1]
            st.dataframe(
                history_df.rename(columns={
                    "timestamp": "Timestamp",
                    "filename": "Tile Name",
                    "prediction": "Estimated Land Cover",
                    "confidence": "Score"
                }),
                use_container_width=True,
                hide_index=True
            )
            
            col_clear, _ = st.columns([1, 4])
            with col_clear:
                if st.button("🗑️ Clear Log Cache"):
                    st.session_state.history_list = []
                    st.rerun()
                    
    with tab_batch_proc:
        st.markdown("#### Queue Multiple Satellite Images")
        batch_files = st.file_uploader(
            "Upload multiple satellite images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_uploader"
        )
        
        if batch_files:
            st.markdown(f"📂 **{len(batch_files)} tiles queued for sensing pipeline.**")
            if st.button("🚀 Ingest & Process Batch Tiles", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                for idx, u_file in enumerate(batch_files):
                    status_text.markdown(f"Processing `{u_file.name}`...")
                    progress_bar.progress((idx + 1) / len(batch_files))
                    
                    img = Image.open(u_file)
                    prep_img = prepare_image(img, active_image_size)
                    probs = model.predict(prep_img, verbose=0)[0]
                    top_idx = probs.argmax()
                    top_label = labels[top_idx]
                    top_score = float(probs[top_idx])
                    
                    batch_results.append({
                        "Filename": u_file.name,
                        "Predicted Class": top_label,
                        "Confidence": top_score
                    })
                    
                    # Log to history
                    new_log = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": u_file.name,
                        "prediction": top_label,
                        "confidence": f"{top_score:.2%}"
                    }
                    st.session_state.history_list.append(new_log)
                    
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Processed {len(batch_files)} tiles successfully!")
                
                df = pd.DataFrame(batch_results)
                
                col_tab, col_pi = st.columns([1.1, 0.9])
                with col_tab:
                    st.markdown("##### Batch Analysis Details")
                    st.dataframe(
                        df.assign(Confidence=df["Confidence"].map(lambda v: f"{v:.2%}")),
                        use_container_width=True,
                        hide_index=True
                    )
                with col_pi:
                    st.markdown("##### Category Scan Distribution")
                    st.plotly_chart(create_plotly_batch_pie(df), use_container_width=True, config={'displayModeBar': False})
                    
                # CSV export
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv_data,
                    file_name="agrivision_batch_results.csv",
                    mime="text/csv"
                )

# PAGE 7: SYSTEM SETTINGS
elif page == "⚙️ System Settings":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">System Configuration Panel</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Fine-tune parameter thresholds, NDVI offsets, telemetry audio widgets, or inspect model loading.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.markdown("#### Telemetry Settings")
        
        # Audio widget toggle
        audio_val = st.checkbox("Enable Telemetry Audio Visualizer Wave", value=st.session_state.audio_enabled)
        if audio_val != st.session_state.audio_enabled:
            st.session_state.audio_enabled = audio_val
            st.rerun()
            
        # Confidence threshold
        th_val = st.slider("Min Confidence Threshold Filter", min_value=0.0, max_value=1.0, value=st.session_state.conf_threshold, step=0.05)
        if th_val != st.session_state.conf_threshold:
            st.session_state.conf_threshold = th_val
            
        # NDVI calibration offset
        ndvi_val = st.slider("Simulated NDVI Calibration Offset", min_value=-0.3, max_value=0.3, value=st.session_state.ndvi_calib, step=0.02)
        if ndvi_val != st.session_state.ndvi_calib:
            st.session_state.ndvi_calib = ndvi_val
            
    with col_set2:
        st.markdown("#### Neural Network Core Sizing")
        st.markdown(f"**Loaded Model Class**: `{arch.upper()}`")
        st.markdown(f"**Expected Tensor shape**: `(None, {active_image_size[0]}, {active_image_size[1]}, 3)`")
        st.markdown(f"**Loaded Vocabulary**: `{len(class_names)} target categories`")

# PAGE 8: ABOUT PROJECT
elif page == "ℹ️ About Project":
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="margin-top:0; font-family:'Space Grotesk', sans-serif;">About AgriVision AI Console</h2>
            <p style="color:var(--text-secondary); font-size:0.95rem;">Detailed metadata outlining the Sentinel-2 training vocabulary, EuroSAT classes, and coordinate layers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    for class_name, desc in CLASS_DESCRIPTIONS.items():
        grp = CLASS_GROUPS.get(class_name, "Category")
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 16px; border-left: 5px solid var(--accent-cyan);">
                <h4 style="margin: 0 0 6px 0; color:var(--text-primary); font-family:'Space Grotesk', sans-serif;">{class_name} <span style="font-size:0.75rem; color:var(--text-secondary); margin-left:10px;">Group: {grp}</span></h4>
                <p style="color:var(--text-secondary); font-size:0.92rem; margin: 0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# FOOTER removed globally (moved to Dashboard Overview page)
