import streamlit as st
import os
import logging

# Step 1: Test basic imports
st.write("🔄 Testing basic imports...")
try:
    import uuid
    import time
    import json
    import threading
    from typing import List, Dict, Optional, Any, Union
    from datetime import datetime, timedelta
    from dataclasses import dataclass, field
    from collections import defaultdict
    import requests
    import jwt
    from enum import Enum
    import io
    import re
    import sys
    from urllib.parse import urlparse
    import html
    st.write("✅ Basic imports successful")
except Exception as e:
    st.error(f"❌ Basic imports failed: {e}")
    st.stop()

# Step 2: Test PDF imports
st.write("🔄 Testing PDF imports...")
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import black, grey, lightgrey
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    st.write("✅ PDF imports successful")
except Exception as e:
    st.error(f"❌ PDF imports failed: {e}")

# Step 3: Test AI imports
st.write("🔄 Testing AI imports...")
try:
    import openai
    st.write("✅ OpenAI import successful")
except Exception as e:
    st.error(f"❌ OpenAI import failed: {e}")

# Step 4: Test SQLite Cloud
st.write("🔄 Testing SQLite Cloud...")
try:
    import sqlitecloud
    st.write("✅ SQLite Cloud import successful")
except Exception as e:
    st.error(f"❌ SQLite Cloud import failed: {e}")

# Step 5: Test LangChain imports (THIS IS LIKELY THE CULPRIT)
st.write("🔄 Testing LangChain imports...")
try:
    from langchain_openai import ChatOpenAI
    st.write("✅ langchain_openai successful")
    
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    st.write("✅ langchain_core.messages successful")
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    st.write("✅ langchain_core.prompts successful")
    
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    st.write("✅ langchain.agents successful")
    
    from langchain_community.tools.tavily_search import TavilySearchResults
    st.write("✅ langchain_community.tools successful")
    
    from langchain_core.tools import tool
    st.write("✅ langchain_core.tools successful")
    
except Exception as e:
    st.error(f"❌ LangChain imports failed: {e}")
    st.exception(e)

# Step 6: Test Pinecone
st.write("🔄 Testing Pinecone...")
try:
    from pinecone import Pinecone
    st.write("✅ Pinecone import successful")
except Exception as e:
    st.error(f"❌ Pinecone import failed: {e}")

# Step 7: Test Tavily
st.write("🔄 Testing Tavily...")
try:
    from tavily import TavilyClient
    st.write("✅ Tavily import successful")
except Exception as e:
    st.error(f"❌ Tavily import failed: {e}")

# Step 8: Test configuration loading
st.write("🔄 Testing configuration...")
try:
    # Simple config test
    jwt_secret = st.secrets.get("JWT_SECRET", "test-secret")
    st.write(f"✅ Config loaded, JWT secret length: {len(jwt_secret)}")
except Exception as e:
    st.error(f"❌ Config loading failed: {e}")

st.success("🎉 All import tests completed!")
st.info("Check above for any ❌ errors that might be causing the st.get_option() issue.")

# Test if we can create basic classes
st.write("🔄 Testing class creation...")
try:
    @dataclass
    class TestSession:
        session_id: str = "test"
        messages: List[Dict[str, Any]] = field(default_factory=list)
    
    test_session = TestSession()
    st.write("✅ Dataclass creation successful")
except Exception as e:
    st.error(f"❌ Class creation failed: {e}")

st.write("📋 Debug complete. Any errors above indicate the problematic component.")
