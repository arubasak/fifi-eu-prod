import streamlit as st
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuid

# All the imports that passed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from pinecone import Pinecone
from tavily import TavilyClient

st.title("🧪 Class Instantiation Test")

# Test 1: Basic dataclass (this should work)
st.write("🔄 Testing basic dataclass...")
try:
    @dataclass
    class UserSession:
        session_id: str
        user_type: str = "guest"
        messages: List[Dict[str, Any]] = field(default_factory=list)
        created_at: datetime = field(default_factory=datetime.now)
    
    session = UserSession(session_id=str(uuid.uuid4()))
    st.write("✅ UserSession creation successful")
except Exception as e:
    st.error(f"❌ UserSession creation failed: {e}")
    st.exception(e)

# Test 2: ChatOpenAI initialization (SUSPECT #1)
st.write("🔄 Testing ChatOpenAI initialization...")
try:
    # Try with minimal parameters first
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="fake-key-for-testing",  # Just testing initialization
        temperature=0.7
    )
    st.write("✅ ChatOpenAI initialization successful")
except Exception as e:
    st.error(f"❌ ChatOpenAI initialization failed: {e}")
    st.exception(e)

# Test 3: TavilySearchResults initialization (SUSPECT #2)
st.write("🔄 Testing TavilySearchResults initialization...")
try:
    search_tool = TavilySearchResults(
        api_key="fake-key-for-testing",  # Just testing initialization
        max_results=5
    )
    st.write("✅ TavilySearchResults initialization successful")
except Exception as e:
    st.error(f"❌ TavilySearchResults initialization failed: {e}")
    st.exception(e)

# Test 4: Pinecone initialization (SUSPECT #3)
st.write("🔄 Testing Pinecone initialization...")
try:
    pc = Pinecone(api_key="fake-key-for-testing")
    st.write("✅ Pinecone initialization successful")
except Exception as e:
    st.error(f"❌ Pinecone initialization failed: {e}")
    st.exception(e)

# Test 5: TavilyClient initialization (SUSPECT #4)
st.write("🔄 Testing TavilyClient initialization...")
try:
    tavily_client = TavilyClient(api_key="fake-key-for-testing")
    st.write("✅ TavilyClient initialization successful")
except Exception as e:
    st.error(f"❌ TavilyClient initialization failed: {e}")
    st.exception(e)

# Test 6: Our custom classes from the original code
st.write("🔄 Testing custom Config class...")
try:
    class MinimalConfig:
        def __init__(self):
            self.JWT_SECRET = "test-secret"
            self.WORDPRESS_URL = "https://example.com"
            self.OPENAI_API_KEY = "fake-key"
            self.SQLITE_CLOUD_CONNECTION = None
    
    config = MinimalConfig()
    st.write("✅ MinimalConfig creation successful")
except Exception as e:
    st.error(f"❌ MinimalConfig creation failed: {e}")
    st.exception(e)

# Test 7: DatabaseManager simulation
st.write("🔄 Testing DatabaseManager simulation...")
try:
    class MockDatabaseManager:
        def __init__(self):
            self.local_sessions = {}
            self.use_cloud = False
        
        def save_session(self, session):
            self.local_sessions[session.session_id] = session
        
        def load_session(self, session_id):
            return self.local_sessions.get(session_id)
    
    db_manager = MockDatabaseManager()
    st.write("✅ MockDatabaseManager creation successful")
except Exception as e:
    st.error(f"❌ MockDatabaseManager creation failed: {e}")
    st.exception(e)

st.success("🎉 Class instantiation test completed!")
st.info("The first ❌ error above shows which class is calling st.get_option() internally.")
