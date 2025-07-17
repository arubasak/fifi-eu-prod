import streamlit as st
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuid

# Import our working classes
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone
from tavily import TavilyClient

st.title("🧪 Method Call Test")

# Set fake environment variables for testing
os.environ['OPENAI_API_KEY'] = 'fake-key'
os.environ['TAVILY_API_KEY'] = 'fake-key'
os.environ['PINECONE_API_KEY'] = 'fake-key'

# Test 1: ChatOpenAI method calls
st.write("🔄 Testing ChatOpenAI method calls...")
try:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="fake-key", temperature=0.7)
    
    # Test invoke method (this might call st.get_option internally)
    st.write("  - Testing invoke method...")
    try:
        # This will fail due to fake API key, but we're testing for st.get_option() error
        response = llm.invoke([HumanMessage(content="test")])
    except Exception as e:
        if "get_option() takes 1 positional argument but 2 were given" in str(e):
            st.error(f"❌ Found st.get_option() error in ChatOpenAI.invoke: {e}")
        else:
            st.write(f"  - Expected API error (not st.get_option): {type(e).__name__}")
    
    st.write("✅ ChatOpenAI method testing completed")
except Exception as e:
    st.error(f"❌ ChatOpenAI method test failed: {e}")
    st.exception(e)

# Test 2: TavilyClient method calls
st.write("🔄 Testing TavilyClient method calls...")
try:
    tavily_client = TavilyClient(api_key="fake-key")
    
    # Test search method
    st.write("  - Testing search method...")
    try:
        response = tavily_client.search(query="test", max_results=1)
    except Exception as e:
        if "get_option() takes 1 positional argument but 2 were given" in str(e):
            st.error(f"❌ Found st.get_option() error in TavilyClient.search: {e}")
        else:
            st.write(f"  - Expected API error (not st.get_option): {type(e).__name__}")
    
    st.write("✅ TavilyClient method testing completed")
except Exception as e:
    st.error(f"❌ TavilyClient method test failed: {e}")
    st.exception(e)

# Test 3: Our session state operations
st.write("🔄 Testing session state operations...")
try:
    # Test session state assignments that might trigger the error
    st.write("  - Testing deployment origin detection...")
    
    # This is from our original code - might be the culprit
    is_streamlit_cloud = (
        'streamlit.app' in os.environ.get('STREAMLIT_SERVER_ADDRESS', '') or
        'streamlit.app' in os.environ.get('HOSTNAME', '') or
        os.environ.get('STREAMLIT_SHARING_MODE') == 'true'
    )
    
    st.session_state.test_deployment_origin = 'https://fifi-co-pilot.streamlit.app'
    st.session_state.test_deployment_env = 'streamlit_cloud'
    
    st.write("✅ Session state operations completed")
except Exception as e:
    if "get_option() takes 1 positional argument but 2 were given" in str(e):
        st.error(f"❌ Found st.get_option() error in session state operations: {e}")
    st.exception(e)

# Test 4: Our original _get_current_origin method
st.write("🔄 Testing origin detection method...")
try:
    def test_get_current_origin():
        try:
            # Use session state if available
            if hasattr(st.session_state, 'deployment_origin'):
                return st.session_state.deployment_origin
            
            # Simple environment detection
            is_streamlit_cloud = (
                'streamlit.app' in os.environ.get('STREAMLIT_SERVER_ADDRESS', '') or
                'streamlit.app' in os.environ.get('HOSTNAME', '') or
                os.environ.get('STREAMLIT_SHARING_MODE') == 'true'
            )
            
            if is_streamlit_cloud:
                return 'https://fifi-co-pilot.streamlit.app'
            else:
                return 'https://fifi-co-pilot-v1-121263692901.europe-west4.run.app'
                
        except Exception as e:
            return 'https://fifi-co-pilot.streamlit.app'
    
    origin = test_get_current_origin()
    st.write(f"✅ Origin detection successful: {origin}")
except Exception as e:
    if "get_option() takes 1 positional argument but 2 were given" in str(e):
        st.error(f"❌ Found st.get_option() error in origin detection: {e}")
    st.exception(e)

# Test 5: Session initialization simulation
st.write("🔄 Testing session initialization simulation...")
try:
    # Simulate our init_session_state function
    if 'test_initialized' not in st.session_state:
        st.session_state.test_deployment_origin = 'https://fifi-co-pilot.streamlit.app'
        st.session_state.test_deployment_env = 'streamlit_cloud'
        st.session_state.test_db_manager = None
        st.session_state.test_initialized = True
    
    st.write("✅ Session initialization simulation completed")
except Exception as e:
    if "get_option() takes 1 positional argument but 2 were given" in str(e):
        st.error(f"❌ Found st.get_option() error in session initialization: {e}")
    st.exception(e)

st.success("🎉 Method call test completed!")
st.info("Check above for any st.get_option() errors to identify the exact source.")
