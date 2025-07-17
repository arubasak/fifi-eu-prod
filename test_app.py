import streamlit as st
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuidimport streamlit as st
import os
import uuid
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    pass

# Simple configuration
class Config:
    def __init__(self):
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.WORDPRESS_URL = st.secrets.get("WORDPRESS_URL", "https://example.com")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

config = Config()

@dataclass
class UserSession:
    session_id: str
    user_type: str = "guest"
    email: Optional[str] = None
    first_name: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class SimpleSessionManager:
    """Simple session manager with in-memory storage."""
    
    def __init__(self):
        self.sessions = {}
    
    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            return session
        
        return self.create_guest_session()
    
    def create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.sessions[session.session_id] = session
        st.session_state.current_session_id = session.session_id
        return session
    
    def clear_chat_history(self, session: UserSession):
        session.messages = []
        session.last_activity = datetime.now()

class SimpleAI:
    """Simple AI interface with multiple backends."""
    
    def __init__(self):
        self.openai_client = None
        self.langchain_llm = None
        
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
        
        if LANGCHAIN_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.langchain_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=config.OPENAI_API_KEY,
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"LangChain LLM initialization failed: {e}")
    
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> str:
        """Get AI response using available backend."""
        try:
            # Try OpenAI client first
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
            
            # Try LangChain as fallback
            elif self.langchain_llm:
                from langchain_core.messages import HumanMessage
                response = self.langchain_llm.invoke([HumanMessage(content=prompt)])
                return response.content
            
            else:
                return "AI services are not available. Please configure your OpenAI API key."
        
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "I'm having trouble processing your request right now. Please try again."

def init_session_state():
    """Initialize session state safely."""
    if 'initialized' not in st.session_state:
        try:
            # Simple initialization without complex detection
            st.session_state.session_manager = SimpleSessionManager()
            st.session_state.ai = SimpleAI()
            st.session_state.page = "chat"  # Start directly in chat
            st.session_state.initialized = True
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.session_state.initialized = False

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant")
    
    session = st.session_state.session_manager.get_session()
    
    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))
            if "source" in msg:
                st.caption(f"Source: {msg['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about food & beverage sourcing..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        session.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ai.get_response(prompt, session.messages)
                st.markdown(response)
        
        # Add AI response to history
        session.messages.append({
            "role": "assistant",
            "content": response,
            "source": "FiFi AI",
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session
        session.last_activity = datetime.now()
        
        st.rerun()

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.title("Chat Controls")
        
        session = st.session_state.session_manager.get_session()
        
        # Session info
        st.subheader("Session Info")
        st.write(f"**ID:** {session.session_id[:8]}...")
        st.write(f"**Type:** {session.user_type}")
        if session.email:
            st.write(f"**Email:** {session.email}")
        st.write(f"**Messages:** {len(session.messages)}")
        
        # Status indicators
        st.subheader("System Status")
        st.write(f"**OpenAI:** {'✅' if OPENAI_AVAILABLE and config.OPENAI_API_KEY else '❌'}")
        st.write(f"**LangChain:** {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
        st.write(f"**SQLite Cloud:** {'✅' if SQLITECLOUD_AVAILABLE else '❌'}")
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear History"):
            st.session_state.session_manager.clear_chat_history(session)
            st.rerun()
        
        if st.button("🔄 New Session"):
            if 'current_session_id' in st.session_state:
                del st.session_state.current_session_id
            st.rerun()
        
        # Feature status
        st.subheader("Available Features")
        st.write("✅ Basic Chat")
        st.write("✅ Session Management")
        st.write("✅ OpenAI Integration")
        if LANGCHAIN_AVAILABLE:
            st.write("✅ LangChain Support")
        else:
            st.write("❌ LangChain (install required)")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="FiFi AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    try:
        # Initialize session state
        init_session_state()
        
        if not st.session_state.get('initialized', False):
            st.error("⚠️ Application initialization failed")
            st.info("Please refresh the page.")
            return
        
        # Render interface
        render_sidebar()
        render_chat_interface()
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        st.error("🚨 Application Error")
        st.text(f"Error: {str(e)}")
        
        if st.button("🔄 Restart App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

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
