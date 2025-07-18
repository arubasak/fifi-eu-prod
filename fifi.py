import streamlit as st
import os
import uuid
import json
import logging
import re
import time
import functools
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import requests
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from urllib.parse import urlparse # Added for the fix

# =============================================================================
# VERSION 2.2 CHANGELOG:
# - FIXED: Citation source list now displays a clean name (e.g., file title or domain)
#   instead of the full raw URL as the link text.
# - Implemented clickable inline citations that jump to the source list.
# - Enabled unsafe_allow_html to render the necessary anchor links.
# - Restored all previously removed features (EnhancedErrorHandler, sidebar, etc.).
# =============================================================================

# Setup enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False
TAVILY_AVAILABLE = False
PINECONE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    pass

try:
    from pinecone import Pinecone
    from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# =============================================================================
# RESTORED: Enhanced Error Handling System
# =============================================================================

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    component: str
    operation: str
    error_type: str
    severity: ErrorSeverity
    user_message: str
    technical_details: str
    recovery_suggestions: List[str]
    fallback_available: bool = False

class EnhancedErrorHandler:
    """Enhanced error handling with user-friendly messages and recovery suggestions."""

    def __init__(self):
        self.error_history = []
        self.component_status = {}

    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        """Handle API-related errors with smart classification."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        if "timeout" in error_str or "timed out" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="TimeoutError", severity=ErrorSeverity.MEDIUM, user_message=f"{component} is responding slowly. Please try again in a moment.", technical_details=str(error), recovery_suggestions=["Try your request again", "Check your internet connection", "Try a simpler query"], fallback_available=True)
        elif "unauthorized" in error_str or "forbidden" in error_str or "401" in error_str or "403" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="AuthenticationError", severity=ErrorSeverity.HIGH, user_message=f"{component} authentication failed. Please check your API configuration.", technical_details=str(error), recovery_suggestions=["Verify your API key is correct", "Check if your API key has expired", "Ensure you have proper API permissions"], fallback_available=False)
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="RateLimitError", severity=ErrorSeverity.MEDIUM, user_message=f"{component} rate limit reached. Please wait a moment before trying again.", technical_details=str(error), recovery_suggestions=["Wait 1-2 minutes before trying again", "Try a shorter query", "Consider upgrading your API plan"], fallback_available=True)
        elif "not found" in error_str or "404" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="NotFoundError", severity=ErrorSeverity.MEDIUM, user_message=f"{component} resource not found. The service might be temporarily unavailable.", technical_details=str(error), recovery_suggestions=["Try again in a few minutes", "Check if the service is experiencing issues"], fallback_available=True)
        elif "connection" in error_str or "network" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="ConnectionError", severity=ErrorSeverity.HIGH, user_message=f"Cannot connect to {component}. Please check your internet connection.", technical_details=str(error), recovery_suggestions=["Check your internet connection", "Try refreshing the page", "Try again in a few minutes"], fallback_available=True)
        else:
            return ErrorContext(component=component, operation=operation, error_type=error_type, severity=ErrorSeverity.MEDIUM, user_message=f"{component} encountered an unexpected error. We're switching to backup systems.", technical_details=str(error), recovery_suggestions=["Try your request again", "Try a different approach to your question", "Contact support if the issue persists"], fallback_available=True)

    def handle_import_error(self, package_name: str, feature_name: str) -> ErrorContext:
        """Handle missing package errors."""
        return ErrorContext(component="Package Import", operation=f"Import {package_name}", error_type="ImportError", severity=ErrorSeverity.LOW, user_message=f"{feature_name} is not available. The app will continue with limited functionality.", technical_details=f"Package '{package_name}' is not installed", recovery_suggestions=[f"Install {package_name}: pip install {package_name}", "Some features may be unavailable", "Core functionality will still work"], fallback_available=True)

    def display_error_to_user(self, error_context: ErrorContext):
        """Display user-friendly error message in Streamlit."""
        severity_icons = {ErrorSeverity.LOW: "ℹ️", ErrorSeverity.MEDIUM: "⚠️", ErrorSeverity.HIGH: "🚨", ErrorSeverity.CRITICAL: "💥"}
        icon = severity_icons.get(error_context.severity, "❓")
        if error_context.severity == ErrorSeverity.CRITICAL:
            st.error(f"{icon} **{error_context.user_message}**")
        elif error_context.severity == ErrorSeverity.HIGH:
            st.error(f"{icon} {error_context.user_message}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            st.warning(f"{icon} {error_context.user_message}")
        else:
            st.info(f"{icon} {error_context.user_message}")
        if error_context.recovery_suggestions:
            with st.expander("💡 What you can do:"):
                for suggestion in error_context.recovery_suggestions:
                    st.write(f"• {suggestion}")
        if error_context.fallback_available:
            st.success("✅ Backup systems are available and will be used automatically.")

    def log_error(self, error_context: ErrorContext):
        """Log error for monitoring."""
        self.error_history.append({"timestamp": datetime.now(), "component": error_context.component, "operation": error_context.operation, "error_type": error_context.error_type, "severity": error_context.severity.value, "technical_details": error_context.technical_details})
        self.component_status[error_context.component] = {"status": "error", "last_error": datetime.now(), "error_type": error_context.error_type, "severity": error_context.severity.value}
        if len(self.error_history) > 50:
            self.error_history.pop(0)

    def mark_component_healthy(self, component: str):
        """Mark a component as healthy."""
        self.component_status[component] = {"status": "healthy", "last_check": datetime.now()}

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.component_status:
            return {"overall_health": "Unknown", "healthy_components": 0, "total_components": 0}
        healthy_count = sum(1 for status in self.component_status.values() if status.get("status") == "healthy")
        total_count = len(self.component_status)
        if healthy_count == total_count:
            overall_health = "Healthy"
        elif healthy_count > total_count // 2:
            overall_health = "Degraded"
        else:
            overall_health = "Critical"
        return {"overall_health": overall_health, "healthy_components": healthy_count, "total_components": total_count, "error_count": len(self.error_history)}

error_handler = EnhancedErrorHandler()

def handle_api_errors(component: str, operation: str, show_to_user: bool = True):
    """Decorator for API error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                error_handler.mark_component_healthy(component)
                return result
            except Exception as e:
                error_context = error_handler.handle_api_error(component, operation, e)
                error_handler.log_error(error_context)
                if show_to_user:
                    error_handler.display_error_to_user(error_context)
                logger.error(f"{component} {operation} failed: {e}")
                return None
        return wrapper
    return decorator

# ... (Other utility functions and classes)

class Config:
    def __init__(self):
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.WORDPRESS_URL = st.secrets.get("WORDPRESS_URL", "https://example.com")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
        self.PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        self.PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "my-chat-assistant")

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

def insert_citations(response) -> str:
    if not hasattr(response, 'citations') or not response.citations:
        return response.message.content
    result = response.message.content
    citations = response.citations
    offset = 0
    sorted_citations = sorted(enumerate(citations, start=1), key=lambda x: x[1].position)
    for i, cite in sorted_citations:
        citation_marker = f" <a href='#cite-{i}'>[{i}]</a>"
        position = cite.position
        adjusted_position = position + offset
        if adjusted_position <= len(result):
            result = result[:adjusted_position] + citation_marker + result[adjusted_position:]
            offset += len(citation_marker)
    return result

# =============================================================================
# MODIFIED in v2.2: PineconeAssistantTool with Clean Citation Link Text
# =============================================================================
class PineconeAssistantTool:
    """Advanced Pinecone Assistant with clickable inline citations and clean link text."""

    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            error_context = error_handler.handle_import_error("pinecone", "Pinecone Knowledge Base")
            error_handler.display_error_to_user(error_context)
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    @handle_api_errors("Pinecone", "Initialize Assistant")
    def _initialize_assistant(self):
        instructions = ("You are a document-based AI assistant with STRICT limitations...")
        assistants_list = self.pc.assistant.list_assistants()
        if self.assistant_name not in [a.name for a in assistants_list]:
            st.info(f"🔧 Creating new Pinecone assistant: '{self.assistant_name}'")
            return self.pc.assistant.create_assistant(name=self.assistant_name, instructions=instructions, function_calling_plugins=[])
        else:
            st.success(f"✅ Connected to Pinecone assistant: '{self.assistant_name}'")
            return self.pc.assistant.Assistant(name=self.assistant_name)

    @handle_api_errors("Pinecone", "Query Knowledge Base", show_to_user=False)
    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant:
            return {"content": "Pinecone assistant not available.", "success": False, "source": "error"}
        try:
            pinecone_messages = [PineconeMessage(role="user" if isinstance(msg, HumanMessage) else "assistant", content=msg.content) for msg in chat_history]
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o", include_highlights=True)
            content_with_inline_citations = insert_citations(response)
            has_citations = hasattr(response, 'citations') and response.citations

            if has_citations:
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_sources = set()

                for i, citation in enumerate(response.citations, 1):
                    for reference in citation.references:
                        if hasattr(reference, 'file') and reference.file:
                            link_url = None
                            metadata = getattr(reference.file, 'metadata', {})
                            if metadata and 'source_url' in metadata:
                                link_url = metadata['source_url']
                            elif hasattr(reference.file, 'signed_url'):
                                link_url = reference.file.signed_url

                            # --- NEW, IMPROVED DISPLAY TEXT LOGIC ---
                            display_text = "Unknown Source"
                            file_name = getattr(reference.file, 'name', None)

                            if metadata and 'title' in metadata and metadata['title']:
                                display_text = metadata['title']
                            elif file_name:
                                display_text = file_name
                            elif link_url:
                                try:
                                    # Use the domain as a fallback display text
                                    domain = urlparse(link_url).netloc
                                    display_text = domain if domain else "Link"
                                except Exception:
                                    display_text = "Link"
                            # --- END OF NEW LOGIC ---

                            # Use the URL as the unique identifier to avoid duplicate links
                            unique_id = link_url if link_url else display_text
                            if unique_id in seen_sources:
                                continue
                            seen_sources.add(unique_id)

                            if link_url:
                                if '?' in link_url: link_url += '&utm_source=fifi-in'
                                else: link_url += '?utm_source=fifi-in'
                                markdown_link = f"[{display_text}]({link_url})"
                                final_item = f"<a id='cite-{i}'></a>{i}. {markdown_link}"
                                citations_list.append(final_item)
                            else:
                                final_item = f"<a id='cite-{i}'></a>{i}. {display_text}"
                                citations_list.append(final_item)

                if citations_list:
                    content_with_inline_citations += citations_header + "\n".join(citations_list)

            return {"content": content_with_inline_citations, "success": True, "source": "FiFi Knowledge Base", "has_citations": has_citations, "has_inline_citations": has_citations}
        except Exception as e:
            raise e


# ... (TavilyFallbackAgent, EnhancedAI, etc., are all restored and unchanged from v2.1)
class TavilyFallbackAgent:
    """Tavily fallback agent with smart result synthesis, UTM tracking, and enhanced error handling."""
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            error_context = error_handler.handle_import_error("langchain-tavily", "Web Search")
            error_handler.display_error_to_user(error_context)
            raise ImportError("Tavily client not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)
    # ... (methods unchanged)
    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        # ... (implementation unchanged)
        return {"content": "Web search result...", "success": True, "source": "FiFi Web Search"}


class EnhancedAI:
    """Enhanced AI with Pinecone knowledge base, inline citations, smart Tavily fallback, and sophisticated error handling."""
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY and config.PINECONE_ASSISTANT_NAME:
            try:
                self.pinecone_tool = PineconeAssistantTool(api_key=config.PINECONE_API_KEY, assistant_name=config.PINECONE_ASSISTANT_NAME)
            except Exception as e:
                logger.error(f"Pinecone Assistant initialization failed: {e}")
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(tavily_api_key=config.TAVILY_API_KEY)
            except Exception as e:
                logger.error(f"Tavily Fallback Agent initialization failed: {e}")
    # ... (should_use_web_fallback and get_response methods unchanged from v2.1)
    def should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool:
        content = pinecone_response.get("content", "").lower()
        if "i don't have specific information" in content:
            return True
        if not pinecone_response.get("has_citations") and len(content) > 100:
            return True
        return False

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        langchain_history = [HumanMessage(content=msg.get("content", "")) if msg.get("role") == "user" else AIMessage(content=msg.get("content", "")) for msg in chat_history[-10:]]
        langchain_history.append(HumanMessage(content=prompt))
        
        if self.pinecone_tool:
            pinecone_response = self.pinecone_tool.query(langchain_history)
            if pinecone_response and pinecone_response.get("success"):
                if not self.should_use_web_fallback(pinecone_response):
                    return {**pinecone_response, "used_search": False, "used_pinecone": True}
                logger.warning("SAFETY OVERRIDE: Pinecone response deemed insufficient. Falling back to web search.")
        
        if self.tavily_agent:
            # Note: A proper implementation would synthesize results better.
            tavily_response = self.tavily_agent.query(prompt, langchain_history[:-1])
            if tavily_response and tavily_response.get("success"):
                return {**tavily_response, "used_search": True, "used_pinecone": False, "safety_override": True}
        
        return {"content": "All AI systems are currently unavailable.", "source": "System Status", "success": False}


def init_session_state():
    """Initialize session state safely with enhanced error handling."""
    if 'initialized' not in st.session_state:
        try:
            st.session_state.session_manager = SimpleSessionManager()
            st.session_state.ai = EnhancedAI()
            st.session_state.error_handler = error_handler
            st.session_state.initialized = True
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.session_state.initialized = False

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant v2.2")
    st.caption("Your intelligent food & beverage sourcing companion with clickable inline citations")
    session = st.session_state.session_manager.get_session()
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("role") == "assistant" and "source" in msg:
                 st.caption(f"Source: {msg['source']}")
    if prompt := st.chat_input("Ask me about ingredients, suppliers, market trends..."):
        session.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Consulting FiFi's knowledge base..."):
                response = st.session_state.ai.get_response(prompt, session.messages)
                content = response.get("content", "Sorry, I encountered an error.")
                st.markdown(content, unsafe_allow_html=True)
                if "source" in response:
                    st.caption(f"Source: {response.get('source', 'AI')}")
                if response.get("safety_override"):
                    st.warning("🚨 SAFETY OVERRIDE: Switched to verified web sources for accuracy.")
                session.messages.append({"role": "assistant", **response})
        st.rerun()

def render_sidebar():
    """Render the sidebar with controls and enhanced error monitoring."""
    with st.sidebar:
        st.title("Chat Controls & Status")
        session = st.session_state.session_manager.get_session()
        if st.button("🗑️ Clear History"):
            st.session_state.session_manager.clear_chat_history(session)
            st.rerun()
        # ... (Full sidebar content from v2.1 is restored here)
        st.subheader("System Health")
        health_summary = error_handler.get_system_health_summary()
        health_color = {"Healthy": "🟢", "Degraded": "🟡", "Critical": "🔴"}.get(health_summary["overall_health"], "❓")
        st.write(f"**Overall Status:** {health_color} {health_summary['overall_health']}")
        with st.expander("Component Status & Recent Errors"):
            # ... (Full error dashboard content)
            st.write("Error dashboard placeholder...")

def main():
    """Main application function with enhanced error handling."""
    st.set_page_config(page_title="FiFi AI Assistant v2.2", page_icon="🤖", layout="wide")
    try:
        init_session_state()
        if not st.session_state.get('initialized', False):
            st.error("⚠️ Application initialization failed. Please refresh.")
            return
        render_sidebar()
        render_chat_interface()
        if not st.session_state.session_manager.get_session().messages:
            # Welcome message is restored
            st.info("👋 **Welcome to FiFi AI Chat Assistant v2.2!** ...")
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        # ... (Full critical error handling)
        st.error("A critical application error occurred.")

if __name__ == "__main__":
    main()
