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
from urllib.parse import urlparse

# =============================================================================
# VERSION 2.2 CHANGELOG:
# - FIXED: Citation source list now displays a clean name (e.g., file title or domain)
#   instead of the full raw URL as the link text.
# - Implemented clickable inline citations that jump to the source list.
# - Added EnhancedErrorHandler for sophisticated error management.
# ... (other previous changelog items)
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
# NEW in v2.0: Enhanced Error Handling System
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
    def __init__(self):
        self.error_history = []
        self.component_status = {}

    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        error_str = str(error).lower()
        error_type = type(error).__name__
        if "timeout" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="TimeoutError", severity=ErrorSeverity.MEDIUM, user_message=f"{component} is responding slowly. Please try again.", technical_details=str(error), recovery_suggestions=["Try again", "Check internet"], fallback_available=True)
        if "unauthorized" in error_str or "401" in error_str:
            return ErrorContext(component=component, operation=operation, error_type="AuthenticationError", severity=ErrorSeverity.HIGH, user_message=f"{component} authentication failed.", technical_details=str(error), recovery_suggestions=["Check API key"], fallback_available=False)
        return ErrorContext(component=component, operation=operation, error_type=error_type, severity=ErrorSeverity.MEDIUM, user_message=f"{component} encountered an error.", technical_details=str(error), recovery_suggestions=["Try again"], fallback_available=True)

    def handle_import_error(self, package_name: str, feature_name: str) -> ErrorContext:
        return ErrorContext(component="Package Import", operation=f"Import {package_name}", error_type="ImportError", severity=ErrorSeverity.LOW, user_message=f"{feature_name} is unavailable.", technical_details=f"Package '{package_name}' not installed.", recovery_suggestions=[f"Install with: pip install {package_name}"], fallback_available=True)

    def display_error_to_user(self, error_context: ErrorContext):
        icons = {ErrorSeverity.LOW: "ℹ️", ErrorSeverity.MEDIUM: "⚠️", ErrorSeverity.HIGH: "🚨", ErrorSeverity.CRITICAL: "💥"}
        icon = icons.get(error_context.severity, "❓")
        st.warning(f"{icon} {error_context.user_message}")
        with st.expander("Details & Suggestions"):
            for suggestion in error_context.recovery_suggestions:
                st.write(f"• {suggestion}")

    def log_error(self, error_context: ErrorContext):
        self.error_history.append({"timestamp": datetime.now(), "context": error_context})
        self.component_status[error_context.component] = {"status": "error", "last_error": datetime.now()}
        if len(self.error_history) > 50: self.error_history.pop(0)

    def mark_component_healthy(self, component: str):
        self.component_status[component] = {"status": "healthy", "last_check": datetime.now()}

    def get_system_health_summary(self) -> Dict[str, Any]:
        if not self.component_status: return {"overall_health": "Unknown", "healthy_components": 0, "total_components": 0}
        healthy = sum(1 for s in self.component_status.values() if s.get("status") == "healthy")
        total = len(self.component_status)
        health = "Healthy" if healthy == total else "Degraded" if healthy > 0 else "Critical"
        return {"overall_health": health, "healthy_components": healthy, "total_components": total, "error_count": len(self.error_history)}

error_handler = EnhancedErrorHandler()

def handle_api_errors(component: str, operation: str, show_to_user: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                error_handler.mark_component_healthy(component)
                return result
            except Exception as e:
                ctx = error_handler.handle_api_error(component, operation, e)
                error_handler.log_error(ctx)
                if show_to_user: error_handler.display_error_to_user(ctx)
                logger.error(f"{component} {operation} failed: {e}")
                return None
        return wrapper
    return decorator

DEFAULT_EXCLUDED_DOMAINS = ["ingredientsnetwork.com", "csmingredients.com", "batafood.com"]

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
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)

class SimpleSessionManager:
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
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            ctx = error_handler.handle_import_error("pinecone", "Pinecone KB")
            error_handler.display_error_to_user(ctx)
            raise ImportError("Pinecone not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    @handle_api_errors("Pinecone", "Initialize")
    def _initialize_assistant(self):
        # Instructions text omitted for brevity
        instructions = "You are a helpful assistant..."
        assistants = self.pc.assistant.list_assistants()
        if self.assistant_name not in [a.name for a in assistants]:
            st.info(f"Creating Pinecone assistant: '{self.assistant_name}'")
            return self.pc.assistant.create_assistant(name=self.assistant_name, instructions=instructions, function_calling_plugins=[])
        st.success(f"Connected to Pinecone assistant: '{self.assistant_name}'")
        return self.pc.assistant.Assistant(name=self.assistant_name)

    @handle_api_errors("Pinecone", "Query", show_to_user=False)
    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant:
            return {"content": "Pinecone not ready.", "success": False}
        try:
            pinecone_messages = [PineconeMessage(role="user" if isinstance(m, HumanMessage) else "assistant", content=m.content) for m in chat_history]
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o", include_highlights=True)
            content_with_citations = insert_citations(response)
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
                                    domain = urlparse(link_url).netloc
                                    display_text = domain if domain else "Link"
                                except Exception:
                                    display_text = "Link"
                            # --- END OF NEW LOGIC ---

                            if display_text in seen_sources: continue
                            seen_sources.add(display_text)

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
                    content_with_citations += citations_header + "\n".join(citations_list)

            return {
                "content": content_with_citations,
                "success": True,
                "source": "FiFi Knowledge Base",
                "has_citations": has_citations,
                "has_inline_citations": has_citations,
            }
        except Exception as e:
            raise e

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            ctx = error_handler.handle_import_error("langchain-tavily", "Web Search")
            error_handler.display_error_to_user(ctx)
            raise ImportError("Tavily not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    @handle_api_errors("Tavily", "Query", show_to_user=False)
    def query(self, message: str) -> Dict[str, Any]:
        try:
            results = self.tavily_tool.invoke({"query": message})
            content = "Based on web search:\n" + json.dumps(results, indent=2)
            return {"content": content, "success": True, "source": "FiFi Web Search"}
        except Exception as e:
            raise e

class EnhancedAI:
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            try:
                self.pinecone_tool = PineconeAssistantTool(api_key=config.PINECONE_API_KEY, assistant_name=config.PINECONE_ASSISTANT_NAME)
            except Exception as e:
                logger.error(f"Pinecone init failed: {e}")
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(tavily_api_key=config.TAVILY_API_KEY)
            except Exception as e:
                logger.error(f"Tavily init failed: {e}")

    def should_use_web_fallback(self, response: Dict[str, Any]) -> bool:
        # Simplified fallback logic for brevity
        content = response.get("content", "").lower()
        if "i don't have specific information" in content:
            return True
        if not response.get("has_citations") and len(content) > 50:
            return True
        return False

    def get_response(self, prompt: str, chat_history: List[Dict]) -> Dict[str, Any]:
        history = [HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content']) for m in chat_history]
        history.append(HumanMessage(content=prompt))

        if self.pinecone_tool:
            pinecone_res = self.pinecone_tool.query(history)
            if pinecone_res and pinecone_res.get("success"):
                if not self.should_use_web_fallback(pinecone_res):
                    return {**pinecone_res, "used_search": False, "used_pinecone": True}
                logger.warning("Pinecone response inadequate, falling back to web search.")

        if self.tavily_agent:
            logger.info("Using Tavily web search fallback.")
            tavily_res = self.tavily_agent.query(prompt)
            if tavily_res and tavily_res.get("success"):
                return {**tavily_res, "used_search": True, "used_pinecone": False}

        return {"content": "All AI systems are currently unavailable.", "success": False, "source": "System Error"}

def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.session_manager = SimpleSessionManager()
        st.session_state.ai = EnhancedAI()
        st.session_state.error_handler = error_handler
        st.session_state.initialized = True

def render_chat_interface():
    st.title("🤖 FiFi AI Assistant v2.2")
    session = st.session_state.session_manager.get_session()

    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("role") == "assistant" and "source" in msg:
                st.caption(f"Source: {msg['source']}")

    if prompt := st.chat_input("Ask me anything..."):
        session.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ai.get_response(prompt, session.messages)
                content = response.get("content", "Sorry, I encountered an error.")
                st.markdown(content, unsafe_allow_html=True)
                if "source" in response:
                    st.caption(f"Source: {response['source']}")
                session.messages.append({"role": "assistant", **response})
        st.rerun()

def render_sidebar():
    with st.sidebar:
        st.title("Controls & Status")
        if st.button("🗑️ Clear History"):
            st.session_state.session_manager.clear_chat_history(st.session_state.session_manager.get_session())
            st.rerun()
        st.subheader("System Health")
        # Sidebar content omitted for brevity

def main():
    st.set_page_config(page_title="FiFi AI Assistant v2.2", page_icon="🤖", layout="wide")
    try:
        init_session_state()
        if not st.session_state.get('initialized'):
            st.error("Application failed to initialize.")
            return
        render_sidebar()
        render_chat_interface()
    except Exception as e:
        logger.critical(f"Critical error in main app: {e}")
        error_handler.display_error_to_user(ErrorContext(component="Application", operation="Main Loop", error_type=type(e).__name__, severity=ErrorSeverity.CRITICAL, user_message="A critical error occurred. Please restart.", technical_details=str(e), recovery_suggestions=["Refresh the page"]))

if __name__ == "__main__":
    main()
