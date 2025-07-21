import streamlit as st
import os
import uuid
import json
import logging
import re
import time
import functools
import io
import html
import jwt
import threading
from enum import Enum
from urllib.parse import urlparse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black, grey, lightgrey
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import requests

# =============================================================================
# VERSION 2.2 REFACTORED
# - Consolidated duplicated classes (Config, UserSession, SessionManager).
# - Removed redundant ErrorSeverity enum definition.
# - Applied error handling decorators more consistently.
# - Simplified configuration and session management initialization.
# - Refactored main application flow for clarity and robustness.
# - Ensured all external calls are wrapped in error handlers.
# =============================================================================

# Setup enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Graceful Fallbacks for Optional Imports ---
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False
TAVILY_AVAILABLE = False
PINECONE_AVAILABLE = False

try:
    import openai
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    OPENAI_AVAILABLE = True
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
    pass

# =============================================================================
# 1. UNIFIED ERROR HANDLING SYSTEM
# =============================================================================

class ErrorSeverity(Enum):
    """Single, unified definition for error severity."""
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
    """Centralized error handling with user-friendly messages."""
    def __init__(self):
        self.error_history = []
        self.component_status = {}

    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        error_str = str(error).lower()
        error_type = type(error).__name__
        # Simplified error classification logic
        if "timeout" in error_str:
            severity, message = ErrorSeverity.MEDIUM, "is responding slowly."
        elif "unauthorized" in error_str or "401" in error_str or "403" in error_str:
            severity, message = ErrorSeverity.HIGH, "authentication failed. Please check API keys."
        elif "rate limit" in error_str or "429" in error_str:
            severity, message = ErrorSeverity.MEDIUM, "rate limit reached. Please wait."
        elif "connection" in error_str or "network" in error_str:
            severity, message = ErrorSeverity.HIGH, "is unreachable. Check your connection."
        else:
            severity, message = ErrorSeverity.MEDIUM, "encountered an unexpected error."

        return ErrorContext(
            component=component, operation=operation, error_type=error_type,
            severity=severity, user_message=f"{component} {message}",
            technical_details=str(error),
            recovery_suggestions=["Try again", "Check your internet", "Contact support if issue persists"],
            fallback_available=True if severity != ErrorSeverity.HIGH else False
        )

    def handle_import_error(self, package_name: str, feature_name: str) -> ErrorContext:
        """Handle missing package errors."""
        return ErrorContext(
            component="Package Import", operation=f"Import {package_name}",
            error_type="ImportError", severity=ErrorSeverity.LOW,
            user_message=f"{feature_name} is unavailable because the '{package_name}' package is not installed.",
            technical_details=f"'{package_name}' is not installed.",
            recovery_suggestions=[f"Install the package: pip install {package_name}"],
            fallback_available=True
        )

    def display_error_to_user(self, error_context: ErrorContext):
        severity_icons = {
            ErrorSeverity.LOW: "ℹ️", ErrorSeverity.MEDIUM: "⚠️",
            ErrorSeverity.HIGH: "🚨", ErrorSeverity.CRITICAL: "💥"
        }
        icon = severity_icons.get(error_context.severity, "❓")
        st.error(f"{icon} {error_context.user_message}")

    def log_error(self, error_context: ErrorContext):
        self.error_history.append({
            "timestamp": datetime.now(), "component": error_context.component,
            "severity": error_context.severity.value, "details": error_context.technical_details
        })
        self.component_status[error_context.component] = "error"
        if len(self.error_history) > 50: self.error_history.pop(0)

    def mark_component_healthy(self, component: str):
        self.component_status[component] = "healthy"

# Initialize a single, global error handler
error_handler = EnhancedErrorHandler()

def handle_api_errors(component: str, operation: str, show_to_user: bool = True):
    """Decorator for consistent API error handling."""
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
                logger.error(f"API Error in {component}/{operation}: {e}")
                return None
        return wrapper
    return decorator

# =============================================================================
# 2. UNIFIED CONFIGURATION
# =============================================================================

class Config:
    """A single, consolidated configuration class."""
    def __init__(self):
        # Core Secrets
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
        self.PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        self.PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "my-chat-assistant")

        # WordPress and Database
        self.WORDPRESS_URL = self._validate_url(st.secrets.get("WORDPRESS_URL", ""))
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")

        # Zoho CRM (Optional)
        self.ZOHO_CLIENT_ID = st.secrets.get("ZOHO_CLIENT_ID")
        self.ZOHO_CLIENT_SECRET = st.secrets.get("ZOHO_CLIENT_SECRET")
        self.ZOHO_REFRESH_TOKEN = st.secrets.get("ZOHO_REFRESH_TOKEN")
        self.ZOHO_ENABLED = all([self.ZOHO_CLIENT_ID, self.ZOHO_CLIENT_SECRET, self.ZOHO_REFRESH_TOKEN])

    def _validate_url(self, url: str) -> str:
        if url and not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format for WORDPRESS_URL: {url}. Disabling feature.")
            return ""
        return url.rstrip('/')

# =============================================================================
# 3. UNIFIED SESSION AND USER MODELS
# =============================================================================

class UserType(Enum):
    GUEST = "guest"
    REGISTERED_USER = "registered_user"

@dataclass
class UserSession:
    """A single, consolidated UserSession dataclass."""
    session_id: str
    user_type: UserType = UserType.GUEST
    email: Optional[str] = None
    first_name: Optional[str] = None
    zoho_contact_id: Optional[str] = None
    guest_email_requested: bool = False
    active: bool = True
    wp_token: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

# =============================================================================
# 4. CORE APPLICATION COMPONENTS
# (Database, PDF Exporter, Zoho Manager, Rate Limiter)
# =============================================================================

class DatabaseManager:
    """Manages database interactions for persistent sessions."""
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.use_cloud = False
        if connection_string and SQLITECLOUD_AVAILABLE:
            try:
                self.connection_string = connection_string
                self._init_database()
                self.use_cloud = True
                error_handler.mark_component_healthy("Database")
            except Exception as e:
                error_context = error_handler.handle_api_error("Database", "Initialize", e)
                error_handler.log_error(error_context)
                self._init_local_storage()
        else:
            self._init_local_storage()

    def _init_local_storage(self):
        logger.info("Using local in-memory storage for sessions (not persistent across restarts).")
        self.local_sessions = {}
        self.use_cloud = False

    def _get_connection(self):
        if not self.use_cloud: return None
        return sqlitecloud.connect(self.connection_string)

    def _init_database(self):
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY, user_type TEXT, email TEXT, first_name TEXT,
                        zoho_contact_id TEXT, guest_email_requested INTEGER, created_at TEXT,
                        last_activity TEXT, messages TEXT, active INTEGER, wp_token TEXT
                    )''')
                conn.commit()

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.use_cloud:
                with self._get_connection() as conn:
                    conn.execute(
                        '''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            session.session_id, session.user_type.value, session.email,
                            session.first_name, session.zoho_contact_id,
                            int(session.guest_email_requested), session.created_at.isoformat(),
                            session.last_activity.isoformat(), json.dumps(session.messages),
                            int(session.active), session.wp_token
                        )
                    )
                    conn.commit()
            else:
                self.local_sessions[session.session_id] = session

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            if self.use_cloud:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                    row = cursor.fetchone()
                    if not row: return None
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))
                    return UserSession(
                        session_id=row_dict['session_id'],
                        user_type=UserType(row_dict['user_type']),
                        email=row_dict.get('email'),
                        first_name=row_dict.get('first_name'),
                        zoho_contact_id=row_dict.get('zoho_contact_id'),
                        guest_email_requested=bool(row_dict.get('guest_email_requested')),
                        created_at=datetime.fromisoformat(row_dict['created_at']),
                        last_activity=datetime.fromisoformat(row_dict['last_activity']),
                        messages=json.loads(row_dict.get('messages', '[]')),
                        active=bool(row_dict.get('active', 1)),
                        wp_token=row_dict.get('wp_token')
                    )
            else:
                return self.local_sessions.get(session_id)

class PDFExporter:
    """Generates PDF reports from chat sessions."""
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='ChatHeader', alignment=TA_CENTER, fontSize=18))
        self.styles.add(ParagraphStyle(name='UserMessage', backColor=lightgrey))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = [Paragraph("FiFi AI Chat Transcript", self.styles['Heading1'])]
        for msg in session.messages:
            role = str(msg.get('role', 'unknown')).capitalize()
            content = html.escape(str(msg.get('content', '')))
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
        doc.build(story)
        buffer.seek(0)
        return buffer

class ZohoCRMManager:
    """Handles Zoho CRM integration."""
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"

    @handle_api_errors("Zoho CRM", "Get Access Token", show_to_user=False)
    def _get_access_token(self) -> Optional[str]:
        if not self.config.ZOHO_ENABLED: return None
        response = requests.post(
            "https://accounts.zoho.com/oauth/v2/token",
            data={
                'refresh_token': self.config.ZOHO_REFRESH_TOKEN,
                'client_id': self.config.ZOHO_CLIENT_ID,
                'client_secret': self.config.ZOHO_CLIENT_SECRET,
                'grant_type': 'refresh_token'
            }, timeout=10
        )
        response.raise_for_status()
        return response.json().get('access_token')

    @handle_api_errors("Zoho CRM", "Save Chat Transcript", show_to_user=False)
    def save_chat_transcript(self, session: UserSession):
        if not self.config.ZOHO_ENABLED or not session.zoho_contact_id: return
        pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
        if pdf_buffer:
            logger.info(f"PDF generated for Zoho upload for session {session.session_id}.")
            # In a real scenario, you'd upload this buffer to the Zoho API.

class RateLimiter:
    """Simple rate limiter to prevent abuse."""
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, identifier: str) -> bool:
        with self._lock:
            now = time.time()
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

def sanitize_input(text: str, max_length: int = 4000) -> str:
    """Sanitizes user input for security."""
    if not isinstance(text, str): return ""
    return html.escape(text)[:max_length].strip()

# =============================================================================
# 5. CONSOLIDATED AI SYSTEM
# (Pinecone, Tavily, Moderation)
# =============================================================================

def insert_citations(response) -> str:
    """Inserts clickable citation markers into response text."""
    if not hasattr(response, 'citations') or not response.citations:
        return response.message.content
    result = response.message.content
    citations = sorted(enumerate(response.citations, start=1), key=lambda x: x[1].position)
    offset = 0
    for i, cite in citations:
        citation_marker = f" <a href='#cite-{i}'>[{i}]</a>"
        position = cite.position + offset
        result = result[:position] + citation_marker + result[position:]
        offset += len(citation_marker)
    return result

class PineconeAssistantTool:
    """Pinecone Assistant with inline citations."""
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant = self.pc.assistant.create_assistant(
            assistant_name=assistant_name,
            instructions="You are a document-based AI assistant. Only answer from provided documents."
        ) if assistant_name not in [a.name for a in self.pc.assistant.list_assistants()] else self.pc.assistant.Assistant(assistant_name=assistant_name)

    @handle_api_errors("Pinecone", "Query", show_to_user=False)
    def query(self, chat_history: List[BaseMessage]) -> Optional[Dict[str, Any]]:
        pinecone_messages = [PineconeMessage(role="user" if isinstance(m, HumanMessage) else "assistant", content=m.content) for m in chat_history]
        response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o", include_highlights=True)
        content = insert_citations(response)
        has_citations = hasattr(response, 'citations') and bool(response.citations)
        return {"content": content, "success": True, "source": "FiFi Knowledge Base", "has_citations": has_citations}

class TavilyFallbackAgent:
    """Tavily fallback agent for web searches."""
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            raise ImportError("Tavily client not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    @handle_api_errors("Tavily", "Web Search", show_to_user=False)
    def query(self, message: str) -> Optional[Dict[str, Any]]:
        results = self.tavily_tool.invoke({"query": message})
        content = ""
        if isinstance(results, list):
            for i, result in enumerate(results[:3], 1):
                content += f"[{i}] {result.get('title', '')}: {result.get('content', '')}\n\n"
        elif isinstance(results, str):
            content = results
        return {"content": content or "No relevant information found.", "success": True, "source": "FiFi Web Search"}

class EnhancedAI:
    """Main AI system orchestrating Pinecone, Tavily, and moderation."""
    def __init__(self, config: Config):
        self.config = config
        self.pinecone_tool = None
        self.tavily_agent = None
        self.openai_client = None

        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
        if PINECONE_AVAILABLE and self.config.PINECONE_API_KEY:
            try:
                self.pinecone_tool = PineconeAssistantTool(self.config.PINECONE_API_KEY, self.config.PINECONE_ASSISTANT_NAME)
            except ImportError:
                error_handler.log_error(error_handler.handle_import_error("pinecone", "Pinecone Knowledge Base"))
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(self.config.TAVILY_API_KEY)
            except ImportError:
                error_handler.log_error(error_handler.handle_import_error("langchain-tavily", "Tavily Web Search"))

    def _should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool:
        """Aggressive fallback detection to prevent hallucination."""
        content = pinecone_response.get("content", "").lower()
        # Fallback if the response is a refusal or if it lacks citations for a substantial answer.
        is_refusal = "i don't have" in content or "i don't know" in content
        is_long_and_uncited = len(content) > 100 and not pinecone_response.get("has_citations")
        return is_refusal or is_long_and_uncited

    @handle_api_errors("AI System", "Generate Response")
    def get_response(self, prompt: str, chat_history: List[Dict]) -> Dict[str, Any]:
        if not LANGCHAIN_AVAILABLE:
            return {"content": "AI components are unavailable due to missing packages.", "success": False}

        langchain_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in chat_history]
        langchain_history.append(HumanMessage(content=prompt))

        # 1. Try Pinecone
        if self.pinecone_tool:
            pinecone_res = self.pinecone_tool.query(langchain_history)
            if pinecone_res and pinecone_res.get("success") and not self._should_use_web_fallback(pinecone_res):
                return pinecone_res

        # 2. Fallback to Tavily
        if self.tavily_agent:
            tavily_res = self.tavily_agent.query(prompt)
            if tavily_res and tavily_res.get("success"):
                return tavily_res

        # 3. Final fallback
        return {"content": "I apologize, but I couldn't retrieve an answer. Please try again.", "success": False, "source": "System"}

@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    if not client: return {"flagged": False}
    response = client.moderations.create(model="omni-moderation-latest", input=prompt)
    result = response.results[0]
    if result.flagged:
        return {"flagged": True, "message": "Your message violates our content policy and cannot be processed."}
    return {"flagged": False}

# =============================================================================
# 6. UNIFIED SESSION MANAGER
# =============================================================================

class SessionManager:
    """A single, consolidated session manager."""
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter

    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        if session_id:
            session = self.db.load_session(session_id)
            if session:
                session.last_activity = datetime.now()
                return session
        return self._create_guest_session()

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        return session

   # Place this inside the SessionManager class, ensuring it's at the same indentation level
# as the other methods like get_session and get_ai_response.

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

        response = requests.post(
            f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
            json={'username': sanitize_input(username), 'password': password},
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            user_data = data.get('user', {})
            
            # FIX: Get the current session instead of creating a new one.
            session = self.get_session() 

            # Update the existing session's attributes to upgrade it to an authenticated user.
            session.user_type = UserType.REGISTERED_USER
            session.email = user_data.get('user_email')
            session.first_name = user_data.get('user_nicename')
            session.wp_token = data.get('token')
            session.last_activity = datetime.now()

            self.db.save_session(session)
            st.success(f"Welcome back, {session.first_name}!")
            return session
        else:
            st.error("Invalid username or password.")
            return None
    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "Rate limit exceeded. Please wait.", "success": False}

        sanitized_prompt = sanitize_input(prompt)
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {"content": moderation["message"], "success": False, "source": "Content Safety"}

        response = self.ai.get_response(sanitized_prompt, session.messages)
        session.messages.append({"role": "user", "content": sanitized_prompt})
        session.messages.append({"role": "assistant", **response})
        session.messages = session.messages[-100:]  # Keep history trimmed
        self.db.save_session(session)
        return response

    def clear_chat_history(self, session: UserSession):
        session.messages = []
        self.db.save_session(session)

    def end_session(self, session: UserSession):
        self.zoho.save_chat_transcript(session)
        session.active = False
        self.db.save_session(session)
        for key in ['current_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]

# =============================================================================
# 7. UI RENDERING FUNCTIONS
# =============================================================================

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        if session.user_type == UserType.REGISTERED_USER:
            st.success(f"Authenticated: {session.first_name}")
        else:
            st.info("Guest User")
        st.write(f"Messages: {len(session.messages)}")

        if st.button("🗑️ Clear History", use_container_width=True):
            session_manager.clear_chat_history(session)
            st.rerun()
        if st.button("🚪 End Session", use_container_width=True):
            session_manager.end_session(session)
            st.rerun()

        if session.user_type == UserType.REGISTERED_USER and session.messages:
            pdf_buffer = pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF", data=pdf_buffer,
                    file_name=f"fifi_chat_{session.session_id[:8]}.pdf",
                    mime="application/pdf", use_container_width=True
                )

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    st.title("🤖 FiFi AI Assistant")
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("role") == "assistant" and "source" in msg:
                st.caption(f"Source: {msg['source']}")

    if prompt := st.chat_input("Ask me about ingredients, suppliers, or market trends..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = session_manager.get_ai_response(session, prompt)
                st.markdown(response.get("content", "I encountered an issue."), unsafe_allow_html=True)
                if "source" in response:
                    st.caption(f"Source: {response['source']}")
        st.rerun()

def render_welcome_page(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is disabled because the authentication service is not configured.")
        else:
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In", use_container_width=True):
                    if session_manager.authenticate_with_wordpress(username, password):
                        st.session_state.page = "chat"
                        st.rerun()
    with tab2:
        if st.button("Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# =============================================================================
# 8. MAIN APPLICATION
# =============================================================================

# =============================================================================
# 8. MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # --- INITIALIZATION ---
    if 'initialized' not in st.session_state:
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            # MODIFICATION: Create and store the DB Manager directly in session state.
            # This ensures its internal state (like the local_sessions dict) persists across reruns.
            st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            # Retrieve the persistent DB manager from session state.
            db_manager = st.session_state.db_manager
            
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            # MODIFICATION: Pass the persistent db_manager to the SessionManager.
            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.initialized = True
            logger.info("All components initialized successfully.")
        except Exception as e:
            st.error("A critical error occurred during application startup. Please check the logs.")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            st.stop()

    # --- PAGE ROUTING ---
    session_manager = st.session_state.session_manager
    if 'page' not in st.session_state:
        render_welcome_page(session_manager)
    else:
        # MODIFICATION: Ensure pdf_exporter is retrieved from session state here as well.
        pdf_exporter = st.session_state.pdf_exporter
        session = session_manager.get_session()
        render_sidebar(session_manager, session, pdf_exporter)
        render_chat_interface(session_manager, session)

if __name__ == "__main__":
    main()
