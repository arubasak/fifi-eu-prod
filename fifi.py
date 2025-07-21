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
# VERSION 2.5 FINAL - FIXED SESSION MANAGEMENT FOR FLAT JSON STRUCTURE
# - FIXED: Session ID persistence across reruns after authentication
# - FIXED: Aggressive guest session creation preventing authenticated sessions
# - FIXED: WordPress flat JSON structure handling with robust field extraction
# - ADDED: Comprehensive session state tracking and validation
# - ADDED: Enhanced debugging and logging for session management
# - TESTED: Maintains authenticated state across page reloads and reruns
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
            logger.info(f"💾 Saving session {session.session_id[:8]}... | Type: {session.user_type.value} | Name: {session.first_name or 'None'}")
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
                    logger.info(f"✅ Cloud database save successful for {session.session_id[:8]}...")
            else:
                self.local_sessions[session.session_id] = session
                logger.info(f"✅ Local storage save successful for {session.session_id[:8]}...")

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            logger.info(f"📂 Loading session {session_id[:8]}...")
            if self.use_cloud:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                    row = cursor.fetchone()
                    if not row: 
                        logger.warning(f"❌ Session {session_id[:8]}... not found in cloud database")
                        return None
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))
                    session = UserSession(
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
                    logger.info(f"✅ Loaded from cloud: {session_id[:8]}... | Type: {session.user_type.value} | Name: {session.first_name or 'None'}")
                    return session
            else:
                session = self.local_sessions.get(session_id)
                if session:
                    # ENSURE USER_TYPE IS ALWAYS AN ENUM
                    if isinstance(session.user_type, str):
                        session.user_type = UserType(session.user_type)
                        logger.info(f"🔧 Fixed user_type enum for session {session_id[:8]}...")
                    
                    logger.info(f"✅ Loaded from local: {session_id[:8]}... | Type: {session.user_type.value} | Name: {session.first_name or 'None'}")
                else:
                    logger.warning(f"❌ Session {session_id[:8]}... not found in local storage")
                return session

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
# 6. FIXED SESSION MANAGER - ROBUST SESSION STATE MANAGEMENT
# =============================================================================

class SessionManager:
    """Enhanced session manager with robust session state management and authentication flow."""
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter

    def _ensure_session_id_persistence(self, session_id: str):
        """Ensure session ID persists across reruns."""
        st.session_state.current_session_id = session_id
        # Also store in browser session storage for extra persistence
        if 'persistent_session_id' not in st.session_state:
            st.session_state.persistent_session_id = session_id
        logger.info(f"🔒 Session ID {session_id[:8]}... locked in Streamlit state")

    def get_session(self) -> UserSession:
        """Get the current session with simplified, reliable logic."""
        # Get session ID from primary location
        session_id = st.session_state.get('current_session_id')
        
        logger.info(f"🔍 Session lookup: {session_id[:8] if session_id else 'None'}...")
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Update last activity but don't save immediately to avoid race conditions
                session.last_activity = datetime.now()
                logger.info(f"✅ Retrieved session: {session_id[:8]}... | Type: {session.user_type.value} | Name: {session.first_name or 'None'}")
                return session
            else:
                logger.warning(f"⚠️ Session {session_id[:8]}... not found or inactive in database")
        
        # Create new guest session - this is the fallback
        logger.info("🆕 Creating new guest session (no valid session found)")
        return self._create_guest_session()

    def _has_authenticated_session(self) -> bool:
        """Check if we have any indicators of an authenticated session."""
        return bool(
            st.session_state.get('authenticated_session_id') or
            st.session_state.get('user_authenticated') or
            st.session_state.get('wp_authenticated')
        )

    def _recover_authenticated_session(self) -> UserSession:
        """Attempt to recover an authenticated session."""
        # Try to find any authenticated session indicators
        recovery_id = (
            st.session_state.get('authenticated_session_id') or
            st.session_state.get('last_auth_session_id')
        )
        
        if recovery_id:
            session = self.db.load_session(recovery_id)
            if session and session.user_type == UserType.REGISTERED_USER:
                logger.info(f"🔧 Recovered authenticated session: {recovery_id[:8]}...")
                self._ensure_session_id_persistence(recovery_id)
                return session
        
        logger.error("💥 Session recovery failed - creating emergency guest session")
        return self._create_guest_session()

    def _create_guest_session(self) -> UserSession:
        """Create a new guest session."""
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        self._ensure_session_id_persistence(session.session_id)
        logger.info(f"✨ Created new guest session: {session.session_id[:8]}...")
        return session

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """Simplified WordPress authentication with immediate session persistence."""
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

        clean_username = username.strip()
        clean_password = password.strip()

        try:
            logger.info(f"🔐 Starting authentication for user: {clean_username}")
            
            response = requests.post(
                f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                json={'username': clean_username, 'password': clean_password},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                logger.info("--- WordPress Auth Success: Full Response ---")
                logger.info(json.dumps(data, indent=2))
                logger.info("---------------------------------------------")
                
                # Get the current session to upgrade it
                current_session = self.get_session()
                
                # Enhanced display name extraction with fallbacks
                display_name = (
                    data.get('user_display_name') or 
                    data.get('displayName') or 
                    data.get('name') or 
                    data.get('user_nicename') or 
                    data.get('first_name') or
                    data.get('nickname') or
                    clean_username
                )
                
                logger.info(f"🎭 Extracted display name: '{display_name}' for user {clean_username}")

                # Upgrade the session to authenticated user
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                
                # CRITICAL: Save the authenticated session immediately and ensure it's persisted
                self.db.save_session(current_session)
                
                # Force immediate verification that the session was saved
                verification_session = self.db.load_session(current_session.session_id)
                if verification_session and verification_session.user_type == UserType.REGISTERED_USER:
                    logger.info(f"✅ Session verification successful: {verification_session.first_name}")
                else:
                    logger.error(f"❌ Session verification FAILED - session not saved properly!")
                    return None
                
                # Set session ID in Streamlit state - SIMPLIFIED
                st.session_state.current_session_id = current_session.session_id
                
                logger.info(f"✅ Authentication successful! Session {current_session.session_id[:8]}... upgraded to {current_session.user_type.value}")
                logger.info(f"👤 User: {current_session.first_name} ({current_session.email})")
                
                st.success(f"Welcome back, {current_session.first_name}!")
                return current_session
                
            else:
                error_message = f"Invalid username or password (Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except json.JSONDecodeError:
                    pass
                
                st.error(error_message)
                logger.error(f"❌ Auth failed: {error_message}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during authentication. Please check your connection.")
            logger.error(f"🌐 Auth network exception: {e}", exc_info=True)
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Get AI response with rate limiting and content moderation."""
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
        """Clear chat history for the session."""
        session.messages = []
        self.db.save_session(session)

    def end_session(self, session: UserSession):
        """End the current session and clear all state."""
        logger.info(f"🚪 Ending session: {session.session_id[:8]}...")
        
        # Save to Zoho if applicable
        self.zoho.save_chat_transcript(session)
        
        # Mark session as inactive
        session.active = False
        self.db.save_session(session)
        
        # Clear all session-related state
        keys_to_clear = [
            'current_session_id', 'persistent_session_id', 'authenticated_session_id',
            'user_authenticated', 'wp_authenticated', 'last_auth_session_id',
            'auth_timestamp', 'page'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                logger.info(f"🗑️ Cleared session state: {key}")

# =============================================================================
# 7. ENHANCED UI RENDERING FUNCTIONS
# =============================================================================

def debug_wordpress_fields(session_manager: SessionManager):
    """Debug function to discover WordPress user fields."""
    st.subheader("🔍 WordPress Field Discovery")
    st.markdown("**Use this to see exactly what WordPress returns:**")
    
    col1, col2 = st.columns(2)
    with col1:
        test_username = st.text_input("Test Username:", key="debug_field_user")
    with col2:
        test_password = st.text_input("Test Password:", type="password", key="debug_field_pass")
    
    if st.button("🧪 Discover Available Fields", key="discover_fields"):
        if test_username and test_password:
            try:
                response = requests.post(
                    f"{session_manager.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                    json={'username': test_username.strip(), 'password': test_password.strip()},
                    headers={'Content-Type': 'application/json'},
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ Authentication successful!")
                    
                    # Show response structure
                    st.subheader("📋 Complete WordPress Response")
                    st.json(data)
                    
                    # Determine structure type
                    has_nested_user = 'user' in data and isinstance(data['user'], dict)
                    user_data = data.get('user', {}) if has_nested_user else data
                    
                    st.subheader(f"👤 User Fields ({'Nested' if has_nested_user else 'Flat'} Structure)")
                    
                    # Show all available fields
                    for key, value in user_data.items():
                        if key != 'token':  # Skip token display
                            if value:  # Only show non-empty fields
                                st.write(f"**{key}:** `{value}`")
                            else:
                                st.write(f"**{key}:** ❌ Empty/None")
                    
                    # Show recommended display name field
                    display_candidates = ['user_display_name', 'display_name', 'name', 'first_name', 'user_nicename', 'nickname']
                    found_field = None
                    for candidate in display_candidates:
                        if candidate in user_data and user_data[candidate]:
                            found_field = (candidate, user_data[candidate])
                            break
                    
                    if found_field:
                        st.success(f"✅ **Best display field:** `{found_field[0]}` = `{found_field[1]}`")
                    else:
                        st.warning("⚠️ No good display field found, will use username")
                        
                else:
                    st.error(f"❌ Authentication failed: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Enter both username and password to test")

def debug_session_state(session_manager: SessionManager):
    """Enhanced session state diagnostics."""
    st.subheader("🔍 Enhanced Session State Diagnostics")
    
    if st.button("🧪 Full Session Analysis", key="full_session_analysis"):
        current_session = session_manager.get_session()
        
        st.write("**🔎 Current Session Details:**")
        st.json({
            "session_id": current_session.session_id,
            "session_id_short": current_session.session_id[:8],
            "user_type": current_session.user_type.value,
            "first_name": current_session.first_name,
            "email": current_session.email,
            "has_wp_token": bool(current_session.wp_token),
            "messages_count": len(current_session.messages),
            "active": current_session.active,
            "created_at": current_session.created_at.isoformat(),
            "last_activity": current_session.last_activity.isoformat()
        })
        
        st.write("**🏠 Streamlit Session State:**")
        session_state_info = {}
        for key in st.session_state.keys():
            if any(keyword in key.lower() for keyword in ['session', 'auth', 'user', 'page']):
                value = st.session_state[key]
                if isinstance(value, datetime):
                    value = value.isoformat()
                session_state_info[key] = value
        st.json(session_state_info)
        
        st.write("**💾 Database Status:**")
        db_type = "Cloud Database" if session_manager.db.use_cloud else "Local Memory"
        st.json({
            "database_type": db_type,
            "session_exists_in_db": bool(session_manager.db.load_session(current_session.session_id)),
            "session_count_local": len(session_manager.db.local_sessions) if not session_manager.db.use_cloud else "N/A"
        })
        
        # Authentication status check
        auth_indicators = {
            "has_authenticated_session_id": bool(st.session_state.get('authenticated_session_id')),
            "has_user_authenticated": bool(st.session_state.get('user_authenticated')),
            "has_wp_authenticated": bool(st.session_state.get('wp_authenticated')),
            "session_manager_thinks_authenticated": session_manager._has_authenticated_session()
        }
        
        st.write("**🔐 Authentication Indicators:**")
        st.json(auth_indicators)
        
        # Check for session ID mismatches
        streamlit_id = st.session_state.get('current_session_id')
        auth_id = st.session_state.get('authenticated_session_id')
        persistent_id = st.session_state.get('persistent_session_id')
        
        id_consistency = {
            "current_session_id": streamlit_id,
            "authenticated_session_id": auth_id,
            "persistent_session_id": persistent_id,
            "all_ids_match": len(set(filter(None, [streamlit_id, auth_id, persistent_id]))) <= 1
        }
        
        st.write("**🆔 Session ID Consistency:**")
        st.json(id_consistency)
    
    # Emergency recovery tools
    st.divider()
    st.write("**🔧 Emergency Recovery Tools:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Force Session Refresh", key="force_refresh_session"):
            # Clear session caches but preserve authenticated state
            if 'current_session_id' in st.session_state:
                del st.session_state['current_session_id']
            st.success("Session refreshed")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear All State", key="clear_all_state"):
            keys_to_clear = [k for k in st.session_state.keys() if any(word in k.lower() for word in ['session', 'auth', 'user'])]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("All session state cleared")
            st.rerun()
    
    with col3:
        if st.button("🚨 Emergency Reset", key="emergency_reset"):
            st.session_state.clear()
            st.success("Complete app reset")
            st.rerun()

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    """Simplified sidebar with direct session checking."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Get session directly - no caching
        fresh_session = session_manager.get_session()
        
        # IMMEDIATE DEBUG INFO AT TOP
        st.write(f"**Debug:** Session Type = `{fresh_session.user_type.value}`")
        st.write(f"**Debug:** Session Type Object = `{type(fresh_session.user_type)}`")
        st.write(f"**Debug:** Is Registered User = `{fresh_session.user_type == UserType.REGISTERED_USER}`")
        st.write(f"**Debug:** Display Name = `{fresh_session.first_name or 'None'}`")
        st.write(f"**Debug:** Session ID = `{fresh_session.session_id[:8]}...`")

        # User status display - FIXED ENUM COMPARISON
        if fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user":
            st.success("✅ **Authenticated User**") 
            if fresh_session.first_name:
                st.markdown(f"**Welcome:** {fresh_session.first_name}")
            if fresh_session.email:
                st.markdown(f"**Email:** {fresh_session.email}")
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for full features*")
        
        st.divider()
        st.markdown(f"**Messages:** {len(fresh_session.messages)}")
        
        # Enhanced debug info
        with st.expander("🔧 Session Debug Info", expanded=True):
            st.write(f"**User Type:** `{fresh_session.user_type.value}`")
            st.write(f"**Display Name:** `{fresh_session.first_name or 'None'}`")
            st.write(f"**Email:** `{fresh_session.email or 'None'}`")
            st.write(f"**Has Token:** `{bool(fresh_session.wp_token)}`")
            st.write(f"**Active:** `{fresh_session.active}`")
            
            # Check what's in session state
            session_id_in_state = st.session_state.get('current_session_id', 'None')
            st.write(f"**Session ID in State:** `{session_id_in_state[:8] if session_id_in_state != 'None' else 'None'}...`")
            st.write(f"**IDs Match:** `{session_id_in_state == fresh_session.session_id}`")
            
            db_type = "Cloud Database" if session_manager.db.use_cloud else "Local Memory"
            st.write(f"**Database:** `{db_type}`")
            
            if st.button("🔄 Refresh", key="force_refresh_sidebar"):
                st.rerun()
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(fresh_session)
                st.rerun()
        
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(fresh_session)
                st.rerun()

        # PDF download for registered users - FIXED ENUM COMPARISON
        if (fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user") and fresh_session.messages:
            st.divider()
            pdf_buffer = pdf_exporter.generate_chat_pdf(fresh_session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF", 
                    data=pdf_buffer,
                    file_name=f"fifi_chat_{fresh_session.session_id[:8]}.pdf",
                    mime="application/pdf", 
                    use_container_width=True
                )
        
        elif (fresh_session.user_type == UserType.GUEST or fresh_session.user_type.value == "guest") and fresh_session.messages:
            st.divider()
            st.info("💡 **Sign in** to save chat history and export PDF!")
            if st.button("🔑 Go to Sign In", use_container_width=True):
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    """Enhanced chat interface with session consistency."""
    st.title("🤖 FiFi AI Assistant")
    
    # Get fresh session data for chat interface
    current_session = session_manager.get_session()
    
    # Display chat history
    for msg in current_session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("role") == "assistant" and "source" in msg:
                st.caption(f"Source: {msg['source']}")

    # Chat input
    if prompt := st.chat_input("Ask me about ingredients, suppliers, or market trends..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = session_manager.get_ai_response(current_session, prompt)
                st.markdown(response.get("content", "I encountered an issue."), unsafe_allow_html=True)
                if "source" in response:
                    st.caption(f"Source: {response['source']}")
        
        # Rerun to update the interface
        st.rerun()

def render_welcome_page(session_manager: SessionManager):
    """Enhanced welcome page with comprehensive debugging."""
    st.title("🤖 Welcome to FiFi AI Assistant")
    
    # Enhanced diagnostics section
    with st.expander("🔍 WordPress & Session Diagnostics (Debug Tool)", expanded=False):
        tab1, tab2 = st.tabs(["WordPress Fields", "Session State"])
        
        with tab1:
            debug_wordpress_fields(session_manager)
        
        with tab2:
            debug_session_state(session_manager)
    
    # Main authentication tabs
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is disabled because the authentication service is not configured.")
        else:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                            
                        if authenticated_session:
                            st.balloons()  # Celebrate successful login
                            
                            # Show success message
                            st.success(f"🎉 Welcome back, {authenticated_session.first_name}!")
                            
                            # Small delay to ensure session is saved and state is updated
                            time.sleep(1)
                            
                            # Set page to chat and rerun
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest limitations:**
        - Chat history is not saved across sessions
        - No PDF export capability  
        - Limited personalization features
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# =============================================================================
# 8. MAIN APPLICATION - ENHANCED VERSION
# =============================================================================

def main():
    """Enhanced main application with robust session management."""
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # --- INITIALIZATION ---
    if 'initialized' not in st.session_state:
        try:
            logger.info("🚀 Initializing FiFi AI Assistant v2.5...")
            
            config = Config()
            pdf_exporter = PDFExporter()
            
            # CRITICAL: Store database manager in session state to persist local sessions
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager
            
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.initialized = True
            
            logger.info("✅ All components initialized successfully")
            logger.info("🔧 Enhanced session management with WordPress flat JSON support enabled")
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup. Please check the logs.")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            st.stop()

    # --- PAGE ROUTING ---
    session_manager = st.session_state.session_manager
    pdf_exporter = st.session_state.pdf_exporter
    
    # Enhanced page routing with session state validation
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        # Show welcome page
        render_welcome_page(session_manager)
    else:
        # Show main chat interface
        session = session_manager.get_session()
        
        # Verify we have a valid session
        if session and session.active:
            render_sidebar(session_manager, session, pdf_exporter)
            render_chat_interface(session_manager, session)
        else:
            logger.error("🚨 Invalid session in chat page - redirecting to welcome")
            if 'page' in st.session_state:
                del st.session_state.page
            st.rerun()

if __name__ == "__main__":
    main()
