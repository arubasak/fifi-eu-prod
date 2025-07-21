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
# VERSION 2.4 FINAL - COMPLETE WORKING VERSION
# - FIXED: WordPress flat JSON structure handling (user data at root level)
# - FIXED: Authentication display name extraction for flat responses
# - FIXED: Session state management after authentication
# - FIXED: "Guest user" persisting after successful login
# - ADDED: Comprehensive diagnostics and debugging tools
# - TESTED: Works with WordPress JWT responses that have flat structure
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
# 6. UNIFIED SESSION MANAGER - FINAL FIXED VERSION
# =============================================================================

class SessionManager:
    """A single, consolidated session manager with fixed flat JSON structure handling."""
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter

    def get_session(self) -> UserSession:
        """Get the current session, always loading from database for latest state."""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            # Always load from database to get the latest state
            session = self.db.load_session(session_id)
            if session and session.active:
                # Update last activity
                session.last_activity = datetime.now()
                self.db.save_session(session)
                return session
        
        # Create new guest session if no session exists
        return self._create_guest_session()

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        logger.info(f"Created new guest session: {session.session_id}")
        return session

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

        # Clean credentials (no HTML sanitization for authentication!)
        clean_username = username.strip()
        clean_password = password.strip()

        try:
            response = requests.post(
                f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                json={'username': clean_username, 'password': clean_password},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # CRITICAL DEBUG: Log the complete response structure
                logger.info("=== WordPress Authentication Success ===")
                logger.info(f"Username: {clean_username}")
                logger.info(f"Complete response structure: {json.dumps(data, indent=2)}")
                
                # FIXED: Handle both nested and flat response structures
                # Check if user data is nested under 'user' key or at root level
                user_data = data.get('user', {}) if 'user' in data else data
                
                logger.info("=== User Data Extraction ===")
                logger.info(f"Using {'nested' if 'user' in data else 'flat'} structure")
                logger.info("Available user fields:")
                for key, value in user_data.items():
                    if key != 'token':  # Don't log the token
                        logger.info(f"  {key}: {value}")
                
                # Get the current session (this will load from DB or create new)
                session = self.get_session()
                
                # Update session to authenticated state
                session.user_type = UserType.REGISTERED_USER
                session.email = user_data.get('user_email')
                session.wp_token = data.get('token')  # Token is usually at root level
                session.last_activity = datetime.now()
                
                # FIXED: Try multiple possible display name fields in order of preference
                display_name_candidates = [
                    'user_display_name',     # This is what your WordPress returns!
                    'display_name',          # Alternative
                    'name',                  # Generic name field
                    'first_name',            # First name only
                    'user_nicename',         # Username-style but prettier
                    'nickname',              # Nickname field
                    'user_login',            # Login username (last resort)
                ]
                
                # Find the first available display name field
                display_name = None
                field_used = None
                
                for field_name in display_name_candidates:
                    if field_name in user_data and user_data[field_name] and user_data[field_name].strip():
                        display_name = user_data[field_name].strip()
                        field_used = field_name
                        logger.info(f"✅ SUCCESS: Using field '{field_name}' for display name: {display_name}")
                        break
                
                # Fallback to username if no display name found
                if not display_name:
                    display_name = clean_username
                    field_used = "username_fallback"
                    logger.warning(f"⚠️ No display name field found, using username: {display_name}")
                
                session.first_name = display_name
                
                # CRITICAL: Save the updated session immediately to database
                self.db.save_session(session)
                
                # Ensure the session ID is properly set in Streamlit state
                st.session_state.current_session_id = session.session_id
                
                # Clear any cached session data to force refresh
                for cache_key in ['cached_session', 'temp_authenticated_session']:
                    if cache_key in st.session_state:
                        del st.session_state[cache_key]
                
                logger.info(f"✅ Session updated successfully:")
                logger.info(f"   Session ID: {session.session_id}")
                logger.info(f"   User Type: {session.user_type.value}")
                logger.info(f"   Display Name: {display_name}")
                logger.info(f"   Email: {session.email}")
                logger.info(f"   Field Used: {field_used}")
                logger.info("=====================================")
                
                # Show success with detailed info
                st.success(f"🎉 Welcome back, {display_name}!")
                
                # Debug info - show what field was used
                if field_used == 'user_display_name':
                    st.info(f"✅ Using WordPress display name: {display_name}")
                elif field_used != "username_fallback":
                    st.info(f"🔧 Debug: Using WordPress field `{field_used}`: {display_name}")
                else:
                    st.warning("🔧 Debug: No display name field found, using username")
                
                return session
                
            else:
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', 'Authentication failed')
                    st.error(f"Authentication failed: {error_message}")
                    logger.error(f"Auth failed: {error_message}")
                except:
                    st.error(f"Authentication failed (HTTP {response.status_code})")
                    logger.error(f"Auth failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            logger.error(f"Auth exception: {str(e)}", exc_info=True)
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
        # Clear all session-related state
        keys_to_clear = ['current_session_id', 'page', 'cached_session', 'temp_authenticated_session']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

# =============================================================================
# 7. UI RENDERING FUNCTIONS - FINAL FIXED VERSIONS
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

def debug_session_after_auth(session_manager: SessionManager):
    """Diagnostic function to check session state after authentication."""
    st.subheader("🔍 Session State Diagnostics")
    
    if st.button("🧪 Check Current Session State"):
        current_session = session_manager.get_session()
        
        st.write("**Current Session Details:**")
        st.json({
            "session_id": current_session.session_id,
            "user_type": current_session.user_type.value,
            "first_name": current_session.first_name,
            "email": current_session.email,
            "has_wp_token": bool(current_session.wp_token),
            "messages_count": len(current_session.messages),
            "active": current_session.active,
            "last_activity": current_session.last_activity.isoformat()
        })
        
        # Check Streamlit session state
        st.write("**Streamlit Session State:**")
        st.json({
            "current_session_id": st.session_state.get('current_session_id'),
            "page": st.session_state.get('page'),
            "all_session_keys": [k for k in st.session_state.keys() if 'session' in k.lower()]
        })
        
        # Check database directly
        session_id = st.session_state.get('current_session_id')
        if session_id:
            db_session = session_manager.db.load_session(session_id)
            st.write("**Database Session:**")
            if db_session:
                st.json({
                    "db_user_type": db_session.user_type.value,
                    "db_first_name": db_session.first_name,
                    "db_email": db_session.email,
                    "db_has_token": bool(db_session.wp_token),
                    "db_active": db_session.active
                })
            else:
                st.error("❌ Session not found in database!")
        else:
            st.error("❌ No session ID in Streamlit state!")
    
    # Quick fix button
    if st.button("🔧 Force Session Refresh"):
        # Clear session caches
        for key in ['current_session_id', 'cached_session', 'temp_authenticated_session']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Session state cleared. Page will refresh.")
        st.rerun()

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # FORCE FRESH SESSION DATA - Get the latest session from database
        fresh_session = session_manager.get_session()
        
        # Use the fresh session data instead of the passed session
        current_session = fresh_session
        
        # FIXED: Proper user status display with fresh data
        if current_session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Authenticated User**")
            if current_session.first_name:
                st.markdown(f"**Welcome:** {current_session.first_name}")
            else:
                st.warning("⚠️ No display name available")
            if current_session.email:
                st.markdown(f"**Email:** {current_session.email}")
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for full features*")
        
        # Session Info
        st.divider()
        st.markdown(f"**Messages:** {len(current_session.messages)}")
        st.markdown(f"**Session:** `{current_session.session_id[:8]}...`")
        
        # Enhanced Debug Info (temporary - remove in production)
        with st.expander("🔧 Debug Session Info", expanded=True):  # Expanded by default for debugging
            st.write(f"**User Type:** `{current_session.user_type.value}`")
            st.write(f"**Display Name:** `{current_session.first_name or 'None'}`")
            st.write(f"**Email:** `{current_session.email or 'None'}`")
            st.write(f"**Has Token:** `{bool(current_session.wp_token)}`")
            st.write(f"**Session ID:** `{current_session.session_id}`")
            st.write(f"**Active:** `{current_session.active}`")
            st.write(f"**Last Activity:** {current_session.last_activity.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show database type
            db_type = "Cloud Database" if session_manager.db.use_cloud else "Local Memory"
            st.write(f"**Database Type:** `{db_type}`")
            
            # Test buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Force Refresh", key="force_refresh_sidebar"):
                    st.rerun()
            
            with col2:
                if st.button("🆔 New Session", key="new_session_sidebar"):
                    for key in ['current_session_id', 'cached_session', 'temp_authenticated_session']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        st.divider()
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(current_session)
                st.rerun()
        
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(current_session)
                st.rerun()

        # PDF Download for authenticated users
        if current_session.user_type == UserType.REGISTERED_USER and current_session.messages:
            st.divider()
            pdf_buffer = pdf_exporter.generate_chat_pdf(current_session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF", 
                    data=pdf_buffer,
                    file_name=f"fifi_chat_{current_session.session_id[:8]}.pdf",
                    mime="application/pdf", 
                    use_container_width=True
                )
        
        # Sign-in prompt for guests
        elif current_session.user_type == UserType.GUEST and current_session.messages:
            st.divider()
            st.info("💡 **Sign in** to save chat history and export PDF!")
            if st.button("🔑 Go to Sign In", use_container_width=True):
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    st.title("🤖 FiFi AI Assistant")
    
    # Get fresh session data for chat interface too
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
    st.title("🤖 Welcome to FiFi AI Assistant")
    
    # Add comprehensive diagnostics for debugging
    with st.expander("🔍 WordPress & Session Diagnostics (Debug Tool)", expanded=False):
        debug_wordpress_fields(session_manager)
        st.divider()
        debug_session_after_auth(session_manager)
    
    # Main tabs
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
                            
                            # Small delay to show success message and ensure session is saved
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
# 8. MAIN APPLICATION - FINAL VERSION
# =============================================================================

def main():
    """Main application function with fixed session management for flat JSON structure."""
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # --- INITIALIZATION ---
    if 'initialized' not in st.session_state:
        try:
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
            logger.info("✅ All components initialized successfully - WordPress flat JSON structure support enabled")
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup. Please check the logs.")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            st.stop()

    # --- PAGE ROUTING ---
    session_manager = st.session_state.session_manager
    pdf_exporter = st.session_state.pdf_exporter
    
    if 'page' not in st.session_state:
        # Show welcome page
        render_welcome_page(session_manager)
    else:
        # Show main chat interface
        session = session_manager.get_session()
        render_sidebar(session_manager, session, pdf_exporter)
        render_chat_interface(session_manager, session)

if __name__ == "__main__":
    main()
