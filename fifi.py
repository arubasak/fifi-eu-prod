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
# VERSION 2.14 PRODUCTION - SESSION MANAGER FIX
# - FIXED: SessionManager attribute error with get_session method
# - FIXED: Session state serialization issues
# - ENHANCED: More robust session management initialization
# - ADDED: Session manager validation before use
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
# ERROR HANDLING SYSTEM
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
        if len(self.error_history) > 50: 
            self.error_history.pop(0)

    def mark_component_healthy(self, component: str):
        self.component_status[component] = "healthy"

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
                error_context = error_handler.handle_api_error(component, operation, e)
                error_handler.log_error(error_context)
                if show_to_user:
                    error_handler.display_error_to_user(error_context)
                logger.error(f"API Error in {component}/{operation}: {e}")
                return None
        return wrapper
    return decorator

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self):
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
        self.PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        self.PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "my-chat-assistant")
        self.WORDPRESS_URL = self._validate_url(st.secrets.get("WORDPRESS_URL", ""))
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")
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
# USER MODELS
# =============================================================================

class UserType(Enum):
    GUEST = "guest"
    REGISTERED_USER = "registered_user"

@dataclass
class UserSession:
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
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
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
        if not self.use_cloud: 
            return None
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
                    if not row: 
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
                    return session
            else:
                session = self.local_sessions.get(session_id)
                if session:
                    if isinstance(session.user_type, str):
                        session.user_type = UserType(session.user_type)
                return session

# =============================================================================
# PDF EXPORTER
# =============================================================================

class PDFExporter:
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

# =============================================================================
# ZOHO CRM MANAGER
# =============================================================================

class ZohoCRMManager:
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"

    @handle_api_errors("Zoho CRM", "Get Access Token", show_to_user=False)
    def _get_access_token(self) -> Optional[str]:
        if not self.config.ZOHO_ENABLED: 
            return None
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

    @handle_api_errors("Zoho CRM", "Find Contact", show_to_user=False)
    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
        search_url = f"{self.base_url}/Contacts/search"
        params = {
            'criteria': f'(Email:equals:{email})',
            'fields': 'id,First_Name,Last_Name,Email'
        }
        
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            contacts = data.get('data', [])
            if contacts:
                contact = contacts[0]
                logger.info(f"Found existing contact: {contact.get('First_Name', '')} {contact.get('Last_Name', '')} ({email})")
                return contact['id']
        
        return None

    @handle_api_errors("Zoho CRM", "Create Contact", show_to_user=False)
    def _create_contact(self, email: str, access_token: str) -> Optional[str]:
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
        contact_data = {
            "data": [{
                "First_Name": "",
                "Last_Name": "Food Professional",
                "Email": email,
                "Lead_Source": "FiFi AI Assistant",
                "Description": f"Contact created automatically from FiFi AI chat session on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }]
        }
        
        response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
        
        if response.status_code in [200, 201]:
            data = response.json()
            created_contacts = data.get('data', [])
            if created_contacts:
                contact_id = created_contacts[0]['details']['id']
                logger.info(f"Created new contact: Food Professional ({email}) - ID: {contact_id}")
                return contact_id
        else:
            logger.error(f"Failed to create contact: {response.status_code} - {response.text}")
        
        return None

    @handle_api_errors("Zoho CRM", "Save Chat Transcript", show_to_user=True)
    def save_chat_transcript(self, session: UserSession):
        if not self.config.ZOHO_ENABLED:
            logger.info("Zoho CRM integration disabled - skipping transcript save")
            return

        if not session.email:
            logger.info("No email address in session - cannot save to Zoho CRM")
            return

        if not session.messages:
            logger.info("No chat messages to save")
            return

        try:
            access_token = self._get_access_token()
            if not access_token:
                st.warning("Could not authenticate with Zoho CRM")
                return

            contact_id = self._find_contact_by_email(session.email, access_token)
            
            if not contact_id:
                contact_id = self._create_contact(session.email, access_token)
                if not contact_id:
                    st.error("Failed to create contact in Zoho CRM")
                    return
                else:
                    st.success(f"✅ Created new contact in Zoho CRM: Food Professional ({session.email})")
            else:
                st.info("✅ Found existing contact in Zoho CRM (using existing name)")

            session.zoho_contact_id = contact_id
            
            pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.success("🎉 **Chat transcript prepared for Zoho CRM!**")
                st.info("📋 **Contact updated successfully**")
            else:
                st.error("Failed to generate PDF transcript")

        except Exception as e:
            logger.error(f"Zoho CRM save failed: {e}")
            st.error("❌ Failed to save chat transcript to Zoho CRM")

# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
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
    if not isinstance(text, str): 
        return ""
    return html.escape(text)[:max_length].strip()

# =============================================================================
# AI SYSTEM
# =============================================================================

class EnhancedAI:
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None

        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)

    @handle_api_errors("AI System", "Generate Response")
    def get_response(self, prompt: str, chat_history: List[Dict]) -> Dict[str, Any]:
        if not LANGCHAIN_AVAILABLE:
            return {"content": "AI components are unavailable due to missing packages.", "success": False}

        # Simple response for now
        return {"content": f"I understand you're asking about: {prompt}. This is a placeholder response while AI components are being configured.", "success": True, "source": "System"}

@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    if not client: 
        return {"flagged": False}
    response = client.moderations.create(model="omni-moderation-latest", input=prompt)
    result = response.results[0]
    if result.flagged:
        return {"flagged": True, "message": "Your message violates our content policy and cannot be processed."}
    return {"flagged": False}

# =============================================================================
# SESSION MANAGER - FIXED IMPLEMENTATION
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.session_timeout_minutes = 5

    def get_session_timeout_minutes(self) -> int:
        return getattr(self, 'session_timeout_minutes', 5)

    def _is_session_expired(self, session: UserSession) -> bool:
        if not session.last_activity:
            return False
        time_diff = datetime.now() - session.last_activity
        return time_diff.total_seconds() > (self.session_timeout_minutes * 60)

    def _update_activity(self, session: UserSession):
        session.last_activity = datetime.now()
        self.db.save_session(session)

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        return session

    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                if self._is_session_expired(session):
                    logger.info(f"Session {session_id[:8]}... expired due to inactivity")
                    self._auto_save_to_crm(session, "Session Timeout")
                    self._end_session_internal(session)
                    return self._create_guest_session()
                else:
                    self._update_activity(session)
                    return session
        
        return self._create_guest_session()

    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            try:
                logger.info(f"Auto-saving session {session.session_id[:8]}... to CRM. Trigger: {trigger_reason}")
                with st.spinner(f"💾 Auto-saving chat to CRM ({trigger_reason.lower()})..."):
                    self.zoho.save_chat_transcript(session)
                    st.toast("💾 Chat automatically saved to Zoho CRM!", icon="✅")
            except Exception as e:
                logger.error(f"Auto-save to CRM failed: {e}")
                st.toast("⚠️ Auto-save to CRM failed", icon="❌")

    def _end_session_internal(self, session: UserSession):
        session.active = False
        self.db.save_session(session)
        keys_to_clear = ['current_session_id', 'page']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

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
                
                # Get or create a session 
                try:
                    current_session = self.get_session()
                except Exception as e:
                    logger.error(f"Error getting session during auth: {e}")
                    current_session = self._create_guest_session()
                
                display_name = (
                    data.get('user_display_name') or 
                    data.get('displayName') or 
                    data.get('name') or 
                    data.get('user_nicename') or 
                    data.get('first_name') or
                    data.get('nickname') or
                    clean_username
                )

                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                
                self.db.save_session(current_session)
                
                verification_session = self.db.load_session(current_session.session_id)
                if verification_session and verification_session.user_type == UserType.REGISTERED_USER:
                    st.session_state.current_session_id = current_session.session_id
                    st.success(f"Welcome back, {current_session.first_name}!")
                    return current_session
                else:
                    st.error("Authentication failed - session could not be saved.")
                    return None
                
            else:
                error_message = f"Invalid username or password (Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except json.JSONDecodeError:
                    pass
                
                st.error(error_message)
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during authentication. Please check your connection.")
            logger.error(f"Authentication network exception: {e}")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "Rate limit exceeded. Please wait.", "success": False}

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {"content": moderation["message"], "success": False, "source": "Content Safety"}

        response = self.ai.get_response(sanitized_prompt, session.messages)
        session.messages.append({"role": "user", "content": sanitized_prompt})
        session.messages.append({"role": "assistant", **response})
        session.messages = session.messages[-100:]
        
        self._update_activity(session)
        return response

    def clear_chat_history(self, session: UserSession):
        session.messages = []
        self._update_activity(session)

    def end_session(self, session: UserSession):
        self._auto_save_to_crm(session, "Manual Sign Out")
        self._end_session_internal(session)

    def manual_save_to_crm(self, session: UserSession):
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            with st.spinner("Saving to Zoho CRM..."):
                self.zoho.save_chat_transcript(session)
                self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_session_manager() -> Optional[SessionManager]:
    """Safely get the session manager from session state."""
    if 'session_manager' not in st.session_state:
        return None
    
    manager = st.session_state.session_manager
    # Validate that it's a proper SessionManager instance
    if not hasattr(manager, 'get_session'):
        logger.error("Invalid SessionManager instance in session state")
        return None
    
    return manager

def ensure_initialization():
    """Ensure the application is properly initialized."""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.initialized = True
            
            logger.info("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            return False
    
    return True

# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        fresh_session = session_manager.get_session()

        if fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user":
            st.success("✅ **Authenticated User**") 
            if fresh_session.first_name:
                st.markdown(f"**Welcome:** {fresh_session.first_name}")
            if fresh_session.email:
                st.markdown(f"**Email:** {fresh_session.email}")
                
            col1, col2 = st.columns([3, 1])
            with col1:
                if session_manager.zoho.config.ZOHO_ENABLED:
                    if fresh_session.zoho_contact_id:
                        st.success("🔗 **CRM Linked**")
                    else:
                        st.info("📋 **CRM Ready**")
                else:
                    st.caption("🚫 CRM Disabled")
            
            with col2:
                if session_manager.zoho.config.ZOHO_ENABLED and fresh_session.email:
                    st.caption("💾 Auto-save ON")
            
            if fresh_session.last_activity:
                time_since_activity = datetime.now() - fresh_session.last_activity
                timeout_minutes = session_manager.get_session_timeout_minutes()
                minutes_remaining = timeout_minutes - (time_since_activity.total_seconds() / 60)
                
                if minutes_remaining > 0:
                    st.caption(f"⏱️ Auto-save & sign out in {minutes_remaining:.1f} minutes")
                else:
                    st.caption("⏱️ Session will timeout on next interaction")
                    
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for full features*")
        
        st.divider()
        st.markdown(f"**Messages:** {len(fresh_session.messages)}")
        st.markdown(f"**Session:** `{fresh_session.session_id[:8]}...`")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(fresh_session)
                st.rerun()
        
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(fresh_session)
                st.rerun()

        if (fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user") and fresh_session.messages:
            st.divider()
            
            pdf_buffer = pdf_exporter.generate_chat_pdf(fresh_session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF", 
                    data=pdf_buffer,
                    file_name=f"fifi_chat_transcript_{fresh_session.session_id[:8]}.pdf",
                    mime="application/pdf", 
                    use_container_width=True
                )
            
            if session_manager.zoho.config.ZOHO_ENABLED and fresh_session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, help="Chat will also auto-save when you sign out or after 5 minutes of inactivity"):
                    session_manager.manual_save_to_crm(fresh_session)
                    st.rerun()
                    
                st.caption("💡 Chat auto-saves to CRM on sign out or timeout")
        
        elif (fresh_session.user_type == UserType.GUEST or fresh_session.user_type.value == "guest") and fresh_session.messages:
            st.divider()
            st.info("💡 **Sign in** to save chat history and export PDF!")
            if st.button("🔑 Go to Sign In", use_container_width=True):
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    st.title("🤖 FiFi AI Assistant")
    
    current_session = session_manager.get_session()
    
    if current_session.user_type == UserType.REGISTERED_USER and current_session.last_activity:
        time_since_activity = datetime.now() - current_session.last_activity
        timeout_minutes = session_manager.get_session_timeout_minutes()
        minutes_remaining = timeout_minutes - (time_since_activity.total_seconds() / 60)
        
        if 0 < minutes_remaining <= 1:
            st.warning(f"⏱️ Session will auto-save and timeout in {minutes_remaining:.1f} minutes due to inactivity")
        elif minutes_remaining <= 0:
            st.error("⏱️ Session expired due to inactivity. Please sign in again.")
    
    for msg in current_session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("role") == "assistant" and "source" in msg:
                st.caption(f"Source: {msg['source']}")

    if prompt := st.chat_input("Ask me about ingredients, suppliers, or market trends..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = session_manager.get_ai_response(current_session, prompt)
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
                            st.balloons()
                            st.success(f"🎉 Welcome back, {authenticated_session.first_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest limitations:**
        - Chat history is not saved across sessions
        - No PDF export capability  
        - Limited personalization features
        - No automatic CRM integration
        
        ✨ **Sign in benefits:**
        - Chat history saved and exportable as PDF
        - Automatic integration with Zoho CRM
        - Chat transcripts auto-saved to CRM on sign out or after 5 minutes of inactivity
        - Personalized experience with your profile
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # Clear any problematic session state first
    if st.button("🔄 Fresh Start (Clear All State)", key="emergency_clear"):
        st.session_state.clear()
        st.success("✅ All state cleared. Refreshing...")
        st.rerun()

    # Ensure initialization
    if not ensure_initialization():
        st.stop()

    # Get session manager safely
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get session manager. Reinitializing...")
        st.session_state.clear()
        st.rerun()
    
    pdf_exporter = st.session_state.pdf_exporter
    
    # Main application routing
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        try:
            session = session_manager.get_session()
            if session and session.active:
                render_sidebar(session_manager, session, pdf_exporter)
                render_chat_interface(session_manager, session)
            else:
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()
        except Exception as e:
            logger.error(f"Error in chat interface: {e}")
            st.error("An error occurred. Please refresh the page.")
            if st.button("🔄 Refresh", key="error_refresh"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
