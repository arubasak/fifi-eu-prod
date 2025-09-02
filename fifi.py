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
import copy
import sqlite3
import hashlib
import secrets
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
import streamlit.components.v1 as components
from streamlit_javascript import st_javascript

# NEW: Import for simplified browser reload
try:
    from streamlit_js_eval import streamlit_js_eval
    JS_EVAL_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ streamlit_js_eval available for browser reload")
except ImportError:
    JS_EVAL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ streamlit_js_eval not available, using fallback timeout mechanism")

# =============================================================================
# FINAL INTEGRATED FIFI AI - SIMPLIFIED TIMEOUT SYSTEM
# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO, # Keep at INFO unless debugging specific components
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Graceful fallbacks for optional imports
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False
TAVILY_AVAILABLE = False
PINECONE_AVAILABLE = False
SUPABASE_AVAILABLE = False

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
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    pass

try:
    from pinecone import Pinecone
    from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
    PINECONE_AVAILABLE = True
except ImportError:
    pass

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    logger.info("✅ Supabase client initialized for email verification.")
except ImportError:
    logger.warning("Supabase SDK not available. Email verification will be disabled.")

# Competition exclusion list for web searches
DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com", "csmingredients.com", "batafood.com",
    "nccingredients.com", "prinovaglobal.com", "ingrizo.com",
    "solina.com", "opply.com", "brusco.co.uk", "lehmanningredients.co.uk",
    "i-ingredients.com", "fciltd.com", "lupafoods.com", "tradeingredients.com",
    "peterwhiting.co.uk", "globalgrains.co.uk", "tradeindia.com",
    "udaan.com", "ofbusiness.com", "indiamart.com", "symega.com",
    "meviveinternational.com", "amazon.com", "podfoods.co", "gocheetah.com",
    "foodmaven.com", "connect.kehe.com", "knowde.com", "ingredientsonline.com",
    "sourcegoodfood.com"
]

# Utility for safe JSON loading
def safe_json_loads(data: Optional[str], default_value: Any = None) -> Any:
    """Safely loads JSON string, returning default_value on error or None/empty string."""
    if data is None or data == "":
        return default_value
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to decode JSON: {data[:50]}... Error: {e}")
        return default_value

# =============================================================================
# LOADING STATE MANAGEMENT (NEW)
# =============================================================================

def set_loading_state(loading: bool, message: str = ""):
    """Centralized loading state management"""
    st.session_state.is_loading = loading
    if loading:
        st.session_state.loading_message = message
    else:
        st.session_state.loading_message = ""

def is_loading():
    """Check if system is in loading state"""
    return st.session_state.get('is_loading', False)

def show_loading_overlay():
    """Show loading overlay that blocks all interaction"""
    if is_loading():
        loading_message = st.session_state.get('loading_message', 'Loading...')
        
        # Create a full-screen overlay using HTML/CSS
        overlay_html = f"""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.9);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            font-family: Arial, sans-serif;
        ">
            <div style="
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 400px;
            ">
                <div style="
                    width: 40px;
                    height: 40px;
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #ff6b6b;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 1rem;
                "></div>
                <h3 style="color: #333; margin-bottom: 0.5rem;">🤖 FiFi AI Assistant</h3>
                <p style="color: #666; margin: 0;">{loading_message}</p>
            </div>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """
        st.components.v1.html(overlay_html, height=0)
        return True
    return False

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
        self.SUPABASE_URL = st.secrets.get("SUPABASE_URL")
        self.SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")
        self.SUPABASE_ENABLED = all([SUPABASE_AVAILABLE, self.SUPABASE_URL, self.SUPABASE_ANON_KEY])

    def _validate_url(self, url: str) -> str:
        if url and not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format for WORDPRESS_URL: {url}. Disabling feature.")
            return ""
        return url.rstrip('/')

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
                logger.error(f"API Error in {component}/{operation}: {e}", exc_info=True)
                return None
        return wrapper
    return decorator

# =============================================================================
# USER MODELS & DATABASE
# =============================================================================

class UserType(Enum):
    GUEST = "guest"
    EMAIL_VERIFIED_GUEST = "email_verified_guest"
    REGISTERED_USER = "registered_user"

class BanStatus(Enum):
    NONE = "none"
    ONE_HOUR = "1hour"
    TWENTY_FOUR_HOUR = "24hour"
    EVASION_BLOCK = "evasion_block"

@dataclass
class UserSession:
    session_id: str
    user_type: UserType = UserType.GUEST
    email: Optional[str] = None
    full_name: Optional[str] = None
    zoho_contact_id: Optional[str] = None
    active: bool = True
    wp_token: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    # Changed last_activity default to None for timer start logic
    last_activity: Optional[datetime] = None
    timeout_saved_to_crm: bool = False
    
    # Universal Fingerprinting (ALL sessions)
    fingerprint_id: Optional[str] = None
    fingerprint_method: Optional[str] = None
    visitor_type: str = "new_visitor" # Default value
    recognition_response: Optional[str] = None
    
    # Question Tracking (Activity-Based)
    daily_question_count: int = 0
    total_question_count: int = 0
    last_question_time: Optional[datetime] = None
    question_limit_reached: bool = False
    
    # Ban Management
    ban_status: BanStatus = BanStatus.NONE
    ban_start_time: Optional[datetime] = None
    ban_end_time: Optional[datetime] = None
    ban_reason: Optional[str] = None
    
    # Evasion Tracking
    evasion_count: int = 0
    current_penalty_hours: int = 0
    escalation_level: int = 0
    
    # Multi-Email & Device Tracking
    email_addresses_used: List[str] = field(default_factory=list)
    email_switches_count: int = 0
    
    # Browser Privacy Level (from fingerprinting)
    browser_privacy_level: Optional[str] = None
    
    # Registration Tracking
    registration_prompted: bool = False
    registration_link_clicked: bool = False
    
    # NEW: Soft Clear Mechanism - preserves all messages in DB while clearing UI display
    display_message_offset: int = 0

    # NEW: Fields for Re-verification (to address the security concern)
    reverification_pending: bool = False
    pending_user_type: Optional[UserType] = None
    pending_email: Optional[str] = None
    pending_full_name: Optional[str] = None
    pending_zoho_contact_id: Optional[str] = None
    pending_wp_token: Optional[str] = None

    # NEW: Flag to allow guest questions after declining a recognized email
    declined_recognized_email_at: Optional[datetime] = None


class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        # self.lock = threading.Lock() # REMOVED THIS LINE
        self.conn = None
        self._last_health_check = None
        self._health_check_interval = timedelta(minutes=5)
        logger.info("🔄 INITIALIZING DATABASE MANAGER")
        
        # Prioritize SQLite Cloud if configured and available
        if connection_string and SQLITECLOUD_AVAILABLE:
            self.conn, self.db_type = self._try_sqlite_cloud(connection_string)
        
        # Fallback to local SQLite
        if not self.conn:
            self.conn, self.db_type = self._try_local_sqlite()
        
        # Final fallback to in-memory if all else fails
        if not self.conn:
            logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED. FALLING BACK TO NON-PERSISTENT IN-MEMORY STORAGE.")
            self.db_type = "memory"
            self.local_sessions = {}
        
        # Initialize database schema
        if self.conn:
            try:
                self._init_complete_database()
                logger.info("✅ Database initialization completed successfully")
                error_handler.mark_component_healthy("Database")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}", exc_info=True)
                self.conn = None
                self.db_type = "memory" 
                self.local_sessions = {}
        
    def _try_sqlite_cloud(self, cs: str):
        try:
            conn = sqlitecloud.connect(cs)
            conn.execute("SELECT 1").fetchone()
            logger.info("✅ SQLite Cloud connection established!")
            return conn, "cloud"
        except Exception as e:
            logger.error(f"❌ SQLite Cloud connection failed: {e}")
            return None, None

    def _try_local_sqlite(self):
        try:
            conn = sqlite3.connect("fifi_sessions_v2.db", check_same_thread=False)
            conn.execute("SELECT 1").fetchone()
            logger.info("✅ Local SQLite connection established!")
            return conn, "file"
        except Exception as e:
            logger.error(f"❌ Local SQLite connection failed: {e}")
            return None, None

    def _init_complete_database(self):
        """Initialize database schema with all columns upfront"""
        # with self.lock: # REMOVED THIS LINE
        try:
            if hasattr(self.conn, 'row_factory'): 
                self.conn.row_factory = None

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_type TEXT DEFAULT 'guest',
                    email TEXT,
                    full_name TEXT,
                    zoho_contact_id TEXT,
                    created_at TEXT DEFAULT '',
                    last_activity TEXT, -- Changed to allow NULL initially
                    messages TEXT DEFAULT '[]',
                    active INTEGER DEFAULT 1,
                    fingerprint_id TEXT,
                    fingerprint_method TEXT,
                    visitor_type TEXT DEFAULT 'new_visitor',
                    daily_question_count INTEGER DEFAULT 0,
                    total_question_count INTEGER DEFAULT 0,
                    last_question_time TEXT,
                    question_limit_reached INTEGER DEFAULT 0,
                    ban_status TEXT DEFAULT 'none',
                    ban_start_time TEXT,
                    ban_end_time TEXT,
                    ban_reason TEXT,
                    evasion_count INTEGER DEFAULT 0,
                    current_penalty_hours INTEGER DEFAULT 0,
                    escalation_level INTEGER DEFAULT 0,
                    email_addresses_used TEXT DEFAULT '[]',
                    email_switches_count INTEGER DEFAULT 0,
                    browser_privacy_level TEXT,
                    registration_prompted INTEGER DEFAULT 0,
                    registration_link_clicked INTEGER DEFAULT 0,
                    wp_token TEXT,
                    timeout_saved_to_crm INTEGER DEFAULT 0,
                    recognition_response TEXT,
                    display_message_offset INTEGER DEFAULT 0,
                    -- NEW: Re-verification fields
                    reverification_pending INTEGER DEFAULT 0,
                    pending_user_type TEXT,
                    pending_email TEXT,
                    pending_full_name TEXT,
                    pending_zoho_contact_id TEXT,
                    pending_wp_token TEXT,
                    -- NEW: Declined recognized email
                    declined_recognized_email_at TEXT
                    -- REMOVED pending_daily_question_count, pending_total_question_count, pending_last_question_time
                )
            ''')
            
            # Add new columns if they don't exist (for existing databases)
            # IMPORTANT: Re-run the ALTER TABLE statements to reflect removed pending count fields
            # For existing deployments, these might still be in the DB but will be ignored now.
            new_columns = [
                ("display_message_offset", "INTEGER DEFAULT 0"),
                ("reverification_pending", "INTEGER DEFAULT 0"),
                ("pending_user_type", "TEXT"),
                ("pending_email", "TEXT"),
                ("pending_full_name", "TEXT"),
                ("pending_zoho_contact_id", "TEXT"),
                ("pending_wp_token", "TEXT"),
                ("declined_recognized_email_at", "TEXT")
            ]
            for col_name, col_type in new_columns:
                try:
                    self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}")
                    logger.debug(f"✅ Added {col_name} column to existing database")
                except Exception as alter_error:
                    logger.debug(f"ALTER TABLE for {col_name} failed (likely already exists): {alter_error}")
            
            # Create essential indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON sessions(session_id, active)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_id ON sessions(fingerprint_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON sessions(email)")
            
            self.conn.commit()
            logger.info("✅ Database schema ready and indexes created.")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise

    def _check_connection_health(self) -> bool:
        """Check if database connection is healthy"""
        if not self.conn:
            return False
            
        now = datetime.now()
        if (self._last_health_check and 
            now - self._last_health_check < self._health_check_interval):
            return True
            
        try:
            self.conn.execute("SELECT 1").fetchone()
            self._last_health_check = now
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def _ensure_connection_healthy(self, config_instance: Any):
        """Ensure database connection is healthy, reconnect if needed"""
        if not self._check_connection_health():
            logger.warning("Database connection unhealthy, attempting reconnection...")
            old_conn = self.conn
            self.conn = None
        
            if old_conn:
                try:
                    old_conn.close()
                except Exception as e:
                    logger.debug(f"Error closing old DB connection: {e}")
        
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.db_type == "cloud" and SQLITECLOUD_AVAILABLE and hasattr(config_instance, 'SQLITE_CLOUD_CONNECTION'):
                        self.conn, _ = self._try_sqlite_cloud(config_instance.SQLITE_CLOUD_CONNECTION)
                    elif self.db_type == "file":
                        self.conn, _ = self._try_local_sqlite()
                
                    if self.conn:
                        self.conn.execute("SELECT 1").fetchone()
                        logger.info(f"✅ Database reconnection successful on attempt {attempt + 1}")
                        return
                        
                except Exception as e:
                    logger.error(f"Database reconnection attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
        
            logger.error("All database reconnection attempts failed, falling back to in-memory storage")
            self.db_type = "memory"
            if not hasattr(self, 'local_sessions'):
                self.local_sessions = {}

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with SQLite Cloud compatibility and connection health check"""
        logger.debug(f"💾 SAVING SESSION TO DB: {session.session_id[:8]} | user_type={session.user_type.value} | email={session.email} | messages={len(session.messages)} | daily_q={session.daily_question_count} | fp_id={session.fingerprint_id[:8]} | active={session.active}")
        # with self.lock: # REMOVED THIS LINE
        # Check and ensure connection health before any DB operation
        current_config = st.session_state.get('session_manager').config if st.session_state.get('session_manager') else None
        if current_config:
            self._ensure_connection_healthy(current_config)

        if self.db_type == "memory":
            self.local_sessions[session.session_id] = copy.deepcopy(session)
            logger.debug(f"Saved session {session.session_id[:8]} to in-memory.")
            return
        
        try:
            # NEVER set row_factory for save operations
            if hasattr(self.conn, 'row_factory'): 
                self.conn.row_factory = None
            
            # Validate messages before saving
            if not isinstance(session.messages, list):
                logger.warning(f"Invalid messages field for session {session.session_id[:8]}, resetting to empty list")
                session.messages = []
            
            # Ensure JSON serializable data
            try:
                json_messages = json.dumps(session.messages)
                json_emails_used = json.dumps(session.email_addresses_used)
            except (TypeError, ValueError) as e:
                logger.error(f"Session data not JSON serializable for {session.session_id[:8]}: {e}. Resetting data to empty lists.")
                json_messages = "[]"
                json_emails_used = "[]"
                session.messages = []
                session.email_addresses_used = []
            
            # Handle None for last_activity before saving
            last_activity_iso = session.last_activity.isoformat() if session.last_activity else None
            # Handle None for declined_recognized_email_at before saving
            declined_recognized_email_at_iso = session.declined_recognized_email_at.isoformat() if session.declined_recognized_email_at else None


            self.conn.execute(
                '''REPLACE INTO sessions (session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (session.session_id, session.user_type.value, session.email, session.full_name,
                 session.zoho_contact_id, session.created_at.isoformat(),
                 last_activity_iso, json_messages, int(session.active),
                 session.wp_token, int(session.timeout_saved_to_crm), session.fingerprint_id,
                 session.fingerprint_method, session.visitor_type, # This is the corrected line for fingerprint_method
                 session.daily_question_count,
                 session.total_question_count, 
                 session.last_question_time.isoformat() if session.last_question_time else None,
                 int(session.question_limit_reached), session.ban_status.value,
                 session.ban_start_time.isoformat() if session.ban_start_time else None,
                 session.ban_end_time.isoformat() if session.ban_end_time else None,
                 session.ban_reason, session.evasion_count, session.current_penalty_hours,
                 session.escalation_level, json_emails_used,
                 session.email_switches_count, session.browser_privacy_level, int(session.registration_prompted),
                 int(session.registration_link_clicked), session.recognition_response, session.display_message_offset,
                 int(session.reverification_pending), 
                 session.pending_user_type.value if session.pending_user_type else None,
                 session.pending_email, session.pending_full_name,
                 session.pending_zoho_contact_id, session.pending_wp_token,
                 declined_recognized_email_at_iso)) # NEW field
            self.conn.commit()
            
            logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}, active={session.active}, rev_pending={session.reverification_pending}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id[:8]}: {e}", exc_info=True)
            # Try to fallback to in-memory on save failure
            if not hasattr(self, 'local_sessions'):
                self.local_sessions = {}
            self.local_sessions[session.session_id] = copy.deepcopy(session)
            logger.info(f"Fallback: Saved session {session.session_id[:8]} to in-memory storage")
            raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility and connection health check"""
        # with self.lock: # REMOVED THIS LINE
        # Check and ensure connection health before any DB operation
        current_config = st.session_state.get('session_manager').config if st.session_state.get('session_manager') else None
        if current_config:
            self._ensure_connection_healthy(current_config)

        if self.db_type == "memory":
            session = self.local_sessions.get(session_id)
            if session and isinstance(session.user_type, str):
                try:
                    session.user_type = UserType(session.user_type)
                except ValueError:
                    session.user_type = UserType.GUEST
            
            # Ensure backward compatibility for in-memory sessions too
            if session and not hasattr(session, 'display_message_offset'):
                session.display_message_offset = 0
            if session and not hasattr(session, 'reverification_pending'):
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                logger.debug(f"Added missing re-verification fields to in-memory session {session_id[:8]}")
            if session and not hasattr(session, 'declined_recognized_email_at'): # NEW field
                session.declined_recognized_email_at = None
            
            return copy.deepcopy(session)
        
        try:
            # NEVER set row_factory for cloud connections - always use raw tuples
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None
            
            # Update SELECT statement to include new fields
            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
            row = cursor.fetchone()
            
            if not row: 
                logger.debug(f"No active session found for ID {session_id[:8]}.")
                return None
            
            # Handle as tuple (SQLite Cloud returns tuples)
            # Expected 32 columns before adding re-verification fields. Now 38. Now 39.
            expected_min_cols = 39 # Updated expected columns based on previous version without `pending_daily_question_count`
            if len(row) < expected_min_cols: # Must have at least this many columns for full functionality
                logger.error(f"Row has insufficient columns: {len(row)} (expected at least {expected_min_cols}). Data corruption suspected or old schema.")
                # Attempt to load with missing columns, defaulting new ones
                pass 
                
            try:
                # Safely get display_message_offset, defaulting to 0 if column is missing (backward compatibility)
                loaded_display_message_offset = row[31] if len(row) > 31 else 0
                
                # NEW: Safely get re-verification fields, defaulting if columns are missing
                loaded_reverification_pending = bool(row[32]) if len(row) > 32 else False
                loaded_pending_user_type = UserType(row[33]) if len(row) > 33 and row[33] else None
                loaded_pending_email = row[34] if len(row) > 34 else None
                loaded_pending_full_name = row[35] if len(row) > 35 else None
                loaded_pending_zoho_contact_id = row[36] if len(row) > 36 else None
                loaded_pending_wp_token = row[37] if len(row) > 37 else None
                # NEW: Safely get declined_recognized_email_at
                loaded_declined_recognized_email_at = datetime.fromisoformat(row[38]) if len(row) > 38 and row[38] else None


                # Convert last_activity from ISO format string or None
                loaded_last_activity = datetime.fromisoformat(row[6]) if row[6] else None

                user_session = UserSession(
                    session_id=row[0], 
                    user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                    email=row[2], 
                    full_name=row[3],
                    zoho_contact_id=row[4],
                    created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                    last_activity=loaded_last_activity, # Use the safely loaded value
                    messages=safe_json_loads(row[7], default_value=[]),
                    active=bool(row[8]), 
                    wp_token=row[9],
                    timeout_saved_to_crm=bool(row[10]),
                    fingerprint_id=row[11],
                    fingerprint_method=row[12],
                    visitor_type=row[13] or 'new_visitor', # This line loads the *old* value from DB
                    daily_question_count=row[14] or 0,
                    total_question_count=row[15] or 0,
                    last_question_time=datetime.fromisoformat(row[16]) if row[16] else None,
                    question_limit_reached=bool(row[17]),
                    ban_status=BanStatus(row[18]) if row[18] else BanStatus.NONE,
                    ban_start_time=datetime.fromisoformat(row[19]) if row[19] else None,
                    ban_end_time=datetime.fromisoformat(row[20]) if row[20] else None,
                    ban_reason=row[21],
                    evasion_count=row[22] or 0,
                    current_penalty_hours=row[23] or 0,
                    escalation_level=row[24] or 0,
                    email_addresses_used=safe_json_loads(row[25], default_value=[]),
                    email_switches_count=row[26] or 0,
                    browser_privacy_level=row[27],
                    registration_prompted=bool(row[28]),
                    registration_link_clicked=bool(row[29]),
                    recognition_response=row[30],
                    display_message_offset=loaded_display_message_offset, # Use the safely loaded value
                    reverification_pending=loaded_reverification_pending, # NEW
                    pending_user_type=loaded_pending_user_type, # NEW
                    pending_email=loaded_pending_email, # NEW
                    pending_full_name=loaded_pending_full_name, # NEW
                    pending_zoho_contact_id=loaded_pending_zoho_contact_id, # NEW
                    pending_wp_token=loaded_pending_wp_token, # NEW
                    declined_recognized_email_at=loaded_declined_recognized_email_at # NEW
                )
                
                logger.debug(f"Successfully loaded session {session_id[:8]}: user_type={user_session.user_type.value}, messages={len(user_session.messages)}, active={user_session.active}, rev_pending={user_session.reverification_pending}")
                return user_session
                
            except Exception as e:
                logger.error(f"Failed to create UserSession object from row for session {session_id[:8]}: {e}", exc_info=True)
                logger.error(f"Problematic row data (truncated): {str(row)[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load session {session_id[:8]}: {e}", exc_info=True)
            return None

    @handle_api_errors("Database", "Find by Fingerprint")
    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint_id. Includes inactive sessions."""
        logger.debug(f"🔍 SEARCHING FOR FINGERPRINT: {fingerprint_id[:8]}...")
        # with self.lock: # REMOVED THIS LINE
        current_config = st.session_state.get('session_manager').config if st.session_state.get('session_manager') else None
        if current_config:
            self._ensure_connection_healthy(current_config)

        if self.db_type == "memory":
            sessions = [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
            # Ensure backward compatibility for in-memory sessions
            for session in sessions:
                if not hasattr(session, 'display_message_offset'):
                    session.display_message_offset = 0
                if not hasattr(session, 'reverification_pending'):
                    session.reverification_pending = False
                    session.pending_user_type = None
                    session.pending_email = None
                    session.pending_full_name = None
                    session.pending_zoho_contact_id = None
                    session.pending_wp_token = None
                if not hasattr(session, 'declined_recognized_email_at'): # NEW field
                    session.declined_recognized_email_at = None
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (MEMORY): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            return sessions
        
        try:
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None

            # Query ALL sessions with the fingerprint_id, regardless of active status
            # IMPORTANT: This query correctly does NOT include 'AND active = 1'
            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
            sessions = []
            for row in cursor.fetchall():
                expected_min_cols = 39 # Updated expected columns
                if len(row) < expected_min_cols: # Must have at least this many columns for full functionality
                    logger.warning(f"Row has insufficient columns in find_sessions_by_fingerprint: {len(row)} (expected at least {expected_min_cols}). Skipping row.")
                    continue
                try:
                    # Safely get display_message_offset, defaulting to 0 if column is missing (backward compatibility)
                    loaded_display_message_offset = row[31] if len(row) > 31 else 0
                    
                    # NEW: Safely get re-verification fields, defaulting if columns are missing
                    loaded_reverification_pending = bool(row[32]) if len(row) > 32 else False
                    loaded_pending_user_type = UserType(row[33]) if len(row) > 33 and row[33] else None
                    loaded_pending_email = row[34] if len(row) > 34 else None
                    loaded_pending_full_name = row[35] if len(row) > 35 else None
                    loaded_pending_zoho_contact_id = row[36] if len(row) > 36 else None
                    loaded_pending_wp_token = row[37] if len(row) > 37 else None
                    # NEW: Safely get declined_recognized_email_at
                    loaded_declined_recognized_email_at = datetime.fromisoformat(row[38]) if len(row) > 38 and row[38] else None


                    loaded_last_activity = datetime.fromisoformat(row[6]) if row[6] else None

                    s = UserSession(
                        session_id=row[0], 
                        user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                        email=row[2], 
                        full_name=row[3],
                        zoho_contact_id=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                        last_activity=loaded_last_activity, # Use the safely loaded value
                        messages=safe_json_loads(row[7], default_value=[]),
                        active=bool(row[8]), 
                        wp_token=row[9],
                        timeout_saved_to_crm=bool(row[10]),
                        fingerprint_id=row[11],
                        fingerprint_method=row[12],
                        visitor_type=row[13] or 'new_visitor', # This line loads the *old* value from DB
                        daily_question_count=row[14] or 0,
                        total_question_count=row[15] or 0,
                        last_question_time=datetime.fromisoformat(row[16]) if row[16] else None,
                        question_limit_reached=bool(row[17]),
                        ban_status=BanStatus(row[18]) if row[18] else BanStatus.NONE,
                        ban_start_time=datetime.fromisoformat(row[19]) if row[19] else None,
                        ban_end_time=datetime.fromisoformat(row[20]) if row[20] else None,
                        ban_reason=row[21],
                        evasion_count=row[22] or 0,
                        current_penalty_hours=row[23] or 0,
                        escalation_level=row[24] or 0,
                        email_addresses_used=safe_json_loads(row[25], default_value=[]),
                        email_switches_count=row[26] or 0,
                        browser_privacy_level=row[27],
                        registration_prompted=bool(row[28]),
                        registration_link_clicked=bool(row[29]),
                        recognition_response=row[30],
                        display_message_offset=loaded_display_message_offset, # Use the safely loaded value
                        reverification_pending=loaded_reverification_pending, # NEW
                        pending_user_type=loaded_pending_user_type, # NEW
                        pending_email=loaded_pending_email, # NEW
                        pending_full_name=loaded_pending_full_name, # NEW
                        pending_zoho_contact_id=loaded_pending_zoho_contact_id, # NEW
                        pending_wp_token=loaded_pending_wp_token, # NEW
                        declined_recognized_email_at=loaded_declined_recognized_email_at # NEW
                    )
                    sessions.append(s)
                except Exception as e:
                    logger.error(f"Error converting row to UserSession in find_sessions_by_fingerprint: {e}", exc_info=True)
                    continue
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (DB): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            for s in sessions:
                logger.debug(f"  - {s.session_id[:8]}: type={s.user_type.value}, email={s.email}, daily_q={s.daily_question_count}, total_q={s.total_question_count}, last_activity={s.last_activity}, active={s.active}, rev_pending={s.reverification_pending}")
            return sessions
        except Exception as e:
            logger.error(f"Failed to find sessions by fingerprint '{fingerprint_id[:8]}...': {e}", exc_info=True)
            return []

    # =============================================================================
    # FEATURE MANAGERS (nested within DatabaseManager as they access self.db)
    # =============================================================================

    class FingerprintingManager:
        """Manages browser fingerprinting using external HTML component file."""
        
        def __init__(self):
            self.fingerprint_cache = {}
            self.component_attempts = defaultdict(int)

        def render_fingerprint_component(self, session_id: str):
            """Renders fingerprinting component using external fingerprint_component.html file."""
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                html_file_path = os.path.join(current_dir, 'fingerprint_component.html')
                
                logger.debug(f"🔍 Looking for fingerprint component at: {html_file_path}")
                
                if not os.path.exists(html_file_path):
                    logger.error(f"❌ Fingerprint component file NOT FOUND at {html_file_path}")
                    logger.info(f"📁 Current directory contents: {os.listdir(current_dir)}")
                    return self._generate_fallback_fingerprint()
                
                logger.debug(f"✅ Fingerprint component file found, reading content...")
                
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                logger.debug(f"📄 Read {len(html_content)} characters from fingerprint component file")
                
                # Replace session ID placeholder in the HTML
                original_content = html_content
                html_content = html_content.replace('{SESSION_ID}', session_id)
                
                if original_content == html_content:
                    logger.warning(f"⚠️ No {{SESSION_ID}} placeholder found in HTML content!")
                else:
                    logger.debug(f"✅ Replaced {{SESSION_ID}} placeholder with {session_id[:8]}...")
                
                # Render with minimal visibility (height=0 for silent operation)
                logger.debug(f"🔄 Rendering fingerprint component for session {session_id[:8]}...")
                # REMOVE 'key' ARGUMENT HERE
                st.components.v1.html(html_content, height=0, width=0, scrolling=False)
                
                logger.info(f"✅ External fingerprint component rendered successfully for session {session_id[:8]}")
                return None # Always return None since data comes via redirect
                
            except Exception as e:
                logger.error(f"❌ Failed to render external fingerprint component: {e}", exc_info=True)
                return self._generate_fallback_fingerprint()

        def process_fingerprint_data(self, fingerprint_data: Dict[str, Any]) -> Dict[str, Any]:
            """Processes fingerprint data received from the custom component."""
            if not fingerprint_data or fingerprint_data.get('error'):
                logger.warning("Fingerprint component returned error. Using fallback.")
                return self._generate_fallback_fingerprint()
            
            fingerprint_id = fingerprint_data.get('fingerprint_id')
            fingerprint_method = fingerprint_data.get('fingerprint_method')
            privacy_level = fingerprint_data.get('privacy', 'standard')
            
            if not fingerprint_id or not fingerprint_method:
                logger.warning("Invalid fingerprint data received. Using fallback.")
                return self._generate_fallback_fingerprint()
            
            visitor_type = "returning_visitor" if fingerprint_id in self.fingerprint_cache else "new_visitor"
            self.fingerprint_cache[fingerprint_id] = {'last_seen': datetime.now()}
            
            return {
                'fingerprint_id': fingerprint_id,
                'fingerprint_method': fingerprint_method,
                'visitor_type': visitor_type, # <-- This is where process_fingerprint_data sets visitor_type
                'browser_privacy_level': privacy_level,
                'working_methods': fingerprint_data.get('working_methods', [])
            }
        
        def _generate_fallback_fingerprint(self) -> Dict[str, Any]:
            """Generates a unique fallback fingerprint for cases where real fingerprinting fails."""
            fallback_id = f"fallback_{secrets.token_hex(8)}_{int(time.time())}"
            return {
                'fingerprint_id': fallback_id,
                'fingerprint_method': 'fallback',
                'visitor_type': 'new_visitor',
                'browser_privacy_level': 'high_privacy',
                'working_methods': []
            }

    class EmailVerificationManager:
        """Manages email verification process using Supabase Auth OTP."""
        
        def __init__(self, config: Config):
            self.config = config
            self.supabase = None
            if self.config.SUPABASE_ENABLED:
                try:
                    self.supabase = create_client(self.config.SUPABASE_URL, self.config.SUPABASE_ANON_KEY)
                    logger.info("✅ Supabase client initialized for email verification.")
                except Exception as e:
                    logger.error(f"❌ Supabase client initialization failed: {e}. Email verification will be disabled.")
                    self.supabase = None

        @handle_api_errors("Supabase Auth", "Send Verification Code")
        def send_verification_code(self, email: str) -> bool:
            if not self.supabase:
                st.error("Email verification service is not available (Supabase not configured/failed).")
                return False
            
            try:
                response = self.supabase.auth.sign_in_with_otp({
                    'email': email,
                    'options': {
                        'should_create_user': True,
                        'email_redirect_to': None,
                        'data': {
                            'verification_type': 'email_otp'
                        }
                    }
                })
                
                if response is not None:
                    logger.info(f"Email OTP code sent to {email} via Supabase.")
                    return True
                else:
                    logger.error(f"Supabase OTP send failed - unexpected response from sign_in_with_otp.")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to send verification code via Supabase for {email}: {e}")
                st.error(f"Failed to send verification code: {str(e)}")
                return False

        @handle_api_errors("Supabase Auth", "Verify Code")
        def verify_code(self, email: str, code: str) -> bool:
            if not self.supabase:
                st.error("Email verification service is not available.")
                return False
            
            try:
                response = self.supabase.auth.verify_otp({
                    'email': email,
                    'token': code.strip(),
                    'type': 'email'
                })
                
                if response and response.user:
                    logger.info(f"Email verification successful for {email} (Supabase User ID: {response.user.id}).")
                    return True
                else:
                    logger.warning(f"Email verification failed for {email}: Invalid code or no user returned.")
                    st.error("Invalid verification code. Please check the code and try again.")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to verify code via Supabase for {email}: {e}")
                st.error(f"Verification failed: {str(e)}")
                return False

    class QuestionLimitManager:
        """Enhanced question limit manager with tier system and evasion detection."""
        
        def __init__(self):
            # UPDATED: New tier-based limits for registered users
            self.question_limits = {
                UserType.GUEST.value: 4,
                UserType.EMAIL_VERIFIED_GUEST.value: 10,
                UserType.REGISTERED_USER.value: 20  # CHANGED: Reduced from 40 to 20
            }
            self.evasion_penalties = [24, 48, 96, 192, 336]  # Escalating penalties in hours
        
        def detect_guest_email_evasion(self, session: UserSession, db_manager) -> bool:
            """
            NEW: Detects if a guest is switching emails to evade the 4-question limit.
            Returns True if evasion is detected.
            """
            try:
                # If the user just declined a recognized email and is still within guest limits,
                # temporarily bypass evasion check as a grace period.
                if (session.declined_recognized_email_at and 
                    session.daily_question_count < self.question_limits[UserType.GUEST.value]):
                    logger.info(f"Evasion check bypassed for {session.session_id[:8]}: Just declined recognized email and within guest limit.")
                    return False
                
                # Clear the grace period flag once guest limit is hit, so future attempts trigger evasion
                if session.declined_recognized_email_at and \
                   session.daily_question_count >= self.question_limits[UserType.GUEST.value]:
                    logger.info(f"Evasion grace period ended for {session.session_id[:8]}: Guest limit reached.")
                    session.declined_recognized_email_at = None # Clear the flag
                    db_manager.save_session(session) # Persist this flag clearing
                

                # Only check for guest users who have hit their limit
                if (session.user_type.value != UserType.GUEST.value or # Use .value for robustness
                    session.daily_question_count < self.question_limits[UserType.GUEST.value] or
                    not session.fingerprint_id):
                    return False
                
                # Find all sessions with the same fingerprint
                fingerprint_sessions = db_manager.find_sessions_by_fingerprint(session.fingerprint_id)
                
                if len(fingerprint_sessions) <= 1:
                    return False  # No other sessions to compare
                
                # Look for sessions that hit the guest limit in the last 24 hours
                recent_limit_sessions = []
                now = datetime.now()
                
                for fp_session in fingerprint_sessions:
                    if (fp_session.session_id != session.session_id and
                        fp_session.user_type.value == UserType.GUEST.value and # Use .value for robustness
                        fp_session.daily_question_count >= self.question_limits[UserType.GUEST.value] and
                        fp_session.last_question_time):
                        
                        time_since_limit = now - fp_session.last_question_time
                        if time_since_limit >= timedelta(hours=24): # Only check if the limit was hit and 24 hours have NOT passed
                            continue # Skip if the limit already reset
                        
                        recent_limit_sessions.append(fp_session)
                
                if recent_limit_sessions:
                    logger.warning(f"🚨 EVASION DETECTED: Session {session.session_id[:8]} switching emails after hitting guest limit. Found {len(recent_limit_sessions)} recent limited sessions.")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error in evasion detection for session {session.session_id[:8]}: {e}")
                return False
        
        def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
            """Enhanced limit checking with tier system and evasion detection."""
            user_limit = self.question_limits.get(session.user_type.value, 0)
            
            # Check if any bans are active
            if session.ban_status.value != BanStatus.NONE.value:
                if session.ban_end_time and datetime.now() < session.ban_end_time:
                    time_remaining = session.ban_end_time - datetime.now()
                    return {
                        'allowed': False,
                        'reason': 'banned',
                        'ban_type': session.ban_status.value,
                        'time_remaining': time_remaining,
                        'message': self._get_ban_message(session)
                    }
                else:
                    logger.info(f"Ban for session {session.session_id[:8]} expired. Resetting status and counters.")
                    # ENHANCED: Reset counters when ban expires
                    session.ban_status = BanStatus.NONE
                    session.ban_start_time = None
                    session.ban_end_time = None
                    session.ban_reason = None
                    session.question_limit_reached = False
                    # Reset daily questions for fresh start
                    session.daily_question_count = 0
                    session.last_question_time = None
            
            # Reset daily counters if 24 hours passed
            # IMPORTANT: This logic now applies universally and always resets if 24h passed.
            if session.last_question_time:
                time_since_last = datetime.now() - session.last_question_time
                if time_since_last >= timedelta(hours=24):
                    logger.info(f"Daily question count reset for session {session.session_id[:8]} due to 24-hour window expiration.")
                    session.daily_question_count = 0
                    session.question_limit_reached = False
            
            # ENHANCED: Tier-based logic for registered users
            if session.user_type.value == UserType.REGISTERED_USER.value:
                if session.daily_question_count >= user_limit:  # 20 questions
                    self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Registered user daily limit reached")
                    return {
                        'allowed': False,
                        'reason': 'daily_limit',
                        'message': "Daily limit of 20 questions reached. Please retry in 24 hours."
                    }
                # Tier 1 (Questions 1-10) is handled separately now for the ban delay
                # This only checks if they are _about_ to hit tier 1 ban on the *next* question
                elif session.daily_question_count >= 10:  # Tier 2: Questions 11-20
                    # Still allowed, but close to limit
                    remaining = user_limit - session.daily_question_count
                    return {
                        'allowed': True,
                        'tier': 2,
                        'remaining': remaining,
                        'warning': f"Tier 2: {remaining} questions remaining until 24-hour limit."
                    }
                else:  # Tier 1: Questions 1-10
                    remaining = 10 - session.daily_question_count
                    return {
                        'allowed': True,
                        'tier': 1,
                        'remaining': remaining
                    }
            
            # Original logic for other user types
            if session.user_type.value == UserType.GUEST.value:
                if session.daily_question_count >= user_limit:
                    return {
                        'allowed': False,
                        'reason': 'guest_limit',
                        'message': 'Please provide your email address to continue.'
                    }
            
            elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
                if session.daily_question_count >= user_limit: # user_limit is 10 for EMAIL_VERIFIED_GUEST
                    self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Email-verified daily limit reached")
                    return {
                        'allowed': False,
                        'reason': 'daily_limit',
                        'message': self._get_email_verified_limit_message()
                    }
            
            return {'allowed': True}
        
        def record_question(self, session: UserSession):
            """Records question. Tier-specific ban application is now handled elsewhere."""
            session.daily_question_count += 1
            session.total_question_count += 1  # ADDED: Increment total_question_count
            session.last_question_time = datetime.now()
            
            logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}")
        
        def apply_evasion_penalty(self, session: UserSession) -> int:
            """Applies escalating penalty for evasion attempts."""
            session.evasion_count += 1
            session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
            
            penalty_hours = self.evasion_penalties[session.escalation_level - 1]
            session.current_penalty_hours = penalty_hours
            
            self._apply_ban(session, BanStatus.EVASION_BLOCK, f"Email switching evasion attempt #{session.evasion_count}")
            
            logger.warning(f"🚨 EVASION PENALTY: {penalty_hours}h ban applied to {session.session_id[:8]} (Level {session.escalation_level})")
            return penalty_hours
        
        def _apply_ban(self, session: UserSession, ban_type: BanStatus, reason: str):
            """Applies a ban to the session for a specified duration."""
            ban_hours = {
                BanStatus.ONE_HOUR.value: 1,
                BanStatus.TWENTY_FOUR_HOUR.value: 24,
                BanStatus.EVASION_BLOCK.value: session.current_penalty_hours
            }.get(ban_type.value, 24)

            session.ban_status = ban_type
            session.ban_start_time = datetime.now()
            session.ban_end_time = session.ban_start_time + timedelta(hours=ban_hours)
            session.ban_reason = reason
            session.question_limit_reached = True
            
            logger.info(f"Ban applied to session {session.session_id[:8]}: Type={ban_type.value}, Duration={ban_hours}h, Reason='{reason}'.")
        
        def _get_ban_message(self, session: UserSession) -> str:
            """Provides a user-friendly message for current bans."""
            if session.ban_status.value == BanStatus.EVASION_BLOCK.value:
                return "Access restricted due to policy violation. Please try again later."
            elif session.ban_status.value == BanStatus.ONE_HOUR.value:
                return "You've reached the Tier 1 limit (10 questions). Please wait 1 hour before continuing."
            elif session.user_type.value == UserType.REGISTERED_USER.value:
                return "Daily limit reached. Please retry in 24 hours."
            else:
                return self._get_email_verified_limit_message()
        
        def _get_email_verified_limit_message(self) -> str:
            """Specific message for email-verified guests hitting their daily limit."""
            return ("Our system is very busy and is being used by multiple users. For a fair assessment of our FiFi AI assistant and to provide fair usage to everyone, we can allow a total of 10 questions per day (20 messages). To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

# =============================================================================
# PDF EXPORTER & ZOHO CRM MANAGER (MOVED OUT OF DATABASEMANAGER)
# =============================================================================

class PDFExporter:
    """Handles generation of PDF chat transcripts."""
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        # Modify the existing 'Normal' style from the sample stylesheet
        self.styles['Normal'].fontName = 'Helvetica'
        self.styles['Normal'].fontSize = 10
        self.styles['Normal'].leading = 14  # Increased line spacing (1.4 * font size)
        self.styles['Normal'].spaceAfter = 6 # Space after each paragraph
        
        # Header style (can still be added, as it's a new name)
        self.styles.add(ParagraphStyle(
            name='ChatHeader',
            parent=self.styles['Normal'], # Inherit from the modified 'Normal'
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=12
        ))
        
        # User message style with light grey background
        self.styles.add(ParagraphStyle(
            name='UserMessage',
            parent=self.styles['Normal'], # Inherit from the modified 'Normal'
            backColor=lightgrey,
            leftIndent=5,
            rightIndent=5,
            borderPadding=3,
            borderRadius=3,
            spaceBefore=8, # Add space before user message block
            spaceAfter=8
        ))
        
        # Small caption style for sources
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'], # Inherit from the modified 'Normal'
            fontSize=8,
            leading=10,
            textColor=grey,
            spaceBefore=2,
            spaceAfter=2
        ))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        """Generates a PDF of the chat transcript."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        story = []
        story.append(Paragraph("FiFi AI Chat Transcript", self.styles['ChatHeader']))
        story.append(Spacer(1, 12)) # Additional space after header

        for msg in session.messages:
            role = str(msg.get('role', 'unknown')).capitalize()
            # Clean content from HTML/Markdown before putting into PDF Paragraph
            content = str(msg.get('content', ''))
            # Remove any HTML tags that might be in the content
            content = re.sub(r'<[^>]+>', '', content) 
            # Replace Markdown bold/italics with ReportLab's equivalents or just remove
            content = content.replace('**', '<b>').replace('__', '<b>') # Simple bold conversion
            content = content.replace('*', '<i>').replace('_', '<i>') # Simple italic conversion
            content = html.escape(content) # HTML escape what's left

            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
            
            if msg.get("source"):
                story.append(Paragraph(f"<i>Source: {msg['source']}</i>", self.styles['Caption']))
            
            # Add a bit more space between messages if not handled by spaceAfter
            if style.spaceAfter is None: # Only add if the style doesn't already define spaceAfter
                story.append(Spacer(1, 10)) 
                
        doc.build(story)
        buffer.seek(0)
        return buffer

class ZohoCRMManager:
    """Manages integration with Zoho CRM for contact management and chat transcript saving."""
    
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        self._access_token = None
        self._token_expiry = None

    def _get_access_token(self, force_refresh: bool = False, timeout: int = 15) -> Optional[str]:
        """Retrieves or refreshes the Zoho CRM access token."""
        if not self.config.ZOHO_ENABLED:
            logger.debug("Zoho is not enabled in configuration. Skipping token request.")
            return None

        if not force_refresh and self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
            logger.debug("Using cached Zoho access token.")
            return self._access_token
        
        try:
            logger.info(f"Requesting new Zoho access token with {timeout}s timeout...")
            response = requests.post(
                "https://accounts.zoho.com/oauth/v2/token",
                data={
                    'refresh_token': self.config.ZOHO_REFRESH_TOKEN,
                    'client_id': self.config.ZOHO_CLIENT_ID,
                    'client_secret': self.config.ZOHO_CLIENT_SECRET,
                    'grant_type': 'refresh_token'
                },
                timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            
            self._access_token = data.get('access_token')
            self._token_expiry = datetime.now() + timedelta(minutes=50) 
            
            logger.info("Successfully obtained Zoho access token.")
            return self._access_token
            
        except requests.exceptions.Timeout:
            logger.error(f"Zoho token request timed out after {timeout} seconds.")
            return None
        except Exception as e:
            logger.error(f"Failed to get Zoho access token: {e}", exc_info=True)
            return None

    def _find_contact_by_email(self, email: str) -> Optional[str]:
        """Finds a Zoho contact by email."""
        access_token = self._get_access_token()
        if not access_token: 
            return None
        
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, attempting refresh for _find_contact_by_email...")
                access_token = self._get_access_token(force_refresh=True)
                if not access_token: 
                    return None
                headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data']:
                contact_id = data['data'][0]['id']
                logger.info(f"Found existing Zoho contact: {contact_id}")
                return contact_id
                    
        except Exception as e:
            logger.error(f"Error finding Zoho contact by email {email}: {e}", exc_info=True)
                
        return None

    def _create_contact(self, email: str, full_name: Optional[str]) -> Optional[str]:
        """Creates a new Zoho contact."""
        access_token = self._get_access_token()
        if not access_token: 
            return None

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        contact_data = {
            "data": [{
                "Last_Name": full_name or "Food Professional",
                "Email": email,
                "Lead_Source": "FiFi AI Assistant"
            }]
        }
        
        try:
            response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, attempting refresh for _create_contact...")
                access_token = self._get_access_token(force_refresh=True)
                if not access_token: 
                    return None
                headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                contact_id = data['data'][0]['details']['id']
                logger.info(f"Created new Zoho contact: {contact_id}")
                return contact_id
                    
        except Exception as e:
            logger.error(f"Error creating Zoho contact for {email}: {e}", exc_info=True)
                
        return None

    def _upload_attachment(self, contact_id: str, pdf_buffer: io.BytesIO, filename: str) -> bool:
        """Uploads a PDF attachment to a Zoho contact."""
        access_token = self._get_access_token()
        if not access_token: 
            return False

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        upload_url = f"{self.base_url}/Contacts/{contact_id}/Attachments"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                pdf_buffer.seek(0)
                response = requests.post(
                    upload_url, 
                    headers=headers, 
                    files={'file': (filename, pdf_buffer.read(), 'application/pdf')},
                    timeout=60
                )
                
                if response.status_code == 401:
                    logger.warning("Zoho token expired during upload, attempting refresh...")
                    access_token = self._get_access_token(force_refresh=True)
                    if not access_token: 
                        return False
                    headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                    logger.info(f"Successfully uploaded attachment: {filename}")
                    return True
                else:
                    logger.error(f"Zoho attachment upload failed with response: {data}")
                        
            except requests.exceptions.Timeout:
                logger.error(f"Zoho upload timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error uploading Zoho attachment (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                    
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                    
        return False

    def _add_note(self, contact_id: str, note_title: str, note_content: str) -> bool:
        """Adds a note to a Zoho contact."""
        access_token = self._get_access_token()
        if not access_token:
            return False

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        note_data = {
            "data": [{
                "Parent_Id": contact_id,
                "Note_Title": note_title,
                "Note_Content": note_content,
                "se_module": "Contacts"  # ADDED: Explicitly set the parent module
            }]
        }
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=10)
                
                if response.status_code == 401:
                    logger.warning("Zoho token expired during note add, attempting refresh...")
                    access_token = self._get_access_token(force_refresh=True)
                    if not access_token:
                        return False
                    headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                    continue # Retry with new token
                
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                    logger.info(f"Successfully added note to Zoho contact {contact_id}")
                    return True
                else:
                    logger.error("Failed to add note to Zoho contact.")
                    if attempt < max_retries_note - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error("Max retries for note addition reached. Aborting save.")
                        return False
            except Exception as e:
                logger.error(f"ZOHO NOTE ADD FAILED on attempt {attempt + 1} with an exception.")
                logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries for note addition reached. Aborting save.")
                    return False
        
        return False # Should only be reached if all note retries fail.

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """Synchronously saves the chat transcript to Zoho CRM."""
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        
        if (session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] or 
            not session.email or 
            not session.messages or 
            not self.config.ZOHO_ENABLED):
            logger.info(f"ZOHO SAVE SKIPPED: Not eligible. (UserType: {session.user_type.value}, Email: {bool(session.email)}, Messages: {bool(session.messages)}, Zoho Enabled: {self.config.ZOHO_ENABLED})")
            return False
        
        # Step 1: Find or create contact (once)
        contact_id = self._find_contact_by_email(session.email)
        if not contact_id:
            contact_id = self._create_contact(session.email, session.full_name)
        if not contact_id:
            logger.error("Failed to find or create Zoho contact. Cannot proceed with save.")
            return False
        session.zoho_contact_id = contact_id

        # Step 2: Generate PDF and upload attachment (once)
        pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
        upload_success = False
        if pdf_buffer:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
            upload_success = self._upload_attachment(contact_id, pdf_buffer, pdf_filename)
            if not upload_success:
                logger.warning("Failed to upload PDF attachment to Zoho. Continuing with note only.")

        # Step 3: Add note with retry logic
        max_retries_note = 3 if "timeout" in trigger_reason.lower() or "emergency" in trigger_reason.lower() else 1
        for attempt_note in range(max_retries_note):
            try:
                note_title = f"FiFi AI Chat Transcript from {datetime.now().strftime('%Y-%m-%d %H:%M')} ({trigger_reason})"
                note_content = self._generate_note_content(session, upload_success, trigger_reason)
                note_success = self._add_note(contact_id, note_title, note_content)
                
                if note_success:
                    logger.info("=" * 80)
                    logger.info(f"ZOHO SAVE COMPLETED SUCCESSFULLY on note attempt {attempt_note + 1}")
                    logger.info(f"Contact ID: {contact_id}")
                    logger.info("=" * 80)
                    return True
                else:
                    logger.error("Failed to add note to Zoho contact.")
                    if attempt_note < max_retries_note - 1:
                        time.sleep(2 ** attempt_note)
                    else:
                        logger.error("Max retries for note addition reached. Aborting save.")
                        return False
            except Exception as e:
                logger.error(f"ZOHO NOTE ADD FAILED on attempt {attempt_note + 1} with an exception.")
                logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
                if attempt_note < max_retries_note - 1:
                    time.sleep(2 ** attempt_note)
                else:
                    logger.error("Max retries for note addition reached. Aborting save.")
                    return False
        
        return False # Should only be reached if all note retries fail.

    def _generate_note_content(self, session: UserSession, attachment_uploaded: bool, trigger_reason: str) -> str:
        """Generates the text content for the Zoho CRM note."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        note_content = f"**Session Information:**\n"
        note_content += f"- Session ID: {session.session_id}\n"
        note_content += f"- User: {session.full_name or 'Unknown'} ({session.email})\n"
        note_content += f"- User Type: {session.user_type.value}\n"
        note_content += f"- Save Trigger: {trigger_reason}\n"
        note_content += f"- Timestamp: {timestamp}\n"
        note_content += f"- Total Messages: {len(session.messages)}\n"
        note_content += f"- Questions Asked (Today): {session.daily_question_count}\n\n"
        
        if attachment_uploaded:
            note_content += "✅ **PDF transcript has been attached to this contact.**\n\n"
        else:
            note_content += "⚠️ **PDF attachment upload failed. Full transcript below:**\n\n"
        
        note_content += "**Conversation Summary (truncated):**\n"
        
        for i, msg in enumerate(session.messages):
            role = msg.get("role", "Unknown").capitalize()
            content = re.sub(r'<[^>]+>', '', msg.get("content", ""))
            
            max_msg_length = 500
            if len(content) > max_msg_length:
                content = content[:max_msg_length] + "..."
                
            note_content += f"\n{i+1}. **{role}:** {content}\n"
            
            if msg.get("source"):
                note_content += f"   _Source: {msg['source']}_\n"
                
        return note_content

# =============================================================================
# RATE LIMITER & AI SYSTEM
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter to prevent abuse."""
    def __init__(self, max_requests: int = 2, window_seconds: int = 60): # UPDATED: 2 questions per 60 seconds
        self.requests = defaultdict(list)
        self._lock = threading.Lock() # This is a separate RateLimiter specific lock, not the DB one
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, identifier: str) -> Dict[str, Any]:
        """Returns detailed rate limit information including timer."""
        with self._lock:
            now = time.time()
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                logger.debug(f"Rate limit allowed for {identifier[:8]}... ({len(self.requests[identifier])}/{self.max_requests} within {self.window_seconds}s)")
                return {
                    'allowed': True,
                    'current_count': len(self.requests[identifier]),
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds,
                    'time_until_next': 0
                }
            else:
                # Calculate time until next request is allowed
                oldest_request = min(self.requests[identifier])
                time_until_next = max(0, int((oldest_request + self.window_seconds) - now))
                
                logger.warning(f"Rate limit exceeded for {identifier[:8]}... ({len(self.requests[identifier])}/{self.max_requests} within {self.window_seconds}s)")
                return {
                    'allowed': False,
                    'current_count': len(self.requests[identifier]),
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds,
                    'time_until_next': time_until_next
                }

def sanitize_input(text: str, max_length: int = 4000) -> str:
    """Sanitizes user input to prevent XSS and limit length."""
    if not isinstance(text, str): 
        return ""
    return html.escape(text)[:max_length].strip()

# =============================================================================
# PINECONE ASSISTANT TOOL
# =============================================================================

class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE: 
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self.initialize_assistant()

    def initialize_assistant(self):
        try:
            instructions = (
                "You are a document-based AI assistant with STRICT limitations.\n\n"
                "ABSOLUTE RULES - NO EXCEPTIONS:\n"
                "1. You can ONLY answer using information that exists in your uploaded documents\n"
                "2. If you cannot find the answer in your documents, you MUST respond with EXACTLY: 'I don't have specific information about this topic in my knowledge base.'\n"
                "3. NEVER create fake citations, URLs, or source references\n"
                "4. NEVER create fake file paths, image references (.jpg, .png, etc.), or document names\n"
                "5. NEVER use general knowledge or information not in your documents\n"
                "6. NEVER guess or speculate about anything\n"
                "7. NEVER make up website links, file paths, or citations\n"
                "8. If asked about current events, news, recent information, or anything not in your documents, respond with: 'I don't have specific information about this topic in my knowledge base.'\n"
                "9. Only include citations [1], [2], etc. if they come from your actual uploaded documents\n"
                "10. NEVER reference images, files, or documents that were not actually uploaded to your knowledge base\n\n"
                "REMEMBER: It is better to say 'I don't know' than to provide incorrect information, fake sources, or non-existent file references."
            )
            
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                logger.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions=instructions
                )
            else:
                logger.info(f"Connected to assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone Assistant: {e}")
            return None

    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant: 
            return None
        try:
            pinecone_messages = [
                PineconeMessage(
                    role="user" if isinstance(msg, HumanMessage) else "assistant", 
                    content=msg.content
                ) for msg in chat_history
            ]
            
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")
            content = response.message.content
            has_citations = False
            
            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_items = set()
                
                for citation in response.citations:
                    for reference in citation.references:
                        if hasattr(reference, 'file') and reference.file:
                            link_url = None
                            if hasattr(reference.file, 'metadata') and reference.file.metadata:
                                link_url = reference.file.metadata.get('source_url')
                            if not link_url and hasattr(reference.file, 'signed_url') and reference.file.signed_url:
                                link_url = reference.file.signed_url
                            
                            if link_url:
                                if '?' in link_url:
                                    link_url += '&utm_source=fifi-in'
                                else:
                                    link_url += '?utm_source=fifi-in'
                                
                                display_text = link_url
                                if display_text not in seen_items:
                                    link = f"[{len(seen_items) + 1}] [{display_text}]({link_url})"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                            else:
                                display_text = getattr(reference.file, 'name', 'Unknown Source')
                                if display_text not in seen_items:
                                    link = f"[{len(seen_items) + 1}] {display_text}"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                
                if citations_list:
                    content += citations_header + "\n".join(citations_list)
            
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi",
                "has_citations": has_citations,
                "response_length": len(content),
                "used_pinecone": True,
                "used_search": False,
                "has_inline_citations": bool(citations_list) if has_citations else False,
                "safety_override": False
            }
        except Exception as e:
            logger.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        from tavily import TavilyClient
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    def add_utm_to_links(self, content: str) -> str:
        """Finds all Markdown links in a string and appends the UTM parameters."""
        def replacer(match):
            url = match.group(1)
            utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
            if '?' in url:
                new_url = f"{url}&{utm_params}"
            else:
                new_url = f"{url}?{utm_params}"
            return f"({new_url})"
        return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

    def synthesize_search_results(self, results, query: str) -> str:
        """Synthesize search results from direct Tavily SDK."""
    
    logger.info(f"🔍 SYNTHESIS: Processing SDK results type = {type(results)}")
    
    if not isinstance(results, dict):
        logger.warning("🔍 SYNTHESIS: Results not a dictionary")
        return "I couldn't process the search results properly."
    
    # Direct SDK provides a pre-synthesized answer
    answer = results.get('answer')
    if answer and len(answer.strip()) > 20:
        logger.info(f"🔍 SYNTHESIS: Using Tavily's pre-synthesized answer ({len(answer)} chars)")
        return answer  # Return the answer directly, no "Based on my search:" prefix
    
    # Fallback to manual synthesis if no answer provided
    search_results = results.get('results', [])
    logger.info(f"🔍 SYNTHESIS: Fallback synthesis with {len(search_results)} results")
    
    if not search_results:
        return "I couldn't find any relevant information for your query."
    
    # Process individual results
    relevant_info = []
    sources = []
    
    for i, result in enumerate(search_results[:3], 1):
        if isinstance(result, dict):
            title = result.get('title', f'Result {i}')
            content = result.get('content', '')
            url = result.get('url', '')
            
            if content:
                if len(content) > 400:
                    content = content[:400] + "..."
                relevant_info.append(content)
                
                if url and title:
                    sources.append(f"[{title}]({url})")
    
    if not relevant_info:
        return "I found search results but couldn't extract readable content."
    
    # Build synthesized response
    response_parts = []
    if len(relevant_info) == 1:
        response_parts.append(f"Based on my search: {relevant_info[0]}")
    else:
        response_parts.append("Based on my search:")
        for i, info in enumerate(relevant_info, 1):
            response_parts.append(f"\n\n**{i}.** {info}")
    
    # Add sources
    if sources:
        response_parts.append(f"\n\n**Sources:**")
        for i, source in enumerate(sources, 1):
            response_parts.append(f"\n{i}. {source}")
    
    return "".join(response_parts)
        
        # Fallback for unknown formats
        return "I couldn't find any relevant information for your query."

    # NEW: Determine search strategy based on question and Pinecone error type
    def determine_search_strategy(self, question: str, pinecone_error_type: str = None) -> Dict[str, Any]:
        """Determine whether to use domain-restricted or worldwide search."""
        question_lower = question.lower()
        
        # Check for current information indicators in question
        current_info_indicators = [
            "today", "yesterday", "this week", "this month", "this year", "2025", "2024",
            "current", "latest", "recent", "now", "currently", "updated",
            "news", "weather", "stock", "price", "event", "happening"
        ]
        
        has_time_sensitive_keywords = any(indicator in question_lower for indicator in current_info_indicators)
        
        # For major Pinecone errors, use two-tier approach
        if pinecone_error_type in ["rate_limit", "authentication_error", "payment_required", "server_error"]:
            if has_time_sensitive_keywords:
                # Tier 2: Worldwide search with competitor exclusions
                return {
                    "strategy": "worldwide_with_exclusions",
                    "include_domains": None,
                    "exclude_domains": DEFAULT_EXCLUDED_DOMAINS,
                    "reason": f"Time-sensitive question with Pinecone {pinecone_error_type}"
                }
            else:
                # Tier 1: Domain-restricted to 12taste.com
                return {
                    "strategy": "domain_restricted", 
                    "include_domains": ["12taste.com"],
                    "exclude_domains": None,
                    "reason": f"Domain-restricted fallback for Pinecone {pinecone_error_type}"
                }
        else:
            # Normal fallback - worldwide with exclusions
            return {
                "strategy": "worldwide_with_exclusions",
                "include_domains": None, 
                "exclude_domains": DEFAULT_EXCLUDED_DOMAINS,
                "reason": "Standard safety fallback"
            }

    def query(self, message: str, chat_history: List[BaseMessage], pinecone_error_type: str = None) -> Dict[str, Any]:
        try:
            # Determine search strategy
            strategy = self.determine_search_strategy(message, pinecone_error_type)
            
            # Build Tavily query parameters
            search_params = {"query": message}
            
            if strategy["include_domains"]:
                # Domain-restricted search
                domain_query = f"{message} site:{strategy['include_domains'][0]}"
                search_params["query"] = domain_query
                logger.info(f"🔍 Tavily domain-restricted search: {strategy['include_domains'][0]}")
            elif strategy["exclude_domains"]:
                # Worldwide with exclusions
                search_params["query"] = message
                logger.info(f"🌐 Tavily worldwide search excluding {len(strategy['exclude_domains'])} competitor domains")
            
            # Build search parameters for direct SDK
            sdk_params = {
                "query": search_params["query"],
                "max_results": 5,
                "include_answer": True,  # This is the key parameter!
                "search_depth": "basic"
            }

            # Add domain restrictions if using include/exclude strategy
            if strategy.get("include_domains"):
                sdk_params["include_domains"] = strategy["include_domains"]
            elif strategy.get("exclude_domains"):
                sdk_params["exclude_domains"] = strategy["exclude_domains"]

            logger.info(f"🔍 Direct Tavily SDK call with params: {list(sdk_params.keys())}")
            search_results = self.tavily_client.search(**sdk_params)
            synthesized_content = self.synthesize_search_results(search_results, message)
            final_content = self.add_utm_to_links(synthesized_content)
            
            return {
                "content": final_content,
                "success": True,
                "source": f"FiFi Web Search ({strategy['strategy']})",
                "used_pinecone": False,
                "used_search": True,
                "has_citations": True,
                "has_inline_citations": True,
                "safety_override": False,
                "search_strategy": strategy["strategy"],
                "search_reason": strategy["reason"]
            }
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return None    
class EnhancedAI:
    """Enhanced AI system with improved error handling and bidirectional fallback."""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        self.pinecone_tool = None
        self.tavily_agent = None
        
        # Initialize OpenAI client (for both AI responses and content moderation)
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
        
        # Initialize Pinecone tool
        if PINECONE_AVAILABLE and self.config.PINECONE_API_KEY and self.config.PINECONE_ASSISTANT_NAME:
            try:
                self.pinecone_tool = PineconeAssistantTool(
                    self.config.PINECONE_API_KEY, 
                    self.config.PINECONE_ASSISTANT_NAME
                )
                logger.info("✅ Pinecone Assistant initialized successfully")
                error_handler.mark_component_healthy("Pinecone")
            except Exception as e:
                logger.error(f"Pinecone tool initialization failed: {e}")
                self.pinecone_tool = None
                error_handler.log_error(error_handler.handle_api_error("Pinecone", "Initialize", e))
        
        # Initialize Tavily agent
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(self.config.TAVILY_API_KEY)
                logger.info("✅ Tavily Web Search initialized successfully")
                error_handler.mark_component_healthy("Tavily")
            except Exception as e:
                logger.error(f"Tavily agent initialization failed: {e}")
                self.tavily_agent = None
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Initialize", e))

    def _detect_pinecone_error_type(self, error: Exception) -> str:
        """NEW: Enhanced Pinecone error detection with specific HTTP codes."""
        error_str = str(error).lower()
        
        # Check for specific HTTP status codes that indicate different issues
        if any(code in error_str for code in ['401', '403']):
            error_handler.component_status["Pinecone"] = "authentication_error"
            return "authentication_error"
        elif '402' in error_str:
            error_handler.component_status["Pinecone"] = "payment_required"
            return "payment_required"
        elif '429' in error_str:
            error_handler.component_status["Pinecone"] = "rate_limit"
            return "rate_limit"
        elif any(keyword in error_str for keyword in ['500', '503']): 
            error_handler.component_status["Pinecone"] = "server_error"
            return "server_error"
        elif any(keyword in error_str for keyword in ['timeout', 'connection', 'network']): # FIXED: Changed 'code' to 'keyword' to match other lines
            error_handler.component_status["Pinecone"] = "connectivity_error"
            return "connectivity_error"
        else:
            error_handler.component_status["Pinecone"] = "unknown_error"
            return "unknown_error"

    def _should_use_pinecone_first(self) -> bool:
        """Enhanced routing based on component health status."""
        pinecone_status = error_handler.component_status.get("Pinecone", "healthy")
        tavily_status = error_handler.component_status.get("Tavily", "healthy")
    
        # If Pinecone is healthy, use it first
        if pinecone_status in ["healthy"]:
            return True
    
        # If Pinecone has issues (including rate limits) but Tavily is healthy, use Tavily first
        if pinecone_status in ["authentication_error", "payment_required", "server_error", "rate_limit"] and tavily_status == "healthy":
            return False
    
        # Default to Pinecone first for other scenarios
        return True

    def _get_current_pinecone_error_type(self) -> str:
        """Get current Pinecone error type for Tavily strategy determination."""
        return error_handler.component_status.get("Pinecone", "healthy")

    def should_use_web_fallback(self, pinecone_response: Dict[str, Any], original_question: str) -> bool:
        """EXTREMELY aggressive fallback detection to prevent any hallucination."""
        content = pinecone_response.get("content", "").lower()
        content_raw = pinecone_response.get("content", "")
        question_lower = original_question.lower()  # NEW: Check question instead of response
        
        # PRIORITY 1: Always fallback for current/recent information requests IN QUESTION
        current_info_indicators = [
            "today", "yesterday", "this week", "this month", "this year", "2025", "2024",
            "current", "latest", "recent", "now", "currently", "updated",
            "news", "weather", "stock", "price", "event", "happening"
        ]
        # FIXED: Check original question instead of response content
        if any(indicator in question_lower for indicator in current_info_indicators):
            return True
        
        # PRIORITY 2: Explicit "don't know" statements (allow these to pass)
        explicit_unknown = [
            "i don't have specific information", "i don't know", "i'm not sure",
            "i cannot help", "i cannot provide", "cannot find specific information",
            "no specific information", "no information about", "don't have information",
            "not available in my knowledge", "unable to find", "no data available",
            "insufficient information", "outside my knowledge", "cannot answer"
        ]
        if any(keyword in content for keyword in explicit_unknown):
            return False # Don't fallback for explicit "don't know" responses
        
        # PRIORITY 3: Detect fake files/images/paths (CRITICAL SAFETY)
        fake_file_patterns = [
            ".jpg", ".jpeg", ".png", ".html", ".gif", ".doc", ".docx",
            ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".avi", ".mp3",
            "/uploads/", "/files/", "/images/", "/documents/", "/media/",
            "file://", "ftp://", "path:", "directory:", "folder:"
        ]
        
        has_real_citations = pinecone_response.get("has_citations", False)
        if any(pattern in content_raw for pattern in fake_file_patterns):
            if not has_real_citations:
                logger.warning("🚨 SAFETY: Detected fake file references without real citations")
                return True
        
        # PRIORITY 4: Detect potential fake citations (CRITICAL)
        if "[1]" in content_raw or "**Sources:**" in content_raw:
            suspicious_patterns = [
                "http://", ".org", ".net",
                "example.com", "website.com", "source.com", "domain.com"
            ]
            if not has_real_citations and any(pattern in content_raw for pattern in suspicious_patterns):
                logger.warning("🚨 SAFETY: Detected fake citations")
                return True
        
        # PRIORITY 5: NO CITATIONS = MANDATORY FALLBACK (unless very short or explicit "don't know")
        if not has_real_citations:
            if "[1]" not in content_raw and "**Sources:**" not in content_raw:
                if len(content_raw.strip()) > 30:
                    logger.warning("🚨 SAFETY: Long response without citations")
                    return True
        
        # PRIORITY 6: General knowledge indicators (likely hallucination)
        general_knowledge_red_flags = [
            "generally", "typically", "usually", "commonly", "often", "most",
            "according to", "it is known", "studies show", "research indicates",
            "experts say", "based on", "in general", "as a rule"
        ]
        if any(flag in content for flag in general_knowledge_red_flags):
            logger.warning("🚨 SAFETY: Detected general knowledge indicators")
            return True
        
        # PRIORITY 7: Question-answering patterns that suggest general knowledge
        qa_patterns = [
            "the answer is", "this is because", "the reason", "due to the fact",
            "this happens when", "the cause of", "this occurs"
        ]
        if any(pattern in content for pattern in qa_patterns):
            if not pinecone_response.get("has_citations", False):
                logger.warning("🚨 SAFETY: QA patterns without citations")
                return True
        
        # PRIORITY 8: Response length suggests substantial answer without sources
        response_length = pinecone_response.get("response_length", 0)
        if response_length > 100 and not pinecone_response.get("has_citations", False):
            logger.warning("🚨 SAFETY: Long response without sources")
            return True
        
        return False
    
    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced AI response with bidirectional fallback and intelligent routing."""
        
        # Convert chat history to LangChain format
        if chat_history:
            langchain_history = []
            for msg in chat_history[-10:]: # Limit to last 10 messages
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg.get("content", "")))
            langchain_history.append(HumanMessage(content=prompt))
        else:
            langchain_history = [HumanMessage(content=prompt)]
        
        # Intelligent routing based on component health
        use_pinecone_first = self._should_use_pinecone_first()
        current_pinecone_error = self._get_current_pinecone_error_type()
        
        if use_pinecone_first:
            # Try Pinecone first
            if self.pinecone_tool:
                try:
                    logger.info("🔍 Querying Pinecone knowledge base (primary)...")
                    pinecone_response = self.pinecone_tool.query(langchain_history)
                    
                    if pinecone_response and pinecone_response.get("success"):
                        # Check if we should fallback to web search (FIXED: pass original question)
                        should_fallback = self.should_use_web_fallback(pinecone_response, prompt)
                        
                        if not should_fallback:
                            logger.info("✅ Using Pinecone response (passed safety checks)")
                            error_handler.mark_component_healthy("Pinecone")
                            return pinecone_response
                        else:
                            logger.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switching to verified web sources.")
                            
                except Exception as e:
                    error_type = self._detect_pinecone_error_type(e)
                    logger.error(f"Pinecone query failed ({error_type}): {e}")
                    current_pinecone_error = error_type  # Update error type for Tavily strategy
            
            # Fallback to web search with enhanced strategy
            if self.tavily_agent:
                try:
                    logger.info("🌐 Falling back to web search...")
                    # Pass the error type for strategy determination
                    web_response = self.tavily_agent.query(prompt, langchain_history[:-1], current_pinecone_error)
                    
                    if web_response and web_response.get("success"):
                        logger.info(f"✅ Using web search response: {web_response.get('search_strategy', 'unknown strategy')}")
                        error_handler.mark_component_healthy("Tavily")
                        return web_response
                        
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    error_handler.log_error(error_handler.handle_api_error("Tavily", "Query", e))
            
            # Final Pinecone fallback if Tavily failed
            if not self.tavily_agent:
                if self.pinecone_tool:
                    try:
                        logger.info("🔍 Falling back to Pinecone knowledge base (re-attempt after web search failure/absence)...")
                        pinecone_response = self.pinecone_tool.query(langchain_history)
                        
                        if pinecone_response and pinecone_response.get("success"):
                            should_fallback = self.should_use_web_fallback(pinecone_response, prompt)
                            
                            if not should_fallback:
                                logger.info("✅ Using Pinecone response (fallback)")
                                return pinecone_response
                                
                    except Exception as e:
                        error_type = self._detect_pinecone_error_type(e)
                        logger.error(f"Pinecone fallback also failed ({error_type}): {e}")
        else:
            # Try Tavily first due to Pinecone issues
            if self.tavily_agent:
                try:
                    logger.info("🌐 Querying web search (primary due to Pinecone issues)...")
                    # Pass the error type for strategy determination
                    web_response = self.tavily_agent.query(prompt, langchain_history[:-1], current_pinecone_error)
                    
                    if web_response and web_response.get("success"):
                        logger.info(f"✅ Using web search response (primary): {web_response.get('search_strategy', 'unknown strategy')}")
                        error_handler.mark_component_healthy("Tavily")
                        return web_response
                                
                except Exception as e:
                    logger.error(f"Web search failed: {e}")
                    error_handler.log_error(error_handler.handle_api_error("Tavily", "Query", e))
                
            # Fallback to Pinecone (despite issues)
            if self.pinecone_tool:
                try:
                    logger.info("🔍 Falling back to Pinecone knowledge base (despite initial issues)...")
                    pinecone_response = self.pinecone_tool.query(langchain_history)
                    
                    if pinecone_response and pinecone_response.get("success"):
                        should_fallback = self.should_use_web_fallback(pinecone_response, prompt)
                        
                        if not should_fallback:
                            logger.info("✅ Using Pinecone response (fallback)")
                            return pinecone_response
                            
                except Exception as e:
                    error_type = self._detect_pinecone_error_type(e)
                    logger.error(f"Pinecone fallback also failed ({error_type}): {e}")
        
        # Final fallback - basic response
        logger.warning("⚠️ All AI tools unavailable, using basic fallback")
        return {
            "content": "I apologize, but I'm unable to process your request at the moment due to technical issues. Please try again later.",
            "success": False,
            "source": "System Fallback",
            "used_search": False,
            "used_pinecone": False,
            "has_citations": False,
            "has_inline_citations": False,
            "safety_override": False
        }
@handle_api_errors("Content Moderation", "Check Prompt", show_to_user=False)
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    """Checks user prompt against content moderation guidelines using OpenAI's moderation API."""
    if not client or not hasattr(client, 'moderations') :
        logger.debug("OpenAI client or moderation API not available. Skipping content moderation.")
        return {"flagged": False}
    
    try:
        response = client.moderations.create(model="omni-moderation-latest", input=prompt)
        result = response.results[0]
        
        if result.flagged:
            flagged_categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
            logger.warning(f"Input flagged by moderation for: {', '.join(flagged_categories)}")
            return {
                "flagged": True, 
                "message": "Your message violates our content policy.",
                "categories": flagged_categories
            }
    except Exception as e:
        logger.error(f"Content moderation API call failed: {e}", exc_info=True)
        return {"flagged": False}
    
    return {"flagged": False}

@handle_api_errors("Industry Context Check", "Validate Question Context", show_to_user=False)
def check_industry_context(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    """Checks if user question is relevant to food & beverage ingredients industry using GPT-4o-mini."""
    if not client or not hasattr(client, 'chat'):
        logger.debug("OpenAI client not available for industry context check. Allowing question.")
        return {"relevant": True, "reason": "context_check_unavailable"}
    
    try:
        context_check_prompt = f"""You are an industry context validator for 1-2-Taste (12taste.com), a B2B digital marketplace for food ingredients.

**COMPANY CONTEXT:**
- B2B marketplace serving food & beverage manufacturers, processors, and producers
- Focus on ingredient sourcing, product development, and technical support
- Industries served: bakery, confectionery, beverage, dairy, ice cream, meat processing, snacks, etc.
- Product categories: flavors, colors, sweeteners, proteins, emulsifiers, starches, texturizers, fibers, nutraceuticals, etc.
- Target customers: legitimate businesses in food/beverage industry

**TASK:** Determine if the following user question is relevant to the food & beverage ingredients industry.

**ALLOW questions about:**
- Food ingredients (flavors, colors, sweeteners, proteins, emulsifiers, etc.)
- Beverage ingredients and formulation
- Supplier sourcing and procurement in food industry
- Food safety, regulations, and compliance
- Product development and R&D for food/beverages
- Market trends in food/beverage industry
- Technical formulation and processing questions
- Food manufacturing and production
- Ingredient specifications and applications
- Professional food business operations

**FLAG questions about:**
- Completely unrelated industries (automotive, finance, technology not food-related, etc.)
- Personal consumer cooking advice (unless related to professional manufacturing)
- Non-professional food questions (home recipes, diet advice, etc.)
- Off-topic subjects (politics, sports, entertainment, etc.)
- Medical advice or health claims
- General knowledge unrelated to food industry

**USER QUESTION:** "{prompt}"

**INSTRUCTIONS:**
Respond with ONLY a JSON object in this exact format:
{{
    "relevant": true/false,
    "confidence": 0.0-1.0,
    "category": "food_ingredients" | "beverage_formulation" | "supplier_sourcing" | "technical_support" | "market_trends" | "off_topic" | "personal_cooking" | "unrelated_industry",
    "reason": "brief explanation why relevant or not relevant"
}}

Do NOT include any text outside the JSON object."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an industry context validator. Respond only with valid JSON."},
                {"role": "user", "content": context_check_prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Clean up response (remove any markdown formatting)
        response_content = response_content.replace('```json', '').replace('```', '').strip()
        
        try:
            result = json.loads(response_content)
            
            # Validate required fields
            if not all(key in result for key in ['relevant', 'confidence', 'category', 'reason']):
                logger.warning("Industry context check returned incomplete JSON structure")
                return {"relevant": True, "reason": "context_check_invalid_response"}
                
            logger.info(f"Industry context check: relevant={result['relevant']}, category={result['category']}, confidence={result['confidence']:.2f}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Industry context check returned invalid JSON: {response_content[:100]}... Error: {e}")
            return {"relevant": True, "reason": "context_check_json_error"}
            
    except Exception as e:
        logger.error(f"Industry context check failed: {e}", exc_info=True)
        return {"relevant": True, "reason": "context_check_error"}
# =============================================================================
# SESSION MANAGER - MAIN ORCHESTRATOR CLASS
# =============================================================================

class SessionManager:
    """Main orchestrator class that manages user sessions, integrates all managers, and provides the primary interface for the application."""
    
    def __init__(self, config: Config, db_manager: DatabaseManager, 
                 zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, 
                 rate_limiter: RateLimiter, fingerprinting_manager: DatabaseManager.FingerprintingManager, # Corrected type hint
                 email_verification_manager: DatabaseManager.EmailVerificationManager, # Corrected type hint
                 question_limit_manager: DatabaseManager.QuestionLimitManager): # Corrected type hint
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.fingerprinting = fingerprinting_manager
        self.email_verification = email_verification_manager
        self.question_limits = question_limit_manager
        self._cleanup_interval = timedelta(hours=1)
        self._last_cleanup = datetime.now()
        
        logger.info("✅ SessionManager initialized with all component managers.")

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return 5
    
    # Helper to get privilege level for user types
    def _get_privilege_level(self, user_type: UserType) -> int:
        if user_type == UserType.REGISTERED_USER:
            return 2
        elif user_type == UserType.EMAIL_VERIFIED_GUEST:
            return 1
        else:
            return 0

    def _periodic_cleanup(self):
        """Perform periodic cleanup of memory and resources"""
        now = datetime.now()
        if now - self._last_cleanup < self._cleanup_interval:
            return
            
        try:
            # Clean up fingerprinting cache
            if hasattr(self.fingerprinting, 'fingerprint_cache'):
                old_entries = []
                for fp_id, data in self.fingerprinting.fingerprint_cache.items():
                    if now - data.get('last_seen', now) > timedelta(hours=24):
                        old_entries.append(fp_id)
                
                for old_fp in old_entries:
                    del self.fingerprinting.fingerprint_cache[old_fp]
                
                if old_entries:
                    logger.info(f"Cleaned up {len(old_entries)} old fingerprint cache entries")
            
            # Clean up rate limiter
            if hasattr(self.rate_limiter, 'requests'):
                old_limit_entries = []
                for identifier, timestamps in self.rate_limiter.requests.items():
                    cutoff = time.time() - self.rate_limiter.window_seconds
                    self.rate_limiter.requests[identifier] = [t for t in timestamps if t > cutoff]
                    
                    if not self.rate_limiter.requests[identifier]:
                        old_limit_entries.append(identifier)
                
                for old_id in old_limit_entries:
                    del self.rate_limiter.requests[old_id]
                
                if old_limit_entries:
                    logger.info(f"Cleaned up {len(old_limit_entries)} old rate limiter entries")
            
            # Clean up error history
            if hasattr(st.session_state, 'error_handler') and hasattr(st.session_state.error_handler, 'error_history') and len(st.session_state.error_handler.error_history) > 100:
                st.session_state.error_handler.error_history = st.session_state.error_handler.error_history[-50:]
                logger.info("Cleaned up error history")
            
            self._last_cleanup = now
            logger.debug("Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}", exc_info=True)

    def _update_activity(self, session: UserSession):
        """Updates the session's last activity timestamp and saves it to the DB."""
        session.last_activity = datetime.now()
        
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset 'timeout_saved_to_crm' flag for session {session.session_id[:8]} due to new activity.")
        
        if isinstance(session.user_type, str):
            session.user_type = UserType(session.user_type)
        
        if not isinstance(session.messages, list):
            logger.warning(f"Messages field corrupted for session {session.session_id[:8]}, resetting to empty list")
            session.messages = []

        try:
            self.db.save_session(session)
            logger.debug(f"Activity update saved for {session.session_id[:8]} with {len(session.messages)} messages")
        except Exception as e:
            logger.error(f"Failed to save session during activity update: {e}", exc_info=True)

    def _save_session_with_retry(self, session: UserSession, max_retries: int = 3):
        """Save session with retry logic"""
        for attempt in range(max_retries):
            try:
                self.db.save_session(session)
                return
            except Exception as e:
                logger.warning(f"Session save attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    logger.error(f"All session save attempts failed for {session.session_id[:8]}")
                    raise
    
    def _clear_error_notifications(self):
        """Clear all error notification states when a successful question is processed."""
        error_keys = ['rate_limit_hit', 'moderation_flagged', 'context_flagged']
        cleared_errors = []
    
        for key in error_keys:
            if key in st.session_state:
                del st.session_state[key]
                cleared_errors.append(key)
    
        if cleared_errors:
            logger.debug(f"Auto-cleared error notifications on successful question: {cleared_errors}")
                    

    def _create_new_session(self) -> UserSession:
        """Creates a new user session with temporary fingerprint until JS fingerprinting completes."""
        import traceback
        import inspect
        
        # Get the calling function info
        frame = inspect.currentframe()
        caller = inspect.getframeinfo(frame.f_back)
        
        logger.debug("🚨 NEW SESSION BEING CREATED!")
        logger.debug(f"📍 Called from: {caller.filename}:{caller.lineno} in function {caller.function}")
        logger.debug("📋 Full stack trace:")
        for line in traceback.format_stack():
            logger.debug(line.strip())
        
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id, last_activity=None)
        
        session.fingerprint_id = f"temp_py_{secrets.token_hex(8)}"
        session.fingerprint_method = "temporary_fallback_python"
        
        logger.debug(f"🆔 New session created: {session_id[:8]} (NOT saved to DB yet, will be saved in get_session)")
        return session

    def _check_15min_eligibility(self, session: UserSession) -> bool:
        """Check if session has been active for at least 15 minutes to be eligible for CRM save."""
        try:
            start_time = session.created_at
            if session.last_question_time and session.last_question_time < start_time:
                start_time = session.last_question_time
            
            elapsed_time = datetime.now() - start_time
            elapsed_minutes = elapsed_time.total_seconds() / 60
            
            logger.info(f"15-min eligibility check for {session.session_id[:8]}: {elapsed_minutes:.1f} minutes elapsed")
            return elapsed_minutes >= 15.0
            
        except Exception as e:
            logger.error(f"Error checking 15-min eligibility for {session.session_id[:8]}: {e}")
            return False

    def _is_crm_save_eligible(self, session: UserSession, trigger_reason: str) -> bool:
        """Enhanced eligibility check for CRM saves including new user types and conditions."""
        try:
            if not session.email or not session.messages:
                logger.debug(f"CRM save not eligible - missing email ({bool(session.email)}) or messages ({bool(session.messages)}) for {session.session_id[:8]}")
                return False
            
            if session.timeout_saved_to_crm and "clear_chat" in trigger_reason.lower():
                logger.debug(f"CRM save not eligible - already saved for {session.session_id[:8]}")
                return False
            
            if session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]: # FIXED: Corrected to .value for enum comparison
                logger.debug(f"CRM save not eligible - user type {session.user_type.value} for {session.session_id[:8]}")
                return False
            
            if session.daily_question_count < 1:
                logger.debug(f"CRM save not eligible - no questions asked ({session.daily_question_count}) for {session.session_id[:8]}")
                return False

            logger.info(f"CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Questions={session.daily_question_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    def _is_manual_crm_save_eligible(self, session: UserSession) -> bool:
        """Simple eligibility check for manual CRM saves (Sign Out, Manual Save button) - NO 15-minute requirement."""
        try:
            if not session.email or not session.messages:
                return False
            
            if session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]: # FIXED: Corrected to .value for enum comparison
                return False
            
            if session.daily_question_count < 1:
                return False
            
            logger.info(f"Manual CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Questions={session.daily_question_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking manual CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    def _attempt_fingerprint_inheritance(self, session: UserSession):
        """Attempts to inherit session data from existing sessions with the same fingerprint.
        Prioritizes the highest user type and merges usage statistics.
        Crucially, it now sets a 'reverification_pending' flag for privilege inheritance.
        """
        logger.info(f"🔄 Attempting fingerprint inheritance for session {session.session_id[:8]} with fingerprint {session.fingerprint_id[:8]}...")
        
        try:
            # 1. Get all sessions associated with this fingerprint, including the current one
            historical_fp_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
            
            # Filter out the current session from historical list, if it was already saved
            historical_fp_sessions = [s for s in historical_fp_sessions if s.session_id != session.session_id]

            # Determine visitor_type based on persistent history # <--- ADDED THIS BLOCK
            if historical_fp_sessions:
                session.visitor_type = "returning_visitor"
            else:
                session.visitor_type = "new_visitor" # Explicitly set for clarity, though it's the default
            # END ADDED BLOCK
            
            # Combine current session with historical ones to determine the merged state
            all_sessions_for_merge = [session] + historical_fp_sessions

            # Initialize merged values with current session's values as a baseline
            merged_user_type = session.user_type 
            merged_daily_question_count = session.daily_question_count
            merged_total_question_count = session.total_question_count
            merged_last_question_time = session.last_question_time # Use current session's LQT as baseline
            merged_question_limit_reached = session.question_limit_reached
            merged_ban_status = session.ban_status
            merged_ban_start_time = session.ban_start_time
            merged_ban_end_time = session.ban_end_time
            merged_ban_reason = session.ban_reason
            merged_evasion_count = session.evasion_count
            merged_current_penalty_hours = session.current_penalty_hours
            merged_escalation_level = session.escalation_level
            merged_email_addresses_used = set(session.email_addresses_used) 
            merged_email_switches_count = session.email_switches_count # FIX: Initialize this variable
            
            # Determine the effective 'source' session for identity and core data, favoring highest privilege and recency
            source_for_identity_and_base_data = session 
            
            # Iterate through all sessions to find the most authoritative source for each piece of data
            for s in all_sessions_for_merge:
                # Determine the most authoritative source for user type and identity fields
                if self._get_privilege_level(s.user_type) > self._get_privilege_level(source_for_identity_and_base_data.user_type):
                    source_for_identity_and_base_data = s
                elif self._get_privilege_level(s.user_type) == self._get_privilege_level(source_for_identity_and_base_data.user_type):
                    # If same privilege, prefer more recent
                    if s.last_activity and (not source_for_identity_and_base_data.last_activity or s.last_activity > source_for_identity_and_base_data.last_activity):
                        source_for_identity_and_base_data = s
                
                # Merge usage counts (take max for daily and total)
                # IMPORTANT: Only consider daily_question_count if it's potentially valid (i.e., within 24h or if last_question_time is None)
                # Or if the session itself is active. This prevents inheriting stale/reset counts.
                now = datetime.now()
                if s.daily_question_count is not None:
                    # If the session is active OR its last question time is within 24 hours (i.e. not reset yet)
                    if s.active or (s.last_question_time and (now - s.last_question_time < timedelta(hours=24))):
                        merged_daily_question_count = max(merged_daily_question_count, s.daily_question_count)
                    else:
                        # If the session is inactive and its 24-hour window has passed, its daily_question_count is effectively 0 for inheritance.
                        pass 
                
                if s.total_question_count is not None:
                    merged_total_question_count = max(merged_total_question_count, s.total_question_count)
                
                # Merge last_question_time (most recent from all sessions for the fingerprint)
                if s.last_question_time and (not merged_last_question_time or s.last_question_time > merged_last_question_time):
                    merged_last_question_time = s.last_question_time
                elif merged_last_question_time is None and s.last_question_time is not None:
                    merged_last_question_time = s.last_question_time


                # Merge ban status (most restrictive)
                if s.ban_status != BanStatus.NONE:
                    if merged_ban_status == BanStatus.NONE or \
                       (s.ban_status == BanStatus.EVASION_BLOCK and merged_ban_status != BanStatus.EVASION_BLOCK) or \
                       (s.ban_status == BanStatus.TWENTY_FOUR_HOUR and merged_ban_status == BanStatus.ONE_HOUR):
                        merged_ban_status = s.ban_status
                        merged_ban_start_time = s.ban_start_time
                        merged_ban_end_time = s.ban_end_time
                        merged_ban_reason = s.ban_reason
                        merged_question_limit_reached = s.question_limit_reached 

                # Merge evasion stats (take max)
                if s.evasion_count is not None: merged_evasion_count = max(merged_evasion_count, s.evasion_count)
                if s.current_penalty_hours is not None: merged_current_penalty_hours = max(merged_current_penalty_hours, s.current_penalty_hours)
                if s.escalation_level is not None: merged_escalation_level = max(merged_escalation_level, s.escalation_level)

                # Merge email addresses used
                if s.email_addresses_used:
                    merged_email_addresses_used.update(s.email_addresses_used)
                if s.email_switches_count is not None: merged_email_switches_count = max(merged_email_switches_count, s.email_switches_count)
            
            original_session_user_type_for_log = session.user_type
            
            # Apply merged usage values (always inherited for any session)
            session.daily_question_count = merged_daily_question_count
            session.total_question_count = merged_total_question_count
            session.last_question_time = merged_last_question_time # Set from the most recent activity found
            session.question_limit_reached = merged_question_limit_reached
            session.ban_status = merged_ban_status
            session.ban_start_time = merged_ban_start_time
            session.ban_end_time = merged_ban_end_time
            session.ban_reason = merged_ban_reason
            session.evasion_count = merged_evasion_count
            session.current_penalty_hours = merged_current_penalty_hours
            session.escalation_level = merged_escalation_level
            session.email_addresses_used = list(merged_email_addresses_used)
            session.email_switches_count = merged_email_switches_count
            
            # Determine privilege inheritance based on re-verification need
            if self._get_privilege_level(source_for_identity_and_base_data.user_type) > self._get_privilege_level(session.user_type):
                # We found a higher privilege session for this fingerprint.
                # Do NOT automatically upgrade. Instead, set pending re-verification.
                session.reverification_pending = True
                session.pending_user_type = source_for_identity_and_base_data.user_type
                session.pending_email = source_for_identity_and_base_data.email
                session.pending_full_name = source_for_identity_and_base_data.full_name
                session.pending_zoho_contact_id = source_for_identity_and_base_data.zoho_contact_id
                session.pending_wp_token = source_for_identity_and_base_data.wp_token
                session.browser_privacy_level = source_for_identity_and_base_data.browser_privacy_level
                
                # Also ensure declined_recognized_email_at is cleared if a higher privilege is now pending
                session.declined_recognized_email_at = None
                
                logger.info(f"Re-verification pending for {session.session_id[:8]}. Higher privilege ({source_for_identity_and_base_data.user_type.value}) found for fingerprint, but requires email verification.")
            else:
                # Current session is already at the highest or equal privilege level found for this fingerprint.
                # Or no higher privilege found. Apply identity fields directly.
                session.user_type = source_for_identity_and_base_data.user_type # This ensures GUEST remains GUEST if no higher is found
                session.email = source_for_identity_and_base_data.email
                session.full_name = source_for_identity_and_base_data.full_name
                session.zoho_contact_id = source_for_identity_and_base_data.zoho_contact_id
                session.wp_token = source_for_identity_and_base_data.wp_token
                session.browser_privacy_level = source_for_identity_and_base_data.browser_privacy_level
                session.reverification_pending = False # No pending if already at max privilege or no higher exists
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                # Importantly, do NOT clear declined_recognized_email_at here if it was just set.
                # It will be managed by QuestionLimitManager.detect_guest_email_evasion
                logger.info(f"No re-verification needed for {session.session_id[:8]}. User type set to {session.user_type.value}.")


            # Update fingerprint to the 'real' one if current is temporary and a real one was found
            # The 'real' fingerprint should come from the most authoritative source.
            if session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")) and \
               not source_for_identity_and_base_data.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
               
                logger.info(f"Updating temporary fingerprint {session.fingerprint_id[:8]} to recognized fingerprint {source_for_identity_and_base_data.fingerprint_id[:8]}")
                session.fingerprint_id = source_for_identity_and_base_data.fingerprint_id
                session.fingerprint_method = source_for_identity_and_base_data.fingerprint_method
            
            # Ensure last_activity is updated if it was None or older than source
            # This is critical for the main timeout functionality
            if session.last_activity is None and source_for_identity_and_base_data.last_activity is not None:
                session.last_activity = source_for_identity_and_base_data.last_activity
            elif source_for_identity_and_base_data.last_activity and (not session.last_activity or source_for_identity_and_base_data.last_activity > session.last_activity):
                session.last_activity = source_for_identity_and_base_data.last_activity


            logger.info(f"Inheritance complete for {session.session_id[:8]}: user_type={session.user_type.value} (from {original_session_user_type_for_log.value}), daily_q={session.daily_question_count}, total_q={session.total_question_count}, fp={session.fingerprint_id[:8]}, active={session.active}, rev_pending={session.reverification_pending}")
            
        except Exception as e:
            logger.error(f"Error during fingerprint inheritance for session {session.session_id[:8]}: {e}", exc_info=True)


    def get_session(self) -> Optional[UserSession]:
        """Gets or creates the current user session with enhanced validation."""
        logger.info(f"🔍 get_session() called - current_session_id in state: {st.session_state.get('current_session_id', 'None')}")
        
        # Perform periodic cleanup
        self._periodic_cleanup()

        try:
            session_id = st.session_state.get('current_session_id')
            
            if session_id:
                # ADD THIS: Check if session loading is already in progress
                if st.session_state.get(f'loading_{session_id}', False):
                    logger.warning(f"Session {session_id[:8]} already being loaded, skipping")
                    return None
        
                st.session_state[f'loading_{session_id}'] = True
                session = self.db.load_session(session_id)
                st.session_state[f'loading_{session_id}'] = False  # Clear the flag
                
                if session and session.active:
                    # NEW: Immediately attempt fingerprint inheritance if session has a temporary fingerprint
                    # And this check hasn't been performed for this session yet in the current rerun cycle.
                    fingerprint_checked_key = f'fingerprint_checked_for_inheritance_{session.session_id}'
                    if (session.fingerprint_id and 
                        session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")) and 
                        not st.session_state.get(fingerprint_checked_key, False)):
                        
                        self._attempt_fingerprint_inheritance(session)
                        # Save session after potential inheritance to persist updated user type/counts
                        self.db.save_session(session) # Crucial to save here
                        st.session_state[fingerprint_checked_key] = True # Mark as checked for this session
                        logger.info(f"Fingerprint inheritance check and save completed for {session.session_id[:8]}")

                    # Check limits and handle bans. This is where the 24-hour reset happens.
                    limit_check = self.question_limits.is_within_limits(session)
                    if not limit_check.get('allowed', True):
                        # If the user is being prompted for re-verification due to higher historical privilege,
                        # do not show a ban message, but let the dialog handle it.
                        if session.reverification_pending and limit_check.get('reason') == 'guest_limit': # This condition seems redundant based on the `if not allowed and reason != guest_limit` above
                            logger.info(f"Session {session.session_id[:8]} is pending re-verification, suppressing ban message and allowing dialog.")
                        else:
                            ban_type = limit_check.get('ban_type', 'unknown')
                            message = limit_check.get('message', 'Access restricted due to usage policy.')
                            time_remaining = limit_check.get('time_remaining')
                            
                            st.error(f"🚫 **Access Restricted**")
                            if time_remaining:
                                hours = max(0, int(time_remaining.total_seconds() // 3600))
                                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                                st.error(f"Time remaining: {hours}h {minutes}m")
                            st.info(message)
                            logger.info(f"Session {session_id[:8]} is currently banned: Type={ban_type}, Reason='{message}'.")
                        
                        try:
                            self.db.save_session(session)
                        except Exception as e:
                            logger.error(f"Failed to save banned session {session.session_id[:8]}: {e}", exc_info=True)
                        
                        return session
                    
                    return session
                else:
                    logger.warning(f"Session {session_id[:8]} not found or inactive. Creating new session.")
                    if 'current_session_id' in st.session_state:
                        del st.session_state['current_session_id']

            logger.info(f"Creating new session")
            new_session = self._create_new_session()
            st.session_state.current_session_id = new_session.session_id
            
            # Immediately attempt fingerprint inheritance for the *newly created* session
            # This is critical if a user starts a new session but has an existing fingerprint
            self._attempt_fingerprint_inheritance(new_session) # <--- This call will now also correctly set visitor_type
            st.session_state[f'fingerprint_checked_for_inheritance_{new_session.session_id}'] = True
            
            self.db.save_session(new_session) # Save the new session (potentially updated by inheritance)
            logger.info(f"Created and stored new session {new_session.session_id[:8]} (post-inheritance check), active={new_session.active}, rev_pending={new_session.reverification_pending}")
            return new_session
            
        except Exception as e:
            logger.error(f"Failed to get/create session: {e}", exc_info=True)
            fallback_session = UserSession(session_id=str(uuid.uuid4()), user_type=UserType.GUEST, last_activity=None)
            fallback_session.fingerprint_id = f"emergency_fp_{fallback_session.session_id[:8]}"
            fallback_session.fingerprint_method = "emergency_fallback"
            st.session_state.current_session_id = fallback_session.session_id
            st.error("⚠️ Failed to create or load session. Operating in emergency fallback mode. Chat history may not persist.")
            logger.error(f"Emergency fallback session created {fallback_session.session_id[:8]}")
            return fallback_session

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]) -> bool:
        """Applies fingerprinting data from custom component to the session with better validation."""
        logger.debug(f"🔍 APPLYING FINGERPRINT received from JS: {fingerprint_data.get('fingerprint_id', 'None')[:8]} to session {session.session_id[:8]}")
        
        try:
            if not fingerprint_data or not isinstance(fingerprint_data, dict):
                logger.warning("Invalid fingerprint data provided to apply_fingerprinting")
                return False
            
            old_fingerprint_id = session.fingerprint_id
            old_method = session.fingerprint_method
            
            # Store the *new, real* fingerprint data
            session.fingerprint_id = fingerprint_data.get('fingerprint_id')
            session.fingerprint_method = fingerprint_data.get('fingerprint_method')
            # session.visitor_type is determined by apply_fingerprinting after inheritance check
            session.browser_privacy_level = fingerprint_data.get('browser_privacy_level', 'standard')
            session.recognition_response = None # Clear any previous recognition response
            
            if not session.fingerprint_id or not session.fingerprint_method:
                logger.error("Invalid fingerprint data: missing essential fields from JS. Reverting to old fingerprint.")
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return False
            
            # Now that the current session has its *real* fingerprint from JS,
            # run the inheritance logic to see if this fingerprint has a history.
            # This will update user_type, question counts, etc., based on the definitive fingerprint.
            self._attempt_fingerprint_inheritance(session) # <--- This call will now also update visitor_type
            
            # Save session with new fingerprint data and inherited properties
            try:
                self.db.save_session(session)
                logger.info(f"✅ Fingerprinting applied and inheritance checked for {session.session_id[:8]}: {session.fingerprint_method} (ID: {session.fingerprint_id[:8]}...), active={session.active}, rev_pending={session.reverification_pending}")
            except Exception as e:
                logger.error(f"Failed to save session after fingerprinting (JS data received): {e}")
                # If save fails, revert to old fingerprint to avoid inconsistent state
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return False
        except Exception as e:
            logger.error(f"Fingerprint processing failed: {e}", exc_info=True)
            return False
        return True

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """Check if a fingerprint has historical sessions and return relevant information."""
        try:
            existing_sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if not existing_sessions:
                return {'has_history': False}
            
            # Find the most privileged active/inactive session for history suggestion
            most_privileged_session = None
            for s in existing_sessions:
                if most_privileged_session is None or \
                   self._get_privilege_level(s.user_type) > self._get_privilege_level(most_privileged_session.user_type) or \
                   (self._get_privilege_level(s.user_type) == self._get_privilege_level(most_privileged_session.user_type) and 
                    s.last_activity and (not most_privileged_session.last_activity or s.last_activity > most_privileged_session.last_activity)):
                    
                    if s.email: # Only consider if an email is present
                        most_privileged_session = s

            if most_privileged_session:
                return {
                    'has_history': True,
                    'email': most_privileged_session.email,
                    'full_name': most_privileged_session.full_name,
                    'user_type': most_privileged_session.user_type.value,
                    'last_activity': most_privileged_session.last_activity,
                    'daily_question_count': most_privileged_session.daily_question_count # Also pass the count from history
                }
            
            return {'has_history': False}
            
        except Exception as e:
            logger.error(f"Error checking fingerprint history: {e}")
            return {'has_history': False}

    def _mask_email(self, email: str) -> str:
        """Masks an email address for privacy (shows first 2 chars + domain)."""
        try:
            local, domain = email.split('@')
            if len(local) <= 2:
                masked_local = local[0] + '*' * (len(local) - 1) # Mask remaining chars
            else:
                masked_local = local[:2] + '*' * (len(local) - 2)
            return f"{masked_local}@{domain}"
        except Exception:
            return "****@****.***"

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Enhanced email verification with evasion detection."""
        try:
            sanitized_email = sanitize_input(email, 100).lower().strip()
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', sanitized_email):
                return {'success': False, 'message': 'Please enter a valid email address.'}
            
            # NEW: Check for evasion before processing
            if self.question_limits.detect_guest_email_evasion(session, self.db):
                penalty_hours = self.question_limits.apply_evasion_penalty(session)
                self.db.save_session(session)  # Save the penalty to database
                logger.warning(f"🚨 EMAIL EVASION DETECTED: Applied {penalty_hours}h penalty to session {session.session_id[:8]}")
                return {
                    'success': False, 
                    'message': f'Unusual activity detected. Access temporarily restricted for {penalty_hours} hours.'
                }
            
            # Track email usage for this session
            if sanitized_email not in session.email_addresses_used:
                session.email_addresses_used.append(sanitized_email)
            
            # Update session email (if not already set by re-verification pending)
            if not session.reverification_pending:
                if session.email and session.email != sanitized_email:
                    session.email_switches_count += 1
                session.email = sanitized_email
            elif session.reverification_pending and sanitized_email != session.pending_email:
                # If reverification is pending but they enter a different email, treat as new email path
                # This could be an evasion or legitimate switch. For now, reset pending state.
                session.email_switches_count += 1
                session.email = sanitized_email
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                logger.warning(f"Session {session.session_id[:8]} switched email during pending re-verification. Resetting pending state.")

            # Clear `declined_recognized_email_at` if they are now proceeding with *any* email verification
            session.declined_recognized_email_at = None
            
            # Set last_activity if not already set (first time starting chat)
            if session.last_activity is None:
                session.last_activity = datetime.now()

            # Save session before sending verification
            try:
                self.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session before email verification: {e}")
            
            # Send verification code
            code_sent = self.email_verification.send_verification_code(sanitized_email)
            
            if code_sent:
                return {
                    'success': True,
                    'message': f'Verification code sent to {sanitized_email}. Please check your email.'
                }
            else:
                return {
                    'success': False, 
                    'message': 'Failed to send verification code. Please try again later.'
                }
                
        except Exception as e:
            logger.error(f"Email verification handling failed: {e}")
            return {
                'success': False, 
                'message': 'An error occurred. Please try again.'
            }

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verifies the email verification code and upgrades user status."""
        try:
            email_to_verify = session.pending_email if session.reverification_pending else session.email

            if not email_to_verify:
                return {'success': False, 'message': 'No email address found for verification.'}
            
            sanitized_code = sanitize_input(code, 10).strip()
            
            if not sanitized_code:
                return {'success': False, 'message': 'Please enter the verification code.'}
            
            verification_success = self.email_verification.verify_code(email_to_verify, sanitized_code)
            
            if verification_success:
                logger.info(f"✅ BEFORE email verification upgrade for session {session.session_id[:8]}:")
                logger.info(f"   - User Type: {session.user_type.value}")
                logger.info(f"   - Daily Questions: {session.daily_question_count}")
                logger.info(f"   - Total Questions: {session.total_question_count}")
                logger.info(f"   - Has Messages: {len(session.messages)}")
                logger.info(f"   - Reverification Pending: {session.reverification_pending}")
                
                if session.reverification_pending:
                    # Reclaim pending user type and identity
                    session.user_type = session.pending_user_type if session.pending_user_type else UserType.EMAIL_VERIFIED_GUEST # Fallback
                    session.email = session.pending_email
                    session.full_name = session.pending_full_name
                    session.zoho_contact_id = session.pending_zoho_contact_id
                    session.wp_token = session.pending_wp_token
                    session.reverification_pending = False
                    session.pending_user_type = None
                    session.pending_email = None
                    session.pending_full_name = None
                    session.pending_zoho_contact_id = None
                    session.pending_wp_token = None
                    logger.info(f"✅ User {session.session_id[:8]} reclaimed higher privilege: {session.user_type.value} via re-verification for {session.email}")
                else:
                    # Upgrade user to email verified guest (normal first-time verification)
                    session.user_type = UserType.EMAIL_VERIFIED_GUEST
                    # Email already set in handle_guest_email_verification
                    logger.info(f"✅ User {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST (first time): {session.email}")

                session.question_limit_reached = False  # Reset limit flag
                # Clear declined_recognized_email_at on successful verification
                session.declined_recognized_email_at = None 
                
                # Set last_activity to now (official start for logged-in users)
                if session.last_activity is None:
                    session.last_activity = datetime.now()

                # Save upgraded session
                try:
                    self.db.save_session(session)
                    logger.info(f"✅ AFTER email verification upgrade for session {session.session_id[:8]}:")
                    logger.info(f"   - User Type: {session.user_type.value}")
                    logger.info(f"   - Daily Questions: {session.daily_question_count} (PRESERVED from device history)")
                    logger.info(f"   - Total Questions: {session.total_question_count}")
                    logger.info(f"   - Has Messages: {len(session.messages)}")
                    logger.info(f"   - CRM Eligible: {self._is_crm_save_eligible(session, 'email_verification_success')}")

                except Exception as e:
                    logger.error(f"Failed to save upgraded session: {e}")
                
                return {
                    'success': True,
                    'message': f'✅ Email verified successfully! You now have {10 - session.daily_question_count if session.daily_question_count <= 10 else 0} questions remaining today (total 10 for the day).'
                }
            else:
                return {
                    'success': False,
                    'message': 'Invalid verification code. Please check the code and try again.'
                }
                
        except Exception as e:
            logger.error(f"Email code verification failed: {e}")
            return {
                'success': False,
                'message': 'Verification failed. Please try again.'
            }

    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """Authenticates user with WordPress and creates/updates session."""
        if not self.config.WORDPRESS_URL:
            st.error("WordPress authentication is not configured. Please contact support.")
            logger.warning("WordPress authentication attempted but URL not configured.")
            return None
        
        try:
            auth_url = f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token"
            
            response = requests.post(auth_url, json={
                'username': username,
                'password': password
            }, timeout=10)
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/json' not in content_type:
                    logger.error(f"WordPress authentication received unexpected Content-Type '{content_type}' with 200 status. Expected application/json. Response text: '{response.text[:200]}'", exc_info=True)
                    st.error("Authentication failed: WordPress returned a webpage instead of an API response. This often means the JWT authentication plugin is not active or configured correctly on the WordPress site. Please contact support.")
                    return None

                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError as e:
                    logger.error(f"WordPress authentication received non-JSON response with 200 status, despite Content-Type possibly being JSON. Response text: '{response.text[:200]}'. Error: {e}", exc_info=True)
                    st.error("Authentication failed: Invalid JSON response from WordPress. Please try again or contact support.")
                    return None
                
                wp_token = data.get('token')
                user_email = data.get('user_email')
                user_display_name = data.get('user_display_name')
                
                if wp_token and user_email:
                    current_session = self.get_session()
                    
                    # --- SIMPLIFIED LOGIC FOR daily_question_count ON REGISTERED_USER LOGIN ---
                    # Per user clarification: the 24-hour reset rule for daily_question_count
                    # applies universally. No special "override" to preserve a count beyond
                    # 24 hours just because of REGISTERED_USER login.
                    # current_session.daily_question_count and current_session.last_question_time
                    # will already reflect the correct state due to get_session()'s inheritance
                    # and 24-hour reset logic.
                    
                    # Clear any pending re-verification flags, as direct login supersedes it
                    if current_session.reverification_pending:
                        logger.info(f"REGISTERED_USER login for {current_session.session_id[:8]} supersedes pending re-verification.")
                        current_session.reverification_pending = False
                        current_session.pending_user_type = None
                        current_session.pending_email = None
                        current_session.pending_full_name = None
                        current_session.pending_zoho_contact_id = None
                        current_session.pending_wp_token = None
                    # Clear declined_recognized_email_at on successful login
                    current_session.declined_recognized_email_at = None

                    # Upgrade to registered user
                    current_session.user_type = UserType.REGISTERED_USER
                    current_session.email = user_email
                    current_session.full_name = user_display_name
                    current_session.wp_token = wp_token
                    current_session.question_limit_reached = False # Reset any limit flags
                    
                    # Set last_activity to now (official start for logged-in users)
                    current_session.last_activity = datetime.now()
                    
                    # Save authenticated session
                    try:
                        self.db.save_session(current_session)
                        logger.info(f"User authenticated and upgraded to REGISTERED_USER: {user_email}, daily_q={current_session.daily_question_count}")
                    except Exception as e:
                        logger.error(f"Failed to save authenticated session: {e}")
                        st.error("Authentication succeeded but session could not be saved. Please try again or contact support.")
                        return None
                    
                    return current_session
                else:
                    logger.error(f"WordPress authentication successful (status 200) but missing token or email in response. Response: {data}")
                    st.error("Authentication failed: Incomplete response from WordPress. Please contact support.")
                    return None
            else:
                logger.warning(f"WordPress authentication failed with status: {response.status_code}. Response: {response.text[:200]}")
                st.error("Invalid username or password.")
                return None
                
        except requests.exceptions.Timeout as e:
            logger.error(f"WordPress authentication timed out: {e}")
            st.error("Authentication service timed out. Please try again.")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"WordPress authentication connection error: {e}")
            st.error("Could not connect to authentication service. Check your internet connection or try again later.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during WordPress authentication: {e}", exc_info=True)
            st.error("An unexpected error occurred during authentication. Please try again later.")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Gets AI response and manages session state."""
        try:
            # Handle cases where fingerprint_id might still be temporary or None during early load.
            # Use a fallback ID if the real fingerprint isn't yet established or is a temporary one.
            rate_limiter_id = session.fingerprint_id
            if rate_limiter_id is None or rate_limiter_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                # Fallback to session_id for truly un-fingerprinted or temporary cases,
                # to still apply some level of protection, albeit less robust.
                rate_limiter_id = session.session_id
                logger.warning(f"Rate limiter using session ID as fallback for unconfirmed fingerprint: {rate_limiter_id[:8]}...")

            # Check rate limiting using fingerprint_id (or fallback session_id)
            rate_limit_result = self.rate_limiter.is_allowed(rate_limiter_id)
            if not rate_limit_result['allowed']:
                time_until_next = rate_limit_result.get('time_until_next', 0)
                max_requests = rate_limit_result.get('max_requests', 2)
                window_seconds = rate_limit_result.get('window_seconds', 60)
                
                # Store rate limit info (NO expiry timer - stays until dismissed or success)
                st.session_state.rate_limit_hit = {
                    'timestamp': datetime.now(),
                    'time_until_next': time_until_next,
                    'max_requests': max_requests,
                    'window_seconds': window_seconds
                }
                
                return {
                    'content': f'Rate limit exceeded. Please wait {time_until_next} seconds before asking another question.',
                    'success': False,
                    'source': 'Rate Limiter',
                    'time_until_next': time_until_next
                }
            
            # Content moderation check
            moderation_result = check_content_moderation(prompt, self.ai.openai_client)
            if moderation_result and moderation_result.get("flagged"):
                categories = moderation_result.get('categories', [])
                logger.warning(f"Content moderation flagged input: {categories}")
                
                # Store moderation error info in session state
                st.session_state.moderation_flagged = {
                    'timestamp': datetime.now(),
                    'categories': categories,
                    'message': moderation_result.get("message", "Your message violates our content policy. Please rephrase your question."),
                    
                }
                
                return {
                    "content": moderation_result.get("message", "Your message violates our content policy. Please rephrase your question."),
                    "success": False,
                    "source": "Content Moderation",
                    "used_search": False,
                    "used_pinecone": False,
                    "has_citations": False,
                    "has_inline_citations": False,
                    "safety_override": False
                }

            # Industry context validation check
            context_result = check_industry_context(prompt, self.ai.openai_client)
            if context_result and not context_result.get("relevant", True):
                confidence = context_result.get("confidence", 0.0)
                category = context_result.get("category", "unknown")
                reason = context_result.get("reason", "Not relevant to food & beverage ingredients industry")
                
                logger.warning(f"Industry context check flagged input: category={category}, confidence={confidence:.2f}, reason={reason}")
                
                # Customize response based on category
                if category in ["personal_cooking", "off_topic"]:
                    context_message = "I'm specialized in helping food & beverage industry professionals with ingredient sourcing, formulation, and technical questions. Could you please rephrase your question to focus on professional food ingredient needs?"
                elif category == "unrelated_industry":
                    context_message = "I'm designed to assist with food & beverage ingredients and related industry topics. For questions outside the food industry, please try a general-purpose AI assistant."
                else:
                    context_message = "Your question doesn't seem to be related to food & beverage ingredients. I specialize in helping with ingredient sourcing, formulation, suppliers, and food industry technical questions."
                
                # Store context error info in session state
                st.session_state.context_flagged = {
                    'timestamp': datetime.now(),
                    'category': category,
                    'confidence': confidence,
                    'reason': reason,
                    'message': context_message,
                    
                }
                
                return {
                    "content": context_message,
                    "success": False,
                    "source": "Industry Context Filter",
                    "used_search": False,
                    "used_pinecone": False,
                    "has_citations": False,
                    "has_inline_citations": False,
                    "safety_override": False,
                    "context_category": category,
                    "context_confidence": confidence
                }

            # Rest of the existing function remains the same...
                # Check question limits
            limit_check = self.question_limits.is_within_limits(session)
            # This check for 'allowed' is crucial and must be placed here before recording question
            if not limit_check['allowed']:
                # If it's a hard ban, just return, the message has been rendered by is_within_limits
                if limit_check.get('reason') != 'guest_limit': 
                    return {
                        'banned': True,
                        'content': limit_check.get("message", 'Access restricted.'), # Use 'message' from limit_check
                        'time_remaining': limit_check.get('time_remaining')
                    }
                # If it's 'guest_limit', it means they've hit 4 questions and need to verify
                else: # limit_check.get('reason') == 'guest_limit'
                    # Do NOT set session.question_limit_reached here, it's set on the next rerurn in display_email_prompt_if_needed
                    return {'requires_email': True, 'content': 'Email verification required.'}

            self._clear_error_notifications()
            
            # Sanitize input
            sanitized_prompt = sanitize_input(prompt, 4000)
            if not sanitized_prompt:
                return {
                    'content': 'Please enter a valid question.',
                    'success': False,
                    'source': 'Input Validation'
                }
            
            # Record question (only if allowed to ask)
            self.question_limits.record_question(session)
            
            # Get AI response (now simple call to EnhancedAI's core function)
            ai_response = self.ai.get_response(sanitized_prompt, session.messages) # No extra args needed now
            
            # Handle if ai.get_response() returned None due to its internal errors
            if ai_response is None:
                logger.error(f"EnhancedAI.get_response returned None for session {session.session_id[:8]}")
                return {
                    'content': 'I encountered an internal error while generating a response. Please try again later.',
                    'success': False,
                    'source': 'AI System Internal Error'
                }

            # Add messages to session
            user_message = {'role': 'user', 'content': sanitized_prompt}
            assistant_message = {
                'role': 'assistant',
                'content': ai_response.get('content', 'No response generated.'),
                'source': ai_response.get('source'),
                'used_pinecone': ai_response.get('used_pinecone', False),
                'used_search': ai_response.get('used_search', False),
                'has_citations': ai_response.get('has_citations', False),
                'has_inline_citations': ai_response.get('has_inline_citations', False),
                'safety_override': ai_response.get('safety_override', False)
            }
            
            session.messages.extend([user_message, assistant_message])
            
            # Check for Tier 1 limit AFTER adding the 10th message and applying the ban
            tier1_ban_applied_post_response = False
            if (session.user_type.value == UserType.REGISTERED_USER.value and # Use .value for robustness
                session.daily_question_count == 10 and 
                session.ban_status.value == BanStatus.NONE.value): # Ensure ban isn't already active (use .value)
                
                self.question_limits._apply_ban(session, BanStatus.ONE_HOUR, "Tier 1 limit reached (10 questions)")
                tier1_ban_applied_post_response = True
                logger.info(f"Tier 1 ban applied to registered user {session.session_id[:8]} after 10th question's response was added.")


            # Update activity and save session
            self._update_activity(session)
            
            # Return original AI response, but add tier1_ban_applied_post_response flag
            ai_response['tier1_ban_applied_post_response'] = tier1_ban_applied_post_response
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}", exc_info=True)
            return {
                'content': 'I encountered an error processing your request. Please try again.',
                'success': False,
                'source': 'Error Handler'
            }

    def clear_chat_history(self, session: UserSession):
        """Clears chat history using soft clear mechanism."""
        try:
            # Soft clear: set offset to hide all current messages
            session.display_message_offset = len(session.messages)
            
            # Save the updated session
            self.db.save_session(session)
            
            logger.info(f"Soft cleared chat for session {session.session_id[:8]}: offset set to {session.display_message_offset}")
            
        except Exception as e:
            logger.error(f"Failed to clear chat history for session {session.session_id[:8]}: {e}")

    def manual_save_to_crm(self, session: UserSession):
        """Manually saves chat transcript to CRM (no 15-minute requirement)."""
        try:
            if self._is_manual_crm_save_eligible(session):
                with st.spinner("Saving chat transcript to CRM..."):
                    success = self.zoho.save_chat_transcript_sync(session, "Manual Save")
                    
                if success:
                    st.success("✅ Chat transcript saved to CRM successfully!")
                    logger.info(f"Manual CRM save successful for session {session.session_id[:8]}")
                else:
                    st.error("❌ Failed to save to CRM. Please try again later.")
                    logger.warning(f"Manual CRM save failed for session {session.session_id[:8]}")
            else:
                st.warning("⚠️ CRM save not available. Ensure you have an email verified and at least 1 question asked.")
                logger.info(f"Manual CRM save not eligible for session {session.session_id[:8]}")
                
        except Exception as e:
            logger.error(f"Manual CRM save error: {e}")
            st.error("❌ An error occurred while saving to CRM.")

    def end_session(self, session: UserSession):
        """Ends the current session and performs cleanup."""
        try:
            # Attempt CRM save for eligible users (manual save - no 15-minute requirement)
            if self._is_manual_crm_save_eligible(session):
                with st.spinner("Saving your conversation..."):
                    try:
                        success = self.zoho.save_chat_transcript_sync(session, "Sign Out")
                        if success:
                            st.success("✅ Conversation saved successfully!")
                        else:
                            st.warning("⚠️ Conversation save failed, but sign out will continue.")
                    except Exception as e:
                        logger.error(f"CRM save during sign out failed: {e}")
                        st.warning("⚠️ Conversation save failed, but sign out will continue.")
            
            # Mark session as inactive
            session.active = False
            session.last_activity = datetime.now()
            
            # Save final session state
            try:
                self.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session during end_session: {e}")
            
            # Clear Streamlit session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reset to welcome page
            st.session_state['page'] = None
            
            st.success("👋 You have been signed out successfully!")
            logger.info(f"Session {session.session_id[:8]} ended by user.")
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
            # Force clear session state even if save fails
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['page'] = None

def render_simple_activity_tracker(session_id: str):
    """Renders a simple activity tracker that monitors user interactions."""
    if not session_id:
        logger.warning("render_simple_activity_tracker called without session_id")
        return None
    
    # Create a unique, filesystem-safe key for this session
    safe_session_id = session_id.replace('-', '_')
    component_key = f"activity_tracker_{safe_session_id}"

    simple_tracker_js = f"""
    (() => {{
        const sessionId = "{session_id}";
        const stateKey = 'fifi_activity_{safe_session_id}';
        
        // Initialize or get existing state
        if (!window[stateKey]) {{
            window[stateKey] = {{
                lastActivity: Date.now(),
                listenersInitialized: false,
                sessionId: sessionId
            }};
        }}
        
        const state = window[stateKey];
        
        // Setup activity listeners (only once per browser session lifecycle)
        if (!state.listenersInitialized) {{
            console.log('📍 Simple activity tracker starting for', sessionId.substring(0, 8));
            
            function updateActivity() {{
                state.lastActivity = Date.now();
            }}
            
            // Monitor user activity
            const events = ['mousedown', 'mousemove', 'keydown', 'click', 'scroll', 'touchstart', 'focus'];
            events.forEach(eventType => {{
                document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
            }});
            
            // Try to monitor parent document (for iframes)
            try {{
                if (window.parent && window.parent !== window && window.parent.document) {{
                    events.forEach(eventType => {{
                        window.parent.document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
                    }});
                }}
            }} catch (e) {{
                // Cross-origin restriction, ignore
            }}
            
            state.listenersInitialized = true;
            console.log('✅ Simple activity tracker initialized');
        }}
        
        // Return current activity status
        const now = Date.now();
        const timeSinceActivity = now - state.lastActivity;
        const minutesSinceActivity = Math.floor(timeSinceActivity / 60000);
        
        return {{
            type: 'activity_status',
            session_id: sessionId,
            minutes_inactive: minutesSinceActivity,
            last_activity: state.lastActivity,
            timestamp: now
        }};
    }})()
    """
    
    try:
        result = st_javascript(js_code=simple_tracker_js, key=component_key)
        
        if result and isinstance(result, dict) and result.get('type') == 'activity_status':
            return result
        return None
    except Exception as e:
        logger.error(f"Simple activity tracker failed: {e}")
        logger.error(f"Problematic key in render_simple_activity_tracker: {component_key}")
        return None

def check_timeout_and_trigger_reload(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]) -> bool:
    """Check if timeout has occurred and trigger browser reload if needed."""
    if not session or not session.session_id:
        logger.debug("No valid session for timeout check.")
        return False
    
    # Load fresh session from DB to get the most accurate last_activity
    fresh_session_from_db = session_manager.db.load_session(session.session_id)
    
    if fresh_session_from_db:
        # Update current in-memory session object with latest from DB
        session.last_activity = fresh_session_from_db.last_activity
        session.active = fresh_session_from_db.active
        session.user_type = fresh_session_from_db.user_type
        session.email = fresh_session_from_db.email
        session.full_name = fresh_session_from_db.full_name
        session.zoho_contact_id = fresh_session_from_db.zoho_contact_id
        session.daily_question_count = fresh_session_from_db.daily_question_count
        session.total_question_count = fresh_session_from_db.total_question_count
        session.last_question_time = fresh_session_from_db.last_question_time
        session.display_message_offset = fresh_session_from_db.display_message_offset
        # Also update re-verification flags
        session.reverification_pending = fresh_session_from_db.reverification_pending
        session.pending_user_type = fresh_session_from_db.pending_user_type
        session.pending_email = fresh_session_from_db.pending_email # Corrected variable name from fresh_session_from_rb
        session.pending_full_name = fresh_session_from_db.pending_full_name
        session.pending_zoho_contact_id = fresh_session_from_db.pending_zoho_contact_id
        session.pending_wp_token = fresh_session_from_db.pending_wp_token
        session.declined_recognized_email_at = fresh_session_from_db.declined_recognized_email_at # NEW
    else:
        logger.warning(f"Session {session.session_id[:8]} from st.session_state not found in database. Forcing reset.")
        session.active = False

    # If the session is already inactive, force a reload
    if not session.active:
        logger.info(f"Session {session.session_id[:8]} is inactive. Triggering reload to welcome page.")
        st.error("⏰ **Session Expired**")
        st.info("Your previous session has ended. Please start a new session.")
        
        # Clear Streamlit session state fully
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # NEW: Force welcome page state before any reload attempts
        st.session_state['page'] = None
        st.session_state['session_expired'] = True

        if JS_EVAL_AVAILABLE:
            try:
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                st.stop()
            except Exception as e:
                logger.error(f"Browser reload failed during inactive session handling: {e}")
        
        st.info("🏠 Redirecting to home page...")
        st.rerun()
        st.stop()
        return True

    # Check if last_activity is None (timer hasn't started)
    if session.last_activity is None:
        logger.debug(f"Session {session.session_id[:8]}: last_activity is None, timer has not started.")
        return False
        
    # Update activity from JS component
    if activity_result:
        try:
            js_last_activity_timestamp = activity_result.get('last_activity')
            if js_last_activity_timestamp:
                new_activity_dt = datetime.fromtimestamp(js_last_activity_timestamp / 1000)
                
                if new_activity_dt > session.last_activity:
                    logger.debug(f"Updating last_activity for {session.session_id[:8]} from JS: {session.last_activity.strftime('%H:%M:%S')} -> {new_activity_dt.strftime('%H:%M:%S')}")
                    session.last_activity = new_activity_dt
                    session_manager._save_session_with_retry(session) 
        except Exception as e:
            logger.error(f"Error processing client JS activity timestamp for session {session.session_id[:8]}: {e}")

    # Calculate time since last activity
    time_since_activity = datetime.now() - session.last_activity
    minutes_inactive = time_since_activity.total_seconds() / 60
    
    logger.info(f"TIMEOUT CHECK: Session {session.session_id[:8]} | Inactive: {minutes_inactive:.1f}m | last_activity: {session.last_activity.strftime('%H:%M:%S')}")
    
    # Check if timeout duration has passed
    if minutes_inactive >= session_manager.get_session_timeout_minutes():
        logger.info(f"⏰ TIMEOUT DETECTED: {session.session_id[:8]} inactive for {minutes_inactive:.1f} minutes")
        
        # Perform CRM save if eligible
        if session_manager._is_crm_save_eligible(session, "timeout_auto_reload"):
            logger.info(f"💾 Performing emergency save before auto-reload for {session.session_id[:8]}")
            try:
                emergency_data = {
                    "session_id": session.session_id,
                    "reason": "timeout_auto_reload",
                    "timestamp": int(time.time() * 1000)
                }
                fastapi_url = 'https://fifi-beacon-fastapi-121263692901.europe-west4.run.app/emergency-save'
                response = requests.post(fastapi_url, json=emergency_data, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Emergency save sent to FastAPI successfully")
                else:
                    logger.warning(f"⚠️ FastAPI returned status {response.status_code}, using local fallback")
                    session_manager.zoho.save_chat_transcript_sync(session, "timeout_auto_reload_fallback")
                    session.timeout_saved_to_crm = True
            except Exception as e:
                logger.error(f"❌ Failed to send emergency save to FastAPI: {e}")
                try:
                    logger.info(f"🔄 Using local CRM save as fallback for timeout")
                    session_manager.zoho.save_chat_transcript_sync(session, "timeout_auto_reload_fallback")
                    session.timeout_saved_to_crm = True
                except Exception as save_e:
                    logger.error(f"❌ Local CRM save also failed: {save_e}")

        # Mark session as inactive
        session.active = False
        session.last_activity = datetime.now()
        try:
            session_manager.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to save session during timeout: {e}")
        
        # Clear Streamlit session state fully
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # NEW: Force welcome page state and timeout flag
        st.session_state['page'] = None
        st.session_state['session_expired'] = True
        
        # Show timeout message
        st.error("⏰ **Session Timeout**")
        st.info("Your session has expired due to 5 minutes of inactivity.")
        
        # TRIGGER BROWSER RELOAD using streamlit_js_eval
        if JS_EVAL_AVAILABLE:
            try:
                logger.info(f"🔄 Triggering browser reload for timeout")
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                st.stop()
            except Exception as e:
                logger.error(f"Browser reload failed during inactive session handling: {e}")
        
        st.info("🏠 Redirecting to home page...")
        st.rerun()
        st.stop()
        return True
    
    return False
def render_simplified_browser_close_detection(session_id: str):
    """Enhanced browser close detection with eligibility check."""
    if not session_id:
        return

    # NEW: Check if user is eligible for emergency save before setting up detection
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        logger.debug("No session manager available for browser close detection")
        return
            
    session = session_manager.db.load_session(session_id)
    if not session:
        logger.debug(f"No session found for browser close detection: {session_id[:8]}")
        return
            
    # Check if user is eligible for CRM save - if not, skip emergency save setup entirely
    if not session_manager._is_crm_save_eligible(session, "browser_close_check"):
        logger.info(f"🚫 Session {session_id[:8]} not eligible for CRM save - skipping browser close detection")
        return

    # Only set up emergency save for eligible users
    logger.info(f"✅ Setting up browser close detection for eligible session {session_id[:8]}")

    enhanced_close_js = f"""
    <script>
    (function() {{
        const sessionId = '{session_id}';
        const FASTAPI_URL = 'https://fifi-beacon-fastapi-121263692901.europe-west4.run.app/emergency-save';
        
        if (window.fifi_close_enhanced_initialized) return;
        window.fifi_close_enhanced_initialized = true;
        
        let saveTriggered = false;
        let isTabSwitching = false;
        
        console.log('🛡️ Enhanced browser close detection initialized for eligible user');
        
        // ... rest of the existing JavaScript code remains unchanged ...
        function performActualEmergencySave(reason) {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            console.log('🚨 Confirmed browser/tab close, sending emergency save:', reason);
            
            const emergencyData = JSON.stringify({{
                session_id: sessionId,
                reason: reason,
                timestamp: Date.now()
            }});
            
            // PRIMARY: Try navigator.sendBeacon to FastAPI
            if (navigator.sendBeacon) {{
                try {{
                    const sent = navigator.sendBeacon(
                        FASTAPI_URL,
                        new Blob([emergencyData], {{type: 'application/json'}})
                    );
                    if (sent) {{
                        console.log('✅ Emergency save beacon sent successfully to FastAPI');
                        return;
                    }} else {{
                        console.warn('⚠️ Beacon send returned false, trying fallback...');
                    }}
                }} catch (e) {{
                    console.error('❌ Beacon failed:', e);
                }}
            }}
            
            // FALLBACK 1: Try fetch with very short timeout
            try {{
                fetch(FASTAPI_URL, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: emergencyData,
                    keepalive: true,
                    signal: AbortSignal.timeout(3000)
                }}).then(response => {{
                    if (response.ok) {{
                        console.log('✅ Emergency save via fetch successful');
                    }} else {{
                        console.warn('⚠️ Fetch response not OK, status:', response.status);
                        redirectToStreamlitFallback(reason);
                    }}
                }}).catch(error => {{
                    console.error('❌ Fetch failed:', error);
                    redirectToStreamlitFallback(reason);
                }});
            }} {{ /* no catch here, handled by .catch in the promise chain */ }}
            
            // FALLBACK 2: Always redirect to Streamlit as final backup
            setTimeout(() => {{
                redirectToStreamlitFallback(reason);
            }}, 1000);
        }}
        
        function triggerEmergencySave(reason) {{
            if (isTabSwitching) {{
                console.log('🔍 Potential tab switch detected, delaying emergency save by 150ms...');
                setTimeout(() => {{
                    if (document.visibilityState === 'visible') {{
                        console.log('✅ Tab switch confirmed - CANCELING emergency save');
                        isTabSwitching = false;
                        return;
                    }}
                    console.log('🚨 Real exit confirmed after delay - proceeding with emergency save');
                    performActualEmergencySave(reason);
                }}, 150);
                return;
            }}
            
            performActualEmergencySave(reason);
        }}
        
        // Listen for visibility changes to track tab switching
        document.addEventListener('visibilitychange', function() {{
            if (document.visibilityState === 'hidden') {{
                console.log('📱 Tab became hidden - potential tab switch');
                isTabSwitching = true;
            }} else if (document.visibilityState === 'visible') {{
                console.log('👁️ Tab became visible - confirmed tab switch');
                isTabSwitching = false;
            }}
        }});
        
        // Listen for actual browser close events
        window.addEventListener('beforeunload', () => {{
            triggerEmergencySave('browser_close');
        }}, {{ capture: true, passive: true }});
        
        window.addEventListener('unload', () => {{
            triggerEmergencySave('browser_close');
        }}, {{ capture: true, passive: true }});
        
        // Try to monitor parent window as well
        try {{
            if (window.parent && window.parent !== window) {{
                window.parent.document.addEventListener('visibilitychange', function() {{
                    if (window.parent.document.visibilityState === 'hidden') {{
                        console.log('📱 Parent tab became hidden - potential tab switch');
                        isTabSwitching = true;
                    }} else if (window.parent.document.visibilityState === 'visible') {{
                        console.log('👁️ Parent tab became visible - confirmed tab switch');
                        isTabSwitching = false;
                    }}
                }});
                
                window.parent.addEventListener('beforeunload', () => {{
                    triggerEmergencySave('browser_close');
                }}, {{ capture: true, passive: true }});
            }}
        }} catch (e) {{
            console.debug('Cannot monitor parent events (cross-origin):', e);
        }}
        
        console.log('✅ Enhanced browser close detection ready');
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(enhanced_close_js, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close detection: {e}")

def process_fingerprint_from_query(session_id: str, fingerprint_id: str, method: str, privacy: str, working_methods: List[str]) -> bool:
    """Processes fingerprint data received via URL query parameters."""
    try:
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("❌ Session manager not available during fingerprint processing from query.")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Fingerprint processing: Session '{session_id[:8]}' not found in database.")
            return False
        
        logger.info(f"✅ Processing fingerprint for session '{session_id[:8]}': ID={fingerprint_id[:8]}, Method={method}, Privacy={privacy}")
        
        # The apply_fingerprinting method will now handle both setting the fingerprint
        # and checking for inheritance based on this new 'real' fingerprint.
        processed_data = {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': method,
            'browser_privacy_level': privacy,
            'working_methods': working_methods
            # visitor_type is determined by apply_fingerprinting after inheritance check
        }
        
        success = session_manager.apply_fingerprinting(session, processed_data)
        
        if success:
            logger.info(f"✅ Fingerprint applied successfully to session '{session_id[:8]}'")
            st.session_state.is_chat_ready = True # NEW: Explicitly unlock chat input here
            logger.info(f"Chat input unlocked for session {session_id[:8]} after successful JS fingerprinting.")
            return True
        else:
            logger.warning(f"⚠️ Fingerprint application failed for session '{session_id[:8]}'")
            return False
        
    except Exception as e:
        logger.error(f"Fingerprint processing failed: {e}", exc_info=True)
        return False

def process_emergency_save_from_query(session_id: str, reason: str) -> bool:
    """Processes emergency save request from query parameters."""
    try:
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("❌ Session manager not available during emergency save processing.")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Emergency save: Session '{session_id[:8]}' not found in database.")
            return False
        
        logger.info(f"🚨 Processing emergency save for session '{session_id[:8]}', reason: {reason}")
        
        if session_manager._is_crm_save_eligible(session, f"Emergency Save: {reason}"):
            success = session_manager.zoho.save_chat_transcript_sync(session, f"Emergency Save: {reason}")
            if success:
                session.timeout_saved_to_crm = True
                # NO: session.active = False here, as the FastAPI beacon handles it more robustly.
                # If this is a fallback save, the FastAPI will explicitly mark active=False.
                # If this is a normal save, Streamlit's end_session or timeout handler will mark active=False.
                session.last_activity = datetime.now()
                session_manager.db.save_session(session)
                logger.info(f"✅ Emergency save completed successfully for session {session_id[:8]}")
                return True
            else:
                logger.warning(f"⚠️ Emergency save CRM operation failed for session {session_id[:8]}")
                return False
        else:
            logger.info(f"ℹ️ Emergency save not eligible for CRM save for session {session_id[:8]}")
            return False
        
    except Exception as e:
        logger.error(f"Emergency save processing failed: {e}", exc_info=True)
        return False

def handle_emergency_save_requests_from_query():
    """Checks for and processes emergency save requests sent via URL query parameters."""
    logger.info("🔍 EMERGENCY SAVE HANDLER: Checking for query parameter requests for emergency save...")
    
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    reason = query_params.get("reason", "unknown")
    fallback = query_params.get("fallback", "false")

    if event == "emergency_close" and session_id:
        logger.info("=" * 80)
        logger.info("🚨 EMERGENCY SAVE REQUEST DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}, Reason: {reason}")
        if fallback == "true":
            logger.warning("⚠️ THIS IS A FALLBACK SAVE - FastAPI beacon likely failed!")
        logger.info("=" * 80)
        
        st.error("🚨 **Emergency Save Detected** - Processing browser close save...")
        if fallback == "true":
            st.warning("⚠️ Using backup save method (primary method failed)")
        st.info("Please wait, your conversation is being saved...")
        
        # Clear query parameters to prevent re-triggering on rerun
        params_to_clear = ["event", "session_id", "reason", "fallback"]
        for param in params_to_clear: # Fixed: changed to params_to_clear from params_query_params
            if param in st.query_params:
                del st.query_params[param]
        
        try:
            save_reason = f"{reason}_fallback" if fallback == "true" else reason
            # Call process_emergency_save_from_query which handles the actual saving and active status.
            # This is a Streamlit-level save, intended as a fallback if the beacon failed.
            success = process_emergency_save_from_query(session_id, save_reason)
            
            if success:
                st.success("✅ Emergency save completed successfully!")
                logger.info("✅ Emergency save completed via query parameter successfully.")
            else:
                st.info("ℹ️ Emergency save completed (no CRM save needed or failed).")
                logger.info("ℹ️ Emergency save completed via query parameter (not eligible for CRM save or internal error).")
                
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during emergency save: {str(e)}")
            logger.critical(f"Emergency save processing crashed from query parameter: {e}", exc_info=True)
        
        # Removed time.sleep(2)
        st.stop()
    else:
        logger.debug("ℹ️ No emergency save requests found in current URL query parameters.")

def handle_fingerprint_requests_from_query():
    """Checks for and processes fingerprint data sent via URL query parameters."""
    logger.info("🔍 FINGERPRINT HANDLER: Checking for query parameter fingerprint data...")
    
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event == "fingerprint_complete" and session_id:
        logger.info("=" * 80)
        logger.info("🔍 FINGERPRINT DATA DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}")
        logger.info("=" * 80)
        
        # EXTRACT PARAMETERS BEFORE CLEARING THEM
        fingerprint_id = query_params.get("fingerprint_id")
        method = query_params.get("method")
        privacy = query_params.get("privacy")
        working_methods = query_params.get("working_methods", "").split(",") if query_params.get("working_methods") else []
        
        logger.info(f"Extracted - ID: {fingerprint_id}, Method: {method}, Privacy: {privacy}, Working Methods: {working_methods}")
        
        # Clear query parameters AFTER extraction
        params_to_clear = ["event", "session_id", "fingerprint_id", "method", "privacy", "working_methods", "timestamp"]
        for param in params_to_clear:
            if param in st.query_params:
                del st.query_params[param]
        
        if not fingerprint_id or not method:
            st.error("❌ **Fingerprint Error** - Missing required data in redirect")
            logger.error(f"Missing fingerprint data: ID={fingerprint_id}, Method={method}")
            # Removed time.sleep(2)
            st.rerun()
            return
        
        try:
            success = process_fingerprint_from_query(session_id, fingerprint_id, method, privacy, working_methods)
            logger.info(f"✅ Silent fingerprint processing: {success}")
            
            if success:
                logger.info(f"🔄 Fingerprint processed successfully, stopping execution to preserve page state")
                st.stop()
        except Exception as e:
            logger.error(f"Silent fingerprint processing failed: {e}")
        
        return
    else:
        logger.debug("ℹ️ No fingerprint requests found in current URL query parameters.")

# =============================================================================
# UI COMPONENTS
# =============================================================================

# Modified render_welcome_page function (from prompt)
def render_welcome_page(session_manager: 'SessionManager'):
    """Enhanced welcome page with loading lock."""
    
    # Show loading overlay if in loading state
    if show_loading_overlay():
        return
    
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    st.markdown("---")
    
    # MOVED UP: Sign In/Start as Guest tabs (was previously below tiers)
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is currently disabled because the authentication service (WordPress URL) is not configured in application secrets.")
        else:
            with st.form("login_form", clear_on_submit=True):
                st.markdown("### 🔐 Sign In to Your Account")
                username = st.text_input("Username or Email", help="Enter your WordPress username or email.")
                password = st.text_input("Password", type="password", help="Enter your WordPress password.")
                
                st.markdown("")
                
                col1, col2, col3 = st.columns(3)
                with col2:
                    submit_button = st.form_submit_button("🔐 Sign In", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password to sign in.")
                    else:
                        # Store credentials temporarily and set loading state (NEW)
                        st.session_state.temp_username = username
                        st.session_state.temp_password = password
                        st.session_state.loading_reason = 'authenticate'
                        set_loading_state(True, "Authenticating and preparing your session...")
                        st.rerun()  # Immediately show loading state (NEW)
            
            st.markdown("---")
            st.info("Don't have an account? [Register here](https://www.12taste.com/in/my-account/) to unlock full features!")
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to get a quick start and try FiFi AI Assistant without signing in.
        
        ℹ️ **What to expect as a Guest:**
        - You get an initial allowance of **4 questions** to explore FiFi AI's capabilities.
        - After these 4 questions, **email verification will be required** to continue (unlocks 10 questions/day).
        - Our system utilizes **universal device fingerprinting** for security and to track usage across sessions.
        - You can always choose to **upgrade to a full registration** later for extended benefits.
        """)
        
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("👤 Start as Guest", use_container_width=True):
                # Set loading state and reason BEFORE any session operations (NEW)
                st.session_state.loading_reason = 'start_guest'
                set_loading_state(True, "Setting up your session and initializing AI assistant...")
                st.rerun()  # Immediately show loading state (NEW)

    # MOVED DOWN: Usage tiers explanation (was previously above tabs)
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown("• **4 questions** to try FiFi AI")
        st.markdown("• Email verification required to continue")
        st.markdown("• Quick start, no registration needed")
    
    with col2:
        st.info("📧 **Email Verified Guest**")
        st.markdown("• **10 questions per day** (rolling 24-hour period)")
        st.markdown("• Email verification for access")
        st.markdown("• No full registration required")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown("• **20 questions per day** with tier system:")
        st.markdown("  - **Tier 1**: Questions 1-10 → 1-hour break")
        st.markdown("  - **Tier 2**: Questions 11-20 → 24-hour reset")
        st.markdown("• Cross-device tracking & chat saving")
        st.markdown("• Priority access during high usage")
        
def render_sidebar(session_manager: 'SessionManager', session: UserSession, pdf_exporter: PDFExporter):
    """Enhanced sidebar with tier progression display."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        if session.user_type.value == UserType.REGISTERED_USER.value:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # ENHANCED: Show tier progression
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/20")
            
            if session.daily_question_count <= 10:
                st.progress(min(session.daily_question_count / 10, 1.0), text=f"Tier 1: {session.daily_question_count}/10 questions")
                remaining_tier1 = 10 - session.daily_question_count
                if remaining_tier1 > 0:
                    st.caption(f"⏰ {remaining_tier1} questions until 1-hour break")
                else:
                    st.caption("🚫 Tier 1 complete - 1 hour break required")
            else:
                tier2_progress = min((session.daily_question_count - 10) / 10, 1.0)
                st.progress(tier2_progress, text=f"Tier 2: {session.daily_question_count - 10}/10 questions")
                remaining_tier2 = 20 - session.daily_question_count
                if remaining_tier2 > 0:
                    st.caption(f"⏰ {remaining_tier2} questions until 24-hour reset")
                else:
                    st.caption("🚫 Daily limit reached - 24 hour reset required")
            
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(min(session.daily_question_count / 10, 1.0))
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Daily questions have reset!")
            
        else: # Guest User
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(min(session.daily_question_count / 4, 1.0))
            st.caption("Email verification unlocks 10 questions/day.")
            if session.reverification_pending:
                st.info("💡 An account is available for this device. Re-verify email to reclaim it!")
            elif session.declined_recognized_email_at and session.daily_question_count < session_manager.question_limits.question_limits[UserType.GUEST.value]: # Check for this state
                st.info("💡 You are currently using guest questions. Verify your email to get more.") # Alternative message
                
        # Show fingerprint status
        if session.fingerprint_id:
            if session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                st.markdown("**Device ID:** Identifying...")
                st.caption("Fingerprinting in progress...")                
            else:
                st.markdown(f"**Device ID:** `{session.fingerprint_id[:12]}...`")
                st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        else:
            st.markdown("**Device ID:** Initializing...")
            st.caption("Starting fingerprinting...")

        # Display time since last activity
        if session.last_activity is not None:
            time_since_activity = datetime.now() - session.last_activity
            minutes_inactive = time_since_activity.total_seconds() / 60
            st.caption(f"Last activity: {int(minutes_inactive)} minutes ago")
            
            timeout_duration = session_manager.get_session_timeout_minutes()

            if minutes_inactive >= (timeout_duration - 1) and minutes_inactive < timeout_duration:
                minutes_remaining = timeout_duration - minutes_inactive
                # REMOVED: st.warning(f"⏰ Session expires in: {int(minutes_remaining)}m") # Removed the specific reminder note
            elif minutes_inactive >= timeout_duration:
                st.error(f"🚫 Session is likely expired. Type a question to check.")
        else:
            st.caption("Session timer will start with first interaction.")

        # AI Tools Status
        st.divider()
        st.markdown("**🤖 AI Tools Status**")
        
        ai_system = session_manager.ai
        if ai_system:
            # Enhanced component status display
            pinecone_status = error_handler.component_status.get("Pinecone", "healthy")
            tavily_status = error_handler.component_status.get("Tavily", "healthy")
            
            # Pinecone status with health indicator
            if ai_system.pinecone_tool and ai_system.pinecone_tool.assistant:
                if pinecone_status == "healthy":
                    st.success("🧠 Knowledge Base: Ready")
                elif pinecone_status in ["rate_limit"]:
                    st.warning("🧠 Knowledge Base: Rate Limited")
                else:
                    st.error(f"🧠 Knowledge Base: {pinecone_status.replace('_', ' ').title()}")
            elif ai_system.config.PINECONE_API_KEY:
                st.warning("🧠 Knowledge Base: Error")
            else:
                st.info("🧠 Knowledge Base: Not configured")
            
            # Tavily status with health indicator
            if ai_system.tavily_agent:
                if tavily_status == "healthy":
                    st.success("🌐 Web Search: Ready")
                else:
                    st.warning(f"🌐 Web Search: {tavily_status.replace('_', ' ').title()}")
            elif ai_system.config.TAVILY_API_KEY:
                st.warning("🌐 Web Search: Error")
            else:
                st.info("🌐 Web Search: Not configured")
            
            # OpenAI status
            if ai_system.openai_client:
                st.success("💬 OpenAI: Ready")
            elif ai_system.config.OPENAI_API_KEY:
                st.warning("💬 OpenAI: Error")
            else:
                st.info("💬 OpenAI: Not configured")
        else:
            st.error("🤖 AI System: Not available")
        
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready** (will link on first save)")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM (after inactivity)")
            else:
                st.caption("💾 Auto-save enabled (on sign out or browser/tab close)")
        else: 
            st.caption("🚫 CRM Integration: Registered users & verified guests only")
        
        st.divider()
        
        # Show total messages in session
        total_messages = len(session.messages)
        visible_messages = len(session.messages) - session.display_message_offset
        
        if session.display_message_offset > 0:
            st.markdown(f"**Messages in Chat:** {visible_messages} (Total: {total_messages})")
            st.caption(f"💡 {session.display_message_offset} messages hidden by Clear Chat")
        else:
            st.markdown(f"**Messages in Chat:** {total_messages}")
            
        st.markdown(f"**Current Session ID:** `{session.session_id[:8]}...`")
        
        if session.ban_status.value != BanStatus.NONE.value:
            st.error(f"🚫 **STATUS: RESTRICTED**")
            if session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.markdown(f"**Time Remaining:** {hours}h {minutes}m")
            st.markdown(f"Reason: {session.ban_reason or 'Usage policy violation'}")
        elif session.question_limit_reached and session.user_type.value == UserType.GUEST.value: 
            st.warning("⚠️ **ACTION REQUIRED: Email Verification**")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            clear_chat_help = "Hides all messages from the current conversation display. Messages are preserved in the database and new messages can still be added."
            
            if st.button("🗑️ Clear Chat", use_container_width=True, help=clear_chat_help):
                session_manager.clear_chat_history(session)
                st.success("🗑️ Chat display cleared! Messages preserved in database.")
                st.rerun()
                
        with col2:
            signout_help = "Ends your current session and returns to the welcome page."
            if (session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and # FIXED: Corrected to .value
                session.email and session.messages and session.daily_question_count >= 1):
                signout_help += " Your conversation will be automatically saved to CRM before signing out."
            
            if st.button("🚪 Sign Out", use_container_width=True, help=signout_help):
                session_manager.end_session(session)
                st.rerun()

        if session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and session.messages:
            st.divider()
            
            pdf_buffer = pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download Chat PDF",
                    data=pdf_buffer,
                    file_name=f"fifi_chat_transcript_{session.session_id[:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Download the current chat conversation as a PDF document."
                )
            
            if session_manager.zoho.config.ZOHO_ENABLED and session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, help="Manually save your current chat transcript to your linked Zoho CRM contact."):
                    session_manager.manual_save_to_crm(session)
                st.caption("💡 Chat automatically saves to CRM during Sign Out or browser/tab close.")

# NEW FUNCTION FOR EMAIL PROMPT MANAGEMENT
def display_email_prompt_if_needed(session_manager: 'SessionManager', session: UserSession) -> bool:
    """
    Renders email verification dialog if needed.
    Controls `st.session_state.chat_blocked_by_dialog` and returns if chat input should be disabled.
    """
    
    # Initialize relevant session states if not present
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = None
    if 'guest_continue_active' not in st.session_state:
        st.session_state.guest_continue_active = False

    # Check if a hard block is in place first (non-email-verification related bans)
    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed'] and limit_check.get('reason') != 'guest_limit':
        st.session_state.chat_blocked_by_dialog = True # Hard ban, block everything
        return True # Disable chat input

    # Determine if an email prompt *should* be active
    # --- FIX APPLIED HERE: Compare .value of the enum ---
    user_is_guest = (session.user_type.value == UserType.GUEST.value)
    # --- END FIX ---
    guest_limit_value = session_manager.question_limits.question_limits[UserType.GUEST.value]
    daily_q_value = session.daily_question_count
    daily_q_ge_limit = (daily_q_value >= guest_limit_value)

    logger.debug(f"DEBUG_PROMPT_EVAL_COMPONENTS: SessionID={session.session_id[:8]} | IsGuest={user_is_guest} | DailyQ={daily_q_value} | GuestLimit={guest_limit_value} | DailyQ>=Limit={daily_q_ge_limit}")

    is_guest_limit_hit = (user_is_guest and daily_q_ge_limit) # Use the explicitly evaluated component

    logger.debug(f"DEBUG: display_email_prompt_if_needed: session_id={session.session_id[:8]} | user_type={session.user_type.value} | daily_q={session.daily_question_count} | is_guest_limit_hit={is_guest_limit_hit}")

    should_show_prompt = False

    if session.reverification_pending:
        should_show_prompt = True
        if st.session_state.verification_stage is None: # Initial entry to re-verification
             st.session_state.verification_stage = 'initial_check'
             st.session_state.guest_continue_active = False # Clear any previous guest-continue state
    elif is_guest_limit_hit:
        should_show_prompt = True
        if st.session_state.verification_stage is None or st.session_state.verification_stage == 'declined_recognized_email_prompt_only':
            st.session_state.verification_stage = 'email_entry' # Force email entry if limit hit
            st.session_state.guest_continue_active = False # Clear any previous guest-continue state
    elif session.declined_recognized_email_at and not st.session_state.guest_continue_active:
        # This is the scenario where user declined recognized email, has questions left, but hasn't explicitly
        # chosen "Continue as Guest for Now" in this session state.
        should_show_prompt = True
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'declined_recognized_email_prompt_only'

    # If no prompt should be shown based on conditions, ensure state is clean
    if not should_show_prompt:
        st.session_state.chat_blocked_by_dialog = False
        st.session_state.verification_stage = None # Ensure stage is cleared
        return False # No prompt, chat input enabled

    # If a prompt should be shown, set chat_blocked_by_dialog to True
    st.session_state.chat_blocked_by_dialog = True
    st.error("📧 **Action Required**") # Display prompt header

    current_stage = st.session_state.verification_stage
    disable_chat_input = True # Default to disabling chat input while prompt is active

    if current_stage == 'initial_check':
        email_to_reverify = session.pending_email
        masked_email = session_manager._mask_email(email_to_reverify) if email_to_reverify else "your registered email"
        st.info(f"🤝 **We recognize this device was previously used as a {session.pending_user_type.value.replace('_', ' ').title()} account.**")
        st.info(f"Please verify **{masked_email}** to reclaim your status and higher question limits.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Verify this email", use_container_width=True, key="reverify_yes_btn"):
                session.recognition_response = "yes_reverify"
                session.declined_recognized_email_at = None
                st.session_state.verification_email = email_to_reverify
                st.session_state.verification_stage = "send_code_recognized"
                session_manager.db.save_session(session) # Save response
                st.rerun()
        with col2:
            if st.button("❌ No, I don't recognize the email", use_container_width=True, key="reverify_no_btn"):
                session.recognition_response = "no_declined_reco"
                session.declined_recognized_email_at = datetime.now()
                session.user_type = UserType.GUEST 
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                session_manager.db.save_session(session) # Persist this decision
                st.session_state.guest_continue_active = True # Allow continuing as guest now
                st.session_state.chat_blocked_by_dialog = False # UNBLOCK CHAT
                st.session_state.verification_stage = None # Clear stage to dismiss dialog
                st.success("You can now continue as a Guest.")
                st.rerun()

    elif current_stage == 'send_code_recognized':
        email_to_verify = st.session_state.get('verification_email')
        if email_to_verify:
            with st.spinner(f"Sending verification code to {session_manager._mask_email(email_to_verify)}..."):
                result = session_manager.handle_guest_email_verification(session, email_to_verify)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.verification_stage = "code_entry"
                else:
                    st.error(result['message'])
                    if "unusual activity" in result['message'].lower(): st.stop()
                    st.session_state.verification_stage = "email_entry" # Fallback to manual entry if send fails
            st.rerun()

    elif current_stage == 'email_entry':
        st.info("You've used your 4 free questions. Please verify your email to unlock 10 questions per day.")
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**Please enter your email address to receive a verification code:**")
            current_email_input = st.text_input("Email Address", placeholder="your@email.com", value=st.session_state.get('verification_email', session.email or ""), key="manual_email_input")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if current_email_input:
                    result = session_manager.handle_guest_email_verification(session, current_email_input)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = current_email_input
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                        if "unusual activity" in result['message'].lower(): st.stop()
                else:
                    st.error("Please enter an email address to receive the code.")
        
    elif current_stage == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        st.success(f"📧 A verification code has been sent to **{session_manager._mask_email(verification_email)}**.")
        st.info("Please check your email, including spam/junk folders. The code is valid for 1 minute.")
        
        with st.form("code_verification_form", clear_on_submit=False):
            code = st.text_input("Enter Verification Code", placeholder="e.g., 123456", max_chars=6, key="verification_code_input")
            col_code1, col_code2 = st.columns(2)
            with col_code1:
                submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            with col_code2:
                resend_code = st.form_submit_button("🔄 Resend Code", use_container_width=True)
            
            if resend_code:
                if verification_email:
                    with st.spinner("Resending code..."):
                        verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                        if verification_sent:
                            st.success("Verification code resent successfully!")
                            st.session_state.verification_stage = "code_entry"
                        else:
                            st.error("Failed to resend code. Please try again later.")
                else:
                    st.error("Error: No email address found to resend the code. Please go back and enter your email.")
                    st.session_state.verification_stage = "email_entry"
                st.rerun()

            if submit_code:
                if code:
                    with st.spinner("Verifying code..."):
                        result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        # Clean up verification state on success
                        st.session_state.chat_blocked_by_dialog = False # UNBLOCK CHAT
                        st.session_state.verification_stage = None
                        st.session_state.guest_continue_active = False # Clear this flag too
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code you received.")
    
    elif current_stage == 'declined_recognized_email_prompt_only':
        # This state happens when a user declines a recognized email but still has guest questions left.
        # It's a non-blocking prompt, meaning chat input should remain active.
        disable_chat_input = False # ALLOW CHAT INPUT
        st.session_state.chat_blocked_by_dialog = False # UNBLOCK CHAT (as it's non-blocking visual prompt)

        remaining_questions = session_manager.question_limits.question_limits[UserType.GUEST.value] - session.daily_question_count
        st.info(f"You chose not to verify the recognized email. You can still use your remaining **{remaining_questions} guest questions**.")
        st.info("To ask more questions after this, or to save chat history, please verify your email.")

        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            if st.button("📧 Enter a New Email for Verification", use_container_width=True, key="new_email_opt_btn"):
                st.session_state.verification_email = "" # Clear pre-filled email
                st.session_state.verification_stage = "email_entry"
                st.session_state.guest_continue_active = False # Reset if they change mind and go for new email
                st.rerun()
        with col_opts2:
            if st.button("Continue as Guest for Now", use_container_width=True, key="continue_guest_btn"):
                st.session_state.guest_continue_active = True
                st.session_state.chat_blocked_by_dialog = False # Ensure UNBLOCKING
                st.session_state.verification_stage = None # Clear stage to dismiss dialog
                st.success("You can now continue as a Guest. The email prompt will reappear when your guest questions run out.")
                st.rerun()

    return disable_chat_input

def render_chat_interface_simplified(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]):
    """Chat interface with enhanced tier system notifications."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")

    # NEW: Show fingerprint waiting status
    if not st.session_state.get('is_chat_ready', False) and st.session_state.get('fingerprint_wait_start'):
        current_time_float = time.time() # Use float for direct comparison with time.time()
        wait_start = st.session_state.get('fingerprint_wait_start')
        elapsed = current_time_float - wait_start
        remaining = max(0, 10 - elapsed)
        
        if remaining > 0:
            st.info(f"🔒 **Securing your session...** ({remaining:.0f}s remaining)")
            st.caption("FiFi is setting up device recognition for security and session management.")
        else:
            st.info("🔒 **Finalizing setup...** Almost ready!")
        
        # Add a subtle progress bar
        progress_value = min(elapsed / 10, 1.0)
        st.progress(progress_value, text="Session Security Setup")
        st.markdown("---")

    # Simple activity tracking
    if activity_result:
        js_last_activity_timestamp = activity_result.get('last_activity')
        if js_last_activity_timestamp:
            try:
                new_activity = datetime.fromtimestamp(js_last_activity_timestamp / 1000)
                if session.last_activity is None or new_activity > session.last_activity:
                    session.last_activity = new_activity
                    session_manager._save_session_with_retry(session)
            except Exception as e:
                logger.error(f"Failed to update activity from JavaScript: {e}")

    # Browser close detection for emergency saves
    if session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]:
        try:
            render_simplified_browser_close_detection(session.session_id)
        except Exception as e:
            logger.error(f"Browser close detection failed: {e}")

    # Display email prompt if needed AND get status to disable chat input
    should_disable_chat_input_by_dialog = display_email_prompt_if_needed(session_manager, session)

    # Render chat content ONLY if not blocked by a dialog
    if not st.session_state.get('chat_blocked_by_dialog', False):
        # ENHANCED: Show tier warnings for registered users
        # Note: I've also updated the `is_within_limits` calls to use `.get('allowed')` properly
        # and added `.value` for Enum comparisons for consistency and robustness.
        limit_check_for_display = session_manager.question_limits.is_within_limits(session)
        if (session.user_type.value == UserType.REGISTERED_USER.value and 
            limit_check_for_display.get('allowed') and 
            limit_check_for_display.get('tier')):
            
            tier = limit_check_for_display.get('tier')
            remaining = limit_check_for_display.get('remaining', 0)
            
            if tier == 2 and remaining <= 3:
                st.warning(f"⚠️ **Tier 2 Alert**: Only {remaining} questions remaining until 24-hour reset!")
            elif tier == 1 and remaining <= 2:
                st.info(f"ℹ️ **Tier 1**: {remaining} questions remaining until 1-hour break.")

        # Display chat messages (respects soft clear offset)
        visible_messages = session.messages[session.display_message_offset:]
        for msg in visible_messages:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""), unsafe_allow_html=True)
                
                if msg.get("source"):
                    source_color = {
                        "FiFi": "🧠", "FiFi Web Search": "🌐", 
                        "Content Moderation": "🛡️", "System Fallback": "⚠️",
                        "Error Handler": "❌"
                    }.get(msg['source'], "🤖")
                    st.caption(f"{source_color} Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"): indicators.append("🧠 Knowledge Base")
                if msg.get("used_search"): indicators.append("🌐 Web Search")
                if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")
                
                if msg.get("safety_override"):
                    st.warning("🛡️ Safety Override: Switched to verified sources")
                
                if msg.get("has_citations") and msg.get("has_inline_citations"):
                    st.caption("📚 Response includes verified citations")
                
                # Check for post-response Tier 1 ban notification (for Registered Users only)
                if msg.get('role') == 'assistant' and msg.get('tier1_ban_applied_post_response', False):
                    st.warning("⚠️ **Tier 1 Limit Reached:** You've asked 10 questions. A 1-hour break is now required. You can resume chatting after this period.")
                    st.markdown("---") # Visual separator

    # Chat input section with inline error notifications + manual dismiss
    # MODIFIED: Lock chat input until st.session_state.is_chat_ready is True
    # And combine with other disabled conditions
    overall_chat_disabled = (
        not st.session_state.get('is_chat_ready', False) or 
        should_disable_chat_input_by_dialog or 
        session.ban_status.value != BanStatus.NONE.value
    )

    # Rate limit notification with manual dismiss
    if 'rate_limit_hit' in st.session_state:
        rate_limit_info = st.session_state.rate_limit_hit
        time_until_next = rate_limit_info.get('time_until_next', 0)
        max_requests = rate_limit_info.get('max_requests', 2)
        window_seconds = rate_limit_info.get('window_seconds', 60)
        
        # Calculate remaining time dynamically
        current_time = datetime.now()
        elapsed = (current_time - rate_limit_info['timestamp']).total_seconds()
        remaining_time = max(0, int(time_until_next - elapsed))
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if remaining_time > 0:
                st.error(f"⏱️ **Rate limit exceeded** - Please wait {remaining_time} seconds before asking another question. ({max_requests} questions per {window_seconds} seconds allowed)")
            else:
                st.error(f"⏱️ **Rate limit exceeded** - Please wait a moment before asking another question.")
        with col2:
            if st.button("✕", key="dismiss_rate_limit", help="Dismiss this message", use_container_width=True):
                del st.session_state.rate_limit_hit
                st.rerun()

    # Content moderation notification with manual dismiss
    if 'moderation_flagged' in st.session_state:
        moderation_info = st.session_state.moderation_flagged
        categories = moderation_info.get('categories', [])
        categories_text = ', '.join(categories) if categories else 'policy violation'
        message = moderation_info.get('message', 'Your message violates our content policy.')
        
        col1, col2 = st.columns([5, 1])
        with col1:
            st.error(f"🛡️ **Content Policy Violation** - {categories_text}")
            st.info(f"💡 **Guidance**: {message}")
        with col2:
            if st.button("✕", key="dismiss_moderation", help="Dismiss this message", use_container_width=True):
                del st.session_state.moderation_flagged
                st.rerun()

    # Context error notification with manual dismiss
    if 'context_flagged' in st.session_state:
        context_info = st.session_state.context_flagged
        category = context_info.get('category', 'off-topic')
        confidence = context_info.get('confidence', 0.0)
        message = context_info.get('message', '')
        
        col1, col2 = st.columns([5, 1])
        with col1:
            if category == "unrelated_industry":
                st.warning(f"🏭 **Outside Food Industry** - This question doesn't relate to food & beverage ingredients.")
            elif category in ["personal_cooking", "off_topic"]:
                st.warning(f"👨‍🍳 **Personal vs Professional** - I'm designed for B2B food industry questions.")
            else:
                st.warning(f"🎯 **Off-Topic Question** - Please ask about food ingredients, suppliers, or formulation.")
            
            st.info(f"💡 **Guidance**: {message}")
            st.caption(f"Confidence: {confidence:.1%} | Category: {category}")
        with col2:
            if st.button("✕", key="dismiss_context", help="Dismiss this message", use_container_width=True):
                del st.session_state.context_flagged
                st.rerun()

    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                            disabled=overall_chat_disabled)
    
    if prompt:
        logger.info(f"🎯 Processing question from {session.session_id[:8]}")
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue.")
                        # This should be handled by display_email_prompt_if_needed on next rerun
                        st.session_state.verification_stage = 'email_entry' 
                        st.session_state.chat_blocked_by_dialog = True # Force block chat
                        st.rerun()
                    elif response.get('banned'):
                        st.error(response.get("content", 'Access restricted.'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                        st.rerun()
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            source_color = {
                                "FiFi": "🧠", "FiFi Web Search": "🌐",
                                "Content Moderation": "🛡️", "System Fallback": "⚠️",
                                "Error Handler": "❌"
                            }.get(response['source'], "🤖")
                            st.caption(f"{source_color} Source: {response['source']}")
                        
                        logger.info(f"✅ Question processed successfully")
                        
                        # Only re-run if a ban was just applied post-response to update UI (disable input, show message)
                        if response.get('tier1_ban_applied_post_response', False):
                            logger.info(f"Rerunning to show Tier 1 ban for session {session.session_id[:8]}")
                            st.rerun()
                        
                except Exception as e:
                    logger.error(f"❌ AI response failed: {e}", exc_info=True)
                    st.error("⚠️ I encountered an error. Please try again.")
        
        st.rerun()
# Modified ensure_initialization_fixed to not show spinner (since we have overlay) (from prompt)
def ensure_initialization_fixed():
    """Fixed version without duplicate spinner since we have loading overlay"""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence (no spinner shown, overlay is active)...")
        
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            try:
                db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
                st.session_state.db_manager = db_manager
            except Exception as db_e:
                logger.error(f"Database manager initialization failed: {db_e}", exc_info=True)
                st.session_state.db_manager = type('FallbackDB', (), {
                    'db_type': 'memory',
                    'local_sessions': {},
                    'save_session': lambda self, session: None,
                    'load_session': lambda self, session_id: None,
                    'find_sessions_by_fingerprint': lambda self, fingerprint_id: [],
                    'find_sessions_by_email': lambda self, email: []
                })()
            
            try:
                zoho_manager = ZohoCRMManager(config, pdf_exporter)
            except Exception as e:
                logger.error(f"Zoho manager failed: {e}")
                zoho_manager = type('FallbackZoho', (), {
                    'config': config,
                    'save_chat_transcript_sync': lambda self, session, reason: False
                })()
            
            try:
                ai_system = EnhancedAI(config)
            except Exception as e:
                logger.error(f"AI system failed: {e}")
                ai_system = type('FallbackAI', (), {
                    "openai_client": None,
                    'get_response': lambda self, prompt, history=None: {
                        "content": "AI system temporarily unavailable.",
                        "success": False
                    }
                })()
            
            rate_limiter = RateLimiter(max_requests=2, window_seconds=60)
            fingerprinting_manager = st.session_state.db_manager.FingerprintingManager()
            
            try:
                email_verification_manager = st.session_state.db_manager.EmailVerificationManager(config)
                if hasattr(email_verification_manager, 'supabase') and not email_verification_manager.supabase:
                    email_verification_manager = type('DummyEmail', (), {
                        'send_verification_code': lambda self, email: False,
                        'verify_code': lambda self, email, code: False
                    })()
            except Exception as e:
                logger.error(f"Email verification failed: {e}")
                email_verification_manager = type('DummyEmail', (), {
                    'send_verification_code': lambda self, email: False,
                    'verify_code': lambda self, email, code: False
                })()
            
            question_limit_manager = st.session_state.db_manager.QuestionLimitManager()
            
            st.session_state.session_manager = SessionManager(
                config, st.session_state.db_manager, zoho_manager, ai_system, 
                rate_limiter, fingerprinting_manager, email_verification_manager, 
                question_limit_manager
            )
            
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager

            st.session_state.chat_blocked_by_dialog = False
            st.session_state.verification_stage = None
            st.session_state.guest_continue_active = False
            # NEW: Initialize chat readiness flag
            st.session_state.is_chat_ready = False 
            
            st.session_state.initialized = True
            logger.info("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Critical initialization failure: {e}", exc_info=True)
            st.session_state.initialized = False
            return False
    
    return True

# Modified main function with proper loading state handling (from prompt)
def main_fixed():
    """Main entry point with loading state management"""
    try:
        st.set_page_config(
            page_title="FiFi AI Assistant", 
            page_icon="🤖", 
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {e}")

    # NEW: Check for expired session flag and force welcome page
    if st.session_state.get('session_expired', False):
        logger.info("Session expired flag detected - forcing welcome page")
        st.session_state['page'] = None
        if 'session_expired' in st.session_state:
            del st.session_state['session_expired']
        st.info("⏰ Your session expired. Please start a new session.")

    # Initialize loading state if not already set (for first run)
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
        st.session_state.loading_message = ""
    # NEW: Ensure is_chat_ready is always present and initially False
    if 'is_chat_ready' not in st.session_state:
        st.session_state.is_chat_ready = False


    # Handle loading states first
    loading_state = st.session_state.get('is_loading', False)
    current_page = st.session_state.get('page')
    
    # If we're in loading state, handle the actual initialization
    if loading_state:
        # Show the loading overlay
        if show_loading_overlay():
            pass  # Overlay is shown
        
        # Perform the actual operations based on what triggered the loading
        try:
            # Initialize if not already done
            if not st.session_state.get('initialized', False):
                init_success = ensure_initialization_fixed()  # Remove the spinner wrapper since we have overlay
                if not init_success:
                    set_loading_state(False)
                    st.error("⚠️ Application failed to initialize properly.")
                    st.info("Please refresh the page to try again.")
                    return
            
            session_manager = st.session_state.get('session_manager')
            if not session_manager:
                set_loading_state(False)
                st.error("❌ Session Manager not available. Please refresh the page.")
                return
            
            # Handle different loading scenarios
            loading_reason = st.session_state.get('loading_reason', 'unknown')
            
            session = None # Initialize session to None for scope
            if loading_reason == 'start_guest':
                # Create guest session
                session = session_manager.get_session()
                if session and session.last_activity is None:
                    session.last_activity = datetime.now()
                    session_manager.db.save_session(session)
                st.session_state.page = "chat"
                if 'loading_reason' in st.session_state:
                    del st.session_state['loading_reason']
                
            elif loading_reason == 'authenticate':
                # Handle authentication (you'll need to store username/password temporarily)
                username = st.session_state.get('temp_username', '')
                password = st.session_state.get('temp_password', '')
                
                if username and password:
                    authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                    if authenticated_session:
                        session = authenticated_session # Assign to session variable
                        st.session_state.current_session_id = authenticated_session.session_id
                        st.session_state.page = "chat"
                        # Clear temporary credentials
                        if 'temp_username' in st.session_state:
                            del st.session_state['temp_username']
                        if 'temp_password' in st.session_state:
                            del st.session_state['temp_password']
                        if 'loading_reason' in st.session_state:
                            del st.session_state['loading_reason']
                        st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                        st.balloons()
                    else:
                        set_loading_state(False)
                        # Error message already shown by authenticate_with_wordpress
                        return
                else:
                    set_loading_state(False)
                    st.error("Authentication failed: Missing username or password.")
                    return

            # NEW: Logic to unlock chat input after session is created/authenticated and initial fingerprint check
            if session:
                # Check if the session has a *stable* fingerprint (not a temporary Python fallback 
                # that's still waiting for the JS component to return its data).
                # The `fingerprint_checked_for_inheritance_{session.session_id}` flag, set by get_session(),
                # indicates that the initial inheritance/fingerprint check has occurred.
                fingerprint_is_stable = not session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
                inheritance_checked = st.session_state.get(f'fingerprint_checked_for_inheritance_{session.session_id}', False)

                if fingerprint_is_stable or inheritance_checked:
                    st.session_state.is_chat_ready = True
                    logger.info(f"Chat input unlocked for session {session.session_id[:8]} after initial session/fingerprint setup.")
                else:
                    # If it's still a temporary fingerprint and inheritance check hasn't happened yet, 
                    # keep chat locked. This state implies the JS fingerprint component is expected to run.
                    st.session_state.is_chat_ready = False
                    logger.info(f"Chat input remains locked for session {session.session_id[:8]} pending JS fingerprinting.")
            else:
                st.session_state.is_chat_ready = False # Ensure locked if no session obtained


            # Clear loading state and rerun to show the actual page
            set_loading_state(False)
            st.rerun()
            return
            
        except Exception as e:
            set_loading_state(False)
            st.error(f"⚠️ Error during loading: {str(e)}")
            logger.error(f"Loading state error: {e}", exc_info=True)
            return

    # Normal page rendering (when not in loading state)
    try:
        # Initialize if needed (without loading overlay since it's already done or not triggered)
        if not st.session_state.get('initialized', False):
            # This path handles initial load when no button was pressed, or if initialization failed.
            with st.spinner("Initializing application..."): # Keep a fallback spinner here for initial page load if loading_state was false but init wasn't complete.
                 init_success = ensure_initialization_fixed()
            if not init_success:
                st.error("⚠️ Application failed to initialize properly.")
                st.info("Please refresh the page to try again.")
                return

        # Handle emergency saves and fingerprint data
        handle_emergency_save_requests_from_query()
        handle_fingerprint_requests_from_query()

        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            st.error("❌ Session Manager not available. Please refresh the page.")
            return

        # Route to appropriate page
        if current_page != "chat":
            render_welcome_page(session_manager)
        else:
            # Get existing session (should already exist from loading state or prior direct creation)
            session = session_manager.get_session()
            
            if session is None or not session.active:
                logger.warning(f"Expected active session for 'chat' page but got None or inactive. Forcing welcome page.")
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state['page'] = None
                st.rerun()
                return
            
            # 🔥 RENDER FINGERPRINTING FIRST, BEFORE TIMEOUT LOGIC
            fingerprint_needed = (
                session is not None and (
                    not session.fingerprint_id or
                    session.fingerprint_method == "temporary_fallback_python" or
                    session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
                )
            )
            
            if fingerprint_needed:
                fingerprint_key = f"fingerprint_rendered_{session.session_id}"
                if not st.session_state.get(fingerprint_key, False):
                    session_manager.fingerprinting.render_fingerprint_component(session.session_id)
                    st.session_state[fingerprint_key] = True

            # 🔥 NOW DO TIMEOUT LOGIC AFTER JAVASCRIPT IS RENDERED
            if session:
                fingerprint_is_stable = not session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
                
                if fingerprint_is_stable:
                    # Real fingerprint already obtained, enable chat immediately
                    st.session_state.is_chat_ready = True
                    if 'fingerprint_wait_start' in st.session_state:
                        del st.session_state['fingerprint_wait_start']  # Clear timeout
                else:
                    # Still waiting for JS fingerprinting
                    current_time_float = time.time() # Use float for direct comparison with time.time()
                    wait_start = st.session_state.get('fingerprint_wait_start')
                    
                    if wait_start is None:
                        # First time seeing temp fingerprint, start timeout
                        st.session_state.fingerprint_wait_start = current_time_float
                        st.session_state.is_chat_ready = False
                        logger.info(f"Starting fingerprint wait timer for session {session.session_id[:8]}")
                    elif current_time_float - wait_start > 10:  # ✅ 10 seconds is reasonable
                        # Timeout reached, enable chat with fallback fingerprint
                        st.session_state.is_chat_ready = True
                        logger.warning(f"Fingerprint timeout (10s) - enabling chat with fallback for session {session.session_id[:8]}")
                    else:
                        # Still waiting within timeout period
                        st.session_state.is_chat_ready = False
                        # remaining = 5 - (current_time_float - wait_start) # Original value, if needed
                        # logger.debug(f"Fingerprint wait continues: {remaining:.1f}s remaining for session {session.session_id[:8]}")

            else:
                st.session_state.is_chat_ready = False

            # Right after your timeout logic
            if not st.session_state.get('is_chat_ready', False) and st.session_state.get('fingerprint_wait_start'):
                # Removed the `remaining` calculation, as it's now handled in the toast message or info box directly.
                # Just rerun to keep the UI updating
                st.rerun() 
                return # Stop execution to allow rerun
            
            # Render activity tracker and check for timeout
            activity_data_from_js = None
            if session and session.session_id: 
                activity_tracker_key_state_flag = f'activity_tracker_component_rendered_{session.session_id.replace("-", "_")}'
                if activity_tracker_key_state_flag not in st.session_state or \
                   st.session_state.get(f'{activity_tracker_key_state_flag}_session_id_check') != session.session_id:
                    
                    logger.info(f"Rendering activity tracker component for session {session.session_id[:8]} at top level.")
                    activity_data_from_js = render_simple_activity_tracker(session.session_id)
                    st.session_state[activity_tracker_key_state_flag] = True
                    st.session_state[f'{activity_tracker_key_state_flag}_session_id_check'] = session.session_id
                    st.session_state.latest_activity_data_from_js = activity_data_from_js
                else:
                    activity_data_from_js = st.session_state.latest_activity_data_from_js
            
            timeout_triggered = check_timeout_and_trigger_reload(session_manager, session, activity_data_from_js)
            if timeout_triggered:
                return

            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_simplified(session_manager, session, activity_data_from_js)

    except Exception as e:
        logger.error(f"Main application error: {e}", exc_info=True)
        st.error("⚠️ An unexpected error occurred. Please refresh the page.")
        st.info(f"Error details: {str(e)}")

if __name__ == "__main__":
    main_fixed()
