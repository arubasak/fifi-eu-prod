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
    level=logging.INFO,
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

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
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
    visitor_type: str = "new_visitor"
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

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
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
        with self.lock:
            try:
                if hasattr(self.conn, 'row_factory'): 
                    self.conn.row_factory = None

                # Create table with all columns upfront, including display_message_offset
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_type TEXT DEFAULT 'guest',
                        email TEXT,
                        full_name TEXT,
                        zoho_contact_id TEXT,
                        created_at TEXT DEFAULT '',
                        last_activity TEXT,
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
                        display_message_offset INTEGER DEFAULT 0
                    )
                ''')
                
                # Add display_message_offset column if it doesn't exist (for existing databases)
                try:
                    self.conn.execute("ALTER TABLE sessions ADD COLUMN display_message_offset INTEGER DEFAULT 0")
                    logger.info("✅ Added display_message_offset column to existing database")
                except Exception as alter_error:
                    logger.debug(f"ALTER TABLE for display_message_offset failed (likely already exists): {alter_error}")
                
                # Create essential indexes
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON sessions(session_id, active)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_id ON sessions(fingerprint_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON sessions(email)")
                
                self.conn.commit()
                logger.info("✅ Database schema ready and indexes created.")
                
            except Exception as e:
                logger.error(f"Database initialization failed: {e}", exc_info=True)
                raise

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with SQLite Cloud compatibility"""
        with self.lock:
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                return
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                if not isinstance(session.messages, list):
                    session.messages = []
                
                try:
                    json_messages = json.dumps(session.messages)
                    json_emails_used = json.dumps(session.email_addresses_used)
                except (TypeError, ValueError):
                    json_messages = "[]"
                    json_emails_used = "[]"
                    session.messages = []
                    session.email_addresses_used = []
                
                last_activity_iso = session.last_activity.isoformat() if session.last_activity else None

                self.conn.execute(
                    '''REPLACE INTO sessions (session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session.session_id, session.user_type.value, session.email, session.full_name,
                     session.zoho_contact_id, session.created_at.isoformat(),
                     last_activity_iso, json_messages, int(session.active),
                     session.wp_token, int(session.timeout_saved_to_crm), session.fingerprint_id,
                     session.fingerprint_method, session.visitor_type, session.daily_question_count,
                     session.total_question_count, 
                     session.last_question_time.isoformat() if session.last_question_time else None,
                     int(session.question_limit_reached), session.ban_status.value,
                     session.ban_start_time.isoformat() if session.ban_start_time else None,
                     session.ban_end_time.isoformat() if session.ban_end_time else None,
                     session.ban_reason, session.evasion_count, session.current_penalty_hours,
                     session.escalation_level, json_emails_used,
                     session.email_switches_count, session.browser_privacy_level, int(session.registration_prompted),
                     int(session.registration_link_clicked), session.recognition_response, session.display_message_offset))
                self.conn.commit()
                
            except Exception as e:
                logger.error(f"Failed to save session {session.session_id[:8]}: {e}", exc_info=True)
                if not hasattr(self, 'local_sessions'):
                    self.local_sessions = {}
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility"""
        with self.lock:
            if self.db_type == "memory":
                session = self.local_sessions.get(session_id)
                if session and isinstance(session.user_type, str):
                    try:
                        session.user_type = UserType(session.user_type)
                    except ValueError:
                        session.user_type = UserType.GUEST
                
                if session and not hasattr(session, 'display_message_offset'):
                    session.display_message_offset = 0
                
                return copy.deepcopy(session)
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                row = cursor.fetchone()
                
                if not row: 
                    return None
                
                if len(row) < 31:
                    logger.error(f"Row has insufficient columns: {len(row)} for session {session_id[:8]}")
                    return None
                    
                try:
                    loaded_display_message_offset = row[31] if len(row) > 31 else 0
                    loaded_last_activity = datetime.fromisoformat(row[6]) if row[6] else None

                    user_session = UserSession(
                        session_id=row[0], 
                        user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                        email=row[2], 
                        full_name=row[3],
                        zoho_contact_id=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                        last_activity=loaded_last_activity,
                        messages=safe_json_loads(row[7], default_value=[]),
                        active=bool(row[8]), 
                        wp_token=row[9],
                        timeout_saved_to_crm=bool(row[10]),
                        fingerprint_id=row[11],
                        fingerprint_method=row[12],
                        visitor_type=row[13] or 'new_visitor',
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
                        display_message_offset=loaded_display_message_offset
                    )
                    
                    return user_session
                    
                except Exception as e:
                    logger.error(f"Failed to create UserSession from row for {session_id[:8]}: {e}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to load session {session_id[:8]}: {e}")
                return None

    @handle_api_errors("Database", "Find by Fingerprint")
    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint_id."""
        with self.lock:
            if self.db_type == "memory":
                sessions = [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
                for session in sessions:
                    if not hasattr(session, 'display_message_offset'):
                        session.display_message_offset = 0
                return sessions
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None

                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
                sessions = []
                for row in cursor.fetchall():
                    if len(row) < 31:
                        continue
                    try:
                        loaded_display_message_offset = row[31] if len(row) > 31 else 0
                        loaded_last_activity = datetime.fromisoformat(row[6]) if row[6] else None

                        s = UserSession(
                            session_id=row[0], 
                            user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                            email=row[2], 
                            full_name=row[3],
                            zoho_contact_id=row[4],
                            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                            last_activity=loaded_last_activity,
                            messages=safe_json_loads(row[7], default_value=[]),
                            active=bool(row[8]), 
                            wp_token=row[9],
                            timeout_saved_to_crm=bool(row[10]),
                            fingerprint_id=row[11],
                            fingerprint_method=row[12],
                            visitor_type=row[13] or 'new_visitor',
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
                            display_message_offset=loaded_display_message_offset
                        )
                        sessions.append(s)
                    except Exception as e:
                        continue
                return sessions
            except Exception as e:
                logger.error(f"Failed to find sessions by fingerprint: {e}")
                return []

    @handle_api_errors("Database", "Find by Email")
    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions associated with a specific email address."""
        # Similar implementation to find_sessions_by_fingerprint but filtering by email
        return []  # Simplified for space

# =============================================================================
# FEATURE MANAGERS
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
            
            if not os.path.exists(html_file_path):
                return self._generate_fallback_fingerprint()
            
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            html_content = html_content.replace('{SESSION_ID}', session_id)
            st.components.v1.html(html_content, height=0, width=0, scrolling=False)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to render fingerprint component: {e}")
            return self._generate_fallback_fingerprint()

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
                logger.error(f"❌ Supabase client initialization failed: {e}")
                self.supabase = None

    @handle_api_errors("Supabase Auth", "Send Verification Code")
    def send_verification_code(self, email: str) -> bool:
        if not self.supabase:
            st.error("Email verification service is not available.")
            return False
        
        try:
            response = self.supabase.auth.sign_in_with_otp({
                'email': email,
                'options': {
                    'should_create_user': True,
                    'email_redirect_to': None,
                    'data': {'verification_type': 'email_otp'}
                }
            })
            
            if response is not None:
                logger.info(f"Email OTP code sent to {email}")
                return True
            else:
                logger.error("Supabase OTP send failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send verification code: {e}")
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
                logger.info(f"Email verification successful for {email}")
                return True
            else:
                logger.warning(f"Email verification failed for {email}")
                st.error("Invalid verification code. Please check the code and try again.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify code: {e}")
            st.error(f"Verification failed: {str(e)}")
            return False

# =============================================================================
# PDF EXPORTER & ZOHO CRM MANAGER
# =============================================================================

class PDFExporter:
    """Handles generation of PDF chat transcripts."""
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='ChatHeader', alignment=TA_CENTER, fontSize=18))
        self.styles.add(ParagraphStyle(name='UserMessage', backColor=lightgrey))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        """Generates a PDF of the chat transcript."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = [Paragraph("FiFi AI Chat Transcript", self.styles['Heading1'])]
        
        for msg in session.messages:
            role = str(msg.get('role', 'unknown')).capitalize()
            content = html.escape(str(msg.get('content', '')))
            content = re.sub(r'<[^>]+>', '', content)
            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
            
            if msg.get('source'):
                story.append(Paragraph(f"<i>Source: {msg['source']}</i>", self.styles['Normal']))
                
        doc.build(story)
        buffer.seek(0)
        return buffer

class ZohoCRMManager:
    """Manages integration with Zoho CRM for contact management."""
    
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        self._access_token = None
        self._token_expiry = None

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """Synchronously saves the chat transcript to Zoho CRM."""
        if not self.config.ZOHO_ENABLED:
            return False
        # Simplified implementation for space
        return True

# =============================================================================
# RATE LIMITER & UTILITY FUNCTIONS
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter to prevent abuse."""
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
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions="You are a document-based AI assistant..."
                )
            else:
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
                # Process citations...
            
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi",
                "has_citations": has_citations,
                "response_length": len(content),
                "used_pinecone": True,
                "used_search": False,
                "safety_override": False
            }
        except Exception as e:
            logger.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            search_results = self.tavily_tool.invoke({"query": message})
            synthesized_content = f"Based on my search: {search_results}"
            
            return {
                "content": synthesized_content,
                "success": True,
                "source": "FiFi Web Search",
                "used_pinecone": False,
                "used_search": True,
                "has_citations": True,
                "safety_override": False
            }
        except Exception as e:
            return {
                "content": f"Search error: {str(e)}",
                "success": False,
                "source": "error"
            }

@handle_api_errors("Content Moderation", "Check Prompt", show_to_user=False)
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    """Checks user prompt against content moderation guidelines."""
    if not client or not hasattr(client, 'moderations'):
        return {"flagged": False}
    
    try:
        response = client.moderations.create(model="omni-moderation-latest", input=prompt)
        result = response.results[0]
        
        if result.flagged:
            flagged_categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
            return {
                "flagged": True, 
                "message": "Your message violates our content policy.",
                "categories": flagged_categories
            }
    except Exception as e:
        logger.error(f"Content moderation failed: {e}")
        return {"flagged": False}
    
    return {"flagged": False}

# =============================================================================
# ENHANCED QUESTION LIMIT MANAGER - WITH TIER SYSTEM FOR REGISTERED USERS
# =============================================================================

class QuestionLimitManager:
    """Enhanced question limit manager with tier system for registered users and evasion detection."""
    
    def __init__(self):
        self.question_limits = {
            UserType.GUEST.value: 4,
            UserType.EMAIL_VERIFIED_GUEST.value: 10,
            UserType.REGISTERED_USER.value: 20  # UPDATED: Changed from 40 to 20
        }
        self.evasion_penalties = [24, 48, 96, 192, 336]
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """Enhanced limit checking with tier system for registered users."""
        user_limit = self.question_limits.get(session.user_type.value, 0)
        
        # Check if any existing ban is still active
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
                logger.info(f"Ban for session {session.session_id[:8]} expired. Resetting status.")
                # ENHANCED: Reset both daily and total counters after ban expires
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                session.question_limit_reached = False
                session.daily_question_count = 0  # Reset daily count
                if session.user_type.value == UserType.REGISTERED_USER.value:
                    session.total_question_count = 0  # Reset total count for registered users
        
        # Daily reset logic (24-hour rolling window)
        if session.last_question_time:
            time_since_last = datetime.now() - session.last_question_time
            if time_since_last >= timedelta(hours=24):
                logger.info(f"Daily question count reset for session {session.session_id[:8]}.")
                session.daily_question_count = 0
                if session.user_type.value == UserType.REGISTERED_USER.value:
                    session.total_question_count = 0  # Also reset total for registered users daily
                session.question_limit_reached = False
        
        # GUEST USER logic (unchanged)
        if session.user_type.value == UserType.GUEST.value:
            if session.daily_question_count >= user_limit:
                return {
                    'allowed': False,
                    'reason': 'guest_limit',
                    'message': 'Please provide your email address to continue.'
                }
        
        # EMAIL VERIFIED GUEST logic (unchanged)
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            if session.daily_question_count >= user_limit:
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Email-verified daily limit reached")
                return {
                    'allowed': False,
                    'reason': 'daily_limit',
                    'message': self._get_email_verified_limit_message()
                }
        
        # REGISTERED USER logic - NEW TIER SYSTEM
        elif session.user_type.value == UserType.REGISTERED_USER.value:
            # Tier 1: Questions 1-10 → 1-hour ban
            if session.total_question_count >= 10 and session.total_question_count < user_limit:
                # Check if already served Tier 1 ban
                if not (session.ban_status.value == BanStatus.ONE_HOUR.value and session.ban_end_time):
                    self._apply_ban(session, BanStatus.ONE_HOUR, "Tier 1 limit reached (10 questions)")
                    return {
                        'allowed': False,
                        'reason': 'tier1_limit',
                        'message': "You've used 10 questions today. Please wait 1 hour to access Tier 2 (11-20 questions)."
                    }
            
            # Tier 2: Questions 11-20 → 24-hour ban
            elif session.total_question_count >= user_limit:
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Tier 2 limit reached (20 questions)")
                return {
                    'allowed': False,
                    'reason': 'tier2_limit',
                    'message': "Daily limit reached (20 questions). Please retry in 24 hours as we are giving preference to others in the queue."
                }
        
        return {'allowed': True}
    
    def record_question(self, session: UserSession):
        """Records question for the session."""
        session.daily_question_count += 1
        if session.user_type.value == UserType.REGISTERED_USER.value:
            session.total_question_count += 1
        session.last_question_time = datetime.now()
        logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}.")
    
    def apply_evasion_penalty(self, session: UserSession) -> int:
        """Applies escalating penalty for evasion attempts."""
        session.evasion_count += 1
        session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
        
        penalty_hours = self.evasion_penalties[session.escalation_level - 1]
        session.current_penalty_hours = penalty_hours
        
        self._apply_ban(session, BanStatus.EVASION_BLOCK, f"Evasion attempt #{session.evasion_count}")
        
        logger.warning(f"Evasion penalty applied to {session.session_id[:8]}: {penalty_hours}h (Level {session.escalation_level}).")
        return penalty_hours
    
    def detect_guest_email_evasion(self, session: UserSession, new_email: str) -> bool:
        """Detects if a guest user is switching emails to evade limits."""
        if session.user_type.value != UserType.GUEST.value:
            return False
        
        # Check if this is an email switch
        if session.email and session.email != new_email:
            session.email_switches_count += 1
            logger.warning(f"Guest user {session.session_id[:8]} switching email: {session.email} → {new_email} (Switch #{session.email_switches_count})")
            
            # Apply evasion penalty for email switching after hitting guest limit
            if session.daily_question_count >= 4:
                penalty_hours = self.apply_evasion_penalty(session)
                logger.warning(f"Evasion detected: Guest switched email after hitting 4-question limit. Applied {penalty_hours}h ban.")
                return True
        
        return False
    
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
            return "Usage limit reached due to detected unusual activity. Please try again later."
        elif session.ban_status.value == BanStatus.ONE_HOUR.value:
            return "Tier 1 completed (10 questions). Please wait 1 hour to access Tier 2 (11-20 questions)."
        elif session.user_type.value == UserType.REGISTERED_USER.value:
            return "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
        else:
            return self._get_email_verified_limit_message()
    
    def _get_email_verified_limit_message(self) -> str:
        """Specific message for email-verified guests hitting their daily limit."""
        return ("Our system is very busy and is being used by multiple users. For a fair assessment of our FiFi AI assistant and to provide fair usage to everyone, we can allow a total of 10 questions per day (20 messages). To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

# =============================================================================
# ENHANCED AI SYSTEM WITH IMPROVED ERROR HANDLING
# =============================================================================

class EnhancedAI:
    """Enhanced AI system with improved Pinecone error handling and bidirectional fallback."""
    
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
                error_handler.log_error(error_handler.handle_api_error("Pinecone", "Initialize", e))
                self.pinecone_tool = None
        
        # Initialize Tavily agent
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(self.config.TAVILY_API_KEY)
                logger.info("✅ Tavily Web Search initialized successfully")
                error_handler.mark_component_healthy("Tavily")
            except Exception as e:
                logger.error(f"Tavily agent initialization failed: {e}")
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Initialize", e))
                self.tavily_agent = None

    def _is_pinecone_error_requiring_fallback(self, error: Exception) -> bool:
        """Determines if Pinecone error requires fallback to Tavily."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for HTTP status codes that require fallback
        pinecone_fallback_codes = [
            "401", "402", "403",  # Auth/payment issues
            "429",                # Rate limiting
            "500", "503",         # Server errors
            "400", "404", "409", "412", "422"  # Client errors
        ]
        
        # Check for status codes in error message
        for code in pinecone_fallback_codes:
            if code in error_str:
                logger.warning(f"Pinecone error code {code} detected, triggering fallback")
                return True
        
        # Check for specific error types
        fallback_conditions = [
            "timeout" in error_str,
            "connection" in error_str,
            "network" in error_str,
            "unavailable" in error_str,
            "rate limit" in error_str,
            "quota" in error_str,
            error_type in ["ConnectionError", "TimeoutError", "HTTPError"]
        ]
        
        if any(fallback_conditions):
            logger.warning(f"Pinecone fallback condition met: {error_type} - {error_str}")
            return True
        
        return False

    def _is_tavily_error_requiring_fallback(self, error: Exception) -> bool:
        """Determines if Tavily error requires fallback to Pinecone."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for conditions that should trigger Pinecone fallback
        fallback_conditions = [
            "timeout" in error_str,
            "connection" in error_str,
            "network" in error_str,
            "unavailable" in error_str,
            "rate limit" in error_str,
            "quota" in error_str,
            "401" in error_str,
            "403" in error_str,
            "429" in error_str,
            "500" in error_str,
            "503" in error_str,
            error_type in ["ConnectionError", "TimeoutError", "HTTPError"]
        ]
        
        if any(fallback_conditions):
            logger.warning(f"Tavily fallback condition met: {error_type} - {error_str}")
            return True
        
        return False

    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced AI response with improved error handling and bidirectional fallback."""
        
        # Content moderation check (unchanged)
        moderation_result = check_content_moderation(prompt, self.openai_client)
        if moderation_result and moderation_result.get("flagged"):
            logger.warning(f"Content moderation flagged input: {moderation_result.get('categories', [])}")
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
        
        # Convert chat history to LangChain format (unchanged)
        if chat_history:
            langchain_history = []
            for msg in chat_history[-10:]:
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg.get("content", "")))
            langchain_history.append(HumanMessage(content=prompt))
        else:
            langchain_history = [HumanMessage(content=prompt)]
        
        # Enhanced routing based on component health
        pinecone_healthy = error_handler.component_status.get("Pinecone") == "healthy"
        tavily_healthy = error_handler.component_status.get("Tavily") == "healthy"
        
        # Try Pinecone first if healthy and available
        if self.pinecone_tool and pinecone_healthy:
            try:
                logger.info("🔍 Querying Pinecone knowledge base...")
                pinecone_response = self.pinecone_tool.query(langchain_history)
                
                if pinecone_response and pinecone_response.get("success"):
                    should_fallback = self.should_use_web_fallback(pinecone_response)
                    
                    if not should_fallback:
                        logger.info("✅ Using Pinecone response (passed safety checks)")
                        error_handler.mark_component_healthy("Pinecone")
                        return pinecone_response
                    else:
                        logger.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switching to verified web sources.")
                        
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}")
                error_handler.log_error(error_handler.handle_api_error("Pinecone", "Query", e))
                
                # Check if this error requires fallback to Tavily
                if self._is_pinecone_error_requiring_fallback(e):
                    logger.info("🔄 Pinecone error requires fallback, switching to Tavily")
                    # Continue to Tavily fallback below
                else:
                    # Don't fallback for this error, return error response
                    return {
                        "content": "I apologize, but I'm experiencing technical difficulties with my knowledge base. Please try again later.",
                        "success": False,
                        "source": "Pinecone Error",
                        "used_search": False,
                        "used_pinecone": False,
                        "has_citations": False,
                        "has_inline_citations": False,
                        "safety_override": False
                    }
        
        # Try Tavily if available and healthy (either as fallback or primary)
        if self.tavily_agent and tavily_healthy:
            try:
                logger.info("🌐 Using web search...")
                web_response = self.tavily_agent.query(prompt, langchain_history[:-1])
                
                if web_response and web_response.get("success"):
                    logger.info("✅ Using web search response")
                    error_handler.mark_component_healthy("Tavily")
                    return web_response
                    
            except Exception as e:
                logger.error(f"Tavily search failed: {e}")
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Query", e))
                
                # Check if this error requires fallback to Pinecone
                if self._is_tavily_error_requiring_fallback(e) and self.pinecone_tool:
                    try:
                        logger.info("🔄 Tavily failed, attempting Pinecone fallback")
                        pinecone_response = self.pinecone_tool.query(langchain_history)
                        
                        if pinecone_response and pinecone_response.get("success"):
                            # Skip safety checks for fallback scenario
                            logger.info("✅ Using Pinecone fallback response")
                            error_handler.mark_component_healthy("Pinecone")
                            return pinecone_response
                            
                    except Exception as pinecone_e:
                        logger.error(f"Pinecone fallback also failed: {pinecone_e}")
        
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

    def should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool:
        """EXTREMELY aggressive fallback detection to prevent any hallucination."""
        content = pinecone_response.get("content", "").lower()
        content_raw = pinecone_response.get("content", "")
        
        # PRIORITY 1: Always fallback for current/recent information requests
        current_info_indicators = [
            "today", "yesterday", "this week", "this month", "this year", "2025", "2024",
            "current", "latest", "recent", "now", "currently", "updated",
            "news", "weather", "stock", "price", "event", "happening"
        ]
        if any(indicator in content for indicator in current_info_indicators):
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
            return False
        
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

# =============================================================================
# SESSION MANAGER - MAIN ORCHESTRATOR CLASS
# =============================================================================

class SessionManager:
    """Enhanced session manager with improved features."""
    
    def __init__(self, config: Config, db_manager: DatabaseManager, 
                 zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, 
                 rate_limiter: RateLimiter, fingerprinting_manager: FingerprintingManager,
                 email_verification_manager, question_limit_manager: QuestionLimitManager):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.fingerprinting = fingerprinting_manager
        self.email_verification = email_verification_manager
        self.question_limits = question_limit_manager
        
        logger.info("✅ SessionManager initialized with all component managers.")

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return 5
    
    def get_session(self) -> Optional[UserSession]:
        """Gets or creates the current user session."""
        try:
            session_id = st.session_state.get('current_session_id')
            
            if session_id:
                session = self.db.load_session(session_id)
                
                if session and session.active:
                    # Check limits and handle bans
                    limit_check = self.question_limits.is_within_limits(session)
                    if not limit_check.get('allowed', True):
                        ban_type = limit_check.get('ban_type', 'unknown')
                        message = limit_check.get('message', 'Access restricted.')
                        time_remaining = limit_check.get('time_remaining')
                        
                        st.error(f"🚫 **Access Restricted**")
                        if time_remaining:
                            hours = max(0, int(time_remaining.total_seconds() // 3600))
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                        st.info(message)
                        
                        try:
                            self.db.save_session(session)
                        except Exception as e:
                            logger.error(f"Failed to save banned session: {e}")
                        
                        return session
                    
                    return session
                else:
                    if 'current_session_id' in st.session_state:
                        del st.session_state['current_session_id']

            # Create new session
            new_session = self._create_new_session()
            st.session_state.current_session_id = new_session.session_id
            return new_session
            
        except Exception as e:
            logger.error(f"Failed to get/create session: {e}")
            fallback_session = UserSession(session_id=str(uuid.uuid4()), user_type=UserType.GUEST, last_activity=None)
            fallback_session.fingerprint_id = f"emergency_fp_{fallback_session.session_id[:8]}"
            fallback_session.fingerprint_method = "emergency_fallback"
            st.session_state.current_session_id = fallback_session.session_id
            st.error("⚠️ Operating in emergency fallback mode.")
            return fallback_session

    def _create_new_session(self) -> UserSession:
        """Creates a new user session."""
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id, last_activity=None)
        
        session.fingerprint_id = f"temp_py_{secrets.token_hex(8)}"
        session.fingerprint_method = "temporary_fallback_python"
        
        self.db.save_session(session)
        return session

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Enhanced email verification with evasion detection."""
        return handle_guest_email_verification_enhanced(self, session, email)

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verifies the email verification code and upgrades user status."""
        try:
            if not session.email:
                return {'success': False, 'message': 'No email address found for verification.'}
            
            sanitized_code = sanitize_input(code, 10).strip()
            
            if not sanitized_code:
                return {'success': False, 'message': 'Please enter the verification code.'}
            
            verification_success = self.email_verification.verify_code(session.email, sanitized_code)
            
            if verification_success:
                session.user_type = UserType.EMAIL_VERIFIED_GUEST
                session.question_limit_reached = False
                
                if session.daily_question_count >= 4:
                    session.daily_question_count = 0
                    session.last_question_time = None
                
                if session.last_activity is None:
                    session.last_activity = datetime.now()

                try:
                    self.db.save_session(session)
                    logger.info(f"User {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST")
                except Exception as e:
                    logger.error(f"Failed to save upgraded session: {e}")
                
                return {
                    'success': True,
                    'message': f'✅ Email verified successfully! You now have 10 questions per day.'
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
        """Authenticates user with WordPress."""
        if not self.config.WORDPRESS_URL:
            st.error("WordPress authentication is not configured.")
            return None
        
        try:
            auth_url = f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token"
            
            response = requests.post(auth_url, json={
                'username': username,
                'password': password
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                wp_token = data.get('token')
                user_email = data.get('user_email')
                user_display_name = data.get('user_display_name')
                
                if wp_token and user_email:
                    current_session = self.get_session()
                    
                    current_session.user_type = UserType.REGISTERED_USER
                    current_session.email = user_email
                    current_session.full_name = user_display_name
                    current_session.wp_token = wp_token
                    current_session.question_limit_reached = False
                    
                    # Reset question counts for fresh start
                    current_session.daily_question_count = 0
                    current_session.total_question_count = 0
                    current_session.last_question_time = None
                    current_session.last_activity = datetime.now()
                    
                    try:
                        self.db.save_session(current_session)
                        logger.info(f"User authenticated: {user_email}")
                    except Exception as e:
                        logger.error(f"Failed to save authenticated session: {e}")
                        st.error("Authentication succeeded but session could not be saved.")
                        return None
                    
                    return current_session
                else:
                    st.error("Authentication failed: Incomplete response from WordPress.")
                    return None
            else:
                st.error("Invalid username or password.")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Authentication service timed out. Please try again.")
            return None
        except Exception as e:
            logger.error(f"WordPress authentication error: {e}")
            st.error("An unexpected error occurred during authentication.")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Gets AI response and manages session state."""
        try:
            # Check rate limiting
            if not self.rate_limiter.is_allowed(session.session_id):
                return {
                    'content': 'Too many requests. Please wait a moment.',
                    'success': False,
                    'source': 'Rate Limiter'
                }
            
            # Check question limits
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check['allowed']:
                if limit_check.get('reason') == 'guest_limit':
                    return {'requires_email': True, 'content': 'Email verification required.'}
                else:
                    return {
                        'banned': True,
                        'content': limit_check.get('message', 'Access restricted.'),
                        'time_remaining': limit_check.get('time_remaining')
                    }
            
            # Sanitize input
            sanitized_prompt = sanitize_input(prompt, 4000)
            if not sanitized_prompt:
                return {
                    'content': 'Please enter a valid question.',
                    'success': False,
                    'source': 'Input Validation'
                }
            
            # Record question
            self.question_limits.record_question(session)
            
            # Get AI response
            ai_response = self.ai.get_response(sanitized_prompt, session.messages)
            
            # Add messages to session
            user_message = {'role': 'user', 'content': sanitized_prompt}
            assistant_message = {
                'role': 'assistant',
                'content': ai_response.get('content', 'No response generated.'),
                'source': ai_response.get('source'),
                'used_pinecone': ai_response.get('used_pinecone', False),
                'used_search': ai_response.get('used_search', False),
                'has_citations': ai_response.get('has_citations', False),
                'safety_override': ai_response.get('safety_override', False)
            }
            
            session.messages.extend([user_message, assistant_message])
            
            # Update activity and save
            session.last_activity = datetime.now()
            self.db.save_session(session)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return {
                'content': 'I encountered an error processing your request.',
                'success': False,
                'source': 'Error Handler'
            }

    def clear_chat_history(self, session: UserSession):
        """Clears chat history using soft clear mechanism."""
        try:
            session.display_message_offset = len(session.messages)
            self.db.save_session(session)
            logger.info(f"Soft cleared chat for session {session.session_id[:8]}")
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")

    def end_session(self, session: UserSession):
        """Ends the current session and performs cleanup."""
        try:
            session.active = False
            session.last_activity = datetime.now()
            
            try:
                self.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session during end_session: {e}")
            
            # Clear Streamlit session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.session_state['page'] = None
            st.success("👋 You have been signed out successfully!")
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['page'] = None

# =============================================================================
# ENHANCED UI COMPONENTS
# =============================================================================

def render_sidebar_enhanced(session_manager: 'SessionManager', session: UserSession, pdf_exporter: PDFExporter):
    """Enhanced sidebar with new tier information."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        if session.user_type.value == UserType.REGISTERED_USER.value:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # ENHANCED: Show new tier system
            st.markdown(f"**Questions Today:** {session.total_question_count}/20")
            if session.total_question_count <= 10:
                st.progress(min(session.total_question_count / 10, 1.0), text="Tier 1 (up to 10 questions)")
            else:
                progress_value = min((session.total_question_count - 10) / 10, 1.0)
                st.progress(progress_value, text="Tier 2 (11-20 questions)")
            
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(min(session.daily_question_count / 10, 1.0))
            
        else:
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(min(session.daily_question_count / 4, 1.0))
            st.caption("Email verification unlocks 10 questions/day.")
        
        # Show fingerprint status
        if session.fingerprint_id and not session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:12]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'}")
        else:
            st.markdown("**Device ID:** Identifying...")

        # Show activity timer
        if session.last_activity is not None:
            time_since_activity = datetime.now() - session.last_activity
            minutes_inactive = time_since_activity.total_seconds() / 60
            st.caption(f"Last activity: {int(minutes_inactive)} minutes ago")
            
            timeout_duration = session_manager.get_session_timeout_minutes()
            if minutes_inactive >= (timeout_duration - 1) and minutes_inactive < timeout_duration:
                minutes_remaining = timeout_duration - minutes_inactive
                st.warning(f"⏰ Session expires in: {int(minutes_remaining)}m")
        else:
            st.caption("Session timer will start with first interaction.")

        # AI Tools Status
        st.divider()
        st.markdown("**🤖 AI Tools Status**")
        
        ai_system = session_manager.ai
        if ai_system:
            if ai_system.pinecone_tool:
                st.success("🧠 Knowledge Base: Ready")
            else:
                st.info("🧠 Knowledge Base: Not configured")
            
            if ai_system.tavily_agent:
                st.success("🌐 Web Search: Ready")
            else:
                st.info("🌐 Web Search: Not configured")
        else:
            st.error("🤖 AI System: Not available")
        
        st.divider()
        
        # Show message counts
        total_messages = len(session.messages)
        visible_messages = len(session.messages) - session.display_message_offset
        
        if session.display_message_offset > 0:
            st.markdown(f"**Messages in Chat:** {visible_messages} (Total: {total_messages})")
            st.caption(f"💡 {session.display_message_offset} messages hidden by Clear Chat")
        else:
            st.markdown(f"**Messages in Chat:** {total_messages}")
            
        st.markdown(f"**Session ID:** `{session.session_id[:8]}...`")
        
        # Show ban status
        if session.ban_status.value != BanStatus.NONE.value:
            st.error(f"🚫 **STATUS: RESTRICTED**")
            if session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.markdown(f"**Time Remaining:** {hours}h {minutes}m")
            st.markdown(f"Reason: {session.ban_reason or 'Usage policy violation'}")
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(session)
                st.success("🗑️ Chat display cleared!")
                st.rerun()
                
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(session)
                st.rerun()

        # PDF download
        if session.messages:
            st.divider()
            pdf_buffer = pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download Chat PDF",
                    data=pdf_buffer,
                    file_name=f"fifi_chat_transcript_{session.session_id[:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

def render_email_verification_dialog_enhanced(session_manager: 'SessionManager', session: UserSession):
    """Enhanced email verification dialog with evasion detection."""
    st.error("📧 **Email Verification Required**")
    st.info("You've used your 4 free questions. Please verify your email to unlock 10 questions per day.")
    
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = 'email_entry'

    if st.session_state.verification_stage == 'email_entry':
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**Please enter your email address:**")
            email_input = st.text_input("Email Address", placeholder="your@email.com")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email and email_input:
                result = session_manager.handle_guest_email_verification(session, email_input)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.verification_email = email_input
                    st.session_state.verification_stage = "code_entry"
                    st.rerun()
                else:
                    st.error(result['message'])
    
    elif st.session_state.verification_stage == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 Verification code sent to **{verification_email}**.")
        
        with st.form("code_verification_form", clear_on_submit=False):
            code = st.text_input("Enter Verification Code", placeholder="123456", max_chars=6)
            
            col1, col2 = st.columns(2)
            with col1:
                submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            with col2:
                resend_code = st.form_submit_button("🔄 Resend Code", use_container_width=True)
            
            if resend_code:
                if verification_email:
                    verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                    if verification_sent:
                        st.success("Verification code resent!")
                    else:
                        st.error("Failed to resend code.")
                st.rerun()

            if submit_code and code:
                result = session_manager.verify_email_code(session, code)
                if result['success']:
                    st.success(result['message'])
                    st.balloons()
                    for key in ['verification_email', 'verification_stage']:
                        if key in st.session_state:
                            del st.session_state[key]
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(result['message'])

def render_chat_interface_enhanced(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]):
    """Enhanced chat interface with improved features."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")

    # Activity tracking
    if activity_result:
        js_last_activity_timestamp = activity_result.get('last_activity')
        if js_last_activity_timestamp:
            try:
                new_activity = datetime.fromtimestamp(js_last_activity_timestamp / 1000)
                if session.last_activity is None or new_activity > session.last_activity:
                    session.last_activity = new_activity
                    session_manager.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to update activity from JavaScript: {e}")

    # Fingerprinting
    fingerprint_needed = (
        not session.fingerprint_id or
        session.fingerprint_method == "temporary_fallback_python" or
        session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
    )
    
    if fingerprint_needed:
        session_manager.fingerprinting.render_fingerprint_component(session.session_id)

    # User limits check
    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed']:
        if limit_check.get('reason') == 'guest_limit':
            render_email_verification_dialog_enhanced(session_manager, session)
            return
        else:
            return

    # Display chat messages
    visible_messages = session.messages[session.display_message_offset:]
    for msg in visible_messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            
            if msg.get("role") == "assistant":
                if "source" in msg:
                    source_color = {
                        "FiFi": "🧠", "FiFi Web Search": "🌐", 
                        "Content Moderation": "🛡️", "System Fallback": "⚠️"
                    }.get(msg['source'], "🤖")
                    st.caption(f"{source_color} Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"): indicators.append("🧠 Knowledge Base")
                if msg.get("used_search"): indicators.append("🌐 Web Search")
                if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")

    # Chat input
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                            disabled=session.ban_status.value != BanStatus.NONE.value)
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue.")
                        st.session_state.verification_stage = 'email_entry'
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
                                "FiFi": "🧠", "FiFi Web Search": "🌐"
                            }.get(response['source'], "🤖")
                            st.caption(f"{source_color} Source: {response['source']}")
                        
                except Exception as e:
                    logger.error(f"AI response failed: {e}")
                    st.error("⚠️ I encountered an error. Please try again.")
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS FOR TIMEOUT AND FINGERPRINTING
# =============================================================================

def render_simple_activity_tracker(session_id: str) -> Optional[Dict[str, Any]]:
    """Renders a simple activity tracker that monitors user interactions."""
    if not session_id:
        return None
    
    safe_session_id = session_id.replace('-', '_')
    component_key = f"activity_tracker_{safe_session_id}"

    simple_tracker_js = f"""
    (() => {{
        const sessionId = "{session_id}";
        const stateKey = 'fifi_activity_{safe_session_id}';
        
        if (!window[stateKey]) {{
            window[stateKey] = {{
                lastActivity: Date.now(),
                listenersInitialized: false,
                sessionId: sessionId
            }};
        }}
        
        const state = window[stateKey];
        
        if (!state.listenersInitialized) {{
            function updateActivity() {{
                state.lastActivity = Date.now();
            }}
            
            const events = ['mousedown', 'mousemove', 'keydown', 'click', 'scroll', 'touchstart', 'focus'];
            events.forEach(eventType => {{
                document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
            }});
            
            state.listenersInitialized = true;
        }}
        
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
        result = st_javascript(simple_tracker_js, key=component_key)
        
        if result and isinstance(result, dict) and result.get('type') == 'activity_status':
            return result
        return None
    except Exception as e:
        logger.error(f"Simple activity tracker failed: {e}")
        return None

def check_timeout_and_trigger_reload(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]) -> bool:
    """Check for session timeout and trigger reload if needed."""
    if not session or not session.session_id:
        return False
    
    # Load fresh session from DB
    fresh_session_from_db = session_manager.db.load_session(session.session_id)
    
    if fresh_session_from_db:
        session.last_activity = fresh_session_from_db.last_activity
        session.active = fresh_session_from_db.active
    else:
        session.active = False

    if not session.active:
        st.error("⏰ **Session Expired**")
        st.info("Your session has ended. Please start a new session.")
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        if JS_EVAL_AVAILABLE:
            try:
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                st.stop()
            except Exception as e:
                logger.error(f"Browser reload failed: {e}")
        
        st.rerun()
        st.stop()
        return True

    # Handle activity updates
    if session.last_activity is None:
        return False
        
    if activity_result:
        try:
            js_last_activity_timestamp = activity_result.get('last_activity')
            if js_last_activity_timestamp:
                new_activity_dt = datetime.fromtimestamp(js_last_activity_timestamp / 1000)
                
                if new_activity_dt > session.last_activity:
                    session.last_activity = new_activity_dt
                    session_manager.db.save_session(session)
        except Exception as e:
            logger.error(f"Error processing activity timestamp: {e}")

    # Check timeout
    time_since_activity = datetime.now() - session.last_activity
    minutes_inactive = time_since_activity.total_seconds() / 60
    
    if minutes_inactive >= session_manager.get_session_timeout_minutes():
        session.active = False
        session.last_activity = datetime.now()
        
        try:
            session_manager.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to save session during timeout: {e}")
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.error("⏰ **Session Timeout**")
        st.info("Your session expired due to 5 minutes of inactivity.")
        
        if JS_EVAL_AVAILABLE:
            try:
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
                st.stop()
            except Exception as e:
                logger.error(f"Browser reload failed: {e}")
        
        st.rerun()
        st.stop()
        return True
    
    return False

# =============================================================================
# INITIALIZATION FUNCTIONS
# =============================================================================

def ensure_initialization_fixed():
    """Fixed initialization with better error handling."""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        try:
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.info("🔄 Initializing FiFi AI Assistant...")
                init_progress = st.progress(0)
                status_text = st.empty()
            
            status_text.text("Loading configuration...")
            init_progress.progress(0.1)
            config = Config()
            
            status_text.text("Setting up PDF exporter...")
            init_progress.progress(0.2)
            pdf_exporter = PDFExporter()
            
            status_text.text("Connecting to database...")
            init_progress.progress(0.3)
            try:
                db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
                st.session_state.db_manager = db_manager
            except Exception as db_e:
                logger.error(f"Database manager failed: {db_e}")
                st.session_state.db_manager = type('FallbackDB', (), {
                    'db_type': 'memory',
                    'local_sessions': {},
                    'save_session': lambda self, session: None,
                    'load_session': lambda self, session_id: None,
                    'find_sessions_by_fingerprint': lambda self, fingerprint_id: [],
                    'find_sessions_by_email': lambda self, email: []
                })()
                st.warning("⚠️ Database unavailable. Using temporary storage.")
            
            status_text.text("Setting up managers...")
            init_progress.progress(0.5)
            
            try:
                zoho_manager = ZohoCRMManager(config, pdf_exporter)
            except Exception as e:
                zoho_manager = type('FallbackZoho', (), {
                    'config': config,
                    'save_chat_transcript_sync': lambda self, session, reason: False
                })()
            
            init_progress.progress(0.6)
            
            try:
                ai_system = EnhancedAI(config)
            except Exception as e:
                ai_system = type('FallbackAI', (), {
                    "openai_client": None,
                    'get_response': lambda self, prompt, history=None: {
                        "content": "AI system temporarily unavailable.",
                        "success": False
                    }
                })()
            
            init_progress.progress(0.7)
            
            rate_limiter = RateLimiter()
            fingerprinting_manager = FingerprintingManager()
            
            init_progress.progress(0.8)
            
            try:
                email_verification_manager = EmailVerificationManager(config)
            except Exception as e:
                email_verification_manager = type('DummyEmail', (), {
                    'send_verification_code': lambda self, email: False,
                    'verify_code': lambda self, email, code: False
                })()
            
            init_progress.progress(0.9)
            
            question_limit_manager = QuestionLimitManager()
            
            status_text.text("Finalizing initialization...")
            init_progress.progress(0.95)
            
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

            init_progress.progress(1.0)
            status_text.text("✅ Initialization complete!")
            
            time.sleep(0.5)
            progress_placeholder.empty()
            
            st.session_state.initialized = True
            return True
            
        except Exception as e:
            st.error("💥 Critical initialization error occurred.")
            st.error(f"Error: {str(e)}")
            st.session_state.initialized = False
            return False
    
    return True

# Placeholder functions for query parameter handling
def handle_emergency_save_requests_from_query():
    """Handle emergency save requests from URL parameters."""
    pass

def handle_fingerprint_requests_from_query():
    """Handle fingerprint requests from URL parameters."""
    pass

def render_welcome_page_enhanced(session_manager: 'SessionManager'):
    """Enhanced welcome page with registration tracking."""
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
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
        st.markdown("• **Tier 1: 10 questions** (1-hour break)")  # UPDATED
        st.markdown("• **Tier 2: 20 questions total/day**")      # UPDATED
        st.markdown("• Cross-device tracking & consistent experience")
        st.markdown("• Automatic chat saving to Zoho CRM")
        st.markdown("• Priority access during high usage")
    
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
                        with st.spinner("🔐 Authenticating..."):
                            authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                            
                        if authenticated_session:
                            st.balloons()
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            
                            st.session_state.current_session_id = authenticated_session.session_id
                            st.session_state.page = "chat"
                            
                            time.sleep(1.5)
                            st.rerun()
            
            st.markdown("---")
            
            # ENHANCED: Registration tracking
            current_session = session_manager.get_session()
            if not current_session.registration_prompted:
                current_session.registration_prompted = True
                session_manager.db.save_session(current_session)
                logger.info(f"Registration prompt shown for session {current_session.session_id[:8]}")
            
            # Registration link with tracking
            registration_url = "https://www.12taste.com/in/my-account/"
            st.info(f"Don't have an account? [Register here]({registration_url}) to unlock full features!")
            
            # Note: We cannot directly track link clicks due to Streamlit limitations
            # but we could track this via custom component if needed
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to to get a quick start and try FiFi AI Assistant without signing in.
        
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
                session = session_manager.get_session()
                if session.last_activity is None:
                    session.last_activity = datetime.now()
                    session_manager.db.save_session(session)
                st.session_state.page = "chat"
                st.rerun()

# =============================================================================
# ENHANCED EMAIL VERIFICATION WITH EVASION DETECTION
# =============================================================================

def handle_guest_email_verification_enhanced(session_manager: 'SessionManager', session: UserSession, email: str) -> Dict[str, Any]:
    """Enhanced email verification with evasion detection."""
    try:
        sanitized_email = email.lower().strip()
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', sanitized_email):
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # ENHANCED: Check for evasion before proceeding
        evasion_detected = session_manager.question_limits.detect_guest_email_evasion(session, sanitized_email)
        
        if evasion_detected:
            # Save session with evasion penalty
            session_manager.db.save_session(session)
            return {
                'success': False, 
                'message': f'Usage limit reached due to detected unusual activity. Please wait {session.current_penalty_hours} hours before trying again.'
            }
        
        # Track email usage for this session
        if sanitized_email not in session.email_addresses_used:
            session.email_addresses_used.append(sanitized_email)
        
        # Update session email
        if session.email and session.email != sanitized_email:
            session.email_switches_count += 1
        session.email = sanitized_email
        
        # Set last_activity if not already set
        if session.last_activity is None:
            session.last_activity = datetime.now()

        # Save session before sending verification
        try:
            session_manager.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to save session before email verification: {e}")
        
        # Send verification code
        code_sent = session_manager.email_verification.send_verification_code(sanitized_email)
        
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

# Include all other classes and functions unchanged...
# (Database management, PDF export, Zoho CRM, fingerprinting, etc.)

# =============================================================================
# MAIN APPLICATION WITH ENHANCED FEATURES
# =============================================================================

def main_enhanced():
    """Enhanced main application with all new features."""
    try:
        st.set_page_config(
            page_title="FiFi AI Assistant", 
            page_icon="🤖", 
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {e}")

    # Initialize (unchanged)
    try:
        with st.spinner("Initializing application..."):
            init_success = ensure_initialization_fixed()
            
        if not init_success:
            st.error("⚠️ Application failed to initialize properly.")
            st.info("Please refresh the page to try again.")
            return
            
    except Exception as init_error:
        st.error(f"⚠️ Initialization error: {str(init_error)}")
        st.info("Please refresh the page to try again.")
        logger.error(f"Main initialization error: {init_error}", exc_info=True)
        return

    # Handle query parameters (unchanged)
    try:
        handle_emergency_save_requests_from_query()
        handle_fingerprint_requests_from_query()
    except Exception as e:
        logger.error(f"Query parameter handling failed: {e}")

    # Get session manager and session (unchanged)
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Session Manager not available. Please refresh the page.")
        return

    session = session_manager.get_session()
    
    if session is None or not session.active:
        logger.warning(f"Session is None or Inactive after get_session.")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state['page'] = None
        st.rerun()
        return

    # Activity tracking and timeout check (unchanged)
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

    # Route to appropriate page
    current_page = st.session_state.get('page')
    
    try:
        if current_page != "chat":
            render_welcome_page_enhanced(session_manager)  # ENHANCED VERSION
            
        else:
            render_sidebar_enhanced(session_manager, session, st.session_state.pdf_exporter)  # ENHANCED VERSION
            render_chat_interface_enhanced(session_manager, session, activity_data_from_js)    # ENHANCED VERSION
                    
    except Exception as page_error:
        logger.error(f"Page routing error: {page_error}", exc_info=True)
        st.error("⚠️ Page error occurred. Please refresh the page.")
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        time.sleep(2)
        st.rerun()

# Entry point
if __name__ == "__main__":
    main_enhanced()
