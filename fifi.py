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
# FINAL INTEGRATED FIFI AI - WITH WORKING 5-MINUTE TIMEOUT SYSTEM
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
    last_activity: datetime = field(default_factory=datetime.now)
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

                # Create table with all columns upfront
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_type TEXT DEFAULT 'guest',
                        email TEXT,
                        full_name TEXT,
                        zoho_contact_id TEXT,
                        created_at TEXT DEFAULT '',
                        last_activity TEXT DEFAULT '',
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
                        recognition_response TEXT
                    )
                ''')
                
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
        with self.lock:
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
                
                self.conn.execute(
                    '''REPLACE INTO sessions (session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session.session_id, session.user_type.value, session.email, session.full_name,
                     session.zoho_contact_id, session.created_at.isoformat(),
                     session.last_activity.isoformat(), json_messages, int(session.active),
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
                     int(session.registration_link_clicked), session.recognition_response))
                self.conn.commit()
                
                logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}")
                
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
        with self.lock:
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
                return copy.deepcopy(session)
            
            try:
                # NEVER set row_factory for cloud connections - always use raw tuples
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                row = cursor.fetchone()
                
                if not row: 
                    logger.debug(f"No active session found for ID {session_id[:8]}.")
                    return None
                
                # Handle as tuple (SQLite Cloud returns tuples)
                expected_cols = 31
                if len(row) < expected_cols:
                    logger.error(f"Row has insufficient columns: {len(row)} (expected {expected_cols}) for session {session_id[:8]}. Data corruption suspected.")
                    return None
                    
                try:
                    user_session = UserSession(
                        session_id=row[0], 
                        user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                        email=row[2], 
                        full_name=row[3],
                        zoho_contact_id=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                        last_activity=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
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
                        recognition_response=row[30]
                    )
                    
                    logger.info(f"Successfully loaded session {session_id[:8]}: user_type={user_session.user_type.value}, messages={len(user_session.messages)}")
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
        """Find all sessions with the same fingerprint_id."""
        with self.lock:
            current_config = st.session_state.get('session_manager').config if st.session_state.get('session_manager') else None
            if current_config:
                self._ensure_connection_healthy(current_config)

            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None

                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
                sessions = []
                expected_cols = 31
                for row in cursor.fetchall():
                    if len(row) < expected_cols:
                        logger.warning(f"Row has insufficient columns in find_sessions_by_fingerprint: {len(row)} (expected {expected_cols}). Skipping row.")
                        continue
                    try:
                        s = UserSession(
                            session_id=row[0], 
                            user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                            email=row[2], 
                            full_name=row[3],
                            zoho_contact_id=row[4],
                            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                            last_activity=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
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
                            recognition_response=row[30]
                        )
                        sessions.append(s)
                    except Exception as e:
                        logger.error(f"Error converting row to UserSession in find_sessions_by_fingerprint: {e}", exc_info=True)
                        continue
                return sessions
            except Exception as e:
                logger.error(f"Failed to find sessions by fingerprint '{fingerprint_id[:8]}...': {e}", exc_info=True)
                return []

    @handle_api_errors("Database", "Find by Email")
    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions associated with a specific email address."""
        with self.lock:
            current_config = st.session_state.get('session_manager').config if st.session_state.get('session_manager') else None
            if current_config:
                self._ensure_connection_healthy(current_config)

            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.email == email]
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None

                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response FROM sessions WHERE email = ? ORDER BY last_activity DESC", (email,))
                sessions = []
                expected_cols = 31
                for row in cursor.fetchall():
                    if len(row) < expected_cols:
                        logger.warning(f"Row has insufficient columns in find_sessions_by_email: {len(row)} (expected {expected_cols}). Skipping row.")
                        continue
                    try:
                        s = UserSession(
                            session_id=row[0], 
                            user_type=UserType(row[1]) if row[1] else UserType.GUEST,
                            email=row[2], 
                            full_name=row[3],
                            zoho_contact_id=row[4],
                            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                            last_activity=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
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
                            recognition_response=row[30]
                        )
                        sessions.append(s)
                    except Exception as e:
                        logger.error(f"Error converting row to UserSession in find_sessions_by_email: {e}", exc_info=True)
                        continue
                return sessions
            except Exception as e:
                logger.error(f"Failed to find sessions by email '{email}': {e}", exc_info=True)
                return []

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
            
            logger.info(f"🔍 Looking for fingerprint component at: {html_file_path}")
            
            if not os.path.exists(html_file_path):
                logger.error(f"❌ Fingerprint component file NOT FOUND at {html_file_path}")
                logger.info(f"📁 Current directory contents: {os.listdir(current_dir)}")
                return self._generate_fallback_fingerprint()
            
            logger.info(f"✅ Fingerprint component file found, reading content...")
            
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            logger.info(f"📄 Read {len(html_content)} characters from fingerprint component file")
            
            # Replace session ID placeholder in the HTML
            original_content = html_content
            html_content = html_content.replace('{SESSION_ID}', session_id)
            
            if original_content == html_content:
                logger.warning(f"⚠️ No {{SESSION_ID}} placeholder found in HTML content!")
            else:
                logger.info(f"✅ Replaced {{SESSION_ID}} placeholder with {session_id[:8]}...")
            
            # Render with minimal visibility (height=0 for silent operation)
            logger.info(f"🔄 Rendering fingerprint component for session {session_id[:8]}...")
            st.components.v1.html(html_content, height=0, width=0, scrolling=False)
            
            logger.info(f"✅ External fingerprint component rendered successfully for session {session_id[:8]}")
            return None  # Always return None since data comes via redirect
            
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
            'visitor_type': visitor_type,
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
    """Manages activity-based question limiting and ban statuses for different user tiers."""
    
    def __init__(self):
        self.question_limits = {
            UserType.GUEST.value: 4,
            UserType.EMAIL_VERIFIED_GUEST.value: 10,
            UserType.REGISTERED_USER.value: 40
        }
        self.evasion_penalties = [24, 48, 96, 192, 336]
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """Checks if the current session is within its allowed question limits or if any bans are active."""
        user_limit = self.question_limits.get(session.user_type.value, 0)
        
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
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                session.question_limit_reached = False
        
        if session.last_question_time:
            time_since_last = datetime.now() - session.last_question_time
            if time_since_last >= timedelta(hours=24):
                logger.info(f"Daily question count reset for session {session.session_id[:8]}.")
                session.daily_question_count = 0
                session.question_limit_reached = False
        
        if session.user_type.value == UserType.GUEST.value:
            if session.daily_question_count >= user_limit:
                return {
                    'allowed': False,
                    'reason': 'guest_limit',
                    'message': 'Please provide your email address to continue.'
                }
        
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            if session.daily_question_count >= user_limit:
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Email-verified daily limit reached")
                return {
                    'allowed': False,
                    'reason': 'daily_limit',
                    'message': self._get_email_verified_limit_message()
                }
        
        elif session.user_type.value == UserType.REGISTERED_USER.value:
            if session.total_question_count >= user_limit:
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Registered user total limit reached")
                return {
                    'allowed': False,
                    'reason': 'total_limit',
                    'message': "Usage limit reached. Please retry in 24 hours as we are giving preference to others in the queue."
                }
        
        return {'allowed': True}
    
    def record_question(self, session: UserSession):
        """Increments question counters for the session."""
        session.daily_question_count += 1
        if session.user_type.value == UserType.REGISTERED_USER.value:
            session.total_question_count += 1
        session.last_question_time = datetime.now()
        logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}.")
    
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
    
    def apply_evasion_penalty(self, session: UserSession) -> int:
        """Applies an escalating penalty for evasion attempts."""
        session.evasion_count += 1
        session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
        
        penalty_hours = self.evasion_penalties[session.escalation_level - 1]
        session.current_penalty_hours = penalty_hours
        
        self._apply_ban(session, BanStatus.EVASION_BLOCK, f"Evasion attempt #{session.evasion_count}")
        
        logger.warning(f"Evasion penalty applied to {session.session_id[:8]}: {penalty_hours}h (Level {session.escalation_level}).")
        return penalty_hours
    
    def _get_ban_message(self, session: UserSession) -> str:
        """Provides a user-friendly message for current bans."""
        if session.ban_status.value == BanStatus.EVASION_BLOCK.value:
            return "Usage limit reached due to detected unusual activity. Please try again later."
        elif session.user_type.value == UserType.REGISTERED_USER.value:
            return "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
        else:
            return self._get_email_verified_limit_message()
    
    def _get_email_verified_limit_message(self) -> str:
        """Specific message for email-verified guests hitting their daily limit."""
        return ("Our system is very busy and is being used by multiple users. For a fair assessment of our FiFi AI assistant and to provide fair usage to everyone, we can allow a total of 10 questions per day (20 messages). To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

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
        
        max_content_length = 32000
        if len(note_content) > max_content_length:
            note_content = note_content[:max_content_length - 100] + "\n\n[Content truncated due to size limits]"
        
        note_data = {
            "data": [{
                "Note_Title": note_title,
                "Note_Content": note_content,
                "Parent_Id": {"id": contact_id},
                "se_module": "Contacts"
            }]
        }
        
        try:
            response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=15)
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, attempting refresh for _add_note...")
                access_token = self._get_access_token(force_refresh=True)
                if not access_token: 
                    return False
                headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=15)
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                logger.info(f"Successfully added Zoho note: {note_title}")
                return True
            else:
                logger.error(f"Zoho note creation failed with response: {data}")
                
        except Exception as e:
            logger.error(f"Error adding Zoho note: {e}", exc_info=True)
            
        return False

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
        
        max_retries = 3 if "timeout" in trigger_reason.lower() or "emergency" in trigger_reason.lower() else 1
        
        for attempt in range(max_retries):
            logger.info(f"Zoho Save Attempt {attempt + 1}/{max_retries}")
            try:
                contact_id = self._find_contact_by_email(session.email)
                if not contact_id:
                    contact_id = self._create_contact(session.email, session.full_name)
                if not contact_id:
                    logger.error("Failed to find or create Zoho contact. Cannot proceed with save.")
                    if attempt == max_retries - 1: 
                        return False
                    time.sleep(2 ** attempt)
                    continue 
                session.zoho_contact_id = contact_id

                pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer:
                    logger.error("Failed to generate PDF transcript. Cannot proceed with attachment.")
                
                upload_success = False
                if pdf_buffer:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                    pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                    upload_success = self._upload_attachment(contact_id, pdf_buffer, pdf_filename)
                    if not upload_success:
                        logger.warning("Failed to upload PDF attachment to Zoho. Continuing with note only.")

                note_title = f"FiFi AI Chat Transcript from {datetime.now().strftime('%Y-%m-%d %H:%M')} ({trigger_reason})"
                note_content = self._generate_note_content(session, upload_success, trigger_reason)
                note_success = self._add_note(contact_id, note_title, note_content)
                
                if note_success:
                    logger.info("=" * 80)
                    logger.info(f"ZOHO SAVE COMPLETED SUCCESSFULLY on attempt {attempt + 1}")
                    logger.info(f"Contact ID: {contact_id}")
                    logger.info("=" * 80)
                    return True
                else:
                    logger.error("Failed to add note to Zoho contact.")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached. Aborting save.")
                        return False

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"ZOHO SAVE FAILED on attempt {attempt + 1} with an exception.")
                logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached. Aborting save.")
                    return False
        
        return False

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
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

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
        """Synthesize search results into a coherent response similar to LLM output."""
        
        # Handle string response from Tavily
        if isinstance(results, str):
            return f"Based on my search: {results}"
        
        # Handle dictionary response from Tavily (most common format)
        if isinstance(results, dict):
            # Check if there's a pre-made answer
            if results.get('answer'):
                return f"Based on my search: {results['answer']}"
            
            # Extract the results array
            search_results = results.get('results', [])
            if not search_results:
                return "I couldn't find any relevant information for your query."
            
            # Process the results
            relevant_info = []
            sources = []
            
            for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or 
                             result.get('snippet') or 
                             result.get('description') or 
                             result.get('summary', ''))
                    url = result.get('url', '')
                    
                    if content:
                        # Clean up content
                        if len(content) > 400:
                            content = content[:400] + "..."
                        relevant_info.append(content)
                        
                        if url and title:
                            sources.append(f"[{title}]({url})")
            
            if not relevant_info:
                return "I found search results but couldn't extract readable content. Please try rephrasing your query."
            
            # Build synthesized response
            response_parts = []
            
            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search, here's what I found:")
                for i, info in enumerate(relevant_info, 1):
                    response_parts.append(f"\n\n**{i}.** {info}")
            
            # Add sources
            if sources:
                response_parts.append(f"\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"\n{i}. {source}")
            
            return "".join(response_parts)
        
        # Handle direct list (fallback)
        if isinstance(results, list):
            relevant_info = []
            sources = []
            
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or 
                             result.get('snippet') or 
                             result.get('description', ''))
                    url = result.get('url', '')
                    
                    if content:
                        if len(content) > 400:
                            content = content[:400] + "..."
                        relevant_info.append(content)
                        if url:
                            sources.append(f"[{title}]({url})")
            
            if not relevant_info:
                return "I couldn't find relevant information for your query."
            
            response_parts = []
            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search:")
                for info in relevant_info:
                    response_parts.append(f"\n{info}")
            
            if sources:
                response_parts.append(f"\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"{i}. {source}")
            
            return "".join(response_parts)
        
        # Fallback for unknown formats
        return "I couldn't find any relevant information for your query."

    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            search_results = self.tavily_tool.invoke({"query": message})
            synthesized_content = self.synthesize_search_results(search_results, message)
            final_content = self.add_utm_to_links(synthesized_content)
            
            return {
                "content": final_content,
                "success": True,
                "source": "FiFi Web Search",
                "used_pinecone": False,
                "used_search": True,
                "has_citations": True,
                "has_inline_citations": True,
                "safety_override": False
            }
        except Exception as e:
            return {
                "content": f"I apologize, but an error occurred while searching: {str(e)}",
                "success": False,
                "source": "error",
                "used_pinecone": False,
                "used_search": False,
                "has_citations": False,
                "has_inline_citations": False,
                "safety_override": False
            }
    
class EnhancedAI:
    """Enhanced AI system with Pinecone knowledge base, web search fallback, content moderation, and anti-hallucination safety."""
    
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
            except Exception as e:
                logger.error(f"Pinecone tool initialization failed: {e}")
                self.pinecone_tool = None
        
        # Initialize Tavily agent
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(self.config.TAVILY_API_KEY)
                logger.info("✅ Tavily Web Search initialized successfully")
            except Exception as e:
                logger.error(f"Tavily agent initialization failed: {e}")
                self.tavily_agent = None

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
            return False  # Don't fallback for explicit "don't know" responses
        
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
        """Enhanced AI response with content moderation, Pinecone, web search, and safety features."""
        
        # ADD THIS CONTENT MODERATION CHECK AT THE BEGINNING:
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
        # STEP 2: Convert chat history to LangChain format if needed
        if chat_history:
            # Take only the last message (current prompt) and convert format
            langchain_history = []
            for msg in chat_history[-10:]:  # Limit to last 10 messages
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg.get("content", "")))
            
            # Add current prompt
            langchain_history.append(HumanMessage(content=prompt))
        else:
            langchain_history = [HumanMessage(content=prompt)]
        
        # STEP 3: Try Pinecone first if available
        if self.pinecone_tool:
            try:
                logger.info("🔍 Querying Pinecone knowledge base...")
                pinecone_response = self.pinecone_tool.query(langchain_history)
                
                if pinecone_response and pinecone_response.get("success"):
                    # Check if we should fallback to web search
                    should_fallback = self.should_use_web_fallback(pinecone_response)
                    
                    if not should_fallback:
                        logger.info("✅ Using Pinecone response (passed safety checks)")
                        return pinecone_response
                    else:
                        logger.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switching to verified web sources.")
                        # Mark that safety override occurred but continue to web search
                        
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}")
        
        # STEP 4: Fallback to web search if available
        if self.tavily_agent:
            try:
                logger.info("🌐 Falling back to web search...")
                web_response = self.tavily_agent.query(prompt, langchain_history[:-1])
                
                if web_response and web_response.get("success"):
                    logger.info("✅ Using web search response")
                    return web_response
                    
            except Exception as e:
                logger.error(f"Web search failed: {e}")
        
        # STEP 5: Final fallback - basic response
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

# =============================================================================
# SESSION MANAGER - MAIN ORCHESTRATOR CLASS
# =============================================================================

class SessionManager:
    """Main orchestrator class that manages user sessions, integrates all managers, and provides the primary interface for the application."""
    
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
        self._cleanup_interval = timedelta(hours=1)
        self._last_cleanup = datetime.now()
        
        logger.info("✅ SessionManager initialized with all component managers.")

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return 5  # CHANGED: 5 minute timeout for testing
    
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
            logger.warning(f"Messages field corrupted for session {session.session_id[:8]}, preserving as empty list")
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

    def _create_new_session(self) -> UserSession:
        """Creates a new user session with temporary fingerprint until JS fingerprinting completes."""
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id)
        
        # Apply temporary fingerprint until JS fingerprinting completes
        session.fingerprint_id = f"temp_py_{secrets.token_hex(8)}"
        session.fingerprint_method = "temporary_fallback_python"
        
        # Save to database
        self.db.save_session(session)
        
        logger.info(f"Created new session {session_id[:8]} with temporary fingerprint - waiting for JS fingerprinting")
        return session

    def _check_5min_eligibility(self, session: UserSession) -> bool:
        """Check if session has been active for at least 5 minutes to be eligible for CRM save."""
        try:
            # Use the earliest of session creation time or first question time
            start_time = session.created_at
            if session.last_question_time and session.last_question_time < start_time:
                start_time = session.last_question_time
            
            elapsed_time = datetime.now() - start_time
            elapsed_minutes = elapsed_time.total_seconds() / 60
            
            logger.info(f"5-min eligibility check for {session.session_id[:8]}: {elapsed_minutes:.1f} minutes elapsed")
            return elapsed_minutes >= 5.0  # CHANGED: 5 minute requirement
            
        except Exception as e:
            logger.error(f"Error checking 5-min eligibility for {session.session_id[:8]}: {e}")
            return False

    def _is_crm_save_eligible(self, session: UserSession, trigger_reason: str) -> bool:
        """Enhanced eligibility check for CRM saves including new user types and conditions."""
        try:
            # Basic eligibility requirements
            if not session.email or not session.messages:
                logger.info(f"CRM save not eligible - missing email or messages for {session.session_id[:8]}")
                return False
            
            # Check if already saved to avoid duplicates
            if session.timeout_saved_to_crm and "clear_chat" in trigger_reason.lower():
                logger.info(f"CRM save not eligible - already saved for {session.session_id[:8]}")
                return False
            
            # User type eligibility: registered_user OR email_verified_guest
            if session.user_type not in [UserType.REGISTERED_USER, UserType.EMAIL_VERIFIED_GUEST]:
                logger.info(f"CRM save not eligible - user type {session.user_type.value} for {session.session_id[:8]}")
                return False
            
            # Question count requirement: at least 1 question asked
            if session.daily_question_count < 1:
                logger.info(f"CRM save not eligible - no questions asked for {session.session_id[:8]}")
                return False
            
            # 5-minute eligibility check (CHANGED from 15 minutes)
            if not self._check_5min_eligibility(session):
                logger.info(f"CRM save not eligible - less than 5 minutes active for {session.session_id[:8]}")
                return False
            
            # All conditions met
            logger.info(f"CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Questions={session.daily_question_count}, 5min+")
            return True
            
        except Exception as e:
            logger.error(f"Error checking CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    def get_session(self) -> Optional[UserSession]:
        """Gets or creates the current user session."""
        # Perform periodic cleanup
        self._periodic_cleanup()

        try:
            # Try to get existing session from Streamlit session state
            session_id = st.session_state.get('current_session_id')
            
            if session_id:
                session = self.db.load_session(session_id)
                if session and session.active:
                    session = self._validate_session(session)
                    
                    # Enhanced session recovery - always ensure we have some fingerprint
                    if not session.fingerprint_id:
                        session.fingerprint_id = f"temp_fp_{session.session_id[:8]}"
                        session.fingerprint_method = "temporary_fallback_python"
                        try:
                            self.db.save_session(session)
                            logger.info(f"Applied temporary fallback fingerprint to session {session.session_id[:8]}.")
                        except Exception as e:
                            logger.error(f"Failed to save temporary fingerprint for session {session.session_id[:8]}: {e}", exc_info=True)

                    # Check limits and handle bans
                    limit_check = self.question_limits.is_within_limits(session)
                    if not limit_check.get('allowed', True):
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
                        
                        # Still update activity even if banned
                        try:
                            self._update_activity(session)
                        except Exception as e:
                            logger.error(f"Failed to update activity for banned session {session.session_id[:8]}: {e}", exc_info=True)
                        
                        return session
                    
                    # Update activity for allowed sessions
                    try:
                        self._update_activity(session)
                    except Exception as e:
                        logger.error(f"Failed to update session activity for {session.session_id[:8]}: {e}", exc_info=True)
                    
                    return session
                else:
                    logger.info(f"Session {session_id[:8]} not found or inactive. Creating new session.")
                    if 'current_session_id' in st.session_state:
                        del st.session_state['current_session_id']

            # Create new session if no valid session found
            new_session = self._create_new_session()
            st.session_state.current_session_id = new_session.session_id
            return self._validate_session(new_session)
            
        except Exception as e:
            logger.error(f"Failed to get/create session: {e}", exc_info=True)
            # Create fallback session in case of critical failure
            fallback_session = UserSession(session_id=str(uuid.uuid4()), user_type=UserType.GUEST)
            fallback_session.fingerprint_id = f"emergency_fp_{fallback_session.session_id[:8]}"
            fallback_session.fingerprint_method = "emergency_fallback"
            st.session_state.current_session_id = fallback_session.session_id
            st.error("⚠️ Failed to create or load session. Operating in emergency fallback mode. Chat history may not persist.")
            return fallback_session

    def _validate_session(self, session: UserSession) -> UserSession:
        """Validates and updates session activity."""
        session.last_activity = datetime.now()
        
        # Check for ban expiry
        if (session.ban_status != BanStatus.NONE and 
            session.ban_end_time and 
            datetime.now() >= session.ban_end_time):
            logger.info(f"Ban expired for session {session.session_id[:8]}")
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            session.question_limit_reached = False
        
        # Save updated session
        self.db.save_session(session)
        return session

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]):
        """Applies fingerprinting data from custom component to the session with better validation."""
        try:
            if not fingerprint_data or not isinstance(fingerprint_data, dict):
                logger.warning("Invalid fingerprint data provided to apply_fingerprinting")
                return
            
            old_fingerprint_id = session.fingerprint_id
            old_method = session.fingerprint_method
            
            session.fingerprint_id = fingerprint_data.get('fingerprint_id')
            session.fingerprint_method = fingerprint_data.get('fingerprint_method')
            session.visitor_type = fingerprint_data.get('visitor_type', 'new_visitor')
            session.browser_privacy_level = fingerprint_data.get('browser_privacy_level', 'standard')
            
            # Validate essential fields
            if not session.fingerprint_id or not session.fingerprint_method:
                logger.error("Invalid fingerprint data: missing essential fields")
                # Restore old values if they existed
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return
            
            # Check for existing sessions with same fingerprint
            try:
                existing_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
                if existing_sessions:
                    # Inherit recognition data from most recent session
                    recent_session = max(existing_sessions, key=lambda s: s.last_activity)
                    if recent_session.email and recent_session.user_type != UserType.GUEST:
                        session.visitor_type = "returning_visitor"
            except Exception as e:
                logger.error(f"Failed to check fingerprint history: {e}")
                # Continue without history check
            
            # Save session with new fingerprint data
            try:
                self.db.save_session(session)
                logger.info(f"✅ Fingerprinting applied to {session.session_id[:8]}: {session.fingerprint_method} (ID: {session.fingerprint_id[:8]}...)")
            except Exception as e:
                logger.error(f"Failed to save session after fingerprinting: {e}")
                # Restore old values on save failure
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                
        except Exception as e:
            logger.error(f"Error in apply_fingerprinting for session {session.session_id[:8]}: {e}", exc_info=True)

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """Checks fingerprint history for device recognition."""
        if not fingerprint_id:
            return {'has_history': False}
        
        sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
        if not sessions:
            return {'has_history': False}
        
        # Find most recent session with email
        email_sessions = [s for s in sessions if s.email and s.user_type != UserType.GUEST]
        if email_sessions:
            recent = max(email_sessions, key=lambda s: s.last_activity)
            return {
                'has_history': True,
                'email': recent.email,
                'last_seen': recent.last_activity,
                'user_type': recent.user_type.value
            }
        
        return {'has_history': False}

    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """Authenticates user with WordPress and creates/updates session."""
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
                email = data.get('user_email')
                display_name = data.get('user_display_name')
            
                # Get current session
                session = self.get_session()
            
                # Update session to registered user
                session.user_type = UserType.REGISTERED_USER
                session.email = email
                session.full_name = display_name
                session.wp_token = wp_token
            
                # Add email to email history if not already there
                if email not in session.email_addresses_used:
                    session.email_addresses_used.append(email)
            
                self.db.save_session(session)
            
                logger.info(f"WordPress authentication successful for {email}")
                return session
            
            else:
                st.error("Invalid username or password.")
                return None
            
        except Exception as e:
            logger.error(f"WordPress authentication failed: {e}")
            st.error("Authentication service is temporarily unavailable.")
            return None

    def _mask_email(self, email: str) -> str:
        """Masks an email address for privacy."""
        if '@' not in email:
            return email
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            return f"{local[0]}***@{domain}"
        return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Handles email verification for guest users."""
        try:
            # Update session with email
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email)
            
            self.db.save_session(session)
            
            # Send verification code
            success = self.email_verification.send_verification_code(email)
            if success:
                return {
                    'success': True,
                    'message': f"Verification code sent to {email}. Please check your email including spam folder."
                }
            else:
                return {
                    'success': False,
                    'message': "Failed to send verification code. Please try again later."
                }
                
        except Exception as e:
            logger.error(f"Email verification handling failed: {e}")
            return {
                'success': False,
                'message': "An error occurred while sending verification code."
            }

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verifies email verification code."""
        try:
            if not session.email:
                return {
                    'success': False,
                    'message': "No email address found for verification."
                }
            
            success = self.email_verification.verify_code(session.email, code)
            if success:
                # Upgrade session to email verified guest
                session.user_type = UserType.EMAIL_VERIFIED_GUEST
                session.question_limit_reached = False
                session.daily_question_count = 0  # Reset count after verification
                
                self.db.save_session(session)
                
                return {
                    'success': True,
                    'message': "Email verified successfully! You now have 10 questions per day."
                }
            else:
                return {
                    'success': False,
                    'message': "Invalid verification code. Please check the code and try again."
                }
                
        except Exception as e:
            logger.error(f"Email code verification failed: {e}")
            return {
                'success': False,
                'message': "Verification failed due to a technical error."
            }

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Enhanced version that prevents first-attempt failures"""
        try:
            # STEP A: Session refresh BEFORE any processing (prevents stale session issues)
            try:
                fresh_session = self.db.load_session(session.session_id)
                if fresh_session and fresh_session.active:
                    session = fresh_session
                    session.last_activity = datetime.now()
                    logger.debug(f"✅ Session refreshed before processing: {session.session_id[:8]}")
            except Exception as refresh_error:
                logger.error(f"Session refresh failed: {refresh_error}")

            # STEP B: Database connection health check (prevents connection failures)
            try:
                self.db._ensure_connection_healthy(self.config)
            except Exception as db_error:
                logger.error(f"Database connection check failed: {db_error}")

            # Rest of your existing logic remains the same...
            if not self.rate_limiter.is_allowed(session.session_id):
                return {'content': 'Please slow down - you are sending requests too quickly.', 'success': False}
        
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check['allowed']:
                if limit_check['reason'] == 'guest_limit':
                    return {'requires_email': True}
                elif limit_check['reason'] in ['banned', 'daily_limit', 'total_limit']:
                    return {'banned': True, 'content': limit_check.get('message', 'Access restricted.'), 'time_remaining': limit_check.get('time_remaining')}
        
            sanitized_prompt = sanitize_input(prompt)
            self.question_limits.record_question(session)
        
            user_message = {"role": "user", "content": sanitized_prompt}
            session.messages.append(user_message)
        
            ai_response = self.ai.get_response(sanitized_prompt, session.messages[-10:])
        
            assistant_message = {
                "role": "assistant",
                "content": ai_response.get("content", "No response generated."),
                "source": ai_response.get("source", "FiFi AI"),
                "used_search": ai_response.get("used_search", False),
                "used_pinecone": ai_response.get("used_pinecone", False),
                "has_citations": ai_response.get("has_citations", False),
                "has_inline_citations": ai_response.get("has_inline_citations", False),
                "safety_override": ai_response.get("safety_override", False)
            }
            session.messages.append(assistant_message)
        
            # STEP C: Enhanced session save with retry
            session.last_activity = datetime.now()
            self._save_session_with_retry(session)
        
            return ai_response
        
        except Exception as e:
            logger.error(f"AI response generation failed: {e}", exc_info=True)
        
            try:
                if 'user_message' in locals() and user_message not in session.messages:
                    session.messages.append(user_message)
            
                error_message = {
                    "role": "assistant", "content": "I encountered an error processing your request. Please try again.",
                    "source": "Error Handler", "used_search": False, "used_pinecone": False, 
                    "has_citations": False, "has_inline_citations": False, "safety_override": False
                }
                session.messages.append(error_message)
            
                session.last_activity = datetime.now()
                self._save_session_with_retry(session)
            except Exception as save_error:
                logger.error(f"Failed to save session after error: {save_error}")
        
            return {'content': 'I encountered an error processing your request. Please try again.', 'success': False, 'source': 'Error Handler'}
        
    def clear_chat_history(self, session: UserSession):
        """Enhanced clear chat history with CRM save functionality."""
        try:
            logger.info(f"Clear chat requested for session {session.session_id[:8]} with {len(session.messages)} messages")
            
            # Check if eligible for CRM save before clearing
            if self._is_crm_save_eligible(session, "Clear Chat Request"):
                logger.info(f"Performing CRM save before clearing chat for {session.session_id[:8]}")
                
                try:
                    # Save to CRM directly (no tab switching protection)
                    save_success = self.zoho.save_chat_transcript_sync(session, "Clear Chat Request")
                    
                    if save_success:
                        session.timeout_saved_to_crm = True
                        logger.info(f"✅ CRM save completed successfully before clearing chat for {session.session_id[:8]}")
                    else:
                        logger.warning(f"⚠️ CRM save failed before clearing chat for {session.session_id[:8]}")
                        
                except Exception as crm_error:
                    logger.error(f"CRM save error during clear chat for {session.session_id[:8]}: {crm_error}")
                    # Continue with clearing chat even if CRM save fails
            
            # Clear the chat messages
            session.messages = []
            session.last_activity = datetime.now()
            
            # Save to database (always happens regardless of CRM save status)
            self.db.save_session(session)
            logger.info(f"Chat history cleared and saved to DB for session {session.session_id[:8]}")
            
        except Exception as e:
            logger.error(f"Error clearing chat history for session {session.session_id[:8]}: {e}", exc_info=True)
            # Fallback: just clear and save to DB
            session.messages = []
            session.last_activity = datetime.now()
            try:
                self.db.save_session(session)
            except Exception as db_error:
                logger.error(f"Database save also failed during clear chat fallback: {db_error}")

    def end_session(self, session: UserSession):
        """Ends the current session and performs cleanup."""
        try:
            # Save to CRM if eligible (using the enhanced eligibility check)
            if self._is_crm_save_eligible(session, "Manual Sign Out"):
                logger.info(f"Performing CRM save during session end for {session.session_id[:8]}")
                
                # For sign out, save directly to CRM (no tab switching protection)
                save_success = self.zoho.save_chat_transcript_sync(session, "Manual Sign Out")
                if save_success:
                    session.timeout_saved_to_crm = True
            
            # Mark session as inactive
            session.active = False
            session.last_activity = datetime.now()
            self.db.save_session(session)
            
            # Clear Streamlit session state
            if 'current_session_id' in st.session_state:
                del st.session_state['current_session_id']
            if 'page' in st.session_state:
                del st.session_state['page']
            
            logger.info(f"Session {session.session_id[:8]} ended successfully")
            
        except Exception as e:
            logger.error(f"Error ending session {session.session_id[:8]}: {e}", exc_info=True)

    def manual_save_to_crm(self, session: UserSession):
        """Manually saves chat transcript to CRM."""
        if not session.messages:
            st.warning("No conversation to save.")
            return
        
        if session.user_type not in [UserType.REGISTERED_USER, UserType.EMAIL_VERIFIED_GUEST]:
            st.error("CRM saving is only available for registered users and email-verified guests.")
            return
        
        with st.spinner("Saving conversation to Zoho CRM..."):
            success = self.zoho.save_chat_transcript_sync(session, "Manual Save Request")
            
        if success:
            st.success("✅ Conversation saved to Zoho CRM successfully!")
            session.timeout_saved_to_crm = True
            self.db.save_session(session)
        else:
            st.error("❌ Failed to save to CRM. Please try again later.")

# =============================================================================
# WORKING 5-MINUTE TIMEOUT SYSTEM
# =============================================================================

def render_working_timeout_system(session_id: str):
    """
    WORKING: Complete JavaScript-controlled 5-minute timeout system.
    Shows warnings, countdown, extend button, and handles redirect.
    """
    if not session_id:
        return
    
    timeout_system_js = f"""
    <div id="timeout-overlay" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 10000; color: white; font-family: Arial;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #ff4444; padding: 30px; border-radius: 10px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <h2 style="margin: 0 0 15px 0;">⚠️ Session Expiring Soon!</h2>
            <p style="margin: 0 0 15px 0; font-size: 18px;">Your session expires in <span id="countdown-timer">60</span> seconds</p>
            <button id="extend-btn" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer; margin-right: 10px;">Extend for 5 Minutes</button>
            <button id="signout-btn" style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer;">Sign Out Now</button>
        </div>
    </div>

    <script>
    (function() {{
        const SESSION_ID = '{session_id}';
        const TIMEOUT_MINUTES = 5;  // 5 minute timeout
        const WARNING_SECONDS = 60; // Show warning 60 seconds before timeout
        
        if (window.fifi_timeout_system_active) return;
        window.fifi_timeout_system_active = true;
        
        console.log('🕐 Starting 5-minute timeout system for session:', SESSION_ID.substring(0, 8));
        
        let startTime = Date.now();
        let lastActivity = Date.now();
        let warningShown = false;
        let timeoutTriggered = false;
        let countdownInterval = null;
        
        // Get overlay elements
        const overlay = document.getElementById('timeout-overlay');
        const countdownTimer = document.getElementById('countdown-timer');
        const extendBtn = document.getElementById('extend-btn');
        const signoutBtn = document.getElementById('signout-btn');
        
        function updateActivity() {{
            lastActivity = Date.now();
            console.log('💓 Activity updated');
            
            // Hide warning if shown
            if (warningShown && overlay) {{
                hideWarning();
            }}
        }}
        
        function showWarning() {{
            if (warningShown || timeoutTriggered) return;
            warningShown = true;
            
            console.log('⚠️ Showing timeout warning');
            
            if (overlay) {{
                overlay.style.display = 'block';
                startCountdown();
            }}
        }}
        
        function hideWarning() {{
            if (!warningShown) return;
            warningShown = false;
            
            console.log('✅ Hiding timeout warning');
            
            if (overlay) {{
                overlay.style.display = 'none';
            }}
            
            if (countdownInterval) {{
                clearInterval(countdownInterval);
                countdownInterval = null;
            }}
        }}
        
        function startCountdown() {{
            let secondsLeft = WARNING_SECONDS;
            
            if (countdownTimer) {{
                countdownTimer.textContent = secondsLeft;
            }}
            
            countdownInterval = setInterval(() => {{
                secondsLeft--;
                
                if (countdownTimer) {{
                    countdownTimer.textContent = secondsLeft;
                }}
                
                if (secondsLeft <= 0) {{
                    clearInterval(countdownInterval);
                    triggerTimeout('timeout_5min_automatic');
                }}
            }}, 1000);
        }}
        
        function extendSession() {{
            console.log('🔄 Extending session for 5 minutes');
            
            // Reset timers
            startTime = Date.now();
            lastActivity = Date.now();
            hideWarning();
            
            // Notify Streamlit via query param
            const currentUrl = new URL(window.location.href);
            currentUrl.searchParams.set('extend_session', 'true');
            currentUrl.searchParams.set('session_id', SESSION_ID);
            window.location.href = currentUrl.toString();
        }}
        
        function signOutNow() {{
            console.log('🚪 Manual sign out requested');
            triggerTimeout('manual_signout_from_warning');
        }}
        
        function triggerTimeout(reason) {{
            if (timeoutTriggered) return;
            timeoutTriggered = true;
            
            console.log('🚨 Triggering timeout:', reason);
            
            // First try emergency save
            const emergencyData = JSON.stringify({{
                session_id: SESSION_ID,
                reason: reason,
                timestamp: Date.now()
            }});
            
            if (navigator.sendBeacon) {{
                try {{
                    const sent = navigator.sendBeacon(
                        'https://fifi-beacon-fastapi-121263692901.europe-west4.run.app/emergency-save',
                        new Blob([emergencyData], {{type: 'application/json'}})
                    );
                    console.log('📡 Emergency save beacon sent:', sent);
                }} catch (e) {{
                    console.error('❌ Beacon failed:', e);
                }}
            }}
            
            // Redirect to timeout handler
            const timeoutUrl = new URL(window.location.href);
            timeoutUrl.searchParams.set('event', 'session_timeout_5min');
            timeoutUrl.searchParams.set('session_id', SESSION_ID);
            timeoutUrl.searchParams.set('reason', reason);
            timeoutUrl.searchParams.set('timestamp', Date.now());
            
            window.location.href = timeoutUrl.toString();
        }}
        
        // Set up activity listeners
        const events = ['mousedown', 'mousemove', 'keydown', 'click', 'scroll', 'touchstart', 'focus'];
        events.forEach(eventType => {{
            document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
        }});
        
        // Try to monitor parent document
        try {{
            if (window.parent && window.parent !== window && window.parent.document) {{
                events.forEach(eventType => {{
                    window.parent.document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
                }});
                console.log('📍 Parent activity monitoring enabled');
            }}
        }} catch (e) {{
            console.debug('Cannot monitor parent activity:', e);
        }}
        
        // Set up extend and signout button handlers
        if (extendBtn) {{
            extendBtn.addEventListener('click', extendSession);
        }}
        
        if (signoutBtn) {{
            signoutBtn.addEventListener('click', signOutNow);
        }}
        
        // Main timeout checker - runs every 10 seconds
        const checker = setInterval(() => {{
            if (timeoutTriggered) {{
                clearInterval(checker);
                return;
            }}
            
            const now = Date.now();
            const inactiveTime = now - lastActivity;
            const totalTime = now - startTime;
            const inactiveMinutes = inactiveTime / (1000 * 60);
            const totalMinutes = totalTime / (1000 * 60);
            
            console.log(`🕐 Timer check: ${{inactiveMinutes.toFixed(1)}}min inactive, ${{totalMinutes.toFixed(1)}}min total`);
            
            // Show warning at 4 minutes of inactivity
            if (inactiveMinutes >= 4 && !warningShown) {{
                showWarning();
            }}
            
            // Force timeout at 5 minutes of inactivity
            if (inactiveMinutes >= 5) {{
                clearInterval(checker);
                triggerTimeout('timeout_5min_inactivity');
            }}
        }}, 10000); // Check every 10 seconds
        
        console.log('✅ 5-minute timeout system active');
    }})();
    </script>
