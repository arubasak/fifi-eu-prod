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

# =============================================================================
# FINAL INTEGRATED FIFI AI - FIXED FINGERPRINTING, NO DEBUG PANELS
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
    
    def _ensure_connection(self, config_instance: Any):
        """Ensure database connection is healthy, reconnect if needed"""
        if not self._check_connection_health():
            logger.warning("Database connection unhealthy, attempting reconnection...")
            old_conn = self.conn
            self.conn = None
            
            # Try to close old connection
            if old_conn:
                try:
                    old_conn.close()
                except Exception as e:
                    logger.debug(f"Error closing old DB connection: {e}")
            
            # Attempt reconnection
            if self.db_type == "cloud" and SQLITECLOUD_AVAILABLE:
                self.conn, _ = self._try_sqlite_cloud(config_instance.SQLITE_CLOUD_CONNECTION)
            elif self.db_type == "file":
                self.conn, _ = self._try_local_sqlite()
                
            if not self.conn:
                logger.error("Database reconnection failed, falling back to in-memory storage")
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
                self._ensure_connection(current_config)

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
                self._ensure_connection(current_config)

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
                self._ensure_connection(current_config)

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
                self._ensure_connection(current_config)

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
    """Manages browser fingerprinting using inline JavaScript component."""
    
    def __init__(self):
        self.fingerprint_cache = {}
        self.component_attempts = defaultdict(int)

    def render_fingerprint_component(self, session_id: str):
        """Renders fingerprinting component with inline HTML - no external files needed."""
        try:
            # Create comprehensive inline fingerprinting HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>FiFi Fingerprinting</title>
                <style>
                    body {{ margin: 0; padding: 0; height: 0; overflow: hidden; }}
                </style>
            </head>
            <body>
                <script>
                (function() {{
                    const sessionId = '{session_id}';
                    let fingerprintData = {{}};
                    
                    try {{
                        console.log('🔍 FiFi Fingerprinting: Starting fingerprint generation for session', sessionId.substring(0, 8));
                        
                        // Collect comprehensive browser fingerprint
                        fingerprintData = {{
                            timestamp: Date.now(),
                            userAgent: navigator.userAgent || '',
                            language: navigator.language || '',
                            languages: navigator.languages ? navigator.languages.join(',') : '',
                            platform: navigator.platform || '',
                            cookieEnabled: navigator.cookieEnabled,
                            doNotTrack: navigator.doNotTrack || 'unknown',
                            hardwareConcurrency: navigator.hardwareConcurrency || 0,
                            maxTouchPoints: navigator.maxTouchPoints || 0,
                            screen: {{
                                width: screen.width || 0,
                                height: screen.height || 0,
                                availWidth: screen.availWidth || 0,
                                availHeight: screen.availHeight || 0,
                                colorDepth: screen.colorDepth || 0,
                                pixelDepth: screen.pixelDepth || 0
                            }},
                            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'unknown',
                            timezoneOffset: new Date().getTimezoneOffset(),
                            hasLocalStorage: typeof(Storage) !== "undefined",
                            hasSessionStorage: typeof(Storage) !== "undefined" && typeof(sessionStorage) !== "undefined",
                            hasIndexedDB: typeof(indexedDB) !== "undefined",
                            hasWebGL: false,
                            hasCanvas: false,
                            audioContext: false
                        }};
                        
                        // Test Canvas fingerprinting
                        try {{
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d');
                            if (ctx) {{
                                fingerprintData.hasCanvas = true;
                                ctx.textBaseline = 'top';
                                ctx.font = '14px Arial';
                                ctx.fillText('FiFi Fingerprint Test 🔍', 2, 2);
                                fingerprintData.canvasFingerprint = canvas.toDataURL().substring(0, 100);
                            }}
                        }} catch (e) {{
                            console.debug('Canvas fingerprinting failed:', e);
                        }}
                        
                        // Test WebGL
                        try {{
                            const canvas = document.createElement('canvas');
                            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                            if (gl) {{
                                fingerprintData.hasWebGL = true;
                                const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                                if (debugInfo) {{
                                    fingerprintData.webglVendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || '';
                                    fingerprintData.webglRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '';
                                }}
                            }}
                        }} catch (e) {{
                            console.debug('WebGL fingerprinting failed:', e);
                        }}
                        
                        // Test Audio Context
                        try {{
                            const AudioContext = window.AudioContext || window.webkitAudioContext;
                            if (AudioContext) {{
                                const audioCtx = new AudioContext();
                                fingerprintData.audioContext = true;
                                fingerprintData.audioSampleRate = audioCtx.sampleRate || 0;
                                audioCtx.close();
                            }}
                        }} catch (e) {{
                            console.debug('Audio fingerprinting failed:', e);
                        }}
                        
                        // Generate unique fingerprint ID using collected data
                        const fingerprintString = JSON.stringify(fingerprintData);
                        let hash = 0;
                        for (let i = 0; i < fingerprintString.length; i++) {{
                            const char = fingerprintString.charCodeAt(i);
                            hash = ((hash << 5) - hash) + char;
                            hash = hash & hash; // Convert to 32-bit integer
                        }}
                        
                        const fingerprintId = 'fp_' + Math.abs(hash).toString(36) + '_' + Date.now().toString(36);
                        
                        // Determine method and privacy level
                        let method = 'basic';
                        let workingMethods = ['basic'];
                        
                        if (fingerprintData.hasCanvas) {{
                            method = 'enhanced';
                            workingMethods.push('canvas');
                        }}
                        if (fingerprintData.hasWebGL) {{
                            method = 'comprehensive';
                            workingMethods.push('webgl');
                        }}
                        if (fingerprintData.audioContext) {{
                            method = 'comprehensive';
                            workingMethods.push('audio');
                        }}
                        
                        // Estimate privacy level
                        let privacyLevel = 'standard';
                        if (navigator.doNotTrack === '1' || !navigator.cookieEnabled) {{
                            privacyLevel = 'enhanced';
                        }}
                        if (!fingerprintData.hasCanvas && !fingerprintData.hasWebGL) {{
                            privacyLevel = 'high';
                        }}
                        
                        console.log('✅ FiFi Fingerprinting: Generated fingerprint', fingerprintId, 'using method', method);
                        
                        // Redirect with fingerprint data
                        function getAppUrl() {{
                            try {{
                                if (window.parent && window.parent.location && 
                                    window.parent.location.origin === window.location.origin) {{
                                    return window.parent.location.origin + window.parent.location.pathname;
                                }}
                            }} catch (e) {{
                                console.debug('Parent location access failed, using current location');
                            }}
                            return window.location.origin + window.location.pathname;
                        }}
                        
                        const appUrl = getAppUrl();
                        const redirectUrl = appUrl + 
                            '?event=fingerprint_complete' +
                            '&session_id=' + encodeURIComponent(sessionId) +
                            '&fingerprint_id=' + encodeURIComponent(fingerprintId) +
                            '&method=' + encodeURIComponent(method) +
                            '&privacy=' + encodeURIComponent(privacyLevel) +
                            '&working_methods=' + encodeURIComponent(workingMethods.join(',')) +
                            '&timestamp=' + Date.now();
                        
                        console.log('🔄 FiFi Fingerprinting: Redirecting with data to', redirectUrl.substring(0, 100) + '...');
                        
                        // Perform redirect
                        setTimeout(() => {{
                            try {{
                                if (window.parent && window.parent.location && 
                                    window.parent.location.origin === window.location.origin) {{
                                    window.parent.location.href = redirectUrl;
                                }} else {{
                                    window.location.href = redirectUrl;
                                }}
                            }} catch (e) {{
                                console.error('Redirect failed:', e);
                            }}
                        }}, 500);
                        
                    }} catch (error) {{
                        console.error('🚨 FiFi Fingerprinting: Critical error during fingerprint generation:', error);
                        
                        // Fallback redirect with minimal data
                        const fallbackId = 'fallback_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
                        const appUrl = window.location.origin + window.location.pathname;
                        const fallbackUrl = appUrl + 
                            '?event=fingerprint_complete' +
                            '&session_id=' + encodeURIComponent(sessionId) +
                            '&fingerprint_id=' + encodeURIComponent(fallbackId) +
                            '&method=fallback' +
                            '&privacy=unknown' +
                            '&working_methods=none' +
                            '&timestamp=' + Date.now();
                        
                        setTimeout(() => {{
                            window.location.href = fallbackUrl;
                        }}, 1000);
                    }}
                }})();
                </script>
            </body>
            </html>
            """
            
            # Render with minimal visibility
            st.components.v1.html(html_content, height=0, width=0, scrolling=False)
            
            logger.debug(f"Inline fingerprint component rendered for session {session_id[:8]}")
            return None  # Always return None since data comes via redirect
            
        except Exception as e:
            logger.error(f"Failed to render inline fingerprint component: {e}")
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
        
        if (session.user_type.value != UserType.REGISTERED_USER.value or 
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
    
class EnhancedAI:
    """Placeholder for the AI interaction logic."""
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")

    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Provides a simplified AI response."""
        return {
            "content": f"I understand you're asking about: '{prompt}'. This is the integrated FiFi AI. Your question is processed based on your user tier and system limits.",
            "source": "Integrated FiFi AI System Placeholder",
            "used_search": False,
            "used_pinecone": False,
            "has_citations": False,
            "has_inline_citations": False,
            "safety_override": False,
            "success": True
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
        return 15
    
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
        except Exception as e:    logger.error(f"Failed to save session during activity update: {e}", exc_info=True)

    def _create_new_session(self) -> UserSession:
        """Creates a new user session with immediate working fingerprint."""
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id)
        
        # Apply working fingerprint immediately instead of temporary
        fallback_data = self.fingerprinting._generate_fallback_fingerprint()
        session.fingerprint_id = fallback_data['fingerprint_id']
        session.fingerprint_method = fallback_data['fingerprint_method']
        session.visitor_type = fallback_data['visitor_type']
        session.browser_privacy_level = fallback_data['browser_privacy_level']
        
        # Save to database
        self.db.save_session(session)
        
        logger.info(f"Created new session {session_id[:8]} with working fingerprint {session.fingerprint_id[:12]}...")
        return session

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
            fallback_data = self.fingerprinting._generate_fallback_fingerprint()
            fallback_session.fingerprint_id = fallback_data['fingerprint_id']
            fallback_session.fingerprint_method = fallback_data['fingerprint_method']
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

    def _mask_email(self, email: str) -> str:
        """Masks an email address for privacy."""
        if '@' not in email:
            return email
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            return f"{local[0]}***@{domain}"
        return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"

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
        """Gets AI response for user prompt with all checks and limits."""
        try:
            # Rate limiting check
            if not self.rate_limiter.is_allowed(session.session_id):
                return {
                    'content': 'Please slow down - you are sending requests too quickly.',
                    'success': False
                }
            
            # Question limit check
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check['allowed']:
                if limit_check['reason'] == 'guest_limit':
                    return {'requires_email': True}
                elif limit_check['reason'] in ['banned', 'daily_limit', 'total_limit']:
                    return {
                        'banned': True,
                        'content': limit_check.get('message', 'Access restricted.'),
                        'time_remaining': limit_check.get('time_remaining')
                    }
            
            # Content moderation (if available)
            sanitized_prompt = sanitize_input(prompt)
            moderation_result = check_content_moderation(sanitized_prompt, self.ai.openai_client)
            if moderation_result and moderation_result.get('flagged'):
                return {
                    'content': moderation_result.get('message', 'Your message violates content policy.'),
                    'success': False
                }
            
            # Record the question
            self.question_limits.record_question(session)
            
            # Add user message to session
            user_message = {"role": "user", "content": sanitized_prompt}
            session.messages.append(user_message)
            
            # Get AI response
            ai_response = self.ai.get_response(sanitized_prompt, session.messages[-10:])
            
            # Add AI response to session
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
            
            # Save session with new messages
            session.last_activity = datetime.now()
            self.db.save_session(session)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}", exc_info=True)
            return {
                'content': 'I encountered an error processing your request. Please try again.',
                'success': False,
                'source': 'Error Handler'
            }

    def clear_chat_history(self, session: UserSession):
        """Clears chat history for the session."""
        session.messages = []
        session.last_activity = datetime.now()
        self.db.save_session(session)
        logger.info(f"Chat history cleared for session {session.session_id[:8]}")

    def end_session(self, session: UserSession):
        """Ends the current session and performs cleanup."""
        try:
            # Save to CRM if eligible
            if (session.user_type == UserType.REGISTERED_USER and 
                session.email and 
                session.messages and 
                not session.timeout_saved_to_crm):
                
                logger.info(f"Performing CRM save during session end for {session.session_id[:8]}")
                self.zoho.save_chat_transcript_sync(session, "Manual Sign Out")
            
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
        
        if session.user_type != UserType.REGISTERED_USER:
            st.error("CRM saving is only available for registered users.")
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
# JAVASCRIPT COMPONENTS & EVENT HANDLING
# =============================================================================

def render_activity_timer_component_15min(session_id: str) -> Optional[Dict[str, Any]]:
    """Renders a JavaScript component that tracks user inactivity and triggers an event after 15 minutes."""
    if not session_id:
        logger.warning("❌ Timer component: No session ID provided")
        return None
    
    safe_session_id_js = session_id.replace('-', '_')
    
    js_timer_code = f"""
    (() => {{
        try {{
            const sessionId = "{session_id}";
            const SESSION_TIMEOUT_MS = 900000; // 15 minutes
            
            console.log("🕐 FiFi 15-Minute Timer: Checking session", sessionId.substring(0, 8));
            
            // Initialize or reset timer state
            if (typeof window.fifi_timer_state_{safe_session_id_js} === 'undefined' || 
                window.fifi_timer_state_{safe_session_id_js} === null || 
                window.fifi_timer_state_{safe_session_id_js}.sessionId !== sessionId) {{
                
                console.log("🆕 FiFi 15-Minute Timer: Starting/Resetting for session", sessionId.substring(0, 8)); 
                window.fifi_timer_state_{safe_session_id_js} = {{
                    lastActivityTime: Date.now(),
                    expired: false,
                    listenersInitialized: false,
                    sessionId: sessionId
                }};
            }}
            
            const state = window.fifi_timer_state_{safe_session_id_js};
            
            // Setup activity listeners (only once)
            if (!state.listenersInitialized) {{
                console.log("👂 Setting up FiFi 15-Minute activity listeners...");
                
                function resetActivity() {{
                    try {{
                        const now = Date.now();
                        if (state.lastActivityTime !== now) {{
                            state.lastActivityTime = now;
                            if (state.expired) {{
                                console.log("🔄 Activity detected, resetting expired flag for timer.");
                                state.expired = false;
                            }}
                        }}
                    }} catch (e) {{
                        console.debug("Error in resetActivity:", e);
                    }}
                }}
                
                // Activity event types to monitor
                const events = [
                    'mousedown', 'mousemove', 'mouseup', 'click', 'dblclick',
                    'keydown', 'keyup', 'keypress',
                    'scroll', 'wheel',
                    'touchstart', 'touchmove', 'touchend',
                    'focus'
                ];
                
                // Add listeners to current document
                events.forEach(eventType => {{
                    try {{
                        document.addEventListener(eventType, resetActivity, {{ 
                            passive: true, 
                            capture: true
                        }});
                    }} catch (e) {{
                        console.debug(`Failed to add ${{eventType}} listener:`, e);
                    }}
                }});
                
                // Try to add listeners to parent document (for iframes)
                try {{
                    if (window.parent && window.parent.document && 
                        window.parent.document !== document &&
                        window.parent.location.origin === window.location.origin) {{
                        
                        events.forEach(eventType => {{
                            try {{
                                window.parent.document.addEventListener(eventType, resetActivity, {{ 
                                    passive: true, 
                                    capture: true
                                }});
                            }} catch (e) {{
                                console.debug(`Failed to add parent ${{eventType}} listener:`, e);
                            }}
                        }});
                        console.log("👂 Parent document listeners added successfully.");
                    }}
                }} catch (e) {{
                    console.debug("Cannot access parent document for listeners:", e);
                }}
                
                // Visibility change handlers
                const handleVisibilityChange = () => {{
                    try {{
                        if (document.visibilityState === 'visible') {{
                            resetActivity();
                        }}
                    }} catch (e) {{
                        console.debug("Visibility change error:", e);
                    }}
                }};
                
                document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
                
                try {{
                    if (window.parent && window.parent.document && window.parent.document !== document) {{
                        window.parent.document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
                    }}
                }} catch (e) {{
                    console.debug("Cannot setup parent visibility detection:", e);
                }}
                
                state.listenersInitialized = true;
                console.log("✅ FiFi 15-Minute activity listeners initialized.");
            }}
            
            // Check for timeout
            const currentTime = Date.now();
            const inactiveTimeMs = currentTime - state.lastActivityTime;
            const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
            const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
            
            // Log every 5 minutes for debugging (Python side will also log)
            if (inactiveMinutes > 0 && inactiveMinutes % 5 === 0 && inactiveSeconds < 10) {{
                console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
            }}
            
            // Check if timeout reached
            if (inactiveTimeMs >= SESSION_TIMEOUT_MS && !state.expired) {{
                state.expired = true;
                console.log("🚨 15-MINUTE SESSION TIMEOUT REACHED for session", sessionId.substring(0, 8));
                
                return {{
                    event: "session_timeout_15min",
                    session_id: sessionId,
                    inactive_time_ms: inactiveTimeMs,
                    inactive_minutes: inactiveMinutes,
                    inactive_seconds: inactiveSeconds,
                    timestamp: currentTime
                }};
            }}
            
            return null;
            
        }} catch (error) {{
            console.error("🚨 FiFi 15-Minute Timer component caught a critical error:", error);
            return {{
                event: "timer_error",
                session_id: "{session_id}",
                error: error.message,
                timestamp: Date.now()
            }};
        }}
    }})()
    """
    
    try:
        # Execute the timer JavaScript code
        timer_result = st_javascript(js_timer_code)
        
        # Validate the result
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            return None
        
        if isinstance(timer_result, dict):
            if timer_result.get('event') == "session_timeout_15min":
                if timer_result.get('session_id') == session_id:
                    logger.info(f"✅ Valid 15-min timer event received: {timer_result.get('event')} for session {session_id[:8]}.")
                    return timer_result
                else:
                    logger.warning(f"⚠️ Timer event session ID mismatch: expected {session_id[:8]}, got {timer_result.get('session_id', 'None')[:8]}. Event ignored.")
                    return None
            elif timer_result.get('event') == "timer_error":
                logger.error(f"❌ Timer component error for session {session_id[:8]}: {timer_result.get('error', 'Unknown error')}")
                return None
            else:
                logger.debug(f"Received non-timeout timer result: {timer_result.get('event', 'unknown')}")
                return None
        else:
            logger.debug(f"Received non-dict timer result: {timer_result} (type: {type(timer_result)}).")
            return None
        
    except Exception as e:
        logger.error(f"❌ JavaScript timer component execution error for session {session_id[:8]}: {e}", exc_info=True)
        return None

def render_browser_close_detection_enhanced(session_id: str):
    """Enhanced browser close detection with multiple fallback mechanisms"""
    if not session_id:
        return

    safe_session_id_js = session_id.replace('-', '_')

    js_code = f"""
    <script>
    (function() {{
        const scriptIdentifier = 'fifi_close_enhanced_' + '{safe_session_id_js}';
        if (window[scriptIdentifier]) return;
        window[scriptIdentifier] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false;
        let heartbeatInterval = null;
        
        console.log('🛡️ Enhanced browser close detection initialized for session', sessionId.substring(0, 8));
        
        function getAppUrl() {{
            try {{
                if (window.parent && window.parent.location && 
                    window.parent.location.origin === window.location.origin) {{
                    return window.parent.location.origin + window.parent.location.pathname;
                }}
            }} catch (e) {{
                console.warn("Using current window location as fallback");
            }}
            return window.location.origin + window.location.pathname;
        }}

        function triggerEmergencySave(reason = 'unknown') {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            console.log('🚨 Browser close detected (' + reason + ') - triggering emergency save');
            
            const appUrl = getAppUrl();
            const saveUrl = `${{appUrl}}?event=emergency_close&session_id=${{sessionId}}&reason=${{reason}}`;
            
            if (heartbeatInterval) {{
                clearInterval(heartbeatInterval);
                heartbeatInterval = null;
            }}
            
            try {{
                if (window.parent && window.parent.location && 
                    window.parent.location.origin === window.location.origin) {{
                    window.parent.location.href = saveUrl;
                }} else {{
                    window.location.href = saveUrl;
                }}
            }} catch (e) {{
                console.error('Emergency save redirect failed:', e);
                if (navigator.sendBeacon) {{
                    try {{
                        navigator.sendBeacon(saveUrl.replace('?', '/beacon?'), 
                            'emergency_save=true&session_id=' + sessionId);
                    }} catch (beaconError) {{
                        console.error('Beacon fallback also failed:', beaconError);
                    }}
                }}
            }}
        }}
        
        const events = ['beforeunload', 'pagehide', 'unload'];
        events.forEach(eventType => {{
            try {{
                window.addEventListener(eventType, () => triggerEmergencySave(eventType), {{ 
                    capture: true, passive: true 
                }});
                
                if (window.parent && window.parent !== window) {{
                    window.parent.addEventListener(eventType, () => triggerEmergencySave('parent_' + eventType), {{ 
                        capture: true, passive: true 
                    }});
                }}
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        function handleVisibilityChange() {{
            try {{
                if (document.visibilityState === 'hidden') {{
                    setTimeout(() => {{
                        if (document.hidden) {{
                            triggerEmergencySave('visibility_hidden');
                        }}
                    }}, 2000);
                }}
            }} catch (e) {{
                console.debug('Visibility change handling failed:', e);
            }}
        }}
        
        document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
        try {{
            if (window.parent && window.parent.document && window.parent.document !== document) {{
                window.parent.document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
            }}
        }} catch (e) {{
            console.debug('Parent visibility detection setup failed:', e);
        }}
        
        let lastHeartbeat = Date.now();
        heartbeatInterval = setInterval(() => {{
            const now = Date.now();
            if (now - lastHeartbeat > 60000) {{
                triggerEmergencySave('heartbeat_timeout');
            }}
            lastHeartbeat = now;
        }}, 30000);
        
        let wasVisible = !document.hidden;
        setInterval(() => {{
            const isVisible = !document.hidden;
            if (wasVisible && !isVisible) {{
                setTimeout(() => {{
                    if (document.hidden) {{
                        triggerEmergencySave('tab_hidden_timeout');
                    }}
                }}, 5000);
            }}
            wasVisible = isVisible;
        }}, 1000);
        
        console.log('✅ Enhanced browser close detection fully initialized');
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close component: {e}", exc_info=True)

def handle_timer_event(timer_result: Dict[str, Any], session_manager: 'SessionManager', session: UserSession) -> bool:
    """Processes events triggered by the JavaScript activity timer (e.g., 15-minute timeout)."""
    if not timer_result or not isinstance(timer_result, dict):
        return False
    
    event = timer_result.get('event')
    session_id = timer_result.get('session_id')
    inactive_minutes = timer_result.get('inactive_minutes', 0)
    
    logger.info(f"🎯 Processing timer event: '{event}' for session {session_id[:8] if session_id else 'unknown'}.")
    
    try:
        session = session_manager._validate_session(session)
        
        if event == 'session_timeout_15min':
            st.info(f"⏰ **Session timeout:** Detected {inactive_minutes} minutes of inactivity.")
            
            if (session.user_type.value == UserType.REGISTERED_USER.value and
                session.email and 
                session.messages and
                not session.timeout_saved_to_crm):
                
                with st.spinner("💾 Auto-saving chat to CRM (15-min timeout)..."):
                    try:
                        save_success = session_manager.zoho.save_chat_transcript_sync(session, "15-Minute Session Inactivity Timeout")
                    except Exception as e:
                        logger.error(f"15-min timeout CRM save failed during execution: {e}", exc_info=True)
                        save_success = False
                
                if save_success:
                    st.success("✅ Chat automatically saved to CRM!")
                    session.timeout_saved_to_crm = True
                    session.last_activity = datetime.now()
                    session_manager.db.save_session(session)
                else:
                    st.warning("⚠️ Auto-save to CRM failed. Please check your credentials or contact support if issue persists.")
                
                st.info("ℹ️ You can continue using FiFi AI.")
                return False
            else:
                st.info("ℹ️ Session timeout detected, but no CRM save was performed (e.g., Guest user, no chat history, or already saved).")
                logger.info(f"15-min timeout CRM save eligibility check failed for {session_id[:8]}: UserType={session.user_type.value}, Email={bool(session.email)}, Messages={len(session.messages)}, Saved Status={session.timeout_saved_to_crm}.")
                st.info("ℹ️ You can continue using FiFi AI.")
                return False
                
        else:
            logger.warning(f"⚠️ Received unhandled timer event type: '{event}'.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing timer event '{event}' for session {session_id[:8]}: {e}", exc_info=True)
        st.error(f"⚠️ An internal error occurred while processing activity. Please try refreshing if issues persist.")
        return False

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
        
        # Create processed fingerprint data
        processed_data = {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': method,
            'visitor_type': 'new_visitor',  # Can be enhanced later with cache checking
            'browser_privacy_level': privacy,
            'working_methods': working_methods
        }
        
        # Apply fingerprinting to session
        session_manager.apply_fingerprinting(session, processed_data)
        
        logger.info(f"✅ Fingerprint applied successfully to session '{session_id[:8]}'")
        return True
        
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
        
        # Check if eligible for CRM save
        if (session.user_type.value == UserType.REGISTERED_USER.value and
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            success = session_manager.zoho.save_chat_transcript_sync(session, f"Emergency Save: {reason}")
            if success:
                session.timeout_saved_to_crm = True
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

    if event == "emergency_close" and session_id:
        logger.info("=" * 80)
        logger.info("🚨 EMERGENCY SAVE REQUEST DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}, Reason: {reason}")
        logger.info("=" * 80)
        
        st.error("🚨 **Emergency Save Detected** - Processing browser close save...")
        st.info("Please wait, your conversation is being saved...")
        
        # Clear query parameters to prevent re-triggering on rerun
        if "event" in st.query_params:
            del st.query_params["event"]
        if "session_id" in st.query_params:
            del st.query_params["session_id"]
        if "reason" in st.query_params:
            del st.query_params["reason"]
        
        try:
            success = process_emergency_save_from_query(session_id, reason)
            
            if success:
                st.success("✅ Emergency save completed successfully!")
                logger.info("✅ Emergency save completed via query parameter successfully.")
            else:
                st.info("ℹ️ Emergency save completed (no CRM save needed or failed).")
                logger.info("ℹ️ Emergency save completed via query parameter (not eligible for CRM save or internal error).")
                
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during emergency save: {str(e)}")
            logger.critical(f"Emergency save processing crashed from query parameter: {e}", exc_info=True)
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No emergency save requests found in current URL query parameters.")

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
        
        # DEBUG: Log what we extracted
        logger.info(f"Extracted - ID: {fingerprint_id}, Method: {method}, Privacy: {privacy}, Working Methods: {working_methods}")
        
        # Clear query parameters AFTER extraction
        params_to_clear = ["event", "session_id", "fingerprint_id", "method", "privacy", "working_methods", "timestamp"]
        for param in params_to_clear:
            if param in st.query_params:
                del st.query_params[param]
        
        # Validate we have the required data
        if not fingerprint_id or not method:
            st.error("❌ **Fingerprint Error** - Missing required data in redirect")
            logger.error(f"Missing fingerprint data: ID={fingerprint_id}, Method={method}")
            time.sleep(2) # Give user time to read the error
            st.rerun()
            return
        
        # Process silently - NO user messages, NO st.rerun()
        try:
            success = process_fingerprint_from_query(session_id, fingerprint_id, method, privacy, working_methods)
            logger.info(f"✅ Silent fingerprint processing: {success}")
        except Exception as e:
            logger.error(f"Silent fingerprint processing failed: {e}")
        
        # NO st.rerun() here - let normal flow continue
        return
    else:
        logger.info("ℹ️ No fingerprint requests found in current URL query parameters.")

def global_message_channel_error_handler():
    """Enhanced global error handler for component messages with better communication handling"""
    js_error_handler = """
    (function() {
        if (window.fifi_error_handler_initialized) return;
        window.fifi_error_handler_initialized = true;
        
        // Global error handlers
        window.addEventListener('error', function(e) {
            console.error('🚨 Global JS Error:', e.error, 'at', e.filename, ':', e.lineno);
        });
        
        window.addEventListener('unhandledrejection', function(e) {
            console.error('🚨 Unhandled Promise Rejection:', e.reason);
        });
        
        // Enhanced component communication handler
        window.addEventListener('message', function(event) {
            try {
                if (event.data && typeof event.data === 'object') {
                    if (event.data.type === 'streamlit:setComponentValue') {
                        console.log('📡 Received component message:', event.data.type);
                        
                        // Try to forward to Streamlit if available
                        if (window.Streamlit && window.Streamlit.setComponentValue) {
                            window.Streamlit.setComponentValue(event.data.value);
                        }
                    } else if (event.data.type === 'fingerprint_fallback') {
                        console.log('📡 Received fingerprint fallback message');
                        
                        // Store fallback data for retrieval - this is not used in the current Python code
                        // but keeping it for future potential direct JS fallback integration if needed.
                        window.fingerprint_fallback_data = event.data;
                    }
                }
            } catch (e) {
                console.error('🚨 Message handler error:', e);
            }
        });
        
        // Component readiness checker
        let componentReadyChecks = 0;
        const maxComponentChecks = 50; // 5 seconds max wait
        
        function checkComponentReady() {
            componentReadyChecks++;
            
            if (window.Streamlit && window.Streamlit.setComponentReady) {
                console.log('✅ Streamlit component system ready');
                window.Streamlit.setComponentReady();
                return;
            }
            
            if (componentReadyChecks < maxComponentChecks) {
                setTimeout(checkComponentReady, 100);
            } else {
                console.warn('⚠️ Streamlit component system not ready after 5 seconds');
            }
        }
        
        // Start checking for component readiness
        setTimeout(checkComponentReady, 100);
        
        console.log('✅ Enhanced global error handlers and component communication initialized');
    })();
    """
    
    try:
        # Use st_javascript instead of st.components.v1.html for better reliability
        st_javascript(js_error_handler)
    except Exception as e:
        logger.error(f"Failed to initialize enhanced global error handler: {e}")
        # Fallback to basic version
        try:
            st.components.v1.html(f"<script>{js_error_handler}</script>", height=0, width=0)
        except Exception as fallback_e:
            logger.error(f"Fallback global error handler also failed: {fallback_e}")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_welcome_page(session_manager: 'SessionManager'):
    """Renders the application's welcome page, including sign-in and guest options."""
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
        st.markdown("• **40 questions per day** (across devices)")
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
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
            
            st.markdown("---")
            st.info("Don't have an account? [Register here](https://www.12taste.com/in/my-account/) to unlock full features!")
    
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
                st.session_state.page = "chat"
                st.rerun()

def render_sidebar(session_manager: 'SessionManager', session: UserSession, pdf_exporter: PDFExporter):
    """Renders the application's sidebar."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        if session.user_type.value == UserType.REGISTERED_USER.value:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Questions Today:** {session.total_question_count}/40")
            if session.total_question_count <= 20:
                st.progress(min(session.total_question_count / 20, 1.0), text="Tier 1 (up to 20 questions)")
            else:
                progress_value = min((session.total_question_count - 20) / 20, 1.0)
                st.progress(progress_value, text="Tier 2 (21-40 questions)")
            
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
            
        else:
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(min(session.daily_question_count / 4, 1.0))
            st.caption("Email verification unlocks 10 questions/day.")
        
        # Show working fingerprint ID immediately
        if session.fingerprint_id and not session.fingerprint_id.startswith(("temp_py_", "temp_fp_")):
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:12]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        else:
            st.markdown("**Device ID:** `Ready`")
            st.caption("Fingerprinting active")
        
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type.value == UserType.REGISTERED_USER.value:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready** (will link on first save)")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM (after inactivity)")
            else:
                st.caption("💾 Auto-save enabled (after 15 min inactivity)")
        else: 
            st.caption("🚫 CRM Integration: Registered users only")
        
        st.divider()
        
        st.markdown(f"**Messages in Chat:** {len(session.messages)}")
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
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clears all messages from the current conversation."):
                session_manager.clear_chat_history(session)
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True, help="Ends your current session and returns to the welcome page."):
                session_manager.end_session(session)
                st.rerun()

        if session.user_type.value == UserType.REGISTERED_USER.value and session.messages:
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
                st.caption("💡 Chat automatically saves to CRM after 15 minutes of inactivity.")

def render_email_verification_dialog(session_manager: 'SessionManager', session: UserSession):
    """Renders the email verification dialog for guest users who have hit their initial question limit (4 questions)."""
    st.error("📧 **Email Verification Required**")
    st.info("You've used your 4 free questions. Please verify your email to unlock 10 questions per day.")
    
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = 'initial_check'

    if st.session_state.verification_stage == 'initial_check':
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            st.info(f"🤝 **We seem to recognize this device!**")
            st.markdown(f"Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's my email", use_container_width=True, key="recognize_yes_btn"):
                    session.recognition_response = "yes"
                    st.session_state.verification_email = fingerprint_history['email']
                    st.session_state.verification_stage = "send_code_recognized"
                    st.rerun()
            with col2:
                if st.button("❌ No, use a different email", use_container_width=True, key="recognize_no_btn"):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry"
                    st.rerun()
        else:
            st.session_state.verification_stage = "email_entry"
            st.rerun()

    if st.session_state.verification_stage == 'send_code_recognized':
        email_to_verify = st.session_state.get('verification_email')
        if email_to_verify:
            with st.spinner(f"Sending verification code to {email_to_verify}..."):
                result = session_manager.handle_guest_email_verification(session, email_to_verify)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.verification_stage = "code_entry"
                else:
                    st.error(result['message'])
                    st.session_state.verification_stage = "email_entry"
            st.rerun()
        else:
            st.error("Error: No recognized email found to send the code. Please enter your email manually.")
            st.session_state.verification_stage = "email_entry"
            st.rerun()

    if st.session_state.verification_stage == 'email_entry':
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**Please enter your email address to receive a verification code:**")
            current_email_input = st.text_input("Email Address", placeholder="your@email.com", value=st.session_state.get('verification_email', session.email or ""), key="manual_email_input")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if current_email_input:
                    if session.email and current_email_input != session.email:
                        session.email_switches_count += 1
                        session.email = current_email_input
                        session_manager.db.save_session(session)
                        
                    result = session_manager.handle_guest_email_verification(session, current_email_input)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = current_email_input
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address to receive the code.")
    
    if st.session_state.verification_stage == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 A verification code has been sent to **{verification_email}**.")
        st.info("Please check your email, including spam/junk folders. The code is valid for 10 minutes.")
        
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
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code you received.")

def render_chat_interface(session_manager: 'SessionManager', session: UserSession):    """Renders the main chat interface."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")

    # Check if session needs fingerprinting (only for fallback IDs)
    fingerprint_needed = (
        not session.fingerprint_id or
        session.fingerprint_method == "fallback" or
        session.fingerprint_id.startswith("fallback_")
    )
    
    if fingerprint_needed:
        logger.info(f"🔄 Enhanced fingerprinting needed for session {session.session_id[:8]} (current: {session.fingerprint_id})")
        
        # Render fingerprinting component silently
        session_manager.fingerprinting.render_fingerprint_component(session.session_id)
        # Don't exit here - let the user continue using the app with fallback fingerprint
        # The enhanced fingerprint will be applied on the next redirect

    # Initialize global error handler
    global_message_channel_error_handler()
    
    # Add browser close detection for registered users
    if session.user_type.value == UserType.REGISTERED_USER.value:
        try:
            render_browser_close_detection_enhanced(session.session_id)
        except Exception as e:
            logger.error(f"Failed to render browser close detection for {session.session_id[:8]}: {e}", exc_info=True)

    # Add 15-minute timer for registered users
    if session.user_type.value == UserType.REGISTERED_USER.value:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id)
            if timer_result and handle_timer_event(timer_result, session_manager, session):
                st.rerun()
        except Exception as e:
            logger.error(f"15-minute timer component execution failed for session {session.session_id[:8]}: {e}", exc_info=True)

    # Check user limits
    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed']:
        if limit_check.get('reason') == 'guest_limit':
            render_email_verification_dialog(session_manager, session)
            return
        else:
            return

    # Display chat messages
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            
            if msg.get("role") == "assistant":
                if "source" in msg:
                    st.caption(f"Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"):
                    indicators.append("🧠 Knowledge Base")
                if msg.get("used_search"):
                    indicators.append("🌐 Web Search")
                
                if indicators:
                    st.caption(f"Enhanced with: {', '.join(indicators)}")

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
                        st.error("📧 Please verify your email to continue using FiFi AI.")
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
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Your access has been temporarily restricted.")
                        st.error(f"Penalty duration: {response.get('penalty_hours', 0)} hours.")
                        st.rerun()
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                        
                        indicators = []
                        if response.get("used_pinecone"):
                            indicators.append("🧠 Knowledge Base")
                        if response.get("used_search"):
                            indicators.append("🌐 Web Search")
                        
                        if indicators:
                            st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                except Exception as e:
                    logger.error(f"AI response generation failed due to an unexpected error: {e}", exc_info=True)
                    st.error("⚠️ Sorry, I encountered an unexpected error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# INITIALIZATION & MAIN FUNCTIONS
# =============================================================================

def ensure_initialization_fixed():
    """Fixed version of ensure_initialization with better error handling and timeout prevention"""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence...")
        
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
                logger.error(f"Database manager initialization failed: {db_e}", exc_info=True)
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
                logger.error(f"Zoho manager failed: {e}")
                zoho_manager = type('FallbackZoho', (), {
                    'config': config,
                    'save_chat_transcript_sync': lambda self, session, reason: False
                })()
            
            init_progress.progress(0.6)
            
            try:
                ai_system = EnhancedAI(config)
            except Exception as e:
                logger.error(f"AI system failed: {e}")
                ai_system = type('FallbackAI', (), {
                    'openai_client': None,
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
            logger.info("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            st.error("💥 Critical initialization error occurred.")
            st.error(f"Error: {str(e)}")
            logger.critical(f"Critical initialization failure: {e}", exc_info=True)
            
            st.session_state.initialized = False
            return False
    
    return True

def main_fixed():
    """Fixed main entry point with better error handling and timeout prevention"""
    try:
        st.set_page_config(
            page_title="FiFi AI Assistant", 
            page_icon="🤖", 
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {e}")

    # Initialize
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

    # Handle emergency saves AND fingerprint data first
    try:
        handle_emergency_save_requests_from_query()
        handle_fingerprint_requests_from_query()
    except Exception as e:
        logger.error(f"Query parameter handling failed: {e}")

    # Get session manager
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Session Manager not available. Please refresh the page.")
        return

    # Route to appropriate page
    current_page = st.session_state.get('page')
    
    try:
        if current_page != "chat":
            render_welcome_page(session_manager)
            
        else:
            try:
                session = session_manager.get_session()
                
                if session and session.active:
                    render_sidebar(session_manager, session, st.session_state.pdf_exporter)
                    render_chat_interface(session_manager, session)
                else:
                    st.session_state['page'] = None
                    st.rerun()
                    
            except Exception as session_error:
                logger.error(f"Session handling error: {session_error}", exc_info=True)
                st.error("⚠️ Session error occurred. Redirecting to welcome page...")
                st.session_state['page'] = None
                time.sleep(2)
                st.rerun()
                
    except Exception as page_error:
        logger.error(f"Page routing error: {page_error}", exc_info=True)
        st.error("⚠️ Page error occurred. Please refresh the page.")

# Entry point
if __name__ == "__main__":
    main_fixed()
