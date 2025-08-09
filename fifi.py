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
# FINAL INTEGRATED FIFI AI - WITH TRUE 15-MIN SESSION TIMEOUT
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

    def _check_15min_eligibility(self, session: UserSession) -> bool:
        """Check if session has been active for at least 15 minutes to be eligible for CRM save."""
        try:
            # Use the earliest of session creation time or first question time
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
            
            # 15-minute eligibility check
            if not self._check_15min_eligibility(session):
                logger.info(f"CRM save not eligible - less than 15 minutes active for {session.session_id[:8]}")
                return False
            
            # All conditions met
            logger.info(f"CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Questions={session.daily_question_count}, 15min+")
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
# JAVASCRIPT COMPONENTS & EVENT HANDLING
# =============================================================================

def render_activity_timer_component_15min_fixed_v2(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced timer component V2 that automatically redirects after 15 minutes of inactivity.
    Handles both iframe and non-iframe scenarios by detecting and redirecting the appropriate window.
    """
    logger.info(f"🔍 DEBUG: JavaScript timer component called for {session_id[:8]}")
    
    if not session_id:
        logger.warning("❌ Timer component: No session ID provided")
        return None
    
    safe_session_id_js = session_id.replace('-', '_')
    
    js_timer_code = f"""
    (() => {{
        try {{
            const sessionId = "{session_id}";
            const SESSION_TIMEOUT_MS = 900000; // 15 minutes
            
            console.log("🕐 FiFi 15-Minute Timer V2: Initializing for session", sessionId.substring(0, 8));
            
            // Detect if we're in an iframe
            const isInIframe = () => {{
                try {{
                    return window !== window.parent || window !== window.top;
                }} catch (e) {{
                    // Cross-origin iframe will throw error
                    return true;
                }}
            }};
            
            const inIframe = isInIframe();
            console.log(`📍 Running in ${{inIframe ? 'IFRAME' : 'TOP WINDOW'}}`);
            
            // Initialize or reset timer state
            if (typeof window.fifi_timer_state_{safe_session_id_js} === 'undefined' || 
                window.fifi_timer_state_{safe_session_id_js} === null || 
                window.fifi_timer_state_{safe_session_id_js}.sessionId !== sessionId) {{
                
                console.log("🆕 Starting new timer for session", sessionId.substring(0, 8)); 
                window.fifi_timer_state_{safe_session_id_js} = {{
                    lastActivityTime: Date.now(),
                    expired: false,
                    listenersInitialized: false,
                    timeoutCheckInterval: null,
                    sessionId: sessionId,
                    isInIframe: inIframe
                }};
            }}
            
            const state = window.fifi_timer_state_{safe_session_id_js};
            
            // Setup activity listeners (only once)
            if (!state.listenersInitialized) {{
                console.log("👂 Setting up activity listeners...");
                
                function resetActivity() {{
                    try {{
                        const now = Date.now();
                        if (state.lastActivityTime !== now) {{
                            state.lastActivityTime = now;
                            if (state.expired) {{
                                console.log("🔄 Activity detected, resetting expired flag");
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
                if (inIframe) {{
                    try {{
                        if (window.parent && window.parent.document && 
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
                        console.debug("Cannot access parent document for listeners (likely cross-origin):", e);
                    }}
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
                
                // CRITICAL: Clear any existing interval before setting a new one
                if (state.timeoutCheckInterval) {{
                    clearInterval(state.timeoutCheckInterval);
                }}
                
                // NEW V2: Enhanced automatic timeout checking with iframe handling
                state.timeoutCheckInterval = setInterval(() => {{
                    const currentTime = Date.now();
                    const inactiveTimeMs = currentTime - state.lastActivityTime;
                    const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
                    const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
                    
                    // Log every 5 minutes
                    if (inactiveMinutes > 0 && inactiveMinutes % 5 === 0 && inactiveSeconds < 2) {{
                        console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
                    }}
                    
                    // Check if timeout reached
                    if (inactiveTimeMs >= SESSION_TIMEOUT_MS && !state.expired) {{
                        state.expired = true;
                        console.log("🚨 15-MINUTE TIMEOUT REACHED - TRIGGERING AUTO-REDIRECT");
                        console.log(`📍 Redirect mode: ${{state.isInIframe ? 'IFRAME → PARENT' : 'DIRECT'}}`);
                        
                        // Clear the interval to stop checking
                        clearInterval(state.timeoutCheckInterval);
                        
                        // Build the timeout URL
                        const timeoutUrl = `${{window.location.origin}}${{window.location.pathname}}?event=session_timeout_auto&session_id=${{sessionId}}&inactive_minutes=${{inactiveMinutes}}`;
                        
                        // V2: Smart redirect based on iframe status
                        const performRedirect = () => {{
                            console.log("🔄 Performing redirect to:", timeoutUrl);
                            
                            if (state.isInIframe) {{
                                // We're in an iframe - try to redirect parent/top window
                                console.log("🔄 Attempting parent/top window redirect...");
                                
                                try {{
                                    // Try top window first (handles nested iframes)
                                    if (window.top && window.top.location) {{
                                        console.log("✅ Redirecting TOP window");
                                        window.top.location.href = timeoutUrl;
                                        return;
                                    }}
                                }} catch (e) {{
                                    console.log("⚠️ Cannot access top window (cross-origin?):", e.message);
                                }}
                                
                                try {{
                                    // Fallback to parent window
                                    if (window.parent && window.parent.location) {{
                                        console.log("✅ Redirecting PARENT window");
                                        window.parent.location.href = timeoutUrl;
                                        return;
                                    }}
                                }} catch (e) {{
                                    console.log("⚠️ Cannot access parent window (cross-origin?):", e.message);
                                }}
                                
                                // If we can't access parent/top, try postMessage as last resort
                                try {{
                                    console.log("📡 Attempting postMessage to parent for redirect");
                                    window.parent.postMessage({{
                                        type: 'fifi_timeout_redirect',
                                        url: timeoutUrl,
                                        sessionId: sessionId
                                    }}, '*');
                                    
                                    // Also redirect iframe itself as fallback
                                    setTimeout(() => {{
                                        console.log("🔄 Fallback: Redirecting iframe itself");
                                        window.location.href = timeoutUrl;
                                    }}, 1000);
                                }} catch (e) {{
                                    console.error("❌ postMessage failed:", e);
                                    // Last resort: redirect iframe only
                                    window.location.href = timeoutUrl;
                                }}
                                
                            }} else {{
                                // We're in the top window - direct redirect
                                console.log("✅ Redirecting current window (not in iframe)");
                                window.location.href = timeoutUrl;
                            }}
                        }};
                        
                        try {{
                            performRedirect();
                        }} catch (redirectError) {{
                            console.error("❌ Redirect failed:", redirectError);
                            
                            // Ultimate fallback: Try base URL
                            try {{
                                const fallbackUrl = window.location.origin + window.location.pathname;
                                console.log("🔄 Ultimate fallback redirect to:", fallbackUrl);
                                
                                if (state.isInIframe) {{
                                    try {{
                                        window.top.location.href = fallbackUrl;
                                    }} catch (e) {{
                                        window.parent.location.href = fallbackUrl;
                                    }}
                                }} else {{
                                    window.location.href = fallbackUrl;
                                }}
                            }} catch (fallbackError) {{
                                console.error("❌ All redirect attempts failed:", fallbackError);
                            }}
                        }}
                    }}
                }}, 10000); // Check every 10 seconds for timeout
                
                // V2: Setup parent message listener for cross-origin iframe scenarios
                if (inIframe) {{
                    try {{
                        // Tell parent to listen for timeout messages
                        window.parent.postMessage({{
                            type: 'fifi_timer_ready',
                            sessionId: sessionId
                        }}, '*');
                    }} catch (e) {{
                        console.debug("Could not notify parent of timer readiness:", e);
                    }}
                }}
                
                state.listenersInitialized = true;
                console.log("✅ Activity listeners and auto-timeout checker V2 initialized");
                console.log(`📍 Context: ${{state.isInIframe ? 'IFRAME' : 'TOP'}} | Session: ${{sessionId.substring(0, 8)}}`);
            }}
            
            // Still return timeout status for Python-side checking (backward compatibility)
            const currentTime = Date.now();
            const inactiveTimeMs = currentTime - state.lastActivityTime;
            const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
            const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
            
            if (inactiveTimeMs >= SESSION_TIMEOUT_MS && !state.expired) {{
                state.expired = true;
                return {{
                    event: "session_timeout_15min",
                    session_id: sessionId,
                    inactive_time_ms: inactiveTimeMs,
                    inactive_minutes: inactiveMinutes,
                    inactive_seconds: inactiveSeconds,
                    timestamp: currentTime,
                    is_iframe: state.isInIframe
                }};
            }}
            
            return null;
            
        }} catch (error) {{
            console.error("🚨 Timer component V2 error:", error);
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
        
        # Validate the result (still works if Python checks it)
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            return None
        
        if isinstance(timer_result, dict):
            if timer_result.get('event') == "session_timeout_15min":
                if timer_result.get('session_id') == session_id:
                    logger.info(f"✅ Valid 15-min timer event received: {timer_result.get('event')} for session {session_id[:8]}.")
                    if timer_result.get('is_iframe'):
                        logger.info("📍 Timer detected iframe context")
                    return timer_result
                else:
                    logger.warning(f"⚠️ Timer event session ID mismatch")
                    return None
            elif timer_result.get('event') == "timer_error":
                logger.error(f"❌ Timer component error: {timer_result.get('error', 'Unknown error')}")
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"❌ JavaScript timer component V2 execution error: {e}", exc_info=True)
        return None
        


def handle_auto_timeout_from_query():
    """
    Handles automatic timeout redirects triggered by the JavaScript timer.
    Add this to your query parameter handlers.
    """
    logger.info("🔍 DEBUG: handle_auto_timeout_from_query called")
    
    logger.info("🔍 AUTO-TIMEOUT HANDLER: Checking for timeout requests...")
    
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    inactive_minutes = query_params.get("inactive_minutes", "15")
    
    if event == "session_timeout_auto" and session_id:
        logger.info("=" * 80)
        logger.info("⏰ AUTO-TIMEOUT DETECTED VIA URL REDIRECT!")
        logger.info(f"Session ID: {session_id}, Inactive: {inactive_minutes} minutes")
        logger.info("=" * 80)

        # Set timeout context before any UI changes
        timeout_context_js = """
        <script>
        try {
            sessionStorage.setItem('fifi_timeout_reason', 'session_timeout_15min_inactivity');
            window.postMessage({
                type: 'fifi_timeout_context', 
                reason: 'session_timeout_15min_inactivity'
            }, '*');
            console.log('⏰ Timeout context set: session_timeout_15min_inactivity');
        } catch (e) {
            console.error('Failed to set timeout context:', e);
        }
        </script>
        """
        st.components.v1.html(timeout_context_js, height=0, width=0)
        # Clear query parameters
        for param in ["event", "session_id", "inactive_minutes"]:
            if param in st.query_params:
                del st.query_params[param]
        
        # Get session manager
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("Session manager not available during timeout handling")
            st.session_state['page'] = None
            st.rerun()
            return
        
        # Load and process the session
        try:
            session = session_manager.db.load_session(session_id)
            if session:
                # Save to CRM if eligible
                if session_manager._is_crm_save_eligible(session, "15-Minute Auto Timeout"):
                    logger.info(f"Performing CRM save for auto-timeout session {session_id[:8]}")
                    save_success = session_manager.zoho.save_chat_transcript_sync(session, "15-Minute Auto Timeout")
                    if save_success:
                        session.timeout_saved_to_crm = True
                
                # Mark session as inactive
                session.active = False
                session.last_activity = datetime.now()
                session_manager.db.save_session(session)
                logger.info(f"🔒 Session {session_id[:8]} closed due to auto-timeout")
            
        except Exception as e:
            logger.error(f"Error processing auto-timeout: {e}", exc_info=True)
        
        # Clear session state and redirect to home
        if 'current_session_id' in st.session_state:
            del st.session_state['current_session_id']
        if 'page' in st.session_state:
            del st.session_state['page']
        
        # Show message and redirect
        st.info("⏰ **Session Timeout:** Your session has been closed due to 15 minutes of inactivity.")
        st.info("🏠 Please click 'Start as Guest' or 'Sign In' to begin a new session.")
        
        # Force rerun to show welcome page
        st.rerun()

def render_browser_close_detection_enhanced(session_id: str):
    """Enhanced browser close detection - FIXED to pass timeout context when available"""
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
        const FASTAPI_URL = 'https://fifi-beacon-fastapi-121263692901.europe-west4.run.app/emergency-save';
        
        // ENHANCED: Check for timeout context
        let saveTriggered = false;
        let isTabSwitching = false;
        let tabSwitchTimeout = null;
        let timeoutContext = null; // NEW: Store timeout context
        
        console.log('🛡️ Browser close detection initialized (ENHANCED with timeout context)');

        // NEW: Listen for timeout context from Streamlit
        window.addEventListener('message', function(event) {{
            if (event.data && event.data.type === 'fifi_timeout_context') {{
                timeoutContext = event.data.reason;
                console.log('⏰ Timeout context received:', timeoutContext);
            }}
        }});
        
        // ENHANCED: Check for timeout context in sessionStorage
        function getActualReason(defaultReason) {{
            // Check for timeout context first
            if (timeoutContext) {{
                console.log('✅ Using timeout context:', timeoutContext);
                return timeoutContext;
            }}
            
            // Check sessionStorage for timeout flag
            try {{
                const timeoutFlag = sessionStorage.getItem('fifi_timeout_reason');
                if (timeoutFlag) {{
                    console.log('✅ Found timeout reason in sessionStorage:', timeoutFlag);
                    // sessionStorage.removeItem('fifi_timeout_reason');
                    return timeoutFlag;
                }}
            }} catch (e) {{
                console.debug('SessionStorage not available:', e);
            }}
            
            // Check for timeout indicators in URL
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('timeout_redirect') === 'true') {{
                console.log('✅ Detected timeout redirect in URL');
                return 'session_timeout_redirect';
            }}
            
            console.log('ℹ️ No timeout context found, using browser event:', defaultReason);
            return defaultReason;
        }}

        function performActualEmergencySave(browserReason) {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            // ENHANCED: Get the actual reason (timeout context or browser event)
            const actualReason = getActualReason(browserReason);
            console.log('🚨 Emergency save triggered - Browser:', browserReason, 'Actual:', actualReason);
            
            // PRIMARY METHOD: Send beacon to FastAPI with correct reason
            if (navigator.sendBeacon) {{
                try {{
                    const emergencyData = JSON.stringify({{
                        session_id: sessionId,
                        reason: actualReason, // Use actual reason, not browser reason
                        browser_event: browserReason, // Include browser event for debugging
                        timestamp: Date.now()
                    }});
                    
                    const beaconSent = navigator.sendBeacon(
                        FASTAPI_URL,
                        new Blob([emergencyData], {{type: 'application/json'}})
                    );
                    
                    if (beaconSent) {{
                        console.log('✅ Emergency save beacon sent with correct reason:', actualReason);
                        return;
                    }} else {{
                        console.error('❌ Beacon failed to send');
                    }}
                }} catch (beaconError) {{
                    console.error('❌ Beacon error:', beaconError);
                }}
            }}
            
            // FALLBACK: Redirect to Streamlit with correct reason
            try {{
                console.log('🔄 Beacon failed, trying redirect fallback...');
                const saveUrl = `${{window.location.origin}}${{window.location.pathname}}?event=emergency_close&session_id=${{sessionId}}&reason=${{actualReason}}`;
                window.location.href = saveUrl;
            }} catch (e) {{
                console.error('❌ Both beacon and redirect failed:', e);
            }}
        }}

        // FIXED: Enhanced emergency save that checks for tab switches and timeout context
        function triggerEmergencySave(reason = 'unknown') {{
            if (saveTriggered) return;
            
            // Check for timeout context first
            const actualReason = getActualReason(reason);
            
            // If this is a timeout, skip tab switch detection
            if (actualReason.includes('timeout') || actualReason.includes('inactivity')) {{
                console.log('⏰ Timeout detected, bypassing tab switch detection');
                performActualEmergencySave(reason);
                return;
            }}
            
            // ORIGINAL tab switch detection logic for non-timeout events
            if (isTabSwitching) {{
                console.log('🔍 Potential tab switch detected, delaying emergency save by 100ms...');
                
                setTimeout(() => {{
                    if (document.visibilityState === 'visible') {{
                        console.log('✅ Tab switch confirmed - CANCELING emergency save');
                        isTabSwitching = false;
                        return;
                    }}
                    console.log('🚨 Real exit confirmed after delay - proceeding with emergency save');
                    performActualEmergencySave(reason);
                }}, 100);
                
                return;
            }}
            
            // Immediate save for non-tab-switch scenarios
            performActualEmergencySave(reason);
        }}
        
        // ... rest of the existing browser close detection code ...
        // (Enhanced visibility change tracking, exit event listeners, etc.)
        
        // Enhanced visibility change tracking
        document.addEventListener('visibilitychange', function() {{
            if (document.visibilityState === 'hidden') {{
                console.log('👁️ Tab switched away - marking as potential tab switch');
                isTabSwitching = true;
                
                if (tabSwitchTimeout) {{
                    clearTimeout(tabSwitchTimeout);
                }}
                
                tabSwitchTimeout = setTimeout(() => {{
                    console.log('⏰ Tab switch timeout - assuming real navigation');
                    isTabSwitching = false;
                }}, 2000);
                
            }} else {{
                console.log('👁️ Tab switched back - confirmed tab switch (not real exit)');
                isTabSwitching = false;
                
                if (tabSwitchTimeout) {{
                    clearTimeout(tabSwitchTimeout);
                    tabSwitchTimeout = null;
                }}
            }}
        }}, {{ passive: true }});
        
        // Listen to exit events
        const realExitEvents = ['beforeunload', 'unload'];
        realExitEvents.forEach(eventType => {{
            try {{
                window.addEventListener(eventType, (event) => {{
                    console.log('🚨 Exit event detected:', eventType, 'TabSwitching:', isTabSwitching);
                    triggerEmergencySave(eventType);
                }}, {{ capture: true, passive: true }});
                
                if (window.parent && window.parent !== window) {{
                    window.parent.addEventListener(eventType, (event) => {{
                        console.log('🚨 Parent exit event detected:', eventType, 'TabSwitching:', isTabSwitching);
                        triggerEmergencySave('parent_' + eventType);
                    }}, {{ capture: true, passive: true }});
                }}
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        // LOG pagehide but DON'T trigger saves
        window.addEventListener('pagehide', function(event) {{
            console.log('📄 pagehide detected - persisted:', event.persisted, 'TabSwitching:', isTabSwitching, '(NOT triggering save - relying on beforeunload/unload)');
        }}, {{ passive: true }});
        
        console.log('✅ Enhanced browser close detection ready - Smart timeout context detection enabled');
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close component: {e}", exc_info=True)

def add_activity_detection(session_id: str, session_manager, session):
    """Simple activity detection for meta refresh"""
    if not session_id:
        return
    
    safe_session_id = session_id.replace('-', '_')
    
    activity_js = f"""
    (() => {{
        const sessionId = "{session_id}";
        const stateKey = 'fifi_activity_{safe_session_id}';
        
        if (!window[stateKey]) {{
            window[stateKey] = {{ lastActivity: Date.now(), lastUpdate: 0, initialized: false }};
        }}
        
        const state = window[stateKey];
        
        if (!state.initialized) {{
            function track() {{ state.lastActivity = Date.now(); }}
            ['mousedown', 'keydown', 'click', 'scroll', 'touchstart'].forEach(e => {{
                document.addEventListener(e, track, {{passive: true}});
            }});
            state.initialized = true;
            console.log('✅ Activity tracking enabled');
        }}
        
        const now = Date.now();
        if (now - state.lastUpdate > 30000) {{
            state.lastUpdate = now;
            return {{ type: 'activity_ping', session_id: sessionId, last_activity: state.lastActivity }};
        }}
        return null;
    }})()
    """
    
    try:
        result = st_javascript(activity_js)
        if result and result.get('type') == 'activity_ping':
            js_activity = result.get('last_activity')
            if js_activity:
                try:
                    new_activity = datetime.fromtimestamp(js_activity / 1000)
                    if new_activity > session.last_activity:
                        session.last_activity = new_activity
                        session_manager.db.save_session(session)
                        logger.debug(f"💓 Activity updated for {session_id[:8]}")
                except Exception as e:
                    logger.error(f"Activity processing failed: {e}")
    except Exception as e:
        logger.error(f"Activity detection failed: {e}")
        
def handle_timer_event(timer_result: Dict[str, Any], session_manager: 'SessionManager', session: UserSession) -> bool:
    """Processes events triggered by the JavaScript activity timer with TRUE session timeout."""
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
            st.info("🔄 **Your session is being closed due to inactivity.**")
            
            # Save to CRM if eligible before closing session
            if session_manager._is_crm_save_eligible(session, "15-Minute Session Inactivity Timeout"):
                with st.spinner("💾 Auto-saving chat to CRM before closing session..."):
                    try:
                        save_success = session_manager.zoho.save_chat_transcript_sync(session, "15-Minute Session Inactivity Timeout")
                    except Exception as e:
                        logger.error(f"15-min timeout CRM save failed during execution: {e}", exc_info=True)
                        save_success = False
                
                if save_success:
                    st.success("✅ Chat automatically saved to CRM!")
                    session.timeout_saved_to_crm = True
                else:
                    st.warning("⚠️ Auto-save to CRM failed. Please check your credentials or contact support if issue persists.")
            else:
                st.info("ℹ️ Session timeout detected, but no CRM save was performed (not eligible based on activity, user type, or duration).")
                logger.info(f"15-min timeout CRM save eligibility check failed for {session_id[:8]}: UserType={session.user_type.value}, Email={bool(session.email)}, Messages={len(session.messages)}, Questions={session.daily_question_count}, Saved Status={session.timeout_saved_to_crm}.")
            
            # TRUE SESSION TIMEOUT: Close session and redirect to home
            try:
                # Mark session as inactive
                session.active = False
                session.last_activity = datetime.now()
                session_manager.db.save_session(session)
                
                # Clear Streamlit session state to force new session
                if 'current_session_id' in st.session_state:
                    del st.session_state['current_session_id']
                if 'page' in st.session_state:
                    del st.session_state['page']
                
                logger.info(f"🔒 Session {session_id[:8]} closed due to 15-minute timeout")
                
                # Show redirect message and redirect to home
                st.info("🏠 **Redirecting to home page...**")
                st.info("You can start a new session from the welcome page.")
                
                # Force redirect to home after a brief delay
                time.sleep(2)
                st.rerun()
                
            except Exception as close_error:
                logger.error(f"Error closing session during timeout for {session_id[:8]}: {close_error}")
                # Force redirect even if session close fails
                st.session_state['page'] = None
                st.rerun()
                
            return True  # Indicate that session was closed
                
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
        if session_manager._is_crm_save_eligible(session, f"Emergency Save: {reason}"):
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
        
        # Process silently and stop execution
        try:
            success = process_fingerprint_from_query(session_id, fingerprint_id, method, privacy, working_methods)
            logger.info(f"✅ Silent fingerprint processing: {success}")
            
            if success:
            # Stop execution - user stays on current page, next interaction will show updated fingerprint
                logger.info(f"🔄 Fingerprint processed successfully, stopping execution to preserve page state")
                st.stop()
        except Exception as e:
            logger.error(f"Silent fingerprint processing failed: {e}")
        
        # If we get here, processing failed - let normal flow continue
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

def calculate_seconds_until_timeout(session, timeout_minutes=15):
    """
    Calculates exactly how many seconds until this session will timeout.
    """
    time_since_activity = datetime.now() - session.last_activity
    timeout_seconds = timeout_minutes * 60
    seconds_until_timeout = timeout_seconds - time_since_activity.total_seconds()
    
    # Return at least 5 seconds to prevent immediate refresh loops
    return max(5, int(seconds_until_timeout))


def inject_dynamic_timeout_refresh(session):
    """Enhanced meta refresh that works with activity detection"""
    time_since_activity = datetime.now() - session.last_activity
    timeout_seconds = 15 * 60
    seconds_until_timeout = timeout_seconds - time_since_activity.total_seconds()
    refresh_at = max(5, min(int(seconds_until_timeout) + 1, timeout_seconds))
    
    # Set timeout context
    timeout_context_js = """
    <script>
    try {
        sessionStorage.setItem('fifi_timeout_reason', 'meta_refresh_timeout_15min');
        console.log('⏰ Meta refresh timeout context set');
    } catch (e) { console.error('Failed to set timeout context:', e); }
    </script>
    """
    st.components.v1.html(timeout_context_js, height=0, width=0)
    
    # Enhanced meta refresh
    dynamic_refresh_html = f"""
    <meta http-equiv="refresh" content="{refresh_at}">
    <script>
        console.log('🕐 Meta Refresh: Will redirect in {refresh_at} seconds');
        console.log('📊 Activity age:', Math.floor((Date.now() - {int(session.last_activity.timestamp() * 1000)}) / 1000), 'seconds');
    </script>
    """
    st.markdown(dynamic_refresh_html, unsafe_allow_html=True)
    
    # ORIGINAL META REFRESH WITH TIMEOUT CONTEXT
    dynamic_refresh_html = f"""
    <meta http-equiv="refresh" content="{refresh_at}">
    <script>
        console.log('⏰ Session will be checked for timeout in {refresh_at} seconds');
        
        // Visual countdown (optional - remove if you don't want users to see)
        let secondsLeft = {refresh_at};
        setInterval(() => {{
            secondsLeft--;
            if (secondsLeft <= 300 && secondsLeft > 0) {{ // Last 5 minutes
                console.log(`⏰ Timeout in ${{Math.floor(secondsLeft/60)}}m ${{secondsLeft%60}}s`);
            }}
        }}, 1000);
    </script>
    """
    
    st.markdown(dynamic_refresh_html, unsafe_allow_html=True)

def completely_reset_session():
    """
    Completely clears the session and creates a new one.
    This is like the user closing and reopening the browser.
    """
    logger.info("🔄 Completely resetting session due to timeout")
    
    # Clear ALL session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # This will force a new session to be created on next rerun
    st.session_state.clear()
    
    logger.info("✅ Session state completely cleared")

def check_and_handle_timeout_with_reset(session_manager, session, timeout_minutes=15):
    """
    Checks for timeout and completely resets the session if timed out.
    """
    logger.info(f"🔍 DEBUG: check_and_handle_timeout_with_reset called for {session.session_id[:8] if session else 'None'}")

    if not session or not session.active:
        return True
        
    # Calculate time since last activity
    time_since_activity = datetime.now() - session.last_activity
    
    # Check if timeout reached
    if time_since_activity.total_seconds() > (timeout_minutes * 60):
        logger.info(f"⏰ Session timeout: {session.session_id[:8]} inactive for {time_since_activity}")
        
        # Set timeout context before any UI changes
        timeout_context_js = """
        <script>
        try {
            sessionStorage.setItem('fifi_timeout_reason', 'session_timeout_15min_inactivity');
            window.postMessage({
                type: 'fifi_timeout_context', 
                reason: 'session_timeout_15min_inactivity'
            }, '*');
            console.log('⏰ Timeout context set: session_timeout_15min_inactivity');
        } catch (e) {
            console.error('Failed to set timeout context:', e);
        }
        </script>
        """
        st.components.v1.html(timeout_context_js, height=0, width=0)
        
        # Show timeout message
        st.error("⏰ **Session Timeout**")
        st.info("Your session has expired due to 15 minutes of inactivity. Please start a new session.")
        
        # Save to CRM if eligible (before clearing)
        if session_manager._is_crm_save_eligible(session, "15-Minute Timeout"):
            with st.spinner("Saving conversation before ending session..."):
                try:
                    session_manager.zoho.save_chat_transcript_sync(session, "15-Minute Timeout")
                    st.success("✅ Conversation saved")
                except Exception as e:
                    logger.error(f"CRM save failed: {e}")
        
        # Mark session as inactive in database
        session.active = False
        session_manager.db.save_session(session)
        
        # COMPLETELY RESET SESSION
        completely_reset_session()
        
        # Add a button to start fresh
        if st.button("🔄 Start New Session", type="primary"):
            st.rerun()
        
        st.stop()  # Stop execution
        return True
        
    return False
     

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
                            logger.info(f"🔍 AUTHENTICATION SUCCESS: Setting page to 'chat' for session {authenticated_session.session_id[:8]}")
                            st.session_state.page = "chat"
                            time.sleep(1)
                            logger.info(f"🔍 AUTHENTICATION SUCCESS: About to rerun, page state = '{st.session_state.get('page')}'")
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
                logger.info("🔍 GUEST BUTTON: Setting page to 'chat'")
                st.session_state.page = "chat"
                logger.info(f"🔍 GUEST BUTTON: Page state set to '{st.session_state.get('page')}', about to rerun")
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
        
        # Show fingerprint status properly
        if session.fingerprint_id:
            # Check if it's a temporary or fallback fingerprint
            if session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                st.markdown("**Device ID:** Identifying...")
                st.caption("Fingerprinting in progress...")                
            else:
                # Real fingerprint ID from JavaScript
                st.markdown(f"**Device ID:** `{session.fingerprint_id[:12]}...`")
                st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        else:
            st.markdown("**Device ID:** Initializing...")
            st.caption("Starting fingerprinting...")

        render_timeout_status_sidebar(session)

        # AI Tools Status
        st.divider()
        st.markdown("**🤖 AI Tools Status**")
        
        ai_system = session_manager.ai
        if ai_system:
            # Pinecone status
            if ai_system.pinecone_tool and ai_system.pinecone_tool.assistant:
                st.success("🧠 Knowledge Base: Ready")
            elif ai_system.config.PINECONE_API_KEY:
                st.warning("🧠 Knowledge Base: Error")
            else:
                st.info("🧠 Knowledge Base: Not configured")
            
            # Tavily status  
            if ai_system.tavily_agent:
                st.success("🌐 Web Search: Ready")
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
                st.caption("💾 Auto-save enabled (after 15 min inactivity)")
        else: 
            st.caption("🚫 CRM Integration: Registered users & verified guests only")
        
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
            # Enhanced Clear Chat button with tooltip about CRM save
            clear_chat_help = "Clears all messages from the current conversation."
            if (session.user_type in [UserType.REGISTERED_USER, UserType.EMAIL_VERIFIED_GUEST] and 
                session.email and session.messages and session.daily_question_count >= 1):
                
                # Check 15-minute eligibility
                if session_manager._check_15min_eligibility(session):
                    clear_chat_help += " Your conversation will be automatically saved to CRM before clearing."
                else:
                    clear_chat_help += " (CRM save requires 15+ minutes of activity)"
            
            if st.button("🗑️ Clear Chat", use_container_width=True, help=clear_chat_help):
                # Show appropriate messaging based on CRM save eligibility
                if session_manager._is_crm_save_eligible(session, "Clear Chat Request"):
                    with st.spinner("💾 Saving conversation to CRM before clearing..."):
                        session_manager.clear_chat_history(session)
                    st.success("✅ Chat saved to CRM and cleared!")
                else:
                    session_manager.clear_chat_history(session)
                    st.info("🗑️ Chat history cleared.")
                
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True, help="Ends your current session and returns to the welcome page."):
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

# =============================================================================
# ADD THESE FUNCTIONS ANYWHERE BEFORE render_chat_interface
# (e.g., after your existing JavaScript components section)
# =============================================================================

def inject_activity_heartbeat_monitor(session_id: str, heartbeat_interval_seconds: int = 30):
    """
    Injects JavaScript that monitors activity and sends heartbeats to Python.
    This is more reliable than expecting JS to run for 15 minutes straight.
    """
    
    heartbeat_js = f"""
    (() => {{
        // Unique namespace for this session
        const SESSION_ID = '{session_id}';
        const HEARTBEAT_INTERVAL = {heartbeat_interval_seconds * 1000};
        const TIMEOUT_MS = 900000; // 15 minutes
        
        // Initialize or retrieve state
        if (!window.fifiHeartbeatMonitor) {{
            window.fifiHeartbeatMonitor = {{
                lastActivityTime: Date.now(),
                lastHeartbeatTime: 0,
                isActive: true,
                sessionId: SESSION_ID
            }};
            
            console.log('💓 Heartbeat monitor initialized for session', SESSION_ID.substring(0, 8));
            
            // Activity tracking
            function trackActivity() {{
                window.fifiHeartbeatMonitor.lastActivityTime = Date.now();
                window.fifiHeartbeatMonitor.isActive = true;
            }}
            
            // Monitor all activity types
            const events = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart', 'click', 'focus'];
            events.forEach(event => {{
                document.addEventListener(event, trackActivity, {{ passive: true, capture: true }});
            }});
            
            // Also monitor parent if in iframe
            try {{
                if (window.parent && window.parent !== window) {{
                    events.forEach(event => {{
                        window.parent.document.addEventListener(event, trackActivity, {{ passive: true, capture: true }});
                    }});
                }}
            }} catch (e) {{
                console.debug('Cannot monitor parent activity:', e);
            }}
            
            // Heartbeat sender
            setInterval(() => {{
                const monitor = window.fifiHeartbeatMonitor;
                const now = Date.now();
                const timeSinceActivity = now - monitor.lastActivityTime;
                const timeSinceHeartbeat = now - monitor.lastHeartbeatTime;
                
                // Check if we should send a heartbeat
                if (timeSinceActivity < TIMEOUT_MS && timeSinceHeartbeat >= HEARTBEAT_INTERVAL) {{
                    monitor.lastHeartbeatTime = now;
                    
                    const heartbeatData = {{
                        type: 'activity_heartbeat',
                        session_id: SESSION_ID,
                        last_activity: monitor.lastActivityTime,
                        time_since_activity: timeSinceActivity,
                        timestamp: now
                    }};
                    
                    console.log('💓 Sending heartbeat, inactive for', Math.floor(timeSinceActivity / 1000), 'seconds');
                    
                    // Return data for st_javascript to capture
                    return heartbeatData;
                }}
                
                // Check for client-side timeout
                if (timeSinceActivity >= TIMEOUT_MS && monitor.isActive) {{
                    monitor.isActive = false;
                    console.log('⏰ Client-side timeout detected!');
                    
                    // Try to notify Python about timeout
                    return {{
                        type: 'client_timeout',
                        session_id: SESSION_ID,
                        inactive_minutes: Math.floor(timeSinceActivity / 60000),
                        timestamp: now
                    }};
                }}
            }}, 5000); // Check every 5 seconds
        }}
        
        // Return current status
        const monitor = window.fifiHeartbeatMonitor;
        const now = Date.now();
        const timeSinceActivity = now - monitor.lastActivityTime;
        
        return {{
            type: 'status_check',
            session_id: SESSION_ID,
            time_since_activity: timeSinceActivity,
            is_active: monitor.isActive,
            timestamp: now
        }};
    }})()
    """
    
    # Execute and capture result
    result = st_javascript(heartbeat_js)
    return result


def update_session_heartbeat(session_manager, session, heartbeat_data: dict):
    """
    Updates session based on heartbeat data from JavaScript.
    """
    if not heartbeat_data or not isinstance(heartbeat_data, dict):
        return
    
    heartbeat_type = heartbeat_data.get('type')
    
    if heartbeat_type == 'activity_heartbeat':
        # Update last activity time based on client report
        client_activity_time = heartbeat_data.get('last_activity')
        if client_activity_time:
            # Convert JS timestamp to Python datetime
            client_activity = datetime.fromtimestamp(client_activity_time / 1000)
            
            # Only update if client reports more recent activity
            if client_activity > session.last_activity:
                session.last_activity = client_activity
                session_manager.db.save_session(session)
                logger.debug(f"💓 Heartbeat updated activity for {session.session_id[:8]}")
    
    elif heartbeat_type == 'client_timeout':
        logger.info(f"⏰ Client reported timeout for {session.session_id[:8]}")
        # Client detected timeout, but server makes final decision

def check_server_side_timeout(session_manager, session, timeout_minutes: int = 15) -> bool:
    """ENHANCED: Server-side timeout check with context setting"""
    time_since_activity = datetime.now() - session.last_activity
    timeout_seconds = timeout_minutes * 60
    
    if time_since_activity.total_seconds() > timeout_seconds:
        logger.info(f"⏰ Server-side timeout confirmed for {session.session_id[:8]}")
        
        # ENHANCED: Set timeout context before any redirects
        timeout_context_js = """
        <script>
        try {
            // Set timeout context immediately
            sessionStorage.setItem('fifi_timeout_reason', 'server_side_timeout_15min');
            
            // Send message to browser close detection
            if (window.postMessage) {
                window.postMessage({
                    type: 'fifi_timeout_context',
                    reason: 'server_side_timeout_15min'
                }, '*');
            }
            
            console.log('⏰ Server-side timeout context set');
        } catch (e) {
            console.error('Failed to set server timeout context:', e);
        }
        </script>
        """
        st.components.v1.html(timeout_context_js, height=0, width=0)
        
        # Display timeout message
        st.error("⏰ **Session Timeout:** Your session has expired due to 15 minutes of inactivity.")
        
        # Save to CRM if eligible
        if session_manager._is_crm_save_eligible(session, "15-Minute Inactivity Timeout"):
            with st.spinner("💾 Saving conversation to CRM..."):
                try:
                    save_success = session_manager.zoho.save_chat_transcript_sync(
                        session, 
                        "15-Minute Inactivity Timeout"
                    )
                    if save_success:
                        st.success("✅ Conversation saved to CRM")
                        session.timeout_saved_to_crm = True
                except Exception as e:
                    logger.error(f"CRM save failed during timeout: {e}")
        
        # End the session
        session.active = False
        session.last_activity = datetime.now()
        session_manager.db.save_session(session)
        
        # Clear session state
        for key in ['current_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.info("🏠 Redirecting to home page...")
        
        # ENHANCED: Force redirect with timeout flag
        st.query_params["timeout_redirect"] = "true"
        time.sleep(1)
        st.rerun()
        
        return True
    
    return False

def handle_timeout_redirect():
    """ENHANCED: Set timeout context when redirecting"""
    logger.info("🔍 DEBUG: handle_timeout_redirect called")

    if st.query_params.get("timeout_redirect") == "true":
        # Set timeout context in JavaScript
        timeout_context_js = """
        <script>
        try {
            // Store timeout reason in sessionStorage
            sessionStorage.setItem('fifi_timeout_reason', 'session_timeout_15min_inactivity');
            
            // Also send message to browser close detection
            window.postMessage({
                type: 'fifi_timeout_context',
                reason: 'session_timeout_15min_inactivity'
            }, '*');
            
            console.log('⏰ Timeout context set: session_timeout_15min_inactivity');
        } catch (e) {
            console.error('Failed to set timeout context:', e);
        }
        </script>
        """
        st.components.v1.html(timeout_context_js, height=0, width=0)
        
        # Clear the flag
        if "timeout_redirect" in st.query_params:
            del st.query_params["timeout_redirect"]
        
        # Clear session state to show welcome page
        for key in ['current_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]        
        
def render_timeout_status_sidebar(session):
    """
    Shows timeout countdown in sidebar ONLY in the last 5 minutes.
    """
    TIMEOUT_MINUTES = 15
    time_since_activity = datetime.now() - session.last_activity
    time_remaining = timedelta(minutes=TIMEOUT_MINUTES) - time_since_activity
    
    if time_remaining.total_seconds() > 0:
        total_seconds = int(time_remaining.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        
        # ONLY show warning in last 5 minutes
        if minutes < 5:
            if minutes < 2:
                st.sidebar.error(f"⏰ Session expires in: {minutes}m {seconds}s")
            else:
                st.sidebar.warning(f"⏰ Session expires in: {minutes}m {seconds}s")
            st.sidebar.caption("Any activity resets the timer")

def render_chat_interface_complete_fix(session_manager: 'SessionManager', session: UserSession):
    """Complete fix for all chat issues"""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")

    # ENHANCEMENT A: Session refresh at interface start
    try:
        fresh_session = session_manager.db.load_session(session.session_id)
        if fresh_session and fresh_session.active:
            session = fresh_session
    except Exception as refresh_error:
        logger.error(f"Interface session refresh failed: {refresh_error}")

    # ENHANCEMENT B: Database health check
    try:
        session_manager.db._ensure_connection_healthy(session_manager.config)
    except Exception as db_error:
        logger.error(f"Database check failed: {db_error}")

    # Server timeout check
    time_since_activity = datetime.now() - session.last_activity
    if time_since_activity.total_seconds() > (15 * 60):
        st.error("⏰ **Session Timeout:** Your session expired due to 15 minutes of inactivity.")
        if session_manager._is_crm_save_eligible(session, "Server Timeout"):
            with st.spinner("💾 Saving conversation..."):
                try:
                    session_manager.zoho.save_chat_transcript_sync(session, "Server Timeout")
                    st.success("✅ Conversation saved")
                except Exception as e:
                    logger.error(f"CRM save failed: {e}")
        
        session.active = False
        try:
            session_manager.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to save ended session: {e}")
        
        for key in ['current_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.info("🏠 Please start a new session from the welcome page.")
        time.sleep(2)
        st.rerun()
        return

    # ENHANCEMENT C: Keep your working meta refresh + add activity detection
    inject_dynamic_timeout_refresh(session)
    add_activity_detection(session.session_id, session_manager, session)

    # Your existing functionality (fingerprinting, error handler, etc.)
    fingerprint_needed = (
        not session.fingerprint_id or
        session.fingerprint_method == "temporary_fallback_python" or
        session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
    )
    
    if fingerprint_needed:
        session_manager.fingerprinting.render_fingerprint_component(session.session_id)

    global_message_channel_error_handler()
    
    if session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]:
        try:
            render_browser_close_detection_enhanced(session.session_id)
        except Exception as e:
            logger.error(f"Browser close detection failed: {e}")

    # User limits check
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
            
            if msg.get("role") == "assistant" and "source" in msg:
                source_color = {
                    "FiFi": "🧠", "FiFi Web Search": "🌐", 
                    "Content Moderation": "🛡️", "System Fallback": "⚠️",
                    "Error Handler": "❌"
                }.get(msg['source'], "🤖")
                st.caption(f"{source_color} Source: {msg['source']}")

    # ENHANCEMENT D: Enhanced chat input processing
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                            disabled=session.ban_status.value != BanStatus.NONE.value)
    
    if prompt:
        logger.info(f"🎯 Processing question from {session.session_id[:8]}")
        
        # CRITICAL: Refresh session again before processing
        try:
            fresh_session = session_manager.db.load_session(session.session_id)
            if fresh_session and fresh_session.active:
                session = fresh_session
        except Exception as refresh_error:
            logger.error(f"Pre-processing refresh failed: {refresh_error}")
        
        # Update activity immediately
        session.last_activity = datetime.now()
        try:
            session_manager.db.save_session(session)
        except Exception as save_error:
            logger.error(f"Failed to save activity: {save_error}")
        
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
                                "FiFi": "🧠", "FiFi Web Search": "🌐",
                                "Content Moderation": "🛡️", "System Fallback": "⚠️",
                                "Error Handler": "❌"
                            }.get(response['source'], "🤖")
                            st.caption(f"{source_color} Source: {response['source']}")
                        
                        logger.info(f"✅ Question processed successfully for {session.session_id[:8]}")
                        
                except Exception as e:
                    logger.error(f"❌ AI response failed: {e}", exc_info=True)
                    st.error("⚠️ I encountered an error. Please try again.")
        
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
        handle_auto_timeout_from_query()
        handle_timeout_redirect()
    except Exception as e:
        logger.error(f"Query parameter handling failed: {e}")

    # Get session manager
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Session Manager not available. Please refresh the page.")
        return

    # Route to appropriate page
    current_page = st.session_state.get('page')
    logger.info(f"🔍 MAIN ROUTING: Current page = '{current_page}'")
    
    try:
        if current_page != "chat":
            logger.info("🔍 MAIN ROUTING: Rendering welcome page")
            render_welcome_page(session_manager)
            
        else:
            logger.info("🔍 MAIN ROUTING: Should render chat interface, getting session...")
            try:
                session = session_manager.get_session()
                logger.info(f"🔍 MAIN ROUTING: Got session {session.session_id[:8] if session else 'None'}")
                
                if session and session.active:
                    logger.info(f"🔍 MAIN ROUTING: Session is active, rendering sidebar and chat interface")
                    
                     # ADD THIS CHECK HERE - Check timeout BEFORE rendering anything
                    if check_and_handle_timeout_with_reset(session_manager, session):
                        return  # Session timed out and was reset, stop here
                        
                    render_sidebar(session_manager, session, st.session_state.pdf_exporter)
                    render_chat_interface_complete_fix(session_manager, session)
                else:
                    logger.warning(f"🔍 MAIN ROUTING: Session inactive or None, redirecting to welcome")
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
