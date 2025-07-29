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
# from streamlit_javascript import st_javascript # Not used in current code, keeping for future if needed

# =============================================================================
# CLEAN PROFESSIONAL FIFI AI - DEBUG ELEMENTS REMOVED
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
    # from langchain_openai import ChatOpenAI # Not directly used in EnhancedAI placeholder
    # from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # Not directly used in EnhancedAI placeholder
    OPENAI_AVAILABLE = True
    LANGCHAIN_AVAILABLE = True # Keep this for general readiness
except ImportError:
    pass

try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    pass

try:
    # from langchain_tavily import TavilySearch # Not directly used in EnhancedAI placeholder
    TAVILY_AVAILABLE = True
except ImportError:
    pass

try:
    # from pinecone import Pinecone # Not directly used in EnhancedAI placeholder
    # from pinecone_plugins.assistant.models.chat import Message as PineconeMessage # Not directly used in EnhancedAI placeholder
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
    recognition_response: Optional[str] = None # Added for device recognition logic
    
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
                # SQLite Cloud does not support row_factory globally like this.
                # It's set to None explicitly before save/load where it matters.
                # For schema creation, it's fine.
                if hasattr(self.conn, 'row_factory'): 
                    self.conn.row_factory = None

                # Create table with all columns upfront, matching UserSession fields
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
            # Access config safely, as session_manager might not be fully initialized yet
            current_config = None
            if 'session_manager' in st.session_state and st.session_state.session_manager:
                current_config = st.session_state.session_manager.config
            
            if current_config:
                self._ensure_connection(current_config)

            if self.db_type == "memory":
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                logger.debug(f"Saved session {session.session_id[:8]} to in-memory.")
                return
            
            try:
                # For SQLite Cloud, it's safer to always use raw tuples, so row_factory should be None
                if self.db_type == "cloud" and hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                elif self.db_type == "file":
                    # For local SQLite, row_factory can be used if desired, but for consistency, keep None
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
                # Re-raise to let handle_api_errors decorator catch it and display error
                raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility and connection health check"""
        with self.lock:
            # Check and ensure connection health before any DB operation
            current_config = None
            if 'session_manager' in st.session_state and st.session_state.session_manager:
                current_config = st.session_state.session_manager.config
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
                # For SQLite Cloud, it's safer to always use raw tuples, so row_factory should be None
                if self.db_type == "cloud" and hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                elif self.db_type == "file":
                    # For local SQLite, row_factory can be used if desired, but for consistency, keep None
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
            current_config = None
            if 'session_manager' in st.session_state and st.session_state.session_manager:
                current_config = st.session_state.session_manager.config
            if current_config:
                self._ensure_connection(current_config)

            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
            
            try:
                # For SQLite Cloud, it's safer to always use raw tuples, so row_factory should be None
                if self.db_type == "cloud" and hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                elif self.db_type == "file":
                    # For local SQLite, row_factory can be used if desired, but for consistency, keep None
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
            current_config = None
            if 'session_manager' in st.session_state and st.session_state.session_manager:
                current_config = st.session_state.session_manager.config
            if current_config:
                self._ensure_connection(current_config)

            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.email == email]
            
            try:
                # For SQLite Cloud, it's safer to always use raw tuples, so row_factory should be None
                if self.db_type == "cloud" and hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                elif self.db_type == "file":
                    # For local SQLite, row_factory can be used if desired, but for consistency, keep None
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
# FEATURE MANAGERS - FINGERPRINTING FIXED
# =============================================================================

class FingerprintingManager:
    """Manages browser fingerprinting using redirect method from original code."""
    
    def __init__(self):
        self.fingerprint_cache = {}

    def render_fingerprint_component(self, session_id: str):
        """Renders fingerprinting component that redirects with results."""
        try:
            # Create the fingerprinting HTML component that redirects back with data
            fingerprint_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>FiFi Fingerprinting</title>
                <style>
                    /* Hide the iframe content to prevent flickering */
                    body {{ margin: 0; padding: 0; display: none; }}
                </style>
            </head>
            <body>
                <script>
                (function() {{
                    // Ensure script runs only once per iframe load
                    if (window.fifiFingerprintInitialized) return;
                    window.fifiFingerprintInitialized = true;

                    const sessionId = '{session_id}';
                    
                    function generateFingerprint() {{
                        try {{
                            // Enhanced browser fingerprinting
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d');
                            let canvasData = 'unsupported';
                            if (ctx) {{
                                ctx.textBaseline = 'top';
                                ctx.font = '14px Arial';
                                ctx.fillStyle = '#f60';
                                ctx.fillRect(125, 1, 62, 20);
                                ctx.fillStyle = '#069';
                                ctx.fillText('FiFi Fingerprint Test', 2, 2);
                                ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
                                ctx.fillText('Browser ID: ' + sessionId.substring(0, 8), 4, 17);
                                canvasData = canvas.toDataURL();
                            }}
                            
                            // WebGL fingerprinting
                            let webglInfo = 'none';
                            try {{
                                const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                                if (gl) {{
                                    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
                                    if (debugInfo) {{
                                        webglInfo = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) + '|' + 
                                                   gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                                    }} else {{
                                        webglInfo = gl.getParameter(gl.VENDOR) + '|' + gl.getParameter(gl.RENDERER);
                                    }}
                                }}
                            }} catch (e) {{
                                webglInfo = 'webgl_error';
                            }}
                            
                            // Collect comprehensive browser properties
                            const props = [
                                navigator.userAgent || '',
                                navigator.language || '',
                                (navigator.languages || []).join(','),
                                navigator.platform || '',
                                screen.width + 'x' + screen.height + 'x' + screen.colorDepth,
                                (new Date()).getTimezoneOffset(),
                                Intl.DateTimeFormat().resolvedOptions().timeZone || '',
                                navigator.hardwareConcurrency || 0,
                                navigator.deviceMemory || 0,
                                navigator.maxTouchPoints || 0,
                                navigator.cookieEnabled ? '1' : '0',
                                navigator.doNotTrack || '',
                                canvasData.substring(canvasData.length - Math.min(canvasData.length, 100)), // Last 100 chars of canvas
                                webglInfo
                            ];
                            
                            // Create consistent hash (using MurmurHash2 for better distribution if possible, or simple hash)
                            let hash = 0;
                            const dataString = props.join('|');
                            for (let i = 0; i < dataString.length; i++) {{
                                const char = dataString.charCodeAt(i);
                                hash = ((hash << 5) - hash) + char;
                                hash = hash & hash; // Convert to 32bit integer
                            }}
                            
                            // Convert to hex fingerprint ID (12 characters, padded)
                            const fingerprintId = (Math.abs(hash) >>> 0).toString(16).padStart(12, '0').substring(0, 12);
                            
                            // Determine privacy level
                            let privacyLevel = 'standard';
                            if (navigator.doNotTrack === '1' || navigator.globalPrivacyControl) {{
                                privacyLevel = 'high_privacy';
                            }} else if (!navigator.cookieEnabled) {{
                                privacyLevel = 'medium_privacy';
                            }}
                            
                            // Working methods
                            const workingMethods = ['canvas', 'navigator', 'screen', 'timezone'];
                            if (webglInfo !== 'none' && webglInfo !== 'webgl_error') {{
                                workingMethods.push('webgl');
                            }}
                            if (navigator.hardwareConcurrency) {{
                                workingMethods.push('hardware');
                            }}
                            
                            console.log('✅ FiFi fingerprint generated successfully:', fingerprintId);
                            
                            // Redirect back to app with fingerprint data
                            redirectWithFingerprint(fingerprintId, 'enhanced_browser_fingerprint', privacyLevel, workingMethods);
                            
                        }} catch (error) {{
                            console.error('❌ Advanced fingerprinting failed:', error);
                            
                            // Fallback fingerprinting logic
                            try {{
                                const simpleProps = [
                                    navigator.userAgent || 'unknown',
                                    navigator.language || 'unknown',
                                    navigator.platform || 'unknown',
                                    screen.width || 0,
                                    screen.height || 0,
                                    (new Date()).getTimezoneOffset() || 0
                                ];
                                
                                let simpleHash = 0;
                                const simpleString = simpleProps.join('|');
                                for (let i = 0; i < simpleString.length; i++) {{
                                    simpleHash = ((simpleHash << 5) - simpleHash) + simpleString.charCodeAt(i);
                                    simpleHash = simpleHash & simpleHash;
                                }}
                                
                                const fallbackId = (Math.abs(simpleHash) >>> 0).toString(16).padStart(12, '0').substring(0, 12);
                                
                                console.log('⚠️ Using fallback fingerprint:', fallbackId);
                                redirectWithFingerprint(fallbackId, 'simple_fallback', 'unknown', ['basic_navigator']);
                                
                            }} catch (fallbackError) {{
                                console.error('❌ Even fallback fingerprinting failed:', fallbackError);
                                
                                // Ultimate fallback - timestamp based
                                const timestampId = Date.now().toString(16).substring(-12).padStart(12, '0');
                                redirectWithFingerprint(timestampId, 'timestamp_fallback', 'unknown', ['timestamp_only']);
                            }}
                        }}
                    }}
                    
                    function redirectWithFingerprint(fingerprintId, method, privacy, workingMethods) {{
                        try {{
                            // Get the current app URL (for parent window if in iframe, or current window)
                            let appUrl = window.location.origin + window.location.pathname;
                            
                            // Try to get parent URL if in iframe and same origin
                            try {{
                                if (window.parent && window.parent.location && 
                                    window.parent.location.origin === window.location.origin) {{
                                    appUrl = window.parent.location.origin + window.parent.location.pathname;
                                }}
                            }} catch (e) {{
                                console.debug('Cannot access parent URL, using current URL: ' + e);
                            }}
                            
                            // Build redirect URL with fingerprint data
                            const params = new URLSearchParams({{
                                event: 'fingerprint_complete',
                                session_id: sessionId,
                                fingerprint_id: fingerprintId,
                                method: method,
                                privacy: privacy,
                                working_methods: workingMethods.join(','),
                                timestamp: Date.now()
                            }});
                            
                            const redirectUrl = `${{appUrl}}?${{params.toString()}}`;
                            
                            console.log('🔄 Attempting redirect to app with fingerprint data...');
                            
                            // Redirect to main app
                            if (window.parent && window.parent.location && 
                                window.parent.location.origin === window.location.origin) {{
                                window.parent.location.href = redirectUrl;
                            }} else {{
                                window.location.href = redirectUrl;
                            }}
                            
                        }} catch (redirectError) {{
                            console.error('❌ Redirect failed:', redirectError);
                            // Fallback if redirect itself fails, this is less likely to happen
                        }}
                    }}
                    
                    // Start fingerprinting immediately after DOM is ready
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', generateFingerprint);
                    }} else {{
                        setTimeout(generateFingerprint, 100); // Small delay to ensure Streamlit is ready to receive
                    }}
                    
                }})();
                </script>
            </body>
            </html>
            """
            
            # Render the component invisibly in an iframe
            # Sandbox rules allow scripts, same origin, and top navigation (for redirect)
            st.components.v1.html(
                f'<iframe srcdoc="{html.escape(fingerprint_html)}" style="width:0;height:0;border:0;display:none;" sandbox="allow-scripts allow-same-origin allow-top-navigation"></iframe>',
                height=0, width=0
            )
            
            logger.debug(f"Fingerprint component rendered for session {session_id[:8]} - waiting for redirect...")
            return None  # Always return None since data comes via redirect
            
        except Exception as e:
            logger.error(f"Failed to render fingerprint component: {e}")
            # If rendering fails, provide a Python-side fallback fingerprint
            return self._generate_fallback_fingerprint()

    def process_fingerprint_data(self, fingerprint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes fingerprint data received from the redirect (from original code)."""
        # This method assumes `fingerprint_data` is already from the query parameters.
        # It's primarily used to record the fingerprint and determine visitor type.
        
        fingerprint_id = fingerprint_data.get('fingerprint_id')
        fingerprint_method = fingerprint_data.get('method')
        privacy_level = fingerprint_data.get('privacy', 'standard')
        
        if not fingerprint_id or not fingerprint_method:
            logger.warning("Invalid fingerprint data received for processing. Using internal fallback.")
            return self._generate_fallback_fingerprint()
        
        # Determine visitor type based on internal cache
        visitor_type = "new_visitor"
        if fingerprint_id in self.fingerprint_cache:
            visitor_type = "returning_visitor"
        
        self.fingerprint_cache[fingerprint_id] = {'last_seen': datetime.now()}
        
        return {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': fingerprint_method,
            'visitor_type': visitor_type,
            'browser_privacy_level': privacy_level,
            'working_methods': fingerprint_data.get('working_methods', [])
        }

    def _generate_fallback_fingerprint(self) -> Dict[str, Any]:
        """Generates a unique fallback fingerprint using Python side."""
        # Create a more unique fallback using timestamp and random
        timestamp = str(int(time.time() * 1000))[-8:]  # Last 8 digits of timestamp
        random_part = secrets.token_hex(4)
        fallback_id = f"fb{timestamp}{random_part}"[:16] # Ensure it's 16 chars max
        
        return {
            'fingerprint_id': fallback_id,
            'fingerprint_method': 'secure_fallback_python',
            'visitor_type': 'new_visitor',
            'browser_privacy_level': 'unknown',
            'working_methods': ['timestamp', 'random']
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
            # Supabase OTP sign-in will create user if not exists
            response = self.supabase.auth.sign_in_with_otp({
                'email': email,
                'options': {
                    'should_create_user': True, # Create user if not exists
                    'email_redirect_to': None,  # No redirect needed for OTP
                    'data': {
                        'verification_type': 'email_otp'
                    }
                }
            })
            
            # Supabase sign_in_with_otp returns a data object, not just a response directly
            if response and response.data and response.data.user:
                logger.info(f"Email OTP code sent to {email} via Supabase.")
                return True
            else:
                # Log detailed error if response.data or user is missing
                logger.error(f"Supabase OTP send failed - unexpected response from sign_in_with_otp: {response}")
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
            # Supabase verify_otp response structure: { 'data': { 'user': User, 'session': Session }, 'error': Error }
            response = self.supabase.auth.verify_otp({
                'email': email,
                'token': code.strip(),
                'type': 'email' # The type of OTP to verify (e.g., 'email', 'phone', 'signup')
            })
            
            if response and response.data and response.data.user:
                logger.info(f"Email verification successful for {email} (Supabase User ID: {response.data.user.id}).")
                return True
            else:
                logger.warning(f"Email verification failed for {email}: Invalid code or no user/session returned. Error: {response.error.message if response.error else 'Unknown'}")
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
        # Escalating penalties in hours
        self.evasion_penalties = [24, 48, 96, 192, 336] 
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """Checks if the current session is within its allowed question limits or if any bans are active."""
        user_limit = self.question_limits.get(session.user_type.value, 0)
        
        # Check if ban is active and still within its duration
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
                session.question_limit_reached = False # Reset limit status on ban expiry
        
        # Reset daily question count if 24 hours have passed since last question
        if session.last_question_time:
            time_since_last = datetime.now() - session.last_question_time
            if time_since_last >= timedelta(hours=24):
                logger.info(f"Daily question count reset for session {session.session_id[:8]}.")
                session.daily_question_count = 0
                session.question_limit_reached = False # Reset limit status on daily reset
        
        # Apply limits based on user type
        if session.user_type.value == UserType.GUEST.value:
            if session.daily_question_count >= user_limit:
                session.question_limit_reached = True # Mark that limit is reached
                return {
                    'allowed': False,
                    'reason': 'guest_limit',
                    'message': 'Please provide your email address to continue.'
                }
        
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            if session.daily_question_count >= user_limit:
                session.question_limit_reached = True # Mark that limit is reached
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Email-verified daily limit reached")
                return {
                    'allowed': False,
                    'reason': 'daily_limit',
                    'message': self._get_email_verified_limit_message()
                }
        
        elif session.user_type.value == UserType.REGISTERED_USER.value:
            if session.total_question_count >= user_limit:
                session.question_limit_reached = True # Mark that limit is reached
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
            BanStatus.EVASION_BLOCK.value: session.current_penalty_hours # Use calculated penalty for evasion
        }.get(ban_type.value, 24) # Default to 24 hours if type is unknown

        session.ban_status = ban_type
        session.ban_start_time = datetime.now()
        session.ban_end_time = session.ban_start_time + timedelta(hours=ban_hours)
        session.ban_reason = reason
        session.question_limit_reached = True # Ensure this is set when banned
        
        logger.info(f"Ban applied to session {session.session_id[:8]}: Type={ban_type.value}, Duration={ban_hours}h, Reason='{reason}'.")
    
    def apply_evasion_penalty(self, session: UserSession) -> int:
        """Applies an escalating penalty for evasion attempts."""
        session.evasion_count += 1
        # Cap escalation level to available penalties
        session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
        
        # Get penalty hours based on escalation level
        # Adjusting index to be 0-based
        penalty_hours = self.evasion_penalties[session.escalation_level - 1] if session.escalation_level > 0 else self.evasion_penalties[0]
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
        else: # Covers EMAIL_VERIFIED_GUEST and general daily limits
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
        # Ensure 'Normal' style is available or create a basic one
        if 'Normal' not in self.styles:
            self.styles.add(ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=10, leading=12))

        self.styles.add(ParagraphStyle(name='ChatHeader', alignment=TA_CENTER, fontSize=18))
        self.styles.add(ParagraphStyle(name='UserMessage', backColor=lightgrey, fontName='Helvetica', fontSize=10, leading=12))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        """Generates a PDF of the chat transcript."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = [Paragraph("FiFi AI Chat Transcript", self.styles['ChatHeader']), Spacer(1, 0.2*inch)]
        
        for msg in session.messages:
            role = str(msg.get('role', 'unknown')).capitalize()
            content = str(msg.get('content', ''))
            
            # Simple HTML tag stripping and escaping for PDF safety
            content_plain = re.sub(r'<[^>]+>', '', content) # Remove HTML tags
            content_escaped = html.escape(content_plain) # HTML escape
            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<b>{role}:</b> {content_escaped}", style))
            
            if msg.get('source'):
                story.append(Paragraph(f"<i>Source: {html.escape(str(msg['source']))}</i>", self.styles['Normal']))
                
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f"Error building PDF: {e}", exc_info=True)
            return None

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
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
            
            self._access_token = data.get('access_token')
            # Token usually expires in 1 hour (3600 seconds), set expiry slightly before for safety
            self._token_expiry = datetime.now() + timedelta(minutes=50) 
            
            logger.info("Successfully obtained Zoho access token.")
            return self._access_token
            
        except requests.exceptions.Timeout:
            logger.error(f"Zoho token request timed out after {timeout} seconds.")
            return None
        except Exception as e:
            logger.error(f"Failed to get Zoho access token: {e}", exc_info=True)
            return None

    @handle_api_errors("Zoho CRM", "Save Chat Transcript")
    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """Synchronously saves the chat transcript to Zoho CRM."""
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        
        if (session.user_type != UserType.REGISTERED_USER or 
            not session.email or 
            not session.messages or 
            not self.config.ZOHO_ENABLED):
            logger.info(f"ZOHO SAVE SKIPPED: Not eligible (UserType: {session.user_type.value}, Email: {bool(session.email)}, Messages: {bool(session.messages)}, Zoho Enabled: {self.config.ZOHO_ENABLED}).")
            return False
        
        access_token = self._get_access_token()
        if not access_token:
            logger.error("ZOHO SAVE FAILED: No access token.")
            return False

        # Placeholder for Zoho CRM interaction logic
        # In a real scenario, you'd find/create a contact, upload the PDF, etc.
        logger.info("✅ ZOHO SAVE COMPLETED SUCCESSFULLY (simplified placeholder logic)")
        # Simulate success
        return True

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
    # Use html.escape for basic XSS prevention
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
        """Provides a simplified AI response (placeholder for actual AI logic)."""
        # Simulate AI processing time
        time.sleep(1.5) 
        
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
        # Use a non-deprecated model if available, otherwise "text-moderation-latest"
        model_to_use = "text-moderation-latest" 
        # OpenAI's new moderation endpoint is `moderations.create`
        response = client.moderations.create(input=prompt, model=model_to_use)
        result = response.results[0]
        
        if result.flagged:
            # Dynamically get flagged categories
            flagged_categories = [cat for cat, flagged in result.categories.model_dump().items() if flagged]
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
                 email_verification_manager: Any, # Use Any for type hinting as it can be DummyEmail
                 question_limit_manager: QuestionLimitManager):
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
        # This can be used for CRM auto-save or other inactivity logic
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
                    # Remove entries older than 24 hours
                    if now - data.get('last_seen', now) > timedelta(hours=24):
                        old_entries.append(fp_id)
                
                for old_fp in old_entries:
                    del self.fingerprinting.fingerprint_cache[old_fp]
                
                if old_entries:
                    logger.info(f"Cleaned up {len(old_entries)} old fingerprint cache entries")
            
            self._last_cleanup = now
            logger.debug("Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}", exc_info=True)

    def _update_activity(self, session: UserSession):
        """Updates the session's last activity timestamp and saves it to the DB."""
        session.last_activity = datetime.now()
        
        # Reset timeout_saved_to_crm flag if session becomes active again
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset 'timeout_saved_to_crm' flag for session {session.session_id[:8]} due to new activity.")
        
        # Ensure user_type is an Enum, especially after loading from DB
        if isinstance(session.user_type, str):
            try:
                session.user_type = UserType(session.user_type)
            except ValueError:
                logger.warning(f"Invalid user_type '{session.user_type}' for session {session.session_id[:8]}, defaulting to GUEST.")
                session.user_type = UserType.GUEST
        
        # Ensure messages is a list, especially after loading from DB
        if not isinstance(session.messages, list):
            logger.warning(f"Messages field corrupted for session {session.session_id[:8]}, preserving as empty list")
            session.messages = []

        try:
            self.db.save_session(session)
            logger.debug(f"Activity update saved for {session.session_id[:8]} with {len(session.messages)} messages")
        except Exception as e:
            logger.error(f"Failed to save session during activity update: {e}", exc_info=True)

    def _create_new_session(self) -> UserSession:
        """Creates a new user session with a temporary fallback fingerprint."""
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id)
        
        # Apply temporary fallback fingerprint immediately
        fallback_fp_data = self.fingerprinting._generate_fallback_fingerprint()
        session.fingerprint_id = fallback_fp_data['fingerprint_id']
        session.fingerprint_method = fallback_fp_data['fingerprint_method']
        session.visitor_type = fallback_fp_data['visitor_type']
        session.browser_privacy_level = fallback_fp_data['browser_privacy_level']
        
        # Save to database immediately after creation
        self.db.save_session(session)
        
        logger.info(f"Created new session {session_id[:8]} with user type {session.user_type.value} and temporary fingerprint {session.fingerprint_id[:8]}")
        return session

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]):
        """Applies fingerprinting data (from redirect) to the session."""
        try:
            if not fingerprint_data or not isinstance(fingerprint_data, dict):
                logger.warning("Invalid fingerprint data provided to apply_fingerprinting")
                return
            
            # Store old fingerprint for logging/rollback if needed
            old_fingerprint_id = session.fingerprint_id
            old_method = session.fingerprint_method
            
            # Update session with new fingerprint details
            session.fingerprint_id = fingerprint_data.get('fingerprint_id')
            session.fingerprint_method = fingerprint_data.get('fingerprint_method')
            session.visitor_type = fingerprint_data.get('visitor_type', 'new_visitor')
            session.browser_privacy_level = fingerprint_data.get('browser_privacy_level', 'standard')
            
            # Basic validation after update
            if not session.fingerprint_id or not session.fingerprint_method:
                logger.error("Invalid fingerprint data applied: missing essential fields after processing. Reverting.")
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return
            
            # Check for existing sessions with same fingerprint to determine if returning user
            try:
                existing_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
                # Filter out the current session if it's already in the DB
                existing_sessions = [s for s in existing_sessions if s.session_id != session.session_id]
                
                if existing_sessions:
                    # Inherit recognition data (like email) from the most recent session
                    # This helps in identifying a returning user faster
                    recent_session = max(existing_sessions, key=lambda s: s.last_activity)
                    if recent_session.email and recent_session.user_type != UserType.GUEST:
                        session.visitor_type = "returning_visitor"
                        # Potentially pre-fill email for verification if not already set
                        if not session.email:
                            session.email = recent_session.email
                            if session.email not in session.email_addresses_used:
                                session.email_addresses_used.append(session.email)
                        session.recognition_response = "recognized" # Set for UI
                    else:
                        session.visitor_type = "returning_device" # Device recognized, but no user email
                        session.recognition_response = "recognized_device_only"
                else:
                    session.visitor_type = "new_visitor"
                    session.recognition_response = "new_device"

            except Exception as e:
                logger.error(f"Failed to check fingerprint history during application: {e}")
                # Continue without history check, default to new_visitor

            # Save session with newly applied fingerprint data
            try:
                self.db.save_session(session)
                logger.info(f"✅ Fingerprinting applied to {session.session_id[:8]}: {session.fingerprint_method} (ID: {session.fingerprint_id[:8]}...) Visitor Type: {session.visitor_type}")
            except Exception as e:
                logger.error(f"Failed to save session after fingerprinting: {e}. Reverting fingerprint.")
                # Revert to old values on save failure to maintain consistency
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                # Potentially set status to error on session for UI
                st.session_state.fingerprint_status = 'error' # Direct Streamlit state update
                st.session_state.fingerprint_error = "Failed to save fingerprint"
                
        except Exception as e:
            logger.error(f"Error in apply_fingerprinting for session {session.session_id[:8]}: {e}", exc_info=True)
            st.session_state.fingerprint_status = 'error'
            st.session_state.fingerprint_error = "Unexpected error applying fingerprint"


    def get_session(self) -> Optional[UserSession]:
        """Gets or creates the current user session."""
        # Perform periodic cleanup
        self._periodic_cleanup()

        try:
            # Try to get existing session ID from Streamlit session state
            session_id = st.session_state.get('current_session_id')
            
            session = None
            if session_id:
                session = self.db.load_session(session_id)
                
                # If session is found but inactive, treat as new session
                if session and not session.active:
                    logger.info(f"Loaded session {session_id[:8]} is inactive. Treating as new session.")
                    session = None # Force creation of new session

            if not session:
                # Create new session if no valid session found or if loaded session was inactive
                new_session = self._create_new_session()
                st.session_state.current_session_id = new_session.session_id
                session = new_session

            # Validate and update activity for the active session
            session = self._validate_session(session) # This also saves the session
            
            # Re-check limits and handle bans after activity update and validation
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check.get('allowed', True):
                ban_type = limit_check.get('ban_type', 'unknown')
                message = limit_check.get('message', 'Access restricted due to usage policy.')
                time_remaining = limit_check.get('time_remaining')
                
                # Display ban message to the user directly if not handled elsewhere
                # (This can be refined in UI components, but keeping it here as a fallback)
                st.error(f"🚫 **Access Restricted**")
                if time_remaining:
                    hours = max(0, int(time_remaining.total_seconds() // 3600))
                    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                    st.error(f"Time remaining: {hours}h {minutes}m")
                st.info(message)
                logger.info(f"Session {session.session_id[:8]} is currently banned: Type={ban_type}, Reason='{message}'.")
                
            return session
            
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to get/create session: {e}", exc_info=True)
            # Create a very basic fallback session in case of catastrophic failure
            fallback_session = UserSession(session_id=str(uuid.uuid4()), user_type=UserType.GUEST)
            fallback_session.fingerprint_id = f"emergency_fp_{fallback_session.session_id[:8]}"
            fallback_session.fingerprint_method = "emergency_fallback"
            st.session_state.current_session_id = fallback_session.session_id
            st.error("⚠️ Failed to establish persistent session. Operating in emergency fallback mode. Chat history may not persist.")
            return fallback_session

    def _validate_session(self, session: UserSession) -> UserSession:
        """Validates and updates session activity, including ban expiry."""
        session.last_activity = datetime.now()
        
        # Check for ban expiry and reset if expired
        if (session.ban_status != BanStatus.NONE and 
            session.ban_end_time and 
            datetime.now() >= session.ban_end_time):
            logger.info(f"Ban expired for session {session.session_id[:8]}")
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            session.question_limit_reached = False # Reset limit status as ban is over
        
        # Save updated session to ensure activity and ban status are persisted
        self.db.save_session(session)
        return session

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """Checks fingerprint history for device recognition."""
        if not fingerprint_id or fingerprint_id.startswith(("temp_py_", "emergency_fp_")):
            return {'has_history': False}
        
        sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
        if not sessions:
            return {'has_history': False}
        
        # Find most recent session with a non-guest user type and an email
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
        if not email or '@' not in email:
            return email
        local, domain = email.split('@', 1)
        if len(local) <= 2:
            return f"{local[0]}***@{domain}"
        # Mask all but first and last characters of local part
        return f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}@{domain}"

    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """Authenticates user with WordPress and creates/updates session."""
        if not self.config.WORDPRESS_URL:
            st.error("WordPress authentication is not configured in secrets.")
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
                
                # Get current session to update it
                # Using st.session_state.current_session_id directly to ensure we get the right one
                session = self.db.load_session(st.session_state.get('current_session_id'))
                if not session:
                    logger.error("No active session found during WordPress authentication. Creating new session.")
                    session = self._create_new_session() # Fallback to new session if current is lost
                    st.session_state.current_session_id = session.session_id

                # Update session to registered user
                session.user_type = UserType.REGISTERED_USER
                session.email = email
                session.full_name = display_name
                session.wp_token = wp_token
                
                # Add email to email history if not already there
                if email and email not in session.email_addresses_used:
                    session.email_addresses_used.append(email)
                
                # Reset daily limits and ban status on successful login
                session.daily_question_count = 0
                session.question_limit_reached = False
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None

                self.db.save_session(session)
                
                logger.info(f"WordPress authentication successful for {email}. Session {session.session_id[:8]} upgraded.")
                return session
                
            else:
                error_message = response.json().get('message', 'Authentication failed.')
                st.error(f"Authentication failed: {error_message}")
                logger.warning(f"WordPress authentication failed for user {username}: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Authentication request timed out. Please try again.")
            logger.error(f"WordPress authentication timeout for {username}.")
            return None
        except Exception as e:
            logger.error(f"WordPress authentication failed: {e}", exc_info=True)
            st.error("Authentication service is temporarily unavailable.")
            return None

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Handles email verification for guest users."""
        try:
            # Update session with email (if changed or first time)
            if not session.email or session.email != email:
                if session.email: # If changing email
                    session.email_switches_count += 1
                session.email = email
                if email not in session.email_addresses_used:
                    session.email_addresses_used.append(email)
                self.db.save_session(session) # Save immediately to persist email

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
            logger.error(f"Email verification handling failed: {e}", exc_info=True)
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
                    'message': "No email address found for verification. Please enter your email first."
                }
            
            success = self.email_verification.verify_code(session.email, code)
            if success:
                # Upgrade session to email verified guest
                session.user_type = UserType.EMAIL_VERIFIED_GUEST
                session.question_limit_reached = False # Reset limit flag
                session.daily_question_count = 0  # Reset count after verification for fresh start
                session.ban_status = BanStatus.NONE # Clear any pending bans
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                
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
            logger.error(f"Email code verification failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': "Verification failed due to a technical error."
            }

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Gets AI response for user prompt with all checks and limits."""
        try:
            # Rate limiting check (basic, per session ID)
            if not self.rate_limiter.is_allowed(session.session_id):
                return {
                    'content': 'Please slow down - you are sending requests too quickly.',
                    'success': False,
                    'source': 'Rate Limiter'
                }
            
            # Question limit check (tier-based)
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check['allowed']:
                # Save session to persist any ban updates from `is_within_limits`
                self.db.save_session(session) 
                
                if limit_check['reason'] == 'guest_limit':
                    return {'requires_email': True}
                elif limit_check['reason'] in ['banned', 'daily_limit', 'total_limit']:
                    return {
                        'banned': True,
                        'content': limit_check.get('message', 'Access restricted.'),
                        'time_remaining': limit_check.get('time_remaining')
                    }
            
            # Content moderation (if OpenAI client and API are available)
            sanitized_prompt = sanitize_input(prompt)
            moderation_result = check_content_moderation(sanitized_prompt, self.ai.openai_client)
            if moderation_result and moderation_result.get('flagged'):
                # Apply an evasion penalty if moderation is triggered
                penalty_hours = self.question_limits.apply_evasion_penalty(session)
                self.db.save_session(session) # Save ban status
                return {
                    'content': "Your message violates our content policy. Further attempts may result in temporary restrictions.",
                    "success": False,
                    "evasion_penalty": True,
                    "penalty_hours": penalty_hours,
                    "source": "Content Moderation"
                }
            
            # Record the question count for this session
            self.question_limits.record_question(session)
            
            # Add user message to session's chat history
            user_message = {"role": "user", "content": sanitized_prompt}
            session.messages.append(user_message)
            
            # Get AI response from the AI system
            # Pass only recent messages for context
            ai_response = self.ai.get_response(sanitized_prompt, session.messages[-10:]) 
            
            # Add AI response to session's chat history
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
            
            # Update last activity and save session with new messages and updated limits
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
            # Save to CRM if eligible and not already saved due to timeout
            if (session.user_type == UserType.REGISTERED_USER and 
                session.email and 
                session.messages and 
                not session.timeout_saved_to_crm):
                
                logger.info(f"Performing CRM save during manual session end for {session.session_id[:8]}")
                self.zoho.save_chat_transcript_sync(session, "Manual Sign Out")
                session.timeout_saved_to_crm = True # Mark as saved
            
            # Mark session as inactive in the database
            session.active = False
            session.last_activity = datetime.now()
            self.db.save_session(session)
            
            # Clear Streamlit session state relevant to the current user session
            if 'current_session_id' in st.session_state:
                del st.session_state['current_session_id']
            if 'page' in st.session_state:
                del st.session_state['page'] # Redirect to welcome page
            # Also clear any verification state
            for key in ['verification_stage', 'verification_email']:
                if key in st.session_state:
                    del st.session_state[key]
            
            logger.info(f"Session {session.session_id[:8]} ended successfully.")
            
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
            session.timeout_saved_to_crm = True # Mark as saved
            self.db.save_session(session) # Persist the saved status
        else:
            st.error("❌ Failed to save to CRM. Please try again later.")

# =============================================================================
# QUERY PARAMETER HANDLERS FOR REDIRECT-BASED FINGERPRINTING
# =============================================================================

def process_fingerprint_from_query(session_id: str, fingerprint_id: str, method: str, privacy: str, working_methods: List[str]) -> bool:
    """Processes fingerprint data received via URL query parameters."""
    try:
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("❌ Session manager not available during fingerprint processing from query. Cannot process.")
            return False
        
        # Load the session to apply fingerprint to it
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Fingerprint processing: Session '{session_id[:8]}' not found in database. Cannot apply fingerprint.")
            return False
        
        logger.info(f"✅ Processing fingerprint for session '{session_id[:8]}': ID={fingerprint_id[:8]}, Method={method}, Privacy={privacy}")
        
        # Create processed fingerprint data from query params for apply_fingerprinting
        processed_data = {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': method,
            'browser_privacy_level': privacy,
            'working_methods': working_methods # This will be unused by current `apply_fingerprinting`
        }
        
        # Apply fingerprinting to the loaded session
        session_manager.apply_fingerprinting(session, processed_data)
        
        logger.info(f"✅ Fingerprint applied successfully to session '{session_id[:8]}'")
        return True
        
    except Exception as e:
        logger.error(f"Fingerprint processing failed: {e}", exc_info=True)
        return False

def handle_fingerprint_requests_from_query():
    """Silently processes fingerprint data sent via URL query parameters."""
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    # Check for fingerprint completion event
    if event == "fingerprint_complete" and session_id:
        logger.info("=" * 80)
        logger.info("🔍 FINGERPRINT DATA DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}")
        logger.info("=" * 80)
        
        # EXTRACT ALL RELEVANT PARAMETERS BEFORE CLEARING THEM
        fingerprint_id = query_params.get("fingerprint_id")
        method = query_params.get("method")
        privacy = query_params.get("privacy")
        # Ensure working_methods is a list, even if empty string
        working_methods_str = query_params.get("working_methods", "")
        working_methods = working_methods_str.split(",") if working_methods_str else []
        
        # Clear query parameters immediately to prevent reprocessing on rerun
        # Use a copy of keys to avoid RuntimeError during iteration and modification
        keys_to_clear = list(st.query_params.keys())
        for param in keys_to_clear:
            if param in ["event", "session_id", "fingerprint_id", "method", "privacy", "working_methods", "timestamp"]: # Specific fingerprint params
                del st.query_params[param]
        
        # Process the fingerprint data silently
        if fingerprint_id and method:
            try:
                if process_fingerprint_from_query(session_id, fingerprint_id, method, privacy, working_methods):
                    logger.info(f"✅ Silent fingerprint processing completed for {session_id[:8]}")
                    # Force a rerun to update the UI with the new fingerprint ID
                    st.rerun() 
                else:
                    logger.warning(f"Silent fingerprint processing for {session_id[:8]} returned False.")
            except Exception as e:
                logger.error(f"Silent fingerprint processing failed: {e}", exc_info=True)
        
        # No else block here, as `st.rerun()` handles the UI update.
        # If processing failed, the next rerun will still show pending/error state.
        return
    else:
        logger.debug("ℹ️ No fingerprint requests found in current URL query parameters.")

# =============================================================================
# UI COMPONENTS - CLEAN PROFESSIONAL VERSION
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
                            time.sleep(1) # Small delay for user to see success message
                            st.session_state.page = "chat"
                            st.rerun()
            
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
                st.session_state.page = "chat"
                st.rerun()

def render_sidebar(session_manager: 'SessionManager', session: UserSession, pdf_exporter: PDFExporter):
    """Renders the application's sidebar, displaying session information, user status, and action buttons."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # User Type and Question Limits Display
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Questions Today:** {session.total_question_count}/40")
            # Progress bar for registered users up to 40 questions
            total_limit = session_manager.question_limits.question_limits.get(UserType.REGISTERED_USER.value, 40)
            st.progress(min(session.total_question_count / total_limit, 1.0))
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            daily_limit = session_manager.question_limits.question_limits.get(UserType.EMAIL_VERIFIED_GUEST.value, 10)
            st.progress(min(session.daily_question_count / daily_limit, 1.0))
            
            # Display time until daily reset
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
            guest_limit = session_manager.question_limits.question_limits.get(UserType.GUEST.value, 4)
            st.progress(min(session.daily_question_count / guest_limit, 1.0))
            st.caption("Email verification unlocks 10 questions/day.")
        
        # Enhanced Fingerprint ID Display
        st.markdown("---")
        st.markdown("### 📱 Device Information")

        # Check for fingerprint completion status
        fingerprint_status_message = "Generating..."
        fingerprint_icon = "🔄"
        
        # Determine actual status from session object
        if session.fingerprint_id:
            if session.fingerprint_method in ["enhanced_browser_fingerprint", "enhanced_canvas_webgl"]:
                fingerprint_status_message = f"`{session.fingerprint_id}` ✅"
                fingerprint_icon = ""
                st.caption(f"Method: {session.fingerprint_method.replace('_', ' ').title()} | Privacy: {session.browser_privacy_level.replace('_', ' ').title()}")
            elif session.fingerprint_method in ["simple_fallback", "secure_fallback_python"]:
                fingerprint_status_message = f"`{session.fingerprint_id}` ⚡"
                fingerprint_icon = ""
                st.caption(f"Method: {session.fingerprint_method.replace('_', ' ').title()} | Privacy: {session.browser_privacy_level.replace('_', ' ').title()}")
            elif session.fingerprint_method == "timestamp_fallback":
                fingerprint_status_message = f"`{session.fingerprint_id}` (basic)"
                fingerprint_icon = ""
                st.caption(f"Method: {session.fingerprint_method.replace('_', ' ').title()} | Privacy: {session.browser_privacy_level.replace('_', ' ').title()}")
            elif session.fingerprint_id.startswith(("temp_py_", "emergency_fp_")):
                fingerprint_status_message = "Identifying... 🔄"
                fingerprint_icon = ""
                st.caption("Awaiting full browser fingerprint.")
            else: # Catch-all for unknown, or initial python-generated placeholder
                fingerprint_status_message = f"`{session.fingerprint_id}` (?)"
                fingerprint_icon = ""
                st.caption(f"Method: {session.fingerprint_method or 'Unknown'} | Privacy: {session.browser_privacy_level or 'Unknown'}")
        else: # No fingerprint_id yet means still pending or initial state
            fingerprint_status_message = "Identifying... 🔄"
            fingerprint_icon = ""
            st.caption("Browser fingerprinting in progress.")

        st.markdown(f"**Device ID:** {fingerprint_status_message} {fingerprint_icon}")

        # Display CRM Integration Status
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type == UserType.REGISTERED_USER:
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
        
        # Display Ban/Limit messages
        if session.ban_status != BanStatus.NONE:
            st.error(f"🚫 **STATUS: RESTRICTED**")
            if session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.markdown(f"**Time Remaining:** {hours}h {minutes}m")
            st.markdown(f"Reason: {session.ban_reason or 'Usage policy violation'}")
        elif session.question_limit_reached and session.user_type == UserType.GUEST: 
            st.warning("⚠️ **ACTION REQUIRED: Email Verification**")
        
        st.divider()
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clears all messages from the current conversation."):
                session_manager.clear_chat_history(session)
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True, help="Ends your current session and returns to the welcome page."):
                session_manager.end_session(session)
                st.rerun()

        # Download PDF and Save to CRM (conditional)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
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
    
    # Initialize verification stage if not present
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = 'initial_check'
        # Set a flag to rerun immediately after this to process initial_check
        st.rerun() 

    if st.session_state.verification_stage == 'initial_check':
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            st.info(f"🤝 **We seem to recognize this device!**")
            st.markdown(f"Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's my email", use_container_width=True, key="recognize_yes_btn"):
                    session.recognition_response = "recognized_yes" # User confirms recognition
                    st.session_state.verification_email = fingerprint_history['email']
                    st.session_state.verification_stage = "send_code_recognized"
                    session_manager.db.save_session(session) # Persist recognition response
                    st.rerun()
            with col2:
                if st.button("❌ No, use a different email", use_container_width=True, key="recognize_no_btn"):
                    session.recognition_response = "recognized_no" # User denies recognition
                    st.session_state.verification_stage = "email_entry"
                    session_manager.db.save_session(session) # Persist recognition response
                    st.rerun()
        else:
            session.recognition_response = "no_recognition" # No history found
            st.session_state.verification_stage = "email_entry"
            session_manager.db.save_session(session) # Persist recognition response
            st.rerun() # Rerun to display email_entry form

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
                    st.session_state.verification_stage = "email_entry" # Go back to manual entry on failure
            st.rerun()
        else:
            st.error("Error: No recognized email found to send the code. Please enter your email manually.")
            st.session_state.verification_stage = "email_entry"
            st.rerun()

    if st.session_state.verification_stage == 'email_entry':
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**Please enter your email address to receive a verification code:**")
            # Pre-fill with session.email if it exists, or stored verification_email
            initial_email_value = st.session_state.get('verification_email', session.email or "")
            current_email_input = st.text_input("Email Address", placeholder="your@email.com", value=initial_email_value, key="manual_email_input")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if current_email_input:
                    # Update session.email if different
                    if session.email and current_email_input != session.email:
                        session.email_switches_count += 1
                        session.email = current_email_input
                        session_manager.db.save_session(session) # Save updated email to session
                        
                    result = session_manager.handle_guest_email_verification(session, current_email_input)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = current_email_input # Store for code entry
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address to receive the code.")
    
    if st.session_state.verification_stage == 'code_entry':
        # Ensure email is available for display and resend logic
        verification_email = st.session_state.get('verification_email', session.email)
        if not verification_email:
            st.error("No email address set for verification. Please go back and enter your email.")
            st.session_state.verification_stage = "email_entry"
            st.rerun()
            return
            
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
                # No need to check verification_email again, it's checked above
                with st.spinner("Resending code..."):
                    verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                    if verification_sent:
                        st.success("Verification code resent successfully!")
                        st.session_state.verification_stage = "code_entry" # Stay on this stage
                    else:
                        st.error("Failed to resend code. Please try again later.")
                st.rerun() # Rerun to clear spinner and update UI

            if submit_code:
                if code:
                    with st.spinner("Verifying code..."):
                        result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        # Clear verification-related session state variables
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        time.sleep(1) # Small delay for user to see success
                        st.rerun() # Rerun to exit verification dialog and show chat
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code you received.")

def render_chat_interface(session_manager: 'SessionManager', session: UserSession):
    """Renders the main chat interface with redirect-based fingerprinting."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")
    
    # Handle fingerprinting - render component if needed
    # A temporary fingerprint starts with 'temp_py_' or 'emergency_fp_'
    # A successful fingerprint will not start with these prefixes.
    fingerprint_needed = (
        not session.fingerprint_id or 
        session.fingerprint_id.startswith(("temp_py_", "emergency_fp_"))
    )
    
    if fingerprint_needed:
        st.info("🔄 Identifying your device... Please wait.")
        # Render fingerprinting component - this will cause a redirect when complete
        session_manager.fingerprinting.render_fingerprint_component(session.session_id)
        # Exit current run immediately. The redirect will cause a fresh run.
        return 

    # Check limits BEFORE displaying chat history or input
    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed']:
        # If the reason is guest_limit, render the email verification dialog
        if limit_check.get('reason') == 'guest_limit':
            render_email_verification_dialog(session_manager, session)
            return # Stop rendering chat interface
        else:
            # For other ban types (daily_limit, total_limit, banned), just display the message
            # The message is already displayed by `session_manager.get_session()` at app start
            # or by `get_ai_response`. Here, we just block further interaction.
            return # Stop rendering chat interface

    # Display chat messages (if allowed to proceed)
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            
            if msg.get("role") == "assistant":
                if "source" in msg and msg["source"]:
                    st.caption(f"Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"):
                    indicators.append("🧠 Knowledge Base")
                if msg.get("used_search"):
                    indicators.append("🌐 Web Search")
                
                if indicators:
                    st.caption(f"Enhanced with: {', '.join(indicators)}")

    # Chat input area
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                          disabled=session.ban_status != BanStatus.NONE) # Disable if banned
    
    if prompt:
        # Display user's message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response and update chat
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue using FiFi AI.")
                        st.session_state.verification_stage = 'email_entry'
                        st.rerun() # Rerun to display email verification dialog
                    elif response.get('banned'):
                        st.error(response.get("content", 'Access restricted.'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                        st.rerun() # Rerun to display ban message and block input
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Your access has been temporarily restricted.")
                        st.error(f"Penalty duration: {response.get('penalty_hours', 0)} hours.")
                        st.rerun() # Rerun to display evasion message and block input
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
        
        # Rerun to update the chat history display after processing
        st.rerun()

# =============================================================================
# INITIALIZATION & MAIN FUNCTIONS - CLEAN VERSION
# =============================================================================

def ensure_initialization():
    """Clean initialization of all managers and stores them in session state."""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence...")
        
        try:
            with st.spinner("🔄 Initializing FiFi AI Assistant..."):
                config = Config()
                pdf_exporter = PDFExporter()
                
                # Initialize DatabaseManager first
                try:
                    db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
                    st.session_state.db_manager = db_manager # Store db_manager directly
                except Exception as db_e:
                    logger.critical(f"Database manager initialization failed: {db_e}", exc_info=True)
                    # Create a dummy/fallback DB manager for minimal functionality
                    st.session_state.db_manager = type('FallbackDB', (object,), {
                        'db_type': 'memory',
                        'local_sessions': {},
                        'save_session': lambda s, session: s.local_sessions.update({session.session_id: copy.deepcopy(session)}),
                        'load_session': lambda s, session_id: copy.deepcopy(s.local_sessions.get(session_id)),
                        'find_sessions_by_fingerprint': lambda s, fp_id: [copy.deepcopy(x) for x in s.local_sessions.values() if x.fingerprint_id == fp_id],
                        'find_sessions_by_email': lambda s, email: [copy.deepcopy(x) for x in s.local_sessions.values() if x.email == email]
                    })()
                    st.warning("⚠️ Database connection failed. Using temporary in-memory storage. Data will not persist across runs.")
                
                # Initialize other managers, potentially with fallbacks if dependencies fail
                try:
                    zoho_manager = ZohoCRMManager(config, pdf_exporter)
                except Exception as e:
                    logger.error(f"Zoho manager initialization failed: {e}", exc_info=True)
                    zoho_manager = type('FallbackZoho', (object,), {
                        'config': config,
                        'save_chat_transcript_sync': lambda s, session, reason: False # Always fail gracefully
                    })()
                
                try:
                    ai_system = EnhancedAI(config)
                except Exception as e:
                    logger.error(f"AI system initialization failed: {e}", exc_info=True)
                    ai_system = type('FallbackAI', (object,), {
                        'openai_client': None,
                        'get_response': lambda s, prompt, history=None: {
                            "content": "AI system temporarily unavailable. Please try again later.",
                            "success": False,
                            "source": "AI Error"
                        }
                    })()
                
                rate_limiter = RateLimiter() # This is in-memory, no external dependencies
                fingerprinting_manager = FingerprintingManager() # This is primarily JS/local, no external API dependencies
                
                try:
                    email_verification_manager = EmailVerificationManager(config)
                    # If Supabase client failed inside EmailVerificationManager, it sets self.supabase to None
                    # We can check that here, or rely on its internal checks.
                    if not email_verification_manager.supabase:
                         logger.warning("Email verification will function as a dummy as Supabase is not ready.")
                except Exception as e:
                    logger.error(f"Email verification manager initialization failed: {e}", exc_info=True)
                    email_verification_manager = type('DummyEmail', (object,), {
                        'send_verification_code': lambda s, email: (False, st.error("Email verification service is unavailable.")),
                        'verify_code': lambda s, email, code: (False, st.error("Email verification service is unavailable."))
                    })()
                
                question_limit_manager = QuestionLimitManager() # This is purely logic, no external dependencies
                
                # Finally, initialize the main SessionManager with all its dependencies
                st.session_state.session_manager = SessionManager(
                    config, st.session_state.db_manager, zoho_manager, ai_system, 
                    rate_limiter, fingerprinting_manager, email_verification_manager, 
                    question_limit_manager
                )
                
                # Store other frequently used managers or helpers in session state for easy access
                st.session_state.pdf_exporter = pdf_exporter
                st.session_state.error_handler = error_handler
                # No need to store fingerprinting_manager, email_verification_manager, question_limit_manager separately
                # as they are accessed via st.session_state.session_manager.
                
                st.session_state.initialized = True
                logger.info("✅ Application initialized successfully.")
                return True
            
        except Exception as e:
            st.error("💥 Critical initialization error occurred. Please refresh or reset the app.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Critical initialization failure: {e}", exc_info=True)
            
            st.session_state.initialized = False
            return False
    
    return True # Already initialized

def main():
    """Main application entry point."""
    try:
        st.set_page_config(
            page_title="FiFi AI Assistant", 
            page_icon="🤖", 
            layout="wide",
            initial_sidebar_state="expanded" # Keep sidebar expanded by default
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {e}")

    # Simple reset button (visible only if not in the midst of a redirect or heavy processing)
    # The 'Reset App' button should clear all Streamlit session state and rerun.
    if st.button("🔄 Reset Application", help="Force reset the app if it gets stuck or for a fresh start."):
        st.session_state.clear()
        st.rerun()

    # Step 1: Initialize all managers and components
    try:
        init_success = ensure_initialization()
        if not init_success:
            # If initialization failed, display error and stop further rendering
            st.error("⚠️ Application failed to initialize properly. Please try resetting.")
            return
    except Exception as init_error:
        # Catch any unexpected errors during ensure_initialization
        st.error(f"⚠️ An unexpected error occurred during startup: {str(init_error)}")
        st.info("Please try clicking 'Reset Application' above or refresh the page.")
        logger.critical(f"Unhandled error during ensure_initialization: {init_error}", exc_info=True)
        return

    # Step 2: Handle fingerprint data from URL query parameters.
    # This MUST happen BEFORE fetching the session, as it might update the session.
    try:
        handle_fingerprint_requests_from_query()
    except Exception as e:
        logger.error(f"Error handling fingerprint query parameters: {e}", exc_info=True)
        st.error("⚠️ An issue occurred while processing device information. Please try refreshing.")


    # Step 3: Get the current user session. This will create one if it doesn't exist or load from DB.
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Critical: Session Manager not available after initialization. Please 'Reset Application'.")
        return # Cannot proceed without session_manager

    session = session_manager.get_session()
    if not session or not session.active:
        # If get_session fails or returns an inactive session, reset page and rerun
        logger.warning(f"Session not found or inactive. Resetting page to welcome.")
        st.session_state['page'] = None
        st.rerun()
        return

    # Step 4: Route to the appropriate page (Welcome or Chat)
    current_page = st.session_state.get('page')
    
    try:
        if current_page != "chat":
            render_welcome_page(session_manager)
        else:
            # Ensure session is valid for chat interface rendering
            if session and session.active:
                render_sidebar(session_manager, session, st.session_state.pdf_exporter)
                render_chat_interface(session_manager, session)
            else:
                # Should not happen often if get_session handles it, but as a fallback
                logger.warning(f"Session {session.session_id[:8]} became inactive unexpectedly. Redirecting to welcome.")
                st.session_state['page'] = None
                st.rerun()
                
    except Exception as page_error:
        logger.critical(f"CRITICAL: Page rendering or routing error: {page_error}", exc_info=True)
        st.error("💥 A critical error occurred during page display. Please 'Reset Application'.")
        # Attempt to reset state and redirect to welcome as a recovery
        st.session_state.clear()
        st.rerun()


# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
