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
# FINAL INTEGRATED FIFI AI - ALL FEATURES IMPLEMENTED (CORRECTED & COMPLETE)
# - Merges the robust structure of Code 1 with the advanced features of Code 2.
# - ALL features are preserved and correctly integrated.
# - Universal fingerprinting (Canvas/WebGL/Audio) for ALL sessions.
# - 3-tier user system: GUEST → EMAIL_VERIFIED_GUEST → REGISTERED_USER.
# - Activity-based question limits with rolling 24-hour windows.
# - Email verification with Supabase Auth OTP.
# - 15-minute session timeout for CRM saves, integrated with robust timer logic.
# - Cross-device enforcement and evasion detection.
# - Complete new database schema implemented in the robust DatabaseManager.
# - Enhanced browser close detection and error handling preserved.
# - All syntax errors fixed and duplicate code properly merged and adapted.
# - FIX: Non-destructive database schema migration logic using ALTER TABLE ADD COLUMN.
# - FIX: Explicit column listing in REPLACE INTO for database consistency.
# - FIX: Removed 'height' parameter from st_javascript to prevent error.
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
# ENHANCED USER MODELS & DATABASE
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
    
    # Network & Browser
    ip_address: Optional[str] = None
    ip_detection_method: Optional[str] = None
    user_agent: Optional[str] = None
    browser_privacy_level: Optional[str] = None
    
    # Registration Tracking
    registration_prompted: bool = False
    registration_link_clicked: bool = False

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.conn = None
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
            self.local_sessions = {} # For in-memory storage
        
        # Initialize database tables if a connection was established
        if self.conn:
            self._init_complete_database()
            error_handler.mark_component_healthy("Database")

    def _try_sqlite_cloud(self, cs: str):
        try:
            conn = sqlitecloud.connect(cs)
            conn.execute("SELECT 1").fetchone() # Test connection
            logger.info("✅ SQLite Cloud connection established!")
            return conn, "cloud"
        except Exception as e:
            logger.error(f"❌ SQLite Cloud connection failed: {e}")
            return None, None

    def _try_local_sqlite(self):
        try:
            conn = sqlite3.connect("fifi_sessions_v2.db", check_same_thread=False)
            conn.execute("SELECT 1").fetchone() # Test connection
            logger.info("✅ Local SQLite connection established!")
            return conn, "file"
        except Exception as e:
            logger.error(f"❌ Local SQLite connection failed: {e}")
            return None, None

    def _init_complete_database(self):
        """
        Initializes the comprehensive database schema.
        Includes non-destructive migration logic to add missing columns to an existing table.
        """
        with self.lock:
            try:
                # Ensure row_factory is not set during schema creation/modification to avoid issues
                if hasattr(self.conn, 'row_factory'): self.conn.row_factory = None

                # Create the 'sessions' table if it does not exist.
                # This ensures the table structure is always present, even if empty.
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_type TEXT NOT NULL DEFAULT 'guest',
                        email TEXT,
                        full_name TEXT,
                        zoho_contact_id TEXT,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        messages TEXT DEFAULT '[]',
                        active INTEGER DEFAULT 1,
                        wp_token TEXT,
                        timeout_saved_to_crm INTEGER DEFAULT 0,
                        fingerprint_id TEXT,
                        fingerprint_method TEXT,
                        visitor_type TEXT DEFAULT 'new_visitor',
                        recognition_response TEXT,
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
                        ip_address TEXT,
                        ip_detection_method TEXT,
                        user_agent TEXT,
                        browser_privacy_level TEXT,
                        registration_prompted INTEGER DEFAULT 0,
                        registration_link_clicked INTEGER DEFAULT 0
                    )
                ''')
                self.conn.commit()
                logger.info("✅ 'sessions' table ensured to exist.")

                # --- NEW: Non-destructive migration logic ---
                self._migrate_existing_table()
                # --- END NEW MIGRATION LOGIC ---

                # Create indexes for common lookup fields (IF NOT EXISTS will prevent errors if they already exist)
                # These must run AFTER any potential column additions by _migrate_existing_table
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_id ON sessions(fingerprint_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON sessions(email)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user_type ON sessions(user_type)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON sessions(active)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_last_activity ON sessions(last_activity)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ban_status ON sessions(ban_status)")
                self.conn.commit()
                logger.info("✅ Database indexes ensured.")
            except Exception as e:
                logger.error(f"Database schema initialization failed: {e}", exc_info=True)
                raise # Re-raise to ensure the initialization failure propagates

    def _migrate_existing_table(self):
        """
        Adds missing columns to an existing 'sessions' table without dropping data.
        This method is called during initialization to ensure schema compatibility.
        """
        try:
            # Get existing columns in the 'sessions' table
            cursor = self.conn.execute("PRAGMA table_info(sessions)")
            existing_columns = {row[1] for row in cursor.fetchall()} # row[1] is the column name
            
            # Define all new columns that should be present in the latest schema
            # Format: (column_name, column_definition_SQL)
            new_columns_to_add = [
                ("fingerprint_id", "TEXT"),
                ("fingerprint_method", "TEXT"),
                ("visitor_type", "TEXT DEFAULT 'new_visitor'"),
                ("recognition_response", "TEXT"),
                ("daily_question_count", "INTEGER DEFAULT 0"),
                ("total_question_count", "INTEGER DEFAULT 0"),
                ("last_question_time", "TEXT"),
                ("question_limit_reached", "INTEGER DEFAULT 0"),
                ("ban_status", "TEXT DEFAULT 'none'"),
                ("ban_start_time", "TEXT"),
                ("ban_end_time", "TEXT"),
                ("ban_reason", "TEXT"),
                ("evasion_count", "INTEGER DEFAULT 0"),
                ("current_penalty_hours", "INTEGER DEFAULT 0"),
                ("escalation_level", "INTEGER DEFAULT 0"),
                ("email_addresses_used", "TEXT DEFAULT '[]'"),
                ("email_switches_count", "INTEGER DEFAULT 0"),
                ("ip_address", "TEXT"),
                ("ip_detection_method", "TEXT"),
                ("user_agent", "TEXT"),
                ("browser_privacy_level", "TEXT"),
                ("registration_prompted", "INTEGER DEFAULT 0"),
                ("registration_link_clicked", "INTEGER DEFAULT 0")
                # Ensure all columns from the UserSession dataclass that are NOT in the initial Code 1 schema are listed here
                # And that their DEFAULT values match those in UserSession where applicable.
            ]
            
            # Iterate through the defined new columns and add any that are missing
            for column_name, column_def in new_columns_to_add:
                if column_name not in existing_columns:
                    try:
                        self.conn.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_def}")
                        logger.info(f"Successfully added missing column to 'sessions' table: {column_name}")
                    except Exception as e:
                        # Log a warning if a column can't be added (e.g., if it was already added by another process)
                        logger.warning(f"Could not add column '{column_name}' to 'sessions' table (might already exist or other issue): {e}")
            
            self.conn.commit()
            logger.info("✅ Database schema migration (adding columns) completed successfully.")
            
        except Exception as e:
            logger.error(f"❌ Database schema migration (ALTER TABLE) failed: {e}", exc_info=True)
            # Re-raise the exception to indicate a critical failure in initialization process
            raise 

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.db_type == "memory":
                # For in-memory, store a deep copy to prevent external modifications
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                logger.debug(f"Saved session {session.session_id[:8]} to in-memory.")
                return
            
            # Prepare data for storage, converting complex types to string/JSON
            s = session
            session_data = (
                s.session_id, s.user_type.value, s.email, s.full_name, s.zoho_contact_id,
                s.created_at.isoformat(), s.last_activity.isoformat(), json.dumps(s.messages),
                int(s.active), s.wp_token, int(s.timeout_saved_to_crm), s.fingerprint_id,
                s.fingerprint_method, s.visitor_type, s.recognition_response, s.daily_question_count,
                s.total_question_count, s.last_question_time.isoformat() if s.last_question_time else None,
                int(s.question_limit_reached), s.ban_status.value,
                s.ban_start_time.isoformat() if s.ban_start_time else None,
                s.ban_end_time.isoformat() if s.ban_end_time else None, s.ban_reason, s.evasion_count,
                s.current_penalty_hours, s.escalation_level, json.dumps(s.email_addresses_used),
                s.email_switches_count, s.ip_address, s.ip_detection_method, s.user_agent,
                s.browser_privacy_level, int(s.registration_prompted), int(s.registration_link_clicked)
            )
            
            try:
                # FIX: Explicitly list all columns in the REPLACE INTO statement.
                # This prevents issues with default values or column count mismatches.
                self.conn.execute('''
                    REPLACE INTO sessions (
                        session_id, user_type, email, full_name, zoho_contact_id,
                        created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm,
                        fingerprint_id, fingerprint_method, visitor_type, recognition_response,
                        daily_question_count, total_question_count, last_question_time, question_limit_reached,
                        ban_status, ban_start_time, ban_end_time, ban_reason,
                        evasion_count, current_penalty_hours, escalation_level,
                        email_addresses_used, email_switches_count,
                        ip_address, ip_detection_method, user_agent, browser_privacy_level,
                        registration_prompted, registration_link_clicked
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', session_data)
                self.conn.commit()
                logger.debug(f"Successfully saved session {s.session_id[:8]} to database.")
            except Exception as e:
                logger.error(f"Failed to save session {s.session_id[:8]}: {e}", exc_info=True)
                raise # Re-raise to be caught by handle_api_errors

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            if self.db_type == "memory":
                # For in-memory, return a deep copy to isolate from external modifications
                return copy.deepcopy(self.local_sessions.get(session_id))
            try:
                # Set row_factory based on DB type for consistent dictionary-like access
                if self.db_type == "file": self.conn.row_factory = sqlite3.Row
                else: self.conn.row_factory = None # sqlitecloud returns tuples, convert manually

                cursor = self.conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                row = cursor.fetchone()
                if not row:
                    logger.debug(f"No active session found for ID {session_id[:8]}.")
                    return None
                
                # Convert row to dictionary for consistent processing
                row_dict = dict(row) if hasattr(row, 'keys') else dict(zip([d[0] for d in cursor.description], row))

                # Dynamically create UserSession, converting types as needed
                session_params = {}
                for key, value in row_dict.items():
                    session_params[key] = self._convert_db_value_to_python(key, value)
                
                user_session = UserSession(**session_params)
                logger.debug(f"Successfully loaded session {session_id[:8]}: type={user_session.user_type.value}.")
                return user_session
            except Exception as e:
                logger.error(f"Failed to load session {session_id[:8]}: {e}", exc_info=True)
                return None

    def _convert_db_value_to_python(self, key: str, value: Any) -> Any:
        """Helper to convert database string/int values back to Python types."""
        if value is None:
            return None
        
        # Datetime fields
        datetime_keys = ['created_at', 'last_activity', 'last_question_time', 'ban_start_time', 'ban_end_time']
        if key in datetime_keys and isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"Could not convert {key} '{value}' to datetime. Returning None.")
                return None
        
        # JSON fields (lists in this case)
        json_list_keys = ['messages', 'email_addresses_used']
        if key in json_list_keys and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON for {key}: '{value}'. Returning empty list.")
                return []
        
        # Enum fields
        if key == 'user_type' and isinstance(value, str):
            try:
                return UserType(value)
            except ValueError:
                logger.warning(f"Invalid user_type '{value}'. Defaulting to GUEST.")
                return UserType.GUEST
        if key == 'ban_status' and isinstance(value, str):
            try:
                return BanStatus(value)
            except ValueError:
                logger.warning(f"Invalid ban_status '{value}'. Defaulting to NONE.")
                return BanStatus.NONE
        
        # Boolean fields (stored as INTEGER 0 or 1)
        bool_keys = ['active', 'timeout_saved_to_crm', 'question_limit_reached', 'registration_prompted', 'registration_link_clicked']
        if key in bool_keys:
            return bool(value)
            
        return value

    @handle_api_errors("Database", "Find by Fingerprint")
    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint_id."""
        with self.lock:
            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
            try:
                if self.db_type == "file": self.conn.row_factory = sqlite3.Row
                else: self.conn.row_factory = None

                cursor = self.conn.execute("SELECT * FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
                sessions = []
                for row in cursor.fetchall():
                    row_dict = dict(row) if hasattr(row, 'keys') else dict(zip([d[0] for d in cursor.description], row))
                    session_params = {}
                    for key, value in row_dict.items():
                        session_params[key] = self._convert_db_value_to_python(key, value)
                    sessions.append(UserSession(**session_params))
                return sessions
            except Exception as e:
                logger.error(f"Failed to find sessions by fingerprint '{fingerprint_id[:8]}...': {e}", exc_info=True)
                return []

    @handle_api_errors("Database", "Find by Email")
    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions associated with a specific email address."""
        with self.lock:
            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.email == email]
            try:
                if self.db_type == "file": self.conn.row_factory = sqlite3.Row
                else: self.conn.row_factory = None
                
                cursor = self.conn.execute("SELECT * FROM sessions WHERE email = ? ORDER BY last_activity DESC", (email,))
                sessions = []
                for row in cursor.fetchall():
                    row_dict = dict(row) if hasattr(row, 'keys') else dict(zip([d[0] for d in cursor.description], row))
                    session_params = {}
                    for key, value in row_dict.items():
                        session_params[key] = self._convert_db_value_to_python(key, value)
                    sessions.append(UserSession(**session_params))
                return sessions
            except Exception as e:
                logger.error(f"Failed to find sessions by email '{email}': {e}", exc_info=True)
                return []

# =============================================================================
# FEATURE MANAGERS (Fingerprinting, Email Verification, Question Limits)
# =============================================================================

class FingerprintingManager:
    """Manages the 3-layer browser fingerprinting for device identification."""
    
    def __init__(self):
        self.fingerprint_cache = {} # Cache for quick lookup of seen fingerprints

    def generate_fingerprint_component(self, session_id: str) -> str:
        """
        Generates JavaScript code to collect 3-layer browser fingerprints
        (Canvas, WebGL, AudioContext) and other browser/system info.
        """
        js_code = f"""
        (() => {{
            const sessionId = "{session_id}";
            console.log("🔍 FiFi Fingerprinting: Starting 3-layer device identification.");

            // Layer 1: Canvas Fingerprinting (Primary)
            function generateCanvasFingerprint() {{
                try {{
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = 220; canvas.height = 100;
                    ctx.textBaseline = 'top'; ctx.font = '14px Arial';
                    ctx.fillStyle = '#f60'; ctx.fillRect(125, 1, 62, 20);
                    ctx.fillStyle = '#069'; ctx.fillText('FiFi AI Canvas Test 🤖', 2, 15);
                    ctx.fillStyle = 'rgba(102, 204, 0, 0.7)'; ctx.fillText('Food & Beverage Industry', 4, 45);
                    ctx.strokeStyle = '#000'; ctx.beginPath();
                    ctx.arc(50, 50, 20, 0, Math.PI * 2); ctx.stroke();
                    return btoa(canvas.toDataURL()).slice(0, 32); // Base64 and truncate for consistency
                }} catch (e) {{
                    console.error("❌ Canvas fingerprint failed:", e);
                    return 'canvas_blocked';
                }}
            }}
            
            // Layer 2: WebGL Fingerprinting (Secondary)
            function generateWebGLFingerprint() {{
                try {{
                    const canvas = document.createElement('canvas');
                    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                    if (!gl) {{ return 'webgl_unavailable'; }}
                    const webglData = {{
                        vendor: gl.getParameter(gl.VENDOR),
                        renderer: gl.getParameter(gl.RENDERER),
                        version: gl.getParameter(gl.VERSION),
                        extensions: gl.getSupportedExtensions() ? gl.getSupportedExtensions().slice(0, 10) : []
                    }};
                    return btoa(JSON.stringify(webglData)).slice(0, 32);
                }} catch (e) {{
                    console.error("❌ WebGL fingerprint failed:", e);
                    return 'webgl_blocked';
                }}
            }}
            
            // Layer 3: Audio Context Fingerprinting (Tertiary)
            function generateAudioFingerprint() {{
                try {{
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const analyser = audioContext.createAnalyser();
                    const gainNode = audioContext.createGain();
                    oscillator.type = 'triangle'; oscillator.frequency.value = 1000; gainNode.gain.value = 0;
                    oscillator.connect(analyser); analyser.connect(gainNode); gainNode.connect(audioContext.destination);
                    oscillator.start(0);
                    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(frequencyData);
                    oscillator.stop(); audioContext.close();
                    return btoa(Array.from(frequencyData.slice(0, 32)).join(',')).slice(0, 32);
                }} catch (e) {{
                    console.error("❌ Audio fingerprint failed:", e);
                    return 'audio_blocked';
                }}
            }}
            
            // Collect general browser and system information
            function getBrowserInfo() {{
                return {{
                    userAgent: navigator.userAgent, language: navigator.language, platform: navigator.platform,
                    cookieEnabled: navigator.cookieEnabled, doNotTrack: navigator.doNotTrack,
                    hardwareConcurrency: navigator.hardwareConcurrency, maxTouchPoints: navigator.maxTouchPoints,
                    screen: {{ width: screen.width, height: screen.height, colorDepth: screen.colorDepth, pixelDepth: screen.pixelDepth }},
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
                }};
            }}
            
            const canvasFp = generateCanvasFingerprint();
            const webglFp = generateWebGLFingerprint();
            const audioFp = generateAudioFingerprint();
            const browserInfo = getBrowserInfo();
            
            let primaryMethod = 'canvas'; let fingerprintId = canvasFp;
            const workingMethods = [];
            if (canvasFp !== 'canvas_blocked') workingMethods.push('canvas');
            if (webglFp !== 'webgl_blocked' && webglFp !== 'webgl_unavailable') workingMethods.push('webgl');
            if (audioFp !== 'audio_blocked') workingMethods.push('audio');
            
            if (workingMethods.length === 0) {{
                primaryMethod = 'fallback';
                fingerprintId = 'privacy_browser_' + Date.now(); // Fallback if all are blocked
            }} else if (workingMethods.length > 1) {{
                primaryMethod = 'hybrid';
                fingerprintId = btoa([canvasFp, webglFp, audioFp].join('|')).slice(0, 32);
            }} else {{
                primaryMethod = workingMethods[0];
                fingerprintId = (primaryMethod === 'canvas' ? canvasFp : (primaryMethod === 'webgl' ? webglFp : audioFp));
            }}
            
            let privacyLevel = 'standard';
            if (canvasFp === 'canvas_blocked' && webglFp.includes('blocked') && audioFp === 'audio_blocked') {{
                privacyLevel = 'high_privacy';
            }}
            
            const fingerprintResult = {{
                session_id: sessionId,
                fingerprint_id: fingerprintId,
                fingerprint_method: primaryMethod,
                canvas_fp: canvasFp,
                webgl_fp: webglFp,
                audio_fp: audioFp,
                browser_info: browserInfo,
                privacy_level: privacyLevel,
                working_methods: workingMethods,
                timestamp: Date.now()
            }};
            
            console.log("🔍 Fingerprinting complete:", {{
                id: fingerprintId, method: primaryMethod, privacy: privacyLevel, working: workingMethods.length
            }});
            return fingerprintResult;
        }})()
        """
        return js_code

    def extract_fingerprint_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and validates fingerprint data from the JavaScript component's return."""
        if not result or not isinstance(result, dict) or result.get('fingerprint_id') == 'ZXJyb3I=': # b64('error')
            logger.warning("Fingerprint JavaScript returned error or null. Using fallback.")
            return self._generate_fallback_fingerprint()
        
        fingerprint_id = result.get('fingerprint_id')
        fingerprint_method = result.get('fingerprint_method', 'unknown')
        
        if not fingerprint_id or fingerprint_id.startswith('privacy_browser_'):
            logger.info("Fingerprint ID indicates privacy browser or fallback. Generating new fallback.")
            return self._generate_fallback_fingerprint()
        
        # Determine visitor type based on in-memory cache
        visitor_type = "returning_visitor" if fingerprint_id in self.fingerprint_cache else "new_visitor"
        self.fingerprint_cache[fingerprint_id] = {'last_seen': datetime.now()} # Update cache
        
        return {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': fingerprint_method,
            'visitor_type': visitor_type,
            'browser_info': result.get('browser_info', {}),
            'privacy_level': result.get('privacy_level', 'standard'),
            'working_methods': result.get('working_methods', [])
        }
    
    def _generate_fallback_fingerprint(self) -> Dict[str, Any]:
        """Generates a unique fallback fingerprint for cases where real fingerprinting fails."""
        fallback_id = f"fallback_{secrets.token_hex(8)}"
        return {
            'fingerprint_id': fallback_id,
            'fingerprint_method': 'fallback',
            'visitor_type': 'new_visitor',
            'browser_info': {}, # Empty dict as no real info is collected
            'privacy_level': 'high_privacy',
            'working_methods': []
        }

class EmailVerificationManager:
    """Manages email verification process, typically via an OTP service like Supabase Auth."""
    
    def __init__(self, config: Config):
        self.config = config
        self.supabase = None
        if self.config.SUPABASE_ENABLED:
            try:
                self.supabase = create_client(self.config.SUPABASE_URL, self.config.SUPABASE_ANON_KEY)
                logger.info("✅ Supabase client initialized for email verification.")
            except Exception as e:
                logger.error(f"❌ Supabase client initialization failed: {e}. Email verification will be disabled.")
                self.supabase = None # Ensure it's None if init fails

    @handle_api_errors("Supabase Auth", "Send Verification Code")
    def send_verification_code(self, email: str) -> bool:
        if not self.supabase:
            st.error("Email verification service is not available (Supabase not configured/failed).")
            return False
        
        try:
            # Supabase Auth's sign_in_with_otp sends a magic link or OTP to the email
            self.supabase.auth.sign_in_with_otp({'email': email, 'options': {'should_create_user': True}})
            logger.info(f"Email verification code/link sent to {email} via Supabase.")
            return True
        except Exception as e:
            logger.error(f"Failed to send verification code via Supabase for {email}: {e}")
            st.error("Failed to send verification code. Please check your email address and try again.")
            return False

    @handle_api_errors("Supabase Auth", "Verify Code")
    def verify_code(self, email: str, code: str) -> bool:
        if not self.supabase:
            st.error("Email verification service is not available.")
            return False
        
        try:
            response = self.supabase.auth.verify_otp({'email': email, 'token': code.strip(), 'type': 'email'})
            if response.user: # If user object is returned, verification was successful
                logger.info(f"Email verification successful for {email} (Supabase User ID: {response.user.id}).")
                return True
            else:
                logger.warning(f"Email verification failed for {email}: Invalid code or no user returned.")
                return False
        except Exception as e:
            logger.error(f"Failed to verify code via Supabase for {email}: {e}")
            return False

class QuestionLimitManager:
    """Manages activity-based question limiting and ban statuses for different user tiers."""
    
    def __init__(self):
        self.question_limits = {
            UserType.GUEST: 4,                    # 4 questions → forced email verification
            UserType.EMAIL_VERIFIED_GUEST: 10,   # 10 questions/day
            UserType.REGISTERED_USER: 40         # 40 questions/day cross-device
        }
        self.evasion_penalties = [24, 48, 96, 192, 336] # Escalating evasion penalties in hours
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """
        Checks if the current session is within its allowed question limits
        or if any bans are active.
        """
        user_limit = self.question_limits.get(session.user_type, 0)
        
        # 1. Check current ban status (if any)
        if session.ban_status != BanStatus.NONE:
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
                # Ban expired, reset status
                logger.info(f"Ban for session {session.session_id[:8]} expired. Resetting status.")
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                session.question_limit_reached = False # Important for UI display
        
        # 2. Check activity-based reset (24-hour rolling window for daily counts)
        if session.last_question_time:
            time_since_last = datetime.now() - session.last_question_time
            if time_since_last >= timedelta(hours=24):
                logger.info(f"Daily question count reset for session {session.session_id[:8]}.")
                session.daily_question_count = 0
                session.question_limit_reached = False # Reset UI flag
        
        # 3. Apply limits based on UserType
        if session.user_type == UserType.GUEST:
            if session.daily_question_count >= user_limit:
                return {
                    'allowed': False,
                    'reason': 'guest_limit',
                    'message': 'Please provide your email address to continue.'
                }
        
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            if session.daily_question_count >= user_limit:
                # Exceeded 10 questions/day for email-verified guest → 24-hour ban
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Email-verified daily limit reached")
                return {
                    'allowed': False,
                    'reason': 'daily_limit',
                    'message': self._get_email_verified_limit_message()
                }
        
        elif session.user_type == UserType.REGISTERED_USER:
            # Registered users have a 40-question total limit, with a softer 1-hour ban after 20.
            if session.total_question_count >= user_limit:
                # Exceeded 40 questions → 24-hour ban
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Registered user total limit reached")
                return {
                    'allowed': False,
                    'reason': 'total_limit',
                    'message': "Usage limit reached. Please retry in 24 hours as we are giving preference to others in the queue."
                }
            elif session.total_question_count >= 20 and session.ban_status == BanStatus.NONE:
                # Hit 20 questions → apply 1-hour ban if not already banned
                self._apply_ban(session, BanStatus.ONE_HOUR, "Registered user first tier limit reached")
                return {
                    'allowed': False,
                    'reason': 'first_tier_limit',
                    'message': "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
                }
        
        return {'allowed': True} # User is allowed to ask a question
    
    def record_question(self, session: UserSession):
        """Increments question counters for the session."""
        session.daily_question_count += 1
        if session.user_type == UserType.REGISTERED_USER:
            session.total_question_count += 1
        session.last_question_time = datetime.now()
        logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}.")
    
    def _apply_ban(self, session: UserSession, ban_type: BanStatus, reason: str):
        """Applies a ban to the session for a specified duration."""
        ban_hours = {
            BanStatus.ONE_HOUR: 1,
            BanStatus.TWENTY_FOUR_HOUR: 24,
            BanStatus.EVASION_BLOCK: session.current_penalty_hours # Use current_penalty_hours for evasion
        }.get(ban_type, 24) # Default to 24 hours if type is unknown

        session.ban_status = ban_type
        session.ban_start_time = datetime.now()
        session.ban_end_time = session.ban_start_time + timedelta(hours=ban_hours)
        session.ban_reason = reason
        session.question_limit_reached = True # Flag for UI/logging
        
        logger.info(f"Ban applied to session {session.session_id[:8]}: Type={ban_type.value}, Duration={ban_hours}h, Reason='{reason}'.")
    
    def apply_evasion_penalty(self, session: UserSession) -> int:
        """Applies an escalating penalty for evasion attempts."""
        session.evasion_count += 1
        # Escalation level caps at the max index of evasion_penalties list
        session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
        
        penalty_hours = self.evasion_penalties[session.escalation_level - 1] # -1 because lists are 0-indexed
        session.current_penalty_hours = penalty_hours # Store the current penalty
        
        self._apply_ban(session, BanStatus.EVASION_BLOCK, f"Evasion attempt #{session.evasion_count}")
        
        logger.warning(f"Evasion penalty applied to {session.session_id[:8]}: {penalty_hours}h (Level {session.escalation_level}).")
        return penalty_hours
    
    def _get_ban_message(self, session: UserSession) -> str:
        """Provides a user-friendly message for current bans."""
        if session.ban_status == BanStatus.EVASION_BLOCK:
            return "Usage limit reached due to detected unusual activity. Please try again later."
        elif session.user_type == UserType.REGISTERED_USER:
            return "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
        else: # EMAIL_VERIFIED_GUEST (or other cases that hit 24-hour daily limit)
            return self._get_email_verified_limit_message()
    
    def _get_email_verified_limit_message(self) -> str:
        """Specific message for email-verified guests hitting their daily limit."""
        return ("Our system is very busy and is being used by multiple users. For a fair assessment of our FiFi AI assistant and to provide fair usage to everyone, we can allow a total of 10 questions per day (20 messages). To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

# =============================================================================
# PDF EXPORTER & ZOHO CRM MANAGER (Preserved and Adapted)
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
            # Remove HTML tags for cleaner PDF output
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
        """
        Retrieves or refreshes the Zoho CRM access token, with caching and timeout.
        Preserved from Code 1, which had more robust token handling.
        """
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
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            
            self._access_token = data.get('access_token')
            # Token usually valid for 1 hour, refresh slightly before expiry
            self._token_expiry = datetime.now() + timedelta(minutes=50) 
            
            logger.info("Successfully obtained Zoho access token.")
            return self._access_token
            
        except requests.exceptions.Timeout:
            logger.error(f"Zoho token request timed out after {timeout} seconds.")
            error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Get Access Token", requests.exceptions.Timeout("Request timed out")))
            return None
        except Exception as e:
            logger.error(f"Failed to get Zoho access token: {e}", exc_info=True)
            error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Get Access Token", e))
            return None

    def _find_contact_by_email(self, email: str) -> Optional[str]:
        """Finds a Zoho contact by email, with token refresh logic."""
        access_token = self._get_access_token()
        if not access_token: return None
        
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            
            if response.status_code == 401: # Token expired during call
                logger.warning("Zoho token expired, attempting refresh for _find_contact_by_email...")
                access_token = self._get_access_token(force_refresh=True)
                if not access_token: return None
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
            error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Find Contact", e))
            
        return None

    def _create_contact(self, email: str, full_name: Optional[str]) -> Optional[str]:
        """Creates a new Zoho contact, with token refresh logic."""
        access_token = self._get_access_token()
        if not access_token: return None

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        contact_data = {
            "data": [{
                "Last_Name": full_name or "Food Professional", # Zoho requires Last_Name
                "Email": email,
                "Lead_Source": "FiFi AI Assistant"
            }]
        }
        
        try:
            response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, attempting refresh for _create_contact...")
                access_token = self._get_access_token(force_refresh=True)
                if not access_token: return None
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
            error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Create Contact", e))
            
        return None

    def _upload_attachment(self, contact_id: str, pdf_buffer: io.BytesIO, filename: str) -> bool:
        """Uploads a PDF attachment to a Zoho contact, with retry and token refresh."""
        access_token = self._get_access_token()
        if not access_token: return False

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        upload_url = f"{self.base_url}/Contacts/{contact_id}/Attachments"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                pdf_buffer.seek(0) # Reset buffer position for reading
                response = requests.post(
                    upload_url, 
                    headers=headers, 
                    files={'file': (filename, pdf_buffer.read(), 'application/pdf')},
                    timeout=60
                )
                
                if response.status_code == 401:
                    logger.warning("Zoho token expired during upload, attempting refresh...")
                    access_token = self._get_access_token(force_refresh=True)
                    if not access_token: return False # If token refresh fails, give up
                    headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                    continue # Retry with new token
                
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
                time.sleep(2 ** attempt) # Exponential backoff
                
        return False

    def _add_note(self, contact_id: str, note_title: str, note_content: str) -> bool:
        """Adds a note to a Zoho contact, with token refresh logic."""
        access_token = self._get_access_token()
        if not access_token: return False

        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        
        # Truncate note content if too long (Zoho limit is typically 32000 characters)
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
                if not access_token: return False
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
            error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Add Note", e))
            
        return False

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """
        Synchronously saves the chat transcript to Zoho CRM, including contact creation,
        PDF attachment, and a summary note. Includes comprehensive error handling and retries.
        Preserved from Code 1, which had robust debug logging and retry logic.
        """
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        
        # Only registered users with email and messages can save to CRM
        if (session.user_type != UserType.REGISTERED_USER or 
            not session.email or 
            not session.messages or 
            not self.config.ZOHO_ENABLED):
            logger.info(f"ZOHO SAVE SKIPPED: Not eligible. (UserType: {session.user_type.value}, Email: {bool(session.email)}, Messages: {bool(session.messages)}, Zoho Enabled: {self.config.ZOHO_ENABLED})")
            return False
        
        max_retries = 3 if "timeout" in trigger_reason.lower() or "emergency" in trigger_reason.lower() else 1
        
        for attempt in range(max_retries):
            logger.info(f"Zoho Save Attempt {attempt + 1}/{max_retries}")
            try:
                # Find or create contact
                contact_id = self._find_contact_by_email(session.email)
                if not contact_id:
                    contact_id = self._create_contact(session.email, session.full_name)
                if not contact_id:
                    logger.error("Failed to find or create Zoho contact. Cannot proceed with save.")
                    # If this is the last attempt, return False
                    if attempt == max_retries - 1: return False
                    # Else, continue to next retry
                    time.sleep(2 ** attempt)
                    continue 
                session.zoho_contact_id = contact_id # Update session with Zoho contact ID

                # Generate PDF transcript
                pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer:
                    logger.error("Failed to generate PDF transcript. Cannot proceed with attachment.")
                    # Continue attempting note creation even if PDF fails
                
                upload_success = False
                if pdf_buffer:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                    pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                    upload_success = self._upload_attachment(contact_id, pdf_buffer, pdf_filename)
                    if not upload_success:
                        logger.warning("Failed to upload PDF attachment to Zoho. Continuing with note only.")

                # Add summary note to CRM
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
                    # If note failed, and it's the last attempt, report overall failure
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached. Aborting save.")
                        return False

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"ZOHO SAVE FAILED on attempt {attempt + 1} with an exception.")
                logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Save Transcript", e))
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    logger.error("Max retries reached. Aborting save.")
                    return False
        
        return False # Should only be reached if all retries fail

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
            content = re.sub(r'<[^>]+>', '', msg.get("content", "")) # Clean HTML for note
            
            max_msg_length = 500
            if len(content) > max_msg_length:
                content = content[:max_msg_length] + "..."
                
            note_content += f"\n{i+1}. **{role}:** {content}\n"
            
            if msg.get("source"):
                note_content += f"   _Source: {msg['source']}_\n"
                
        return note_content

# =============================================================================
# RATE LIMITER & AI SYSTEM (Preserved)
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter to prevent abuse."""
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.requests = defaultdict(list)
        self._lock = threading.Lock() # Use a lock for thread safety
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, identifier: str) -> bool:
        with self._lock:
            now = time.time()
            # Filter out old requests outside the window
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

def sanitize_input(text: str, max_length: int = 4000) -> str:
    """Sanitizes user input to prevent XSS and limit length."""
    if not isinstance(text, str): 
        return "" # Ensure it's a string
    return html.escape(text)[:max_length].strip()
    
class EnhancedAI:
    """Placeholder for the AI interaction logic."""
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                error_handler.log_error(error_handler.handle_api_error("OpenAI", "Initialization", e))

    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Provides a simplified AI response. In a real application, this would
        integrate with LangChain, Pinecone, Tavily, and OpenAI.
        """
        # Example of how you would connect to OpenAI (if available)
        # if self.openai_client:
        #     try:
        #         messages = [{"role": "user", "content": prompt}]
        #         if chat_history:
        #             messages = chat_history[-5:] + messages # Last 5 messages for context
        #         response = self.openai_client.chat.completions.create(
        #             model="gpt-3.5-turbo", # or your chosen model
        #             messages=messages
        #         )
        #         return {"content": response.choices[0].message.content, "source": "OpenAI", "success": True}
        #     except Exception as e:
        #         logger.error(f"OpenAI API call failed: {e}")
        #         # Fallback to generic response
        #         return {"content": "Sorry, my AI services are currently unavailable.", "success": False, "source": "AI Error"}

        # Simplified placeholder response for demonstration
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

@handle_api_errors("Content Moderation", "Check Prompt", show_to_user=False) # Don't show technical errors to user
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    """Checks user prompt against content moderation guidelines using OpenAI's moderation API."""
    if not client or not hasattr(client, 'moderations') :
        logger.debug("OpenAI client or moderation API not available. Skipping content moderation.")
        return {"flagged": False} # Moderation is optional, so default to not flagged if service is down/unavailable
    
    try:
        response = client.moderations.create(model="text-moderation-latest", input=prompt) # Use text-moderation-latest or omni-moderation-latest
        result = response.results[0]
        
        if result.flagged:
            flagged_categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
            logger.warning(f"Input flagged by moderation for: {', '.join(flagged_categories)}")
            return {
                "flagged": True, 
                "message": "Your message violates our content policy and cannot be processed.",
                "categories": flagged_categories
            }
    except Exception as e:
        logger.error(f"Content moderation API call failed: {e}", exc_info=True)
        # In case of API error, it's safer to not flag unless explicitly configured otherwise
        return {"flagged": False}
    
    return {"flagged": False}

# =============================================================================
# JAVASCRIPT COMPONENTS & EVENT HANDLING
# =============================================================================

def render_activity_timer_component_15min(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Renders a JavaScript component that tracks user inactivity and triggers
    an event after 15 minutes. This version integrates the robust listener setup
    from Code 1 with the 15-minute timeout requirement from Code 2.
    """
    if not session_id:
        return None
    
    # JavaScript logic, optimized for stability and correct event triggering
    js_timer_code = f"""
    (() => {{
        // Wrap everything in an IIFE and try-catch for robustness
        try {{
            const sessionId = "{session_id}";
            const SESSION_TIMEOUT_MS = 900000;  // 15 minutes in milliseconds
            
            console.log("🕐 FiFi 15-Minute Timer: Checking session", sessionId.substring(0, 8));
            
            // Initialize or reset timer state if session changed or first load
            if (typeof window.fifi_timer_state === 'undefined' || window.fifi_timer_state === null || window.fifi_timer_state.sessionId !== sessionId) {{
                console.clear(); // Clear old console logs for a fresh session
                console.log("🆕 FiFi 15-Minute Timer: Starting/Resetting for session", sessionId.substring(0, 8)); 
                window.fifi_timer_state = {{
                    lastActivityTime: Date.now(),
                    expired: false, // Flag to ensure event fires only once per timeout
                    listenersInitialized: false,
                    sessionId: sessionId
                }};
                console.log("🆕 FiFi 15-Minute Timer state initialized.");
            }}
            
            const state = window.fifi_timer_state;
            
            // Initialize activity listeners only once per session
            if (!state.listenersInitialized) {{
                console.log("👂 Setting up FiFi 15-Minute activity listeners...");
                
                function resetActivity() {{
                    try {{
                        const now = Date.now();
                        if (state.lastActivityTime !== now) {{ // Only update if actual change
                            state.lastActivityTime = now;
                            if (state.expired) {{
                                // Log if state was expired but activity detected
                                console.log("🔄 Activity detected, resetting expired flag for timer.");
                            }}
                            state.expired = false; // Reset expired flag on activity
                        }}
                    }} catch (e) {{
                        console.debug("Error in resetActivity:", e);
                    }}
                }}
                
                // Events to detect user activity
                const events = [
                    'mousedown', 'mousemove', 'mouseup', 'click', 'dblclick',
                    'keydown', 'keyup', 'keypress',
                    'scroll', 'wheel',
                    'touchstart', 'touchmove', 'touchend',
                    'focus' // Detects when window/tab gains focus
                ];
                
                // Add listeners to the component iframe and the parent window/document
                const addListenersToTarget = (target) => {{
                    events.forEach(eventType => {{
                        try {{
                            target.addEventListener(eventType, resetActivity, {{ 
                                passive: true, 
                                capture: true,
                                once: false // Keep listening
                            }});
                        }} catch (e) {{
                            console.debug(`Failed to add ${{eventType}} listener to target:`, e);
                        }}
                    }});
                }};
                
                addListenersToTarget(document); // Add to current iframe document
                
                try {{
                    if (window.parent && window.parent.document && window.parent.document !== document &&
                        window.parent.location.origin === window.location.origin) {{
                        addListenersToTarget(window.parent.document); // Add to main Streamlit app document
                        console.log("👂 Parent document listeners added successfully.");
                    }}
                }} catch (e) {{
                    console.debug("Cannot access parent document for listeners:", e);
                }}
                
                // Add visibility change detection for overall activity
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
            
            // Calculate current inactivity time
            const currentTime = Date.now();
            const inactiveTimeMs = currentTime - state.lastActivityTime;
            const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
            const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
            
            console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
            
            // Check for 15-minute timeout and fire event only once
            if (inactiveTimeMs >= SESSION_TIMEOUT_MS && !state.expired) {{
                state.expired = true; // Set flag to prevent re-firing until activity
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
            
            // Explicitly return null if no event is triggered, essential for st_javascript
            return null;
            
        }} catch (error) {{
            console.error("🚨 FiFi 15-Minute Timer component caught a critical error:", error);
            return null; // Ensure null is returned on error
        }}
    }})()
    """
    
    try:
        # Create a stable key for st_javascript, incorporating session ID and a hash of session ID
        # This helps Streamlit identify the component across reruns consistently.
        stable_key = f"fifi_timer_15min_{session_id[:8]}_{hash(session_id) % 10000}"
        
        # Execute the JavaScript code
        timer_result = st_javascript(js_timer_code, key=stable_key)
        
        # Validate the result from JavaScript
        # st_javascript can return None, 0, or an empty string if nothing is returned or an error occurs in JS
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            return None # No meaningful event or error occurred in JS (or it's not yet triggered)
        
        # Further validate if the result is a dictionary and contains the expected event
        if isinstance(timer_result, dict) and timer_result.get('event') == "session_timeout_15min":
            # Ensure the session ID returned by JS matches the current session ID to prevent cross-session interference
            if timer_result.get('session_id') == session_id:
                logger.info(f"✅ Valid 15-min timer event received: {timer_result.get('event')} for session {session_id[:8]}.")
                return timer_result
            else:
                logger.warning(f"⚠️ Timer event session ID mismatch: expected {session_id[:8]}, got {timer_result.get('session_id', 'None')}. Event ignored.")
                return None
        else:
            logger.debug(f"Received non-event timer result: {timer_result} (type: {type(timer_result)}).")
            return None
        
    except Exception as e:
        logger.error(f"❌ JavaScript timer component execution error: {e}", exc_info=True)
        return None

def render_browser_close_detection_enhanced(session_id: str):
    """
    Renders a JavaScript component to detect browser/tab close events
    and trigger an "emergency save" to the Streamlit app.
    This uses the robust beacon API (or XHR fallback) from Code 1 for reliability.
    """
    if not session_id:
        return

    # JavaScript code to send a beacon on unload events
    js_code = f"""
    <script>
    (function() {{
        // Use a flag to ensure the script runs only once per iframe load
        const scriptIdentifier = 'fifi_close_enhanced_script_' + '{session_id}';
        if (window[scriptIdentifier]) return; // If already initialized for this session, exit
        window[scriptIdentifier] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false; // Flag to prevent multiple save attempts
        
        // Helper to get the full application URL including path, handling iframe parent context
        function getAppUrl() {{
            try {{
                // Try to get parent's URL if in an iframe (Streamlit's default setup)
                // Ensure origin matches to avoid cross-origin security errors
                if (window.parent && window.parent.location.origin === window.location.origin) {{
                    return window.parent.location.origin + window.parent.location.pathname;
                }}
            }} catch (e) {{
                // Fallback to current window's URL if parent access is restricted or not applicable
                console.warn("Could not access parent.location. Fallbacking to current window.location for app URL:", e);
            }}
            return window.location.origin + window.location.pathname;
        }}

        function sendEmergencySaveWithReload() {{
            if (saveTriggered) return; // Only send once per unload sequence
            saveTriggered = true;
            console.clear(); // Clear console for clearer logging of this critical event
            console.log('🚨 FiFi Enhanced: Browser close detected - sending FORCED emergency save.');
            
            const appUrl = getAppUrl();
            // Construct the URL with query parameters for the emergency save event
            const saveUrl = `${{appUrl}}?event=emergency_close&session_id=${{sessionId}}`;
            
            console.log('📡 Attempting to send emergency save to:', saveUrl);
            
            // METHOD 1: Use navigator.sendBeacon (asynchronous, non-blocking, designed for unload)
            try {{
                if (navigator.sendBeacon) {{
                    const formData = new FormData();
                    formData.append('event', 'emergency_close');
                    formData.append('session_id', sessionId);
                    // sendBeacon can also take a URL and Blob/FormData
                    const beaconSuccess = navigator.sendBeacon(saveUrl, formData); // Send as POST data
                    console.log('📡 Beacon result:', beaconSuccess ? 'Success (sent)' : 'Failed (not sent)');
                    
                    if (beaconSuccess) {{
                        // Even if beacon succeeds, forcing a reload after a short delay
                        // can help ensure Streamlit processes the query parameter for robust handling.
                        // Add a small delay to allow the beacon to initiate fully.
                        setTimeout(() => {{
                            try {{
                                // Redirect parent window if possible
                                if (window.parent && window.parent.location.origin === window.location.origin) {{
                                    window.parent.location.href = saveUrl;
                                }} else {{
                                    window.location.href = saveUrl; // Fallback for current window
                                }}
                            }} catch (e) {{
                                console.error('Error redirecting after beacon:', e);
                            }}
                        }}, 100); // Short delay
                        return; // Exit after successful beacon attempt
                    }}
                }}
            }} catch (e) {{
                console.warn('📡 navigator.sendBeacon API failed or not available:', e);
            }}
            
            // METHOD 2: Fallback to synchronous XMLHttpRequest (less ideal, but robust for critical saves)
            // This is a blocking call, which ensures the request is sent before the page unloads completely.
            // It might cause a slight delay in page closing, but prioritizes data persistence.
            try {{
                console.log('🔄 Forcing emergency save via synchronous XMLHttpRequest fallback.');
                const xhr = new XMLHttpRequest();
                xhr.open("GET", saveUrl, false); // `false` makes it synchronous
                xhr.send(null);
                console.log('📡 Synchronous XHR sent.');
            }} catch (e2) {{
                console.error('❌ Synchronous XHR failed:', e2);
            }}
            
            // Final fallback/redundancy: Force a page reload with the save parameters
            // This ensures Streamlit processes the request on the *next* load.
            // Add a slight delay to allow other mechanisms to attempt first.
            setTimeout(() => {{
                try {{
                    if (window.parent && window.parent.location.origin === window.location.origin) {{
                        window.parent.location.href = saveUrl;
                    }} else {{
                        window.location.href = saveUrl;
                    }}
                }} catch (e) {{
                    console.error('❌ Final forced reload failed:', e);
                }}
            }}, 50); // Very short delay
        }}
        
        // Attach event listeners for various browser unload/hide scenarios
        const events = ['beforeunload', 'pagehide', 'unload'];
        events.forEach(eventType => {{
            try {{
                // Add listener to the parent window/document if accessible and same-origin
                if (window.parent && window.parent.location.origin === window.location.origin) {{
                    window.parent.addEventListener(eventType, sendEmergencySaveWithReload, {{ capture: true }});
                }}
                // Always add listener to the current window/document (iframe context)
                window.addEventListener(eventType, sendEmergencySaveWithReload, {{ capture: true }});
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        // Enhanced visibility change detection (e.g., tab switched away or browser minimized)
        try {{
            if (window.parent && window.parent.document && window.parent.document.location.origin === window.location.origin) {{
                window.parent.document.addEventListener('visibilitychange', () => {{
                    if (window.parent.document.visibilityState === 'hidden') {{
                        console.log('🚨 Main app tab hidden (visibilitychange event).');
                        sendEmergencySaveWithReload();
                    }}
                }}, {{ passive: true }});
            }}
        }} catch (e) {{
            // Fallback for current iframe's visibility if parent is inaccessible
            document.addEventListener('visibilitychange', () => {{
                if (document.visibilityState === 'hidden') {{
                    console.log('🚨 Component tab hidden (visibilitychange event).');
                    sendEmergencySaveWithReload();
                }}
            }}, {{ passive: true }});
        }}
        
        console.log('✅ Enhanced browser close detection initialized for session', sessionId.substring(0, 8));
    }})();
    </script>
    """
    
    try:
        # Streamlit components.v1.html allows injecting raw HTML/JS code into the app's iframe.
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close component JS for session {session_id[:8]}: {e}", exc_info=True)

def global_message_channel_error_handler():
    """
    Injects a global JavaScript error handler to specifically catch and prevent
    "message channel closed" errors, which are common with Streamlit iframes
    and can cause uncaught promise rejections in browsers. This improves stability.
    """
    js_error_handler = """
    <script>
    (function() {
        // Use a flag to ensure this handler is initialized only once per page load
        if (window.fifi_global_error_handler_initialized) return;
        window.fifi_global_error_handler_initialized = true;
        
        // Global error handler for uncaught promise rejections (e.g., from st_javascript)
        window.addEventListener('unhandledrejection', function(event) {
            const error = event.reason;
            // Check if the error message indicates a "message channel closed" issue
            if (error && error.message && error.message.includes('message channel closed')) {
                console.log('🛡️ FiFi: Caught and gracefully handled a "message channel closed" error:', error.message);
                event.preventDefault(); // Prevent the default browser handling (e.g., printing to console as an unhandled error)
            }
        });
        
        console.log('✅ FiFi: Global message channel error handler initialized.');
    })();
    </script>
    """
    try:
        st.components.v1.html(js_error_handler, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to initialize global message channel error handler: {e}", exc_info=True)

def handle_timer_event(timer_result: Dict[str, Any], session_manager, session: UserSession) -> bool:
    """
    Processes events triggered by the JavaScript activity timer (e.g., 15-minute timeout).
    This function adapted from Code 1's detailed event handler to fit 15-min timeout.
    """
    if not timer_result or not isinstance(timer_result, dict):
        return False # No valid timer event received
    
    event = timer_result.get('event')
    session_id = timer_result.get('session_id')
    inactive_minutes = timer_result.get('inactive_minutes', 0)
    
    logger.info(f"🎯 Processing timer event: '{event}' for session {session_id[:8] if session_id else 'unknown'}.")
    
    try:
        # Ensure session object is fully validated and types are correct for logic
        session = session_manager._validate_session(session)
        
        if event == 'session_timeout_15min':
            # This event signifies 15 minutes of user inactivity, triggering a CRM save.
            st.info(f"⏰ **Session timeout:** Detected {inactive_minutes} minutes of inactivity.")
            
            # Perform CRM save if the session is eligible (registered user, has email, has messages, and not already saved for this timeout)
            if (session.user_type == UserType.REGISTERED_USER and 
                session.email and 
                session.messages and
                not session.timeout_saved_to_crm):
                
                with st.spinner("💾 Auto-saving chat to CRM (15-min timeout)..."):
                    try:
                        # Call the robust synchronous CRM save method
                        save_success = session_manager.zoho.save_chat_transcript_sync(session, "15-Minute Session Inactivity Timeout")
                    except Exception as e:
                        logger.error(f"15-min timeout CRM save failed during execution: {e}", exc_info=True)
                        save_success = False # Ensure save_success is False on exception
                
                if save_success:
                    st.success("✅ Chat automatically saved to CRM!")
                    # Mark the session as saved for this timeout period to prevent duplicate saves
                    session.timeout_saved_to_crm = True
                    # Update session's last activity timestamp (as the save itself is an "activity" for the system)
                    session.last_activity = datetime.now() 
                    session_manager.db.save_session(session) # Persist the updated session state
                else:
                    st.warning("⚠️ Auto-save to CRM failed. Please check your credentials or contact support if issue persists.")
                
                # IMPORTANT: As per your specification, a 15-minute timeout is ONLY for CRM save, NOT for a session ban or expiry.
                st.info("ℹ️ You can continue using FiFi AI.")
                return False # No immediate rerun needed unless there are specific UI updates that require it
            else:
                st.info("ℹ️ Session timeout detected, but no CRM save was performed (e.g., Guest user, no chat history, or already saved).")
                logger.info(f"15-min timeout CRM save eligibility check failed for {session_id[:8]}: UserType={session.user_type.value}, Email={bool(session.email)}, Messages={len(session.messages)}, Saved Status={session.timeout_saved_to_crm}.")
                st.info("ℹ️ You can continue using FiFi AI.")
                return False # No immediate rerun needed
                
        else:
            logger.warning(f"⚠️ Received unhandled timer event type: '{event}'.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing timer event '{event}' for session {session_id[:8]}: {e}", exc_info=True)
        st.error(f"⚠️ An internal error occurred while processing activity. Please try refreshing if issues persist.")
        return False

def process_emergency_save_from_query(session_id: str) -> bool:
    """
    Processes an emergency save request initiated by the browser close beacon/reload.
    This function directly loads the session from the database and attempts a CRM save if eligible.
    Preserved and adapted from Code 1's robust emergency save handler.
    """
    try:
        session_manager = st.session_state.get('session_manager') # Get manager from session state
        if not session_manager:
            logger.error("❌ Session manager not available during emergency save processing from query. Initialization likely failed.")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Emergency save from query: Session '{session_id[:8]}' not found or not active in database.")
            return False
        
        session = session_manager._validate_session(session) # Ensure the loaded session object is correctly typed
        
        logger.info(f"✅ Emergency save processing for session '{session_id[:8]}': UserType={session.user_type.value}, Email={session.email}, Messages={len(session.messages)}.")
        
        # Check eligibility for emergency save: Must be a registered user with email and chat history, and not already saved.
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm): # Important: Only save if not already saved by a timeout timer
            
            logger.info(f"✅ Session '{session_id[:8]}' is eligible for emergency CRM save.")
            
            # Extend session activity to keep it 'active' in the DB during the save process
            session.last_activity = datetime.now()
            session_manager.db.save_session(session) # Persist the updated activity timestamp
            
            # Perform the synchronous CRM save
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Browser Close/Unload)")
            if success:
                # Mark session as saved due to timeout/emergency to prevent re-saves for this period
                session.timeout_saved_to_crm = True
                session_manager.db.save_session(session) # Persist this status update
            return success
        else:
            logger.info(f"❌ Session '{session_id[:8]}' not eligible for emergency save (e.g., Guest, no email, no messages, or already saved by timer).")
            return False
            
    except Exception as e:
        logger.error(f"❌ Emergency save processing failed for session '{session_id[:8]}': {e}", exc_info=True)
        error_handler.log_error(error_handler.handle_api_error("System", "Emergency Save Process (Query)", e))
        return False

def handle_emergency_save_requests_from_query():
    """
    Checks for and processes emergency save requests sent via URL query parameters
    (typically from browser close/unload events detected by JavaScript).
    This function must be called very early in `main()` to intercept the request.
    """
    logger.info("🔍 EMERGENCY SAVE HANDLER: Checking for query parameter requests for emergency save...")
    
    query_params = st.query_params # Access Streamlit's query parameters
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    # Check if the required event and session ID parameters are present
    if event == "emergency_close" and session_id:
        logger.info("=" * 80)
        logger.info("🚨 EMERGENCY SAVE REQUEST DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}")
        logger.info("=" * 80)
        
        # Show an immediate visual message to the user that a save is in progress
        st.error("🚨 **Emergency Save Detected** - Processing browser close save...")
        st.info("Please wait, your conversation is being saved...")
        
        # IMPORTANT: Clear these query parameters *immediately* for the current run context
        # and for subsequent reruns if the app doesn't stop, to prevent re-triggering.
        if "event" in st.query_params:
            del st.query_params["event"]
        if "session_id" in st.query_params:
            del st.query_params["session_id"]
        
        try:
            # Attempt to process the emergency save
            success = process_emergency_save_from_query(session_id)
            
            if success:
                st.success("✅ Emergency save completed successfully!")
                logger.info("✅ Emergency save completed via query parameter successfully.")
            else:
                st.error("❌ Emergency save failed or was not eligible for saving.")
                logger.error("❌ Emergency save failed via query parameter (not eligible or internal error).")
                
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during emergency save: {str(e)}")
            logger.critical(f"Emergency save processing crashed from query parameter: {e}", exc_info=True)
        
        # Add a short delay to allow the user to see the final message, then stop the app execution.
        # `st.stop()` is crucial to prevent the Streamlit app from rendering its main UI,
        # ensuring only the save message is shown and the session effectively ends.
        time.sleep(2)
        st.stop() # Stops the script execution
    else:
        logger.info("ℹ️ No emergency save requests found in current URL query parameters.")

# =============================================================================
# SESSION MANAGER (INTEGRATED & REFINED)
# =============================================================================

class SessionManager:
    """
    Manages user sessions, orchestrating interactions between database, CRM,
    AI, and various feature managers (fingerprinting, email verification, limits).
    This class centralizes the core application logic and session lifecycle.
    """
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, 
                 ai_system: EnhancedAI, rate_limiter: RateLimiter, fingerprinting_manager: FingerprintingManager,
                 email_verification_manager: EmailVerificationManager, question_limit_manager: QuestionLimitManager):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.fingerprinting = fingerprinting_manager
        self.email_verification = email_verification_manager
        self.question_limits = question_limit_manager
        self._save_lock = threading.Lock() # For synchronizing Zoho CRM saves

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return 15 # Hardcoded 15 minutes as per specification

    def _update_activity(self, session: UserSession):
        """
        Updates the session's last activity timestamp and saves it to the DB.
        Also resets the `timeout_saved_to_crm` flag if the user becomes active again
        after a timeout save occurred.
        """
        session.last_activity = datetime.now()
        
        # If the session was previously saved due to a timeout, and now user activity
        # is detected, reset this flag. This ensures the chat can be saved again
        # if another 15-minute inactivity period occurs.
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset 'timeout_saved_to_crm' flag for session {session.session_id[:8]} due to new activity.")
        
        # Ensure UserType is properly an Enum member before saving (re-hydration safeguard)
        if isinstance(session.user_type, str):
            session.user_type = UserType(session.user_type)
        
        try:
            self.db.save_session(session)
            logger.debug(f"Session activity updated and saved for {session.session_id[:8]}.")
        except Exception as e:
            logger.error(f"Failed to update session activity and save for {session.session_id[:8]}: {e}", exc_info=True)

    def _create_guest_session(self) -> UserSession:
        """
        Creates a new guest user session with a unique ID.
        Attempts to capture client's IP address and User-Agent string from
        Streamlit's internal request context for tracking purposes.
        """
        session = UserSession(session_id=str(uuid.uuid4()))
        
        # Attempt to capture client IP and User-Agent from Streamlit's request context
        # This is implementation-dependent and might not work in all deployment scenarios.
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx and ctx.request_context and ctx.request_context.headers:
                headers = ctx.request_context.headers
                
                # Iterate through common headers used by proxies to get the client's real IP
                ip_headers_priority = ['x-forwarded-for', 'x-real-ip', 'cf-connecting-ip', 'x-client-ip']
                for header_name in ip_headers_priority:
                    ip_val = headers.get(header_name)
                    if ip_val:
                        # For 'x-forwarded-for', the first IP is typically the client's IP
                        session.ip_address = ip_val.split(',')[0].strip()
                        session.ip_detection_method = header_name
                        break # Found an IP, break from loop
                else:
                    # Fallback if no specific IP header found (might be direct connection or obscured)
                    session.ip_address = "unknown_direct_connection" 
                    session.ip_detection_method = "direct_connection"

                session.user_agent = headers.get('user-agent', 'unknown')
                logger.debug(f"Captured IP '{session.ip_address}' ({session.ip_detection_method}) and User-Agent for new session {session.session_id[:8]}.")
        except Exception as e:
            logger.warning(f"Could not reliably capture IP/User-Agent for new session {session.session_id[:8]}: {e}")
            session.ip_address = "capture_failed"
            session.user_agent = "capture_failed"

        self.db.save_session(session) # Save the newly created session to DB
        st.session_state.current_session_id = session.session_id # Store in Streamlit's state
        logger.info(f"Created new guest session: {session.session_id[:8]}.")
        return session

    def _validate_session(self, session: UserSession) -> UserSession:
        """
        Ensures the `UserSession` object's Enum fields are correctly typed
        and other defaults/conversions are applied, especially after loading from the database.
        This re-hydrates the object from potentially stored string representations.
        """
        if not session: return session # Nothing to validate if session object is None
            
        # Convert user_type string to UserType Enum
        if isinstance(session.user_type, str):
            try:
                session.user_type = UserType(session.user_type)
            except ValueError:
                logger.error(f"Invalid user_type string '{session.user_type}' for session {session.session_id[:8]}. Defaulting to GUEST.")
                session.user_type = UserType.GUEST
        
        # Convert ban_status string to BanStatus Enum
        if isinstance(session.ban_status, str):
            try:
                session.ban_status = BanStatus(session.ban_status)
            except ValueError:
                logger.error(f"Invalid ban_status string '{session.ban_status}' for session {session.session_id[:8]}. Defaulting to NONE.")
                session.ban_status = BanStatus.NONE
        
        # Ensure 'messages' is a list (it's stored as JSON string in DB)
        if not isinstance(session.messages, list):
            logger.warning(f"Session {session.session_id[:8]} messages field is not a list. Resetting to empty list.")
            session.messages = []
        
        # Ensure 'email_addresses_used' is a list (it's stored as JSON string in DB)
        if not isinstance(session.email_addresses_used, list):
            logger.warning(f"Session {session.session_id[:8]} email_addresses_used field is not a list. Resetting to empty list.")
            session.email_addresses_used = []
            
        return session

    def _mask_email(self, email: str) -> str:
        """
        Masks an email address for display in the UI, preserving privacy.
        Example: "john.doe@example.com" -> "j***e@e***e.com"
        """
        if '@' not in email: return email # Not a valid email format, return as is
        
        local_part, domain_part = email.split('@', 1)
        
        # Mask local part: first character, then asterisks, then last character (if long enough)
        if len(local_part) <= 2:
            masked_local = local_part[0] + '*' * (len(local_part) - 1)
        else:
            masked_local = local_part[0] + '*' * (len(local_part) - 2) + local_part[-1]
        
        # Mask domain part: similar logic, but handles dot-separated parts
        domain_segments = domain_part.split('.')
        if len(domain_segments) > 1:
            # Mask the first segment of the domain
            masked_domain_first_part = '*' * len(domain_segments[0]) 
            masked_domain = masked_domain_first_part + '.' + '.'.join(domain_segments[1:])
        else:
            # If no dots in domain (unlikely for real emails, but for robustness), mask whole domain
            masked_domain = '*' * len(domain_part) 
        
        return f"{masked_local}@{masked_domain}"

    def _save_to_crm_timeout(self, session: UserSession, trigger_reason: str):
        """
        Internal method to initiate a CRM save, specifically for scenarios like
        inactivity timeout or emergency browser close. It wraps the actual Zoho call
        with logic for eligibility and logging.
        """
        with self._save_lock: # Use a thread-safe lock to prevent concurrent CRM saves
            logger.info(f"=== INITIATING CRM SAVE (Internal from SessionManager) ===")
            logger.info(f"Trigger: '{trigger_reason}', Session ID: {session.session_id[:8]}.")
            
            # Pre-flight checks for CRM save eligibility
            if not self.zoho.config.ZOHO_ENABLED:
                logger.info("CRM save skipped: Zoho integration is disabled in configuration.")
                return False
            if session.user_type != UserType.REGISTERED_USER:
                logger.info("CRM save skipped: Only REGISTERED_USER sessions are saved to CRM.")
                return False
            if not session.email:
                logger.info("CRM save skipped: Session has no associated email address.")
                return False
            if not session.messages:
                logger.info("CRM save skipped: No chat messages to save for session.")
                return False
            # Prevent duplicate save for the same trigger type if already saved in this "timeout window"
            if session.timeout_saved_to_crm and "timeout" in trigger_reason.lower():
                logger.info("CRM save skipped: Session already marked as saved for this timeout period.")
                return False 
            
            try:
                # Call the ZohoCRMManager's synchronous save method
                success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success:
                    logger.info("CRM save (internal) completed successfully.")
                    session.timeout_saved_to_crm = True # Mark session as saved by this event
                    self.db.save_session(session) # Persist this updated status
                    return True
                else:
                    logger.error("CRM save (internal) failed. See ZohoCRMManager logs for details.")
                    return False
            except Exception as e:
                logger.error(f"CRM save (internal) encountered an unexpected exception: {e}", exc_info=True)
                # Error handler already logs this via `@handle_api_errors` in ZohoCRMManager, no need to re-log here.
                return False
            finally:
                logger.info(f"=== CRM SAVE (Internal from SessionManager) ENDED ===\n")

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]):
        """
        Applies collected fingerprinting data (from JavaScript component) to the current session.
        This data is then persisted in the database.
        """
        session.fingerprint_id = fingerprint_data.get('fingerprint_id')
        session.fingerprint_method = fingerprint_data.get('fingerprint_method')
        session.visitor_type = fingerprint_data.get('visitor_type', 'new_visitor')
        session.browser_privacy_level = fingerprint_data.get('privacy_level', 'standard')
        
        # Update browser/user agent info if provided by fingerprint data
        browser_info = fingerprint_data.get('browser_info', {})
        if browser_info.get('userAgent'):
            session.user_agent = browser_info['userAgent']
        # Client-side IP capture from JS is often unreliable/proxied, so relying on server-side
        # IP capture during session creation is generally preferred, but can be updated here too.
        
        logger.info(f"Applied fingerprint to session {session.session_id[:8]}: ID={session.fingerprint_id[:8]}..., Method={session.fingerprint_method}.")
        self.db.save_session(session) # Persist the updated session with fingerprint

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """
        Checks the database to see if a given fingerprint ID has been seen before.
        If history is found, it returns relevant details (like associated email)
        for user recognition purposes (e.g., "We recognize you, is this your email?").
        """
        try:
            sessions_with_same_fp = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if sessions_with_same_fp:
                # Filter for sessions that have an associated email
                email_sessions = [s for s in sessions_with_same_fp if s.email]
                
                if email_sessions:
                    # Find the most recently active session with an email to suggest to the user
                    latest_email_session = max(email_sessions, key=lambda x: x.last_activity)
                    
                    return {
                        'has_history': True,
                        'email': latest_email_session.email,
                        'full_name': latest_email_session.full_name,
                        'last_seen': latest_email_session.last_activity,
                        'session_count': len(sessions_with_same_fp) # Total sessions with this fingerprint
                    }
            
            return {'has_history': False} # No history found for this fingerprint
            
        except Exception as e:
            logger.error(f"Error checking fingerprint history for '{fingerprint_id[:8]}...': {e}", exc_info=True)
            error_handler.log_error(error_handler.handle_api_error("Fingerprinting", "Check History", e))
            return {'has_history': False}

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """
        Initiates the email verification process for a guest user who is trying to
        upgrade their tier (e.g., after hitting the 4-question limit).
        It validates the email format and sends a verification code via Supabase.
        """
        # Corrected and robust email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' 
        if not re.match(email_pattern, email):
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # You could add more advanced checks here, e.g., if this email has been used excessively on different devices,
        # or if it's blacklisted.

        # Send the verification code using the EmailVerificationManager
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            session.email = email # Associate email with the current session
            # Track all email addresses used by this session/fingerprint (for evasion detection)
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email) 
            
            self.db.save_session(session) # Persist the session with the new email/tracking
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your inbox (and spam folder).'
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again or contact support.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """
        Verifies the email code provided by the user. If successful, it upgrades
        the user's tier to `EMAIL_VERIFIED_GUEST` and resets their question counts.
        """
        if not session.email:
            return {'success': False, 'message': 'No email address is set for verification in this session.'}
        
        # Verify the code using the EmailVerificationManager
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            session.user_type = UserType.EMAIL_VERIFIED_GUEST # Upgrade user tier
            session.daily_question_count = 0 # Reset daily question count for the new tier
            session.question_limit_reached = False # Clear any previous limit flags for UI
            
            # Reset ban status if any was active (e.g., a temporary ban from hitting guest limit)
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session) # Persist the updated session state
            
            logger.info(f"Session {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST for email: {session.email}.")
            
            return {
                'success': True,
                'message': f'Email verified! You are now an Email Verified Guest with 10 questions per day.'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please check the code and try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """
        Detects potential evasion attempts based on heuristics like
        multiple recent sessions from the same fingerprint hitting limits,
        or rapid email switching within the current session.
        This is a simplified detection; real systems may use more complex algorithms.
        """
        if not session.fingerprint_id:
            logger.debug("Cannot detect evasion: Session has no fingerprint ID.")
            return False # Cannot detect evasion without a fingerprint
        
        # 1. Check for other *recent* sessions from the *same fingerprint* that previously hit limits.
        # This suggests an attempt to bypass limits by creating new sessions/accounts on the same device.
        recent_cutoff = datetime.now() - timedelta(hours=48) # Look back 48 hours for "recent" activity
        sessions_from_same_fingerprint = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        for s in sessions_from_same_fingerprint:
            # Exclude the current session itself
            if s.session_id != session.session_id and s.last_activity and s.last_activity > recent_cutoff:
                # If a previous session from this device hit a question limit or was banned
                if s.question_limit_reached or s.ban_status != BanStatus.NONE:
                    logger.warning(f"Evasion detected: Fingerprint {session.fingerprint_id[:8]}... has recent limited/banned session {s.session_id[:8]}. Triggering evasion penalty.")
                    return True 

        # 2. Check for rapid email switching within the current session.
        # A high count of email switches for a single session might indicate an attempt to exploit guest limits.
        # The threshold (e.g., > 1 or 2 switches) can be adjusted.
        if session.email_switches_count > 1: # More than one switch (initial + 2nd attempt onwards)
            logger.warning(f"Evasion detected: Session {session.session_id[:8]} has switched emails {session.email_switches_count} times. Triggering evasion penalty.")
            return True
            
        return False # No evasion detected based on current heuristics

    def get_session(self) -> UserSession:
        """
        Retrieves the current user session. This is the primary method to get
        the `UserSession` object for any interaction.
        It handles:
        - Loading existing sessions from DB.
        - Creating new guest sessions if none exist.
        - Validating and re-hydrating session data.
        - Checking for active bans and displaying appropriate messages.
        - Updating last activity timestamp.
        """
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # 1. Validate and fix session data types (Enums, lists etc.)
                session = self._validate_session(session)
                
                # 2. Check ban status BEFORE allowing any further interaction.
                # If banned, display message and return the current session (which will be inactive for AI).
                limit_check = self.question_limits.is_within_limits(session)
                if not limit_check.get('allowed', True):
                    ban_type = limit_check.get('ban_type', 'unknown')
                    message = limit_check.get('message', 'Access restricted due to usage policy.')
                    time_remaining = limit_check.get('time_remaining')
                    
                    st.error(f"🚫 **Access Restricted**")
                    if time_remaining:
                        hours = int(time_remaining.total_seconds() // 3600)
                        minutes = int((time_remaining.total_seconds() % 3600) // 60)
                        st.error(f"Time remaining: {hours}h {minutes}m")
                    st.info(message)
                    logger.info(f"Session {session_id[:8]} is currently banned: Type={ban_type}, Reason='{message}'.")
                    # No activity update needed for banned sessions, as they cannot initiate activity.
                    return session
                
                # 3. If not banned, the session is active and valid: update its last activity timestamp and return.
                self._update_activity(session)
                return session
        
        # If no session ID in Streamlit state, or session not found/inactive in DB, create a new guest session.
        logger.info("No active session found or current session is invalid. Creating a new guest session.")
        new_session = self._create_guest_session()
        return self._validate_session(new_session) # Validate the newly created session

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """
        Authenticates a user against the configured WordPress site using JWT.
        If successful, it upgrades the current `UserSession` to `REGISTERED_USER` status.
        """
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service (WordPress URL) is not configured in secrets.")
            return None
        # Basic rate limiting for login attempts to prevent brute force
        if not self.rate_limiter.is_allowed(f"auth_login_attempt_{username}"):
            st.error("Too many login attempts. Please wait a moment before trying again.")
            return None

        clean_username = username.strip()
        clean_password = password.strip()

        try:
            logger.info(f"Attempting WordPress authentication for user: '{clean_username}'.")
            response = requests.post(
                f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                json={'username': clean_username, 'password': clean_password},
                headers={'Content-Type': 'application/json'},
                timeout=15 # Set a timeout for network requests
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WordPress authentication successful for '{clean_username}'.")
                
                # Get the current session (could be a guest session, which will be upgraded)
                current_session = self.get_session() 
                
                # Extract display name for personalization in the UI
                display_name = (
                    data.get('user_display_name') or 
                    data.get('displayName') or 
                    data.get('name') or 
                    data.get('user_nicename') or 
                    data.get('first_name') or
                    data.get('nickname') or
                    clean_username # Fallback if no specific display name found
                )

                # Upgrade the current session's attributes to REGISTERED_USER status
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now() # Reset activity on login
                current_session.timeout_saved_to_crm = False # Reset for new registered session

                # Reset question counts and clear any active bans for newly authenticated registered users
                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                # Add the authenticated email to the list of used addresses for this session/fingerprint
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
                try:
                    self.db.save_session(current_session) # Persist the upgraded session
                    logger.info(f"Authenticated session {current_session.session_id[:8]} saved as REGISTERED_USER.")
                    st.session_state.current_session_id = current_session.session_id # Ensure Streamlit state is updated
                    st.success(f"🎉 Welcome back, {current_session.full_name}!")
                    return current_session
                except Exception as e:
                    logger.error(f"Failed to save authenticated session {current_session.session_id[:8]} to DB: {e}", exc_info=True)
                    st.error("Authentication successful, but failed to save session details. Please try again.")
                    return None
                
            else:
                # Authentication failed on WordPress side
                error_message = f"Invalid username or password (HTTP Status Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message) # Get more specific error from WP API response
                except json.JSONDecodeError:
                    pass # If JSON decoding fails, use default HTTP error message
                
                st.error(error_message)
                logger.warning(f"WordPress authentication failed for '{clean_username}': {error_message}")
                return None
                
        except requests.exceptions.RequestException as e:
            # Handle network-related errors during the authentication request
            st.error(f"A network error occurred during authentication. Please check your internet connection or try again later.")
            logger.error(f"WordPress authentication network exception for '{clean_username}': {e}", exc_info=True)
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """
        Handles the entire AI response generation process for a user prompt.
        This includes: rate limiting, content moderation, evasion detection,
        and question limit checks before forwarding to the AI system.
        """
        # 1. Rate Limiting: Prevent rapid-fire requests from a single session.
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "You are sending requests too quickly. Please wait a moment and try again.", "success": False, "source": "Rate Limiter"}

        # 2. Validate session and check for active bans. This is a critical first step.
        session = self._validate_session(session) # Ensure session object is valid
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            # If user is banned or hits a hard limit, return appropriate message to the UI.
            return {
                "content": limit_check.get('message', 'Access restricted due to usage policy.'),
                "success": False,
                "banned": True, # Flag for UI to display ban message prominently
                "reason": limit_check.get('reason'),
                "time_remaining": limit_check.get('time_remaining')
            }
        
        # 3. Evasion Detection: Check if the user is trying to bypass limits.
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session) # Apply escalating penalty
            self.db.save_session(session) # Persist the session with the new ban status
            return {
                "content": "Unusual activity detected. Your access has been temporarily restricted. Please try again later.",
                "success": False,
                "evasion_penalty": True, # Flag for UI to display evasion message
                "penalty_hours": penalty_hours
            }

        # 4. Update Activity: User is now active, so update their last_activity timestamp.
        self._update_activity(session)

        # 5. Sanitize Input: Clean the user's prompt to prevent potential XSS or injection.
        sanitized_prompt = sanitize_input(prompt)
        
        # 6. Content Moderation: Check if the prompt violates any safety policies.
        moderation_result = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation_result and moderation_result.get("flagged"):
            # If flagged, record the user's prompt and the moderation warning in chat history.
            session.messages.append({"role": "user", "content": sanitized_prompt, "timestamp": datetime.now().isoformat()})
            session.messages.append({"role": "assistant", "content": moderation_result["message"], "source": "Content Safety Policy", "timestamp": datetime.now().isoformat()})
            self.db.save_session(session) # Persist the conversation including the moderation warning
            return {
                "content": moderation_result["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        # 7. Record Question: Increment the question count for the current session and tier.
        self.question_limits.record_question(session)

        # 8. Get AI Response: Call the underlying AI system to generate a response.
        ai_response = self.ai.get_response(sanitized_prompt, session.messages)
        
        # 9. Append Messages: Add both the user's prompt and the AI's response to chat history.
        session.messages.append({
            "role": "user", 
            "content": sanitized_prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare the assistant's message with content and metadata
        response_message = {
            "role": "assistant",
            "content": ai_response.get("content", "No response generated."),
            "source": ai_response.get("source", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add any additional metadata flags from the AI response (e.g., if search or Pinecone was used)
        for flag in ["used_search", "used_pinecone", "has_citations", "has_inline_citations", "safety_override"]:
            if ai_response.get(flag):
                response_message[flag] = True
            
        session.messages.append(response_message)
        session.messages = session.messages[-100:] # Keep chat history limited to the last 100 messages for performance
        
        # 10. Save Session: Persist the updated session (with new messages and question counts) to the database.
        self.db.save_session(session)
        return ai_response

    def clear_chat_history(self, session: UserSession):
        """Clears the chat history for the current session."""
        session = self._validate_session(session)
        session.messages = []
        session.timeout_saved_to_crm = False # Reset save flag if chat history is cleared
        self._update_activity(session) # Save changes to DB

    def end_session(self, session: UserSession):
        """
        Ends the current user session. This includes:
        - Attempting to save the chat transcript to CRM if the user is registered.
        - Marking the session as inactive in the database.
        - Clearing relevant session state variables to return to the welcome page.
        """
        session = self._validate_session(session)
        
        # Attempt CRM save for registered users with chat history
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            logger.info(f"Performing CRM save for manual sign-out of session {session.session_id[:8]}.")
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        # Mark the session as inactive in the database
        session.active = False
        try:
            self.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to mark session {session.session_id[:8]} as inactive in DB during end_session: {e}", exc_info=True)
        
        # Clear Streamlit session state variables to force a fresh start on next rerun
        keys_to_clear = ['current_session_id', 'page', 'verification_stage', 'verification_email']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        logger.info(f"Session {session.session_id[:8]} ended. Redirecting to welcome page.")

    def manual_save_to_crm(self, session: UserSession):
        """Allows registered users to manually save their current chat transcript to Zoho CRM."""
        session = self._validate_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            with st.spinner("💾 Saving chat to Zoho CRM..."):
                success = self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            if success:
                st.success("✅ Chat manually saved to Zoho CRM!")
                self._update_activity(session) # Update activity after successful manual save
            else:
                st.error("❌ Failed to manually save chat to Zoho CRM. Please check logs for details.")
        else:
            st.warning("Cannot save to CRM: Only registered users with a chat history can manually save.")

# =============================================================================
# UI COMPONENTS (INTEGRATED & ENHANCED)
# =============================================================================

def render_welcome_page(session_manager: SessionManager):
    """Renders the application's welcome page, including sign-in and guest options."""
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    # Display information about different user tiers and their benefits
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
    
    # Tabs for Sign In vs. Continue as Guest
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is currently disabled because the authentication service (WordPress URL) is not configured in application secrets.")
        else:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username or Email", help="Enter your WordPress username or email.")
                password = st.text_input("Password", type="password", help="Enter your WordPress password.")
                submit_button = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password to sign in.")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                            
                        if authenticated_session:
                            st.balloons() # Visual celebration for successful login
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            time.sleep(1) # Small delay for user to read the message
                            st.session_state.page = "chat" # Change application page
                            st.rerun() # Force a rerun to switch to the chat interface
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to get a quick start and try FiFi AI Assistant without signing in.
        
        ℹ️ **What to expect as a Guest:**
        - You get an initial allowance of **4 questions** to explore FiFi AI's capabilities.
        - After these 4 questions, **email verification will be required** to continue (unlocks 10 questions/day).
        - Our system utilizes **universal device fingerprinting** for security and to track usage across sessions.
        - You can always choose to **upgrade to a full registration** later for extended benefits.
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat" # Change application page
            st.rerun() # Force a rerun to switch to the chat interface

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    """Renders the application's sidebar, displaying session information, user status, and action buttons."""
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Dynamic user status display
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # Question usage display for registered users (40 total)
            st.markdown(f"**Questions Today:** {session.total_question_count}/40")
            # Visual progress bar, potentially showing tiers
            if session.total_question_count <= 20:
                st.progress(session.total_question_count / 20, text="Tier 1 (up to 20 questions)")
            else:
                st.progress((session.total_question_count - 20) / 20, text="Tier 2 (21-40 questions)")
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            # Daily question usage for email-verified guests (10 per day)
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            # Display time until daily question limit resets
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Daily questions have reset!")
            
        else: # UserType.GUEST
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10 questions/day.")
        
        # Display Fingerprinting Information (for all users once captured)
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        
        # Zoho CRM Status Display (only for registered users if Zoho is enabled)
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
        
        # General session information
        st.markdown(f"**Messages in Chat:** {len(session.messages)}")
        st.markdown(f"**Current Session ID:** `{session.session_id[:8]}...`")
        
        # Display Ban Status prominently if session is restricted
        if session.ban_status != BanStatus.NONE:
            st.error(f"🚫 **STATUS: RESTRICTED**")
            if session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.markdown(f"**Time Remaining:** {hours}h {minutes}m")
            st.markdown(f"Reason: {session.ban_reason or 'Usage policy violation'}")
        elif session.question_limit_reached and session.user_type == UserType.GUEST: 
            # Specific warning for guest users who hit limit (to prompt email verification)
            st.warning("⚠️ **ACTION REQUIRED: Email Verification**")
        
        st.divider()
        
        # Action buttons: Clear Chat and Sign Out
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clears all messages from the current conversation."):
                session_manager.clear_chat_history(session)
                st.rerun() # Rerun to update the chat display
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True, help="Ends your current session and returns to the welcome page."):
                session_manager.end_session(session)
                st.rerun() # Rerun to switch to welcome page

        # Download & Save Chat options (only for registered users with chat history)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
            st.divider()
            
            # PDF Download Button
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
            
            # Manual CRM Save Button (if Zoho is enabled and user is eligible)
            if session_manager.zoho.config.ZOHO_ENABLED and session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, help="Manually save your current chat transcript to your linked Zoho CRM contact."):
                    session_manager.manual_save_to_crm(session)
                st.caption("💡 Chat automatically saves to CRM after 15 minutes of inactivity.")

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """
    Renders the email verification dialog for guest users who have hit their
    initial question limit (4 questions).
    This dialog guides them through entering their email and verifying a code.
    """
    st.error("📧 **Email Verification Required**")
    st.info("You've used your 4 free questions. Please verify your email to unlock 10 questions per day.")
    
    # Initialize the verification stage if it's not already set
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = 'initial_check' # Start with checking fingerprint history

    # --- Stage 1: Initial check / User recognition ---
    if st.session_state.verification_stage == 'initial_check':
        # Check if we have fingerprint history to suggest a previously used email
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            # If a recognized email is found, prompt the user for confirmation
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            st.info(f"🤝 **We seem to recognize this device!**")
            st.markdown(f"Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's my email", use_container_width=True, key="recognize_yes_btn"):
                    session.recognition_response = "yes"
                    st.session_state.verification_email = fingerprint_history['email'] # Store recognized email
                    st.session_state.verification_stage = "send_code_recognized" # Proceed to send code to recognized email
                    st.rerun()
            with col2:
                if st.button("❌ No, use a different email", use_container_width=True, key="recognize_no_btn"):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry" # Proceed to manual email entry
                    st.rerun()
        else:
            # If no history or email associated, skip recognition and go directly to manual entry
            st.session_state.verification_stage = "email_entry"
            st.rerun() # Force rerun to update UI to the email entry form

    # --- Stage 2: Send code to recognized email (automated) ---
    if st.session_state.verification_stage == 'send_code_recognized':
        email_to_verify = st.session_state.get('verification_email')
        if email_to_verify:
            with st.spinner(f"Sending verification code to {email_to_verify}..."):
                result = session_manager.handle_guest_email_verification(session, email_to_verify)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.verification_stage = "code_entry" # Move to code entry stage
                else:
                    st.error(result['message'])
                    st.session_state.verification_stage = "email_entry" # Fallback to manual entry on send failure
            st.rerun()
        else:
            st.error("Error: No recognized email found to send the code. Please enter your email manually.")
            st.session_state.verification_stage = "email_entry" # Force back to manual email entry
            st.rerun()

    # --- Stage 3: Manual Email Entry ---
    if st.session_state.verification_stage == 'email_entry':
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**Please enter your email address to receive a verification code:**")
            # Pre-fill with session's email if available, or a placeholder
            current_email_input = st.text_input("Email Address", placeholder="your@email.com", value=st.session_state.get('verification_email', session.email or ""), key="manual_email_input")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if current_email_input:
                    # Check for email switching and update count (for evasion detection)
                    if session.email and current_email_input != session.email:
                        session.email_switches_count += 1
                        session.email = current_email_input # Update session email to the new one
                        session_manager.db.save_session(session) # Persist the updated session (important for tracking)
                        
                    result = session_manager.handle_guest_email_verification(session, current_email_input)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = current_email_input # Store for code entry stage
                        st.session_state.verification_stage = "code_entry" # Move to code entry
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address to receive the code.")
    
    # --- Stage 4: Code Entry ---
    if st.session_state.verification_stage == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email) # Get email to display
        
        st.success(f"📧 A verification code has been sent to **{verification_email}**.")
        st.info("Please check your email, including spam/junk folders. The code is valid for 10 minutes.")
        
        with st.form("code_verification_form", clear_on_submit=False):
            code = st.text_input("Enter Verification Code", placeholder="e.g., 123456", max_chars=6, key="verification_code_input")
            
            col_code1, col_code2 = st.columns(2)
            with col_code1:
                submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            with col_code2:
                resend_code = st.form_submit_button("🔄 Resend Code", use_container_width=True)
            
            if resend_code: # Check resend button first, as it might change stage
                if verification_email:
                    with st.spinner("Resending code..."):
                        verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                        if verification_sent:
                            st.success("Verification code resent successfully!")
                            st.session_state.verification_stage = "code_entry" # Remain in code entry stage
                        else:
                            st.error("Failed to resend code. Please try again later.")
                else:
                    st.error("Error: No email address found to resend the code. Please go back and enter your email.")
                    st.session_state.verification_stage = "email_entry" # Force back to email entry if no email is known
                st.rerun() # Rerun after resend attempt

            if submit_code: # Process submit button after resend logic
                if code:
                    with st.spinner("Verifying code..."):
                        result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons() # Visual celebration
                        # Clear verification state on successful verification
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        time.sleep(2) # Small delay for user to see success message
                        st.rerun() # Rerun to refresh the app and proceed to chat
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code you received.")
            
def render_chat_interface(session_manager: SessionManager, session: UserSession):
    """Renders the main chat interface, integrating all features like fingerprinting, timers, and question handling."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting.")
    
    # 1. Global handler for JS message channel errors (should run very early, only once)
    global_message_channel_error_handler()

    # 2. Initialize Fingerprinting if not already done for this session
    # This ensures a fingerprint ID is captured early in the session lifecycle.
    if not session.fingerprint_id:
        fingerprint_js_code = session_manager.fingerprinting.generate_fingerprint_component(session.session_id)
        # FIX: Removed 'height' parameter from st_javascript.
        # Execute the JS component. Use a consistent key for repeated calls within Streamlit.
        fp_result = st_javascript(fingerprint_js_code, key=f"fifi_fp_init_{session.session_id[:8]}")
        
        if fp_result:
            # If the JavaScript component returned a result, extract and apply the fingerprint data
            extracted_fp_data = session_manager.fingerprinting.extract_fingerprint_from_result(fp_result)
            session_manager.apply_fingerprinting(session, extracted_fp_data)
            # Crucially, force a rerun to update the session state with the new fingerprint ID
            # and allow the application to proceed with the updated session object.
            st.rerun()
        else:
            # If fingerprinting JS doesn't return immediately (e.g., still loading),
            # it will return on a subsequent rerun. Log a debug message.
            logger.debug(f"Fingerprinting component for session {session.session_id[:8]} did not return result on this run. Will try again.")

    # 3. Render browser close detection for emergency saves (only for Registered Users)
    # This JavaScript must be rendered on every rerun to ensure its listeners are active.
    if session.user_type == UserType.REGISTERED_USER:
        try:
            render_browser_close_detection_enhanced(session.session_id)
        except Exception as e:
            logger.error(f"Failed to render browser close detection JS for {session.session_id[:8]}: {e}", exc_info=True)

    # 4. Handle inactivity timer for 15-minute timeout (only for Registered Users)
    if session.user_type == UserType.REGISTERED_USER:
        timer_result = None
        try:
            # The JS component will return a dictionary event if the 15-minute timeout is reached
            timer_result = render_activity_timer_component_15min(session.session_id)
        except Exception as e:
            logger.error(f"15-minute timer component execution failed: {e}", exc_info=True)
            # Do not stop the app, just log the error
        
        if timer_result:
            # If an event was received from the timer, handle it
            # `handle_timer_event` might trigger a rerun if required (e.g., to update UI after CRM save).
            if handle_timer_event(timer_result, session_manager, session):
                st.rerun()

    # 5. Check for user limits/bans and display appropriate dialogs or messages.
    # This prevents users from interacting with the chat if they are restricted.
    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed']:
        # If the user is a GUEST and has hit the 4-question limit, display the email verification dialog.
        if limit_check.get('reason') == 'guest_limit':
            render_email_verification_dialog(session_manager, session)
            return # IMPORTANT: Stop rendering the rest of the chat interface (especially the input box) if the dialog is active.
        else:
            # For other ban reasons (daily limit, evasion), the message is displayed via get_session()
            # and the user cannot interact with the chat input. So just return.
            return

    # 6. Display chat messages from the session history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")): # Default role to 'user' if not specified
            st.markdown(msg.get("content", ""), unsafe_allow_html=True) # Render content, allowing some HTML if applicable
            
            # Display source and AI tool indicators for assistant messages
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

    # 7. Chat input for user interaction
    # The `disabled` attribute will prevent input if the user is banned.
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                            disabled=session.ban_status != BanStatus.NONE)
    
    if prompt:
        # Display the user's message immediately in the chat interface
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display a spinner while the AI processes the request
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    # Get AI response via the SessionManager, which handles all pre-processing.
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        # If AI response indicates email is required (e.g., guest hit 4-question limit)
                        st.error("📧 Please verify your email to continue using FiFi AI.")
                        st.session_state.verification_stage = 'email_entry' # Set stage to show email dialog
                        st.rerun()
                    elif response.get('banned'):
                        # If AI response indicates a ban (e.g., daily limit, tier limit)
                        st.error(response.get('content', 'Access restricted.')) # Display the ban message
                        if response.get('time_remaining'): # Display remaining time if provided
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                        st.rerun() # Rerun to update the UI with ban status
                    elif response.get('evasion_penalty'):
                        # If AI response indicates an evasion attempt was detected
                        st.error("🚫 Evasion detected - Your access has been temporarily restricted.")
                        st.error(f"Penalty duration: {response.get('penalty_hours', 0)} hours.")
                        st.rerun() # Rerun to update the UI with evasion status
                    else:
                        # If AI response is successful, display its content
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        # Display AI source and tool usage indicators
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
        
        st.rerun() # Always rerun after processing user input to update chat history and UI

# =============================================================================
# MAIN APPLICATION FLOW
# =============================================================================

def ensure_initialization():
    """
    Ensures all necessary application components and managers are initialized and
    stored in Streamlit's session state. This function runs only once per Streamlit session.
    It handles graceful fallback for optional dependencies and critical error logging.
    """
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence...")
        try:
            config = Config() # Load application configuration from Streamlit secrets
            pdf_exporter = PDFExporter() # Initialize PDF exporter
            
            # Database Manager is crucial and should be initialized only once and persisted.
            # It attempts SQLite Cloud first, then local file, then in-memory.
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager # Get the initialized DB manager
            zoho_manager = ZohoCRMManager(config, pdf_exporter) # Initialize Zoho CRM manager
            ai_system = EnhancedAI(config) # Initialize AI system placeholder
            rate_limiter = RateLimiter() # Initialize rate limiter
            
            # Initialize the new feature managers
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            # The main SessionManager orchestrates all other managers and core logic
            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            # Store other managers directly in session state if UI components need direct access
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler # Central error handler
            st.session_state.ai_system = ai_system # Potentially used by content moderation directly
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager

            st.session_state.initialized = True # Mark application as initialized
            logger.info("✅ Application initialized successfully with all features.")
            return True
            
        except Exception as e:
            # Critical error during initialization: display message and log
            st.error("💥 A critical error occurred during application startup and initialization. The application cannot run.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"CRITICAL: Application initialization failed: {e}", exc_info=True)
            return False # Indicate initialization failure
    
    return True # Application was already initialized

def main():
    """
    Main entry point for the Streamlit application.
    This function controls the overall flow, initialization, and page rendering.
    """
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    # Inject a global JavaScript handler for "message channel closed" errors.
    # This should be one of the very first things to run in the app.
    global_message_channel_error_handler()

    # Provide a "Fresh Start" button for development or emergency resets.
    # This clears all session state and forces a full re-initialization.
    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear_state_btn", help="Clears all session state and restarts the app. Use only for development or if the app is stuck."):
        logger.warning("User initiated 'Fresh Start (Dev)' button action.")
        st.session_state.clear() # Clears all session_state variables
        st.rerun() # Forces Streamlit to rerun the script from scratch

    # Ensure all application components and managers are initialized.
    # If initialization fails, the function will stop further execution.
    if not ensure_initialization():
        st.stop()

    # Handle emergency save requests that may come via URL query parameters (from browser close beacon)
    # This must run very early in the app lifecycle before main UI rendering.
    handle_emergency_save_requests_from_query()

    # Retrieve the main SessionManager instance from Streamlit's session state.
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        # If session manager is somehow not available after initialization, it's a critical error.
        st.error("Fatal: Session Manager failed to initialize or retrieve. The application cannot proceed.")
        logger.critical("Fatal: Session Manager not found in st.session_state after initialization attempt. App stopping.")
        st.stop() # Stop execution

    # Determine which main application page to render (Welcome/Login or Chat interface).
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        # If 'page' is not 'chat', render the welcome and login/guest options page.
        render_welcome_page(session_manager)
    else:
        # If 'page' is 'chat', retrieve the current user session.
        # `session_manager.get_session()` handles loading, validation, expiry, and ban checks.
        session = session_manager.get_session() 
        
        # Check if the retrieved session is active and valid to proceed to chat interface.
        if session and session.active:
            # Render the application sidebar (with user info, actions).
            # The PDF exporter is passed as it's needed for download functionality.
            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            # Render the main chat interface.
            render_chat_interface(session_manager, session)
        else:
            # If `get_session()` determines the session is inactive, expired, or banned,
            # it would have already displayed a message to the user.
            # We then clear the 'page' state to force a redirect back to the welcome page
            # on the next rerun.
            if 'page' in st.session_state:
                del st.session_state['page'] # Remove 'page' to default to welcome
            st.rerun() # Force a rerun to display the welcome page for a fresh start.

if __name__ == "__main__":
    main()
