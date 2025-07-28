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

# REMOVED: import signal (as it doesn't work in Streamlit's threading model)

# =============================================================================
# FINAL INTEGRATED FIFI AI - ALL FEATURES IMPLEMENTED (CLEANED & COMPLETE)
# - All previous fixes including enum comparison.
# - OpenAI content moderation model name corrected.
# - All temporary debug logging removed for production use.
# - FIX: Password field DOM warning and console visibility.
# - FIX: Question progress bar affecting chat visibility (sidebar layout).
# - FIX: Simplified browser close detection (redirect-only).
# - FIX: render_welcome_page function correctly replaced.
# - FIX: Database initialization hang (signal module issue) resolved by removing timeout mechanism.
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
        
        # Initialize database schema (SIMPLIFIED - no signal timeout needed)
        if self.conn: # Only attempt if a connection was established
            try:
                self._init_complete_database()
                logger.info("✅ Database initialization completed successfully")
                error_handler.mark_component_healthy("Database")
                
            except Exception as e:
                logger.error(f"Database initialization failed: {e}", exc_info=True)
                self.conn = None
                self.db_type = "memory" 
                self.local_sessions = {} # Fallback to in-memory on any init error
        
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
        Simplified database initialization to avoid hanging issues with ALTER TABLE.
        Creates the full schema upfront.
        """
        with self.lock:
            try:
                if hasattr(self.conn, 'row_factory'): 
                    self.conn.row_factory = None

                # Create table with all columns upfront to avoid ALTER TABLE operations later
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
                        ip_address TEXT,
                        ip_detection_method TEXT,
                        user_agent TEXT,
                        browser_privacy_level TEXT,
                        registration_prompted INTEGER DEFAULT 0,
                        registration_link_clicked INTEGER DEFAULT 0,
                        wp_token TEXT,
                        timeout_saved_to_crm INTEGER DEFAULT 0,
                        recognition_response TEXT
                    )
                ''')
                
                # Create essential indexes only
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON sessions(session_id, active)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint_id ON sessions(fingerprint_id)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_email ON sessions(email)")
                
                self.conn.commit()
                logger.info("✅ Simplified database schema ready and essential indexes created.")
                
            except Exception as e:
                logger.error(f"Database initialization failed: {e}", exc_info=True)
                raise # Re-raise to indicate a critical failure

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.db_type == "memory":
                # For in-memory, store a deep copy to prevent external modifications
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                logger.debug(f"Saved session {session.session_id[:8]} to in-memory.")
                return

            try:
                # Get the actual column names from the database table at runtime
                cursor = self.conn.execute("PRAGMA table_info(sessions)")
                db_column_names = [row[1] for row in cursor.fetchall()] # row[1] is the column name
                
                # Map UserSession attributes to a dictionary for easy lookup by column name
                s = session
                session_attr_to_value = {
                    "session_id": s.session_id,
                    "user_type": s.user_type.value,
                    "email": s.email,
                    "full_name": s.full_name,
                    "zoho_contact_id": s.zoho_contact_id,
                    "created_at": s.created_at.isoformat(),
                    "last_activity": s.last_activity.isoformat(),
                    "messages": json.dumps(s.messages),
                    "active": int(s.active),
                    "wp_token": s.wp_token,
                    "timeout_saved_to_crm": int(s.timeout_saved_to_crm),
                    "fingerprint_id": s.fingerprint_id,
                    "fingerprint_method": s.fingerprint_method,
                    "visitor_type": s.visitor_type,
                    "recognition_response": s.recognition_response,
                    "daily_question_count": s.daily_question_count,
                    "total_question_count": s.total_question_count,
                    "last_question_time": s.last_question_time.isoformat() if s.last_question_time else None,
                    "question_limit_reached": int(s.question_limit_reached),
                    "ban_status": s.ban_status.value,
                    "ban_start_time": s.ban_start_time.isoformat() if s.ban_start_time else None,
                    "ban_end_time": s.ban_end_time.isoformat() if s.ban_end_time else None,
                    "ban_reason": s.ban_reason,
                    "evasion_count": s.evasion_count,
                    "current_penalty_hours": s.current_penalty_hours,
                    "escalation_level": s.escalation_level,
                    "email_addresses_used": json.dumps(s.email_addresses_used),
                    "email_switches_count": s.email_switches_count,
                    "ip_address": s.ip_address,
                    "ip_detection_method": s.ip_detection_method,
                    "user_agent": s.user_agent,
                    "browser_privacy_level": s.browser_privacy_level,
                    "registration_prompted": int(s.registration_prompted),
                    "registration_link_clicked": int(s.registration_link_clicked)
                }

                # Construct the list of values to insert in the exact order of db_column_names.
                values_to_insert = [
                    session_attr_to_value.get(col_name) for col_name in db_column_names
                ]
                
                logger.debug(f"DB Columns (Actual): Count={len(db_column_names)} -> {db_column_names}")
                logger.debug(f"Values to Insert: Count={len(values_to_insert)}")

                # Construct the REPLACE INTO statement dynamically using discovered column names
                columns_sql = ", ".join(db_column_names)
                placeholders_sql = ", ".join(["?"] * len(db_column_names))

                sql_statement = f'''
                    REPLACE INTO sessions ({columns_sql}) VALUES ({placeholders_sql})
                '''
                
                self.conn.execute(sql_statement, values_to_insert)
                self.conn.commit()
                logger.debug(f"Successfully saved session {s.session_id[:8]} to database using dynamic columns.")
            except Exception as e:
                logger.error(f"Failed to save session {s.session_id[:8]}: {e}", exc_info=True)
                raise # Re-raise to be caught by handle_api_errors

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            if self.db_type == "memory":
                return copy.deepcopy(self.local_sessions.get(session_id))
            try:
                if self.db_type == "file": self.conn.row_factory = sqlite3.Row
                else: self.conn.row_factory = None

                cursor = self.conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                row = cursor.fetchone()
                if not row:
                    logger.debug(f"No active session found for ID {session_id[:8]}.")
                    return None
                
                row_dict = dict(row) if hasattr(row, 'keys') else dict(zip([d[0] for d in cursor.description], row))
                
                session_params = {}
                session_params['session_id'] = row_dict.get('session_id', session_id)
                
                for key, value in row_dict.items():
                    if key == 'session_id':
                        continue
                        
                    if hasattr(UserSession, key):
                        session_params[key] = self._convert_db_value_to_python(key, value)
                    else:
                        logger.debug(f"Skipping unknown DB column '{key}' during session load for {session_id[:8]} (not in UserSession dataclass).")
                
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
        
        datetime_keys = ['created_at', 'last_activity', 'last_question_time', 'ban_start_time', 'ban_end_time']
        if key in datetime_keys and isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                logger.warning(f"Could not convert {key} '{value}' to datetime. Returning None.")
                return None
        
        json_list_keys = ['messages', 'email_addresses_used']
        if key in json_list_keys and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON for {key}: '{value}'. Returning empty list.")
                return []
        
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
                    session_params['session_id'] = row_dict.get('session_id', str(uuid.uuid4()))
                    for key, value in row_dict.items():
                        if key == 'session_id': continue
                        if hasattr(UserSession, key):
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
                    session_params['session_id'] = row_dict.get('session_id', str(uuid.uuid4()))
                    for key, value in row_dict.items():
                        if key == 'session_id': continue
                        if hasattr(UserSession, key):
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
        self.fingerprint_cache = {}

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
                fingerprintId = 'privacy_browser_' + Date.now();
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
        if not result or not isinstance(result, dict) or result.get('fingerprint_id') == 'ZXJyb3I=':
            logger.warning("Fingerprint JavaScript returned error or null. Using fallback.")
            return self._generate_fallback_fingerprint()
        
        fingerprint_id = result.get('fingerprint_id')
        fingerprint_method = result.get('fingerprint_method', 'unknown')
        
        if not fingerprint_id or fingerprint_id.startswith('privacy_browser_'):
            logger.info("Fingerprint ID indicates privacy browser or fallback. Generating new fallback.")
            return self._generate_fallback_fingerprint()
        
        visitor_type = "returning_visitor" if fingerprint_id in self.fingerprint_cache else "new_visitor"
        self.fingerprint_cache[fingerprint_id] = {'last_seen': datetime.now()}
        
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
            'browser_info': {},
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
                self.supabase = None

    @handle_api_errors("Supabase Auth", "Send Verification Code")
    def send_verification_code(self, email: str) -> bool:
        if not self.supabase:
            st.error("Email verification service is not available (Supabase not configured/failed).")
            return False
        
        try:
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
            if response.user:
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
            UserType.GUEST.value: 4,
            UserType.EMAIL_VERIFIED_GUEST.value: 10,
            UserType.REGISTERED_USER.value: 40
        }
        self.evasion_penalties = [24, 48, 96, 192, 336]
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """
        Checks if the current session is within its allowed question limits
        or if any bans are active.
        """
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
            elif session.total_question_count >= 20 and session.ban_status.value == BanStatus.NONE.value:
                self._apply_ban(session, BanStatus.ONE_HOUR, "Registered user first tier limit reached")
                return {
                    'allowed': False,
                    'reason': 'first_tier_limit',
                    'message': "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
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
        Retrieves or refreshes the Zoho CRM access token.
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
            response.raise_for_status()
            data = response.json()
            
            self._access_token = data.get('access_token')
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
        """Finds a Zoho contact by email."""
        access_token = self._get_access_token()
        if not access_token: return None
        
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            
            if response.status_code == 401:
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
        """Creates a new Zoho contact."""
        access_token = self._get_access_token()
        if not access_token: return None

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
        """Uploads a PDF attachment to a Zoho contact."""
        access_token = self._get_access_token()
        if not access_token: return False

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
                    if not access_token: return False
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
        if not access_token: return False

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
                if not access_token: return None
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
        Synchronously saves the chat transcript to Zoho CRM.
        """
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
                    if attempt == max_retries - 1: return False
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
                error_handler.log_error(error_handler.handle_api_error("Zoho CRM", "Save Transcript", e))
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
# RATE LIMITER & AI SYSTEM (Preserved)
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
                error_handler.log_error(error_handler.handle_api_error("OpenAI", "Initialization", e))

    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Provides a simplified AI response.
        """
        # In a real application, this would integrate with LangChain, Pinecone, Tavily, and OpenAI.
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
                "message": "Your message violates our content policy and cannot be processed.",
                "categories": flagged_categories
            }
    except Exception as e:
        logger.error(f"Content moderation API call failed: {e}", exc_info=True)
        return {"flagged": False}
    
    return {"flagged": False}

# =============================================================================
# JAVASCRIPT COMPONENTS & EVENT HANDLING
# =============================================================================

def render_activity_timer_component_15min(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Renders a JavaScript component that tracks user inactivity and triggers
    an event after 15 minutes.
    """
    if not session_id:
        return None
    
    js_timer_code = f"""
    (() => {{
        try {{
            const sessionId = "{session_id}";
            const SESSION_TIMEOUT_MS = 900000;
            
            console.log("🕐 FiFi 15-Minute Timer: Checking session", sessionId.substring(0, 8));
            
            if (typeof window.fifi_timer_state === 'undefined' || window.fifi_timer_state === null || window.fifi_timer_state.sessionId !== sessionId) {{
                console.clear();
                console.log("🆕 FiFi 15-Minute Timer: Starting/Resetting for session", sessionId.substring(0, 8)); 
                window.fifi_timer_state = {{
                    lastActivityTime: Date.now(),
                    expired: false,
                    listenersInitialized: false,
                    sessionId: sessionId
                }};
                console.log("🆕 FiFi 15-Minute Timer state initialized.");
            }}
            
            const state = window.fifi_timer_state;
            
            if (!state.listenersInitialized) {{
                console.log("👂 Setting up FiFi 15-Minute activity listeners...");
                
                function resetActivity() {{
                    try {{
                        const now = Date.now();
                        if (state.lastActivityTime !== now) {{
                            state.lastActivityTime = now;
                            if (state.expired) {{
                                console.log("🔄 Activity detected, resetting expired flag for timer.");
                            }}
                            state.expired = false;
                        }}
                    }} catch (e) {{
                        console.debug("Error in resetActivity:", e);
                    }}
                }}
                
                const events = [
                    'mousedown', 'mousemove', 'mouseup', 'click', 'dblclick',
                    'keydown', 'keyup', 'keypress',
                    'scroll', 'wheel',
                    'touchstart', 'touchmove', 'touchend',
                    'focus'
                ];
                
                const addListenersToTarget = (target) => {{
                    events.forEach(eventType => {{
                        try {{
                            target.addEventListener(eventType, resetActivity, {{ 
                                passive: true, 
                                capture: true,
                                once: false
                            }});
                        }} catch (e) {{
                            console.debug(`Failed to add ${{eventType}} listener to target:`, e);
                        }}
                    }});
                }};
                
                addListenersToTarget(document);
                
                try {{
                    if (window.parent && window.parent.document && window.parent.document !== document &&
                        window.parent.location.origin === window.location.origin) {{
                        addListenersToTarget(window.parent.document);
                        console.log("👂 Parent document listeners added successfully.");
                    }}
                }} catch (e) {{
                    console.debug("Cannot access parent document for listeners:", e);
                }}
                
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
            
            const currentTime = Date.now();
            const inactiveTimeMs = currentTime - state.lastActivityTime;
            const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
            const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
            
            console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
            
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
            return null;
        }}
    }})()
    """
    
    try:
        stable_key = f"fifi_timer_15min_{session_id[:8]}_{hash(session_id) % 10000}"
        timer_result = st_javascript(js_timer_code, key=stable_key)
        
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            return None
        
        if isinstance(timer_result, dict) and timer_result.get('event') == "session_timeout_15min":
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

def render_browser_close_detection_simplified(session_id: str):
    """
    Simplified browser close detection using redirect only.
    No POST requests - just redirects to trigger emergency save.
    """
    if not session_id:
        return

    js_code = f"""
    <script>
    (function() {{
        const scriptIdentifier = 'fifi_close_simple_' + '{session_id}';
        if (window[scriptIdentifier]) return;
        window[scriptIdentifier] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false;
        
        function getAppUrl() {{
            try {{
                if (window.parent && window.parent.location.origin === window.location.origin) {{
                    return window.parent.location.origin + window.parent.location.pathname;
                }}
            }} catch (e) {{
                console.warn("Using current window location as fallback");
            }}
            return window.location.origin + window.location.pathname;
        }}

        function triggerEmergencySave() {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            console.log('🚨 Browser close detected - triggering emergency save via redirect');
            
            const appUrl = getAppUrl();
            const saveUrl = `${{appUrl}}?event=emergency_close&session_id=${{sessionId}}`;
            
            try {{
                if (window.parent && window.parent.location.origin === window.location.origin) {{
                    window.parent.location.href = saveUrl;
                }} else {{
                    window.location.href = saveUrl;
                }}
            }} catch (e) {{
                console.error('Emergency save redirect failed:', e);
            }}
        }}
        
        const events = ['beforeunload', 'pagehide', 'unload'];
        events.forEach(eventType => {{
            try {{
                if (window.parent && window.parent.location.origin === window.location.origin) {{
                    window.parent.addEventListener(eventType, triggerEmergencySave, {{ capture: true }});
                }}
                window.addEventListener(eventType, triggerEmergencySave, {{ capture: true }});
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        try {{
            const handleVisibilityChange = () => {{
                if (document.visibilityState === 'hidden') {{
                    triggerEmergencySave();
                }}
            }};
            
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('visibilitychange', handleVisibilityChange);
            }}
            document.addEventListener('visibilitychange', handleVisibilityChange);
        }} catch (e) {{
            console.debug('Visibility change detection setup failed:', e);
        }}
        
        console.log('✅ Simplified browser close detection initialized');
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render simplified browser close component: {e}", exc_info=True)

def global_message_channel_error_handler():
    """
    Injects a global JavaScript error handler to specifically catch and prevent
    "message channel closed" errors.
    """
    js_error_handler = """
    <script>
    (function() {
        if (window.fifi_global_error_handler_initialized) return;
        window.fifi_global_error_handler_initialized = true;
        
        window.addEventListener('unhandledrejection', function(event) {
            const error = event.reason;
            if (error && error.message && error.message.includes('message channel closed')) {
                console.log('🛡️ FiFi: Caught and gracefully handled a "message channel closed" error:', error.message);
                event.preventDefault();
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
    """
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

def process_emergency_save_from_query(session_id: str) -> bool:
    """
    Processes an emergency save request initiated by the browser close beacon/reload.
    """
    try:
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("❌ Session manager not available during emergency save processing from query. Initialization likely failed.")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Emergency save from query: Session '{session_id[:8]}' not found or not active in database.")
            return False
        
        session = session_manager._validate_session(session)
        
        logger.info(f"✅ Emergency save processing for session '{session_id[:8]}': UserType={session.user_type.value}, Email={session.email}, Messages={len(session.messages)}.")
        
        if (session.user_type.value == UserType.REGISTERED_USER.value and
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            logger.info(f"✅ Session '{session_id[:8]}' is eligible for emergency CRM save.")
            
            session.last_activity = datetime.now()
            session_manager.db.save_session(session)
            
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Browser Close/Unload)")
            if success:
                session.timeout_saved_to_crm = True
                session_manager.db.save_session(session)
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
    Checks for and processes emergency save requests sent via URL query parameters.
    """
    logger.info("🔍 EMERGENCY SAVE HANDLER: Checking for query parameter requests for emergency save...")
    
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event == "emergency_close" and session_id:
        logger.info("=" * 80)
        logger.info("🚨 EMERGENCY SAVE REQUEST DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}")
        logger.info("=" * 80)
        
        st.error("🚨 **Emergency Save Detected** - Processing browser close save...")
        st.info("Please wait, your conversation is being saved...")
        
        if "event" in st.query_params:
            del st.query_params["event"]
        if "session_id" in st.query_params:
            del st.query_params["session_id"]
        
        try:
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
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No emergency save requests found in current URL query parameters.")

# =============================================================================
# SESSION MANAGER (INTEGRATED & REFINED)
# =============================================================================

class SessionManager:
    """
    Manages user sessions, orchestrating interactions between database, CRM,
    AI, and various feature managers.
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
        self._save_lock = threading.Lock()

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return 15

    def _update_activity(self, session: UserSession):
        """
        Updates the session's last activity timestamp and saves it to the DB.
        Also resets the `timeout_saved_to_crm` flag if the user becomes active again.
        """
        session.last_activity = datetime.now()
        
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset 'timeout_saved_to_crm' flag for session {session.session_id[:8]} due to new activity.")
        
        if isinstance(session.user_type, str):
            session.user_type = UserType(session.user_type)
        
        try:
            self.db.save_session(session)
            logger.debug(f"Session activity updated and saved for {session.session_id[:8]}.")
        except Exception as e:
            logger.error(f"Failed to update session activity and save for {session.session_id[:8]}: {e}", exc_info=True)

    def _create_guest_session(self) -> UserSession:
        """
        Creates a new guest user session with a unique ID and attempts to capture client info.
        """
        session = UserSession(session_id=str(uuid.uuid4()))
        
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx and ctx.request_context and ctx.request_context.headers:
                headers = ctx.request_context.headers
                
                ip_headers_priority = ['x-forwarded-for', 'x-real-ip', 'cf-connecting-ip', 'x-client-ip']
                for header_name in ip_headers_priority:
                    ip_val = headers.get(header_name)
                    if ip_val:
                        session.ip_address = ip_val.split(',')[0].strip()
                        session.ip_detection_method = header_name
                        break
                else:
                    session.ip_address = "unknown_direct_connection" 
                    session.ip_detection_method = "direct_connection"

                session.user_agent = headers.get('user-agent', 'unknown')
                logger.debug(f"Captured IP '{session.ip_address}' ({session.ip_detection_method}) and User-Agent for new session {session.session_id[:8]}.")
        except Exception as e:
            logger.warning(f"Could not reliably capture IP/User-Agent for new session {session.session_id[:8]}: {e}")
            session.ip_address = "capture_failed"
            session.user_agent = "capture_failed"

        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        logger.info(f"Created new guest session: {session.session_id[:8]}.")
        return session

    def _validate_session(self, session: UserSession) -> UserSession:
        """
        Ensures the `UserSession` object's Enum fields are correctly typed
        and other defaults/conversions are applied.
        """
        if not session: return session
            
        if isinstance(session.user_type, str):
            try:
                session.user_type = UserType(session.user_type)
            except ValueError:
                logger.error(f"Invalid user_type string '{session.user_type}' for session {session.session_id[:8]}. Defaulting to GUEST.")
                session.user_type = UserType.GUEST
        
        if isinstance(session.ban_status, str):
            try:
                session.ban_status = BanStatus(session.ban_status)
            except ValueError:
                logger.error(f"Invalid ban_status string '{session.ban_status}' for session {session.session_id[:8]}. Defaulting to NONE.")
                session.ban_status = BanStatus.NONE
        
        if not isinstance(session.messages, list):
            logger.warning(f"Session {session.session_id[:8]} messages field is not a list. Resetting to empty list.")
            session.messages = []
        
        if not isinstance(session.email_addresses_used, list):
            logger.warning(f"Session {session.session_id[:8]} email_addresses_used field is not a list. Resetting to empty list.")
            session.email_addresses_used = []
            
        return session

    def _mask_email(self, email: str) -> str:
        """
        Masks an email address for display in the UI, preserving privacy.
        """
        if '@' not in email: return email
        
        local_part, domain_part = email.split('@', 1)
        
        if len(local_part) <= 2:
            masked_local = local_part[0] + '*' * (len(local_part) - 1)
        else:
            masked_local = local_part[0] + '*' * (len(local_part) - 2) + local_part[-1]
        
        domain_segments = domain_part.split('.')
        if len(domain_segments) > 1:
            masked_domain_first_part = '*' * len(domain_segments[0]) 
            masked_domain = masked_domain_first_part + '.' + '.'.join(domain_segments[1:])
        else:
            masked_domain = '*' * len(domain_part) 
        
        return f"{masked_local}@{masked_domain}"

    def _save_to_crm_timeout(self, session: UserSession, trigger_reason: str):
        """
        Internal method to initiate a CRM save, specifically for scenarios like
        inactivity timeout or emergency browser close.
        """
        with self._save_lock:
            logger.info(f"=== INITIATING CRM SAVE (Internal from SessionManager) ===")
            logger.info(f"Trigger: '{trigger_reason}', Session ID: {session.session_id[:8]}.")
            
            if (session.user_type.value != UserType.REGISTERED_USER.value or 
                not session.email or 
                not session.messages or 
                not self.config.ZOHO_ENABLED):
                logger.info("CRM save skipped: Not eligible.")
                return False
            if session.timeout_saved_to_crm and "timeout" in trigger_reason.lower():
                logger.info("CRM save skipped: Session already marked as saved for this timeout period.")
                return False 
            
            try:
                success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success:
                    logger.info("CRM save (internal) completed successfully.")
                    session.timeout_saved_to_crm = True
                    self.db.save_session(session)
                    return True
                else:
                    logger.error("CRM save (internal) failed. See ZohoCRMManager logs for details.")
                    return False
            except Exception as e:
                logger.error(f"CRM save (internal) encountered an unexpected exception: {e}", exc_info=True)
                return False
            finally:
                logger.info(f"=== CRM SAVE (Internal from SessionManager) ENDED ===\n")

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]):
        """
        Applies collected fingerprinting data (from JavaScript component) to the current session.
        """
        session.fingerprint_id = fingerprint_data.get('fingerprint_id')
        session.fingerprint_method = fingerprint_data.get('fingerprint_method')
        session.visitor_type = fingerprint_data.get('visitor_type', 'new_visitor')
        session.browser_privacy_level = fingerprint_data.get('privacy_level', 'standard')
        
        browser_info = fingerprint_data.get('browser_info', {})
        if browser_info.get('userAgent'):
            session.user_agent = browser_info['userAgent']
        
        logger.info(f"Applied fingerprint to session {session.session_id[:8]}: ID={session.fingerprint_id[:8]}..., Method={session.fingerprint_method}.")
        self.db.save_session(session)

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """
        Checks the database to see if a given fingerprint ID has been seen before.
        """
        try:
            sessions_with_same_fp = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if sessions_with_same_fp:
                email_sessions = [s for s in sessions_with_same_fp if s.email]
                
                if email_sessions:
                    latest_email_session = max(email_sessions, key=lambda x: x.last_activity)
                    
                    return {
                        'has_history': True,
                        'email': latest_email_session.email,
                        'full_name': latest_email_session.full_name,
                        'last_seen': latest_email_session.last_activity,
                        'session_count': len(sessions_with_same_fp)
                    }
            
            return {'has_history': False}
            
        except Exception as e:
            logger.error(f"Error checking fingerprint history for '{fingerprint_id[:8]}...': {e}", exc_info=True)
            error_handler.log_error(error_handler.handle_api_error("Fingerprinting", "Check History", e))
            return {'has_history': False}

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """
        Initiates the email verification process for a guest user.
        """
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$' 
        if not re.match(email_pattern, email):
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email) 
            
            self.db.save_session(session)
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your inbox (and spam folder).'
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again or contact support.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """
        Verifies the email code provided by the user. If successful, it upgrades
        the user's tier.
        """
        if not session.email:
            return {'success': False, 'message': 'No email address is set for verification in this session.'}
        
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            session.user_type = UserType.EMAIL_VERIFIED_GUEST
            session.daily_question_count = 0
            session.question_limit_reached = False
            
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session)
            
            logger.info(f"Session {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST for email: {session.email}.")
            
            return {
                'success': True,
                'message': f'Email verified! You are now an Email Verified Guest with 10 questions per day.'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please check the code and try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """
        Detects potential evasion attempts based on heuristics.
        """
        if not session.fingerprint_id:
            logger.debug("Cannot detect evasion: Session has no fingerprint ID.")
            return False
        
        recent_cutoff = datetime.now() - timedelta(hours=48)
        sessions_from_same_fingerprint = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        for s in sessions_from_same_fingerprint:
            if s.session_id != session.session_id and s.last_activity and s.last_activity > recent_cutoff:
                if s.question_limit_reached or s.ban_status.value != BanStatus.NONE.value:
                    logger.warning(f"Evasion detected: Fingerprint {session.fingerprint_id[:8]}... has recent limited/banned session {s.session_id[:8]}. Triggering evasion penalty.")
                    return True 

        if session.email_switches_count > 1:
            logger.warning(f"Evasion detected: Session {session.session_id[:8]} has switched emails {session.email_switches_count} times. Triggering evasion penalty.")
            return True
            
        return False

    def get_session(self) -> UserSession:
        """
        Retrieves the current user session.
        """
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                session = self._validate_session(session)
                
                if not session.fingerprint_id:
                    session.fingerprint_id = f"temp_fp_{session.session_id[:8]}"
                    session.fingerprint_method = "temporary_fallback_python"
                    session.visitor_type = "new_visitor_fallback"
                    self.db.save_session(session)
                    logger.info(f"Applied temporary fallback fingerprint to session {session.session_id[:8]} (JS fingerprinting might be failing).")
                
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
                    return session
                
                self._update_activity(session)
                return session
        
        logger.info("No active session found or current session is invalid. Creating a new guest session.")
        new_session = self._create_guest_session()
        if not new_session.fingerprint_id:
            new_session.fingerprint_id = f"temp_fp_new_{new_session.session_id[:8]}"
            new_session.fingerprint_method = "temporary_fallback_python_new_session"
            new_session.visitor_type = "new_visitor_fallback"
            self.db.save_session(new_session)
            logger.info(f"Applied temporary fingerprint to NEW session {new_session.session_id[:8]}")
        
        limit_check = self.question_limits.is_within_limits(new_session)
        
        return self._validate_session(new_session)

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """
        Authenticates a user against the configured WordPress site using JWT.
        """
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service (WordPress URL) is not configured in secrets.")
            return None
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
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WordPress authentication successful for '{clean_username}'.")
                
                current_session = self.get_session() 
                
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
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False

                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
                try:
                    self.db.save_session(current_session)
                    logger.info(f"Authenticated session {current_session.session_id[:8]} saved as REGISTERED_USER.")
                    st.session_state.current_session_id = current_session.session_id
                    st.success(f"🎉 Welcome back, {current_session.full_name}!")
                    return current_session
                except Exception as e:
                    logger.error(f"Failed to save authenticated session {current_session.session_id[:8]} to DB: {e}", exc_info=True)
                    st.error("Authentication successful, but failed to save session details. Please try again.")
                    return None
                
            else:
                error_message = f"Invalid username or password (HTTP Status Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except json.JSONDecodeError:
                    pass
                
                st.error(error_message)
                logger.warning(f"WordPress authentication failed for '{clean_username}': {error_message}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during authentication. Please check your internet connection or try again later.")
            logger.error(f"WordPress authentication network exception for '{clean_username}': {e}", exc_info=True)
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """
        Handles the entire AI response generation process for a user prompt.
        """
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "You are sending requests too quickly. Please wait a moment and try again.", "success": False, "source": "Rate Limiter"}

        session = self._validate_session(session)
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            return {
                "content": limit_check.get('message', 'Access restricted due to usage policy.'),
                "success": False,
                "banned": True,
                "reason": limit_check.get('reason'),
                "time_remaining": limit_check.get('time_remaining')
            }
        
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session)
            self.db.save_session(session)
            return {
                "content": "Unusual activity detected. Your access has been temporarily restricted. Please try again later.",
                "success": False,
                "evasion_penalty": True,
                "penalty_hours": penalty_hours
            }

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation_result = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation_result and moderation_result.get("flagged"):
            session.messages.append({"role": "user", "content": sanitized_prompt, "timestamp": datetime.now().isoformat()})
            session.messages.append({"role": "assistant", "content": moderation_result["message"], "source": "Content Safety Policy", "timestamp": datetime.now().isoformat()})
            self.db.save_session(session)
            return {
                "content": moderation_result["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        self.question_limits.record_question(session)

        ai_response = self.ai.get_response(sanitized_prompt, session.messages)
        
        session.messages.append({
            "role": "user", 
            "content": sanitized_prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        response_message = {
            "role": "assistant",
            "content": ai_response.get("content", "No response generated."),
            "source": ai_response.get("source", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        for flag in ["used_search", "used_pinecone", "has_citations", "has_inline_citations", "safety_override"]:
            if ai_response.get(flag):
                response_message[flag] = True
            
        session.messages.append(response_message)
        session.messages = session.messages[-100:]
        
        self.db.save_session(session)
        return ai_response

    def clear_chat_history(self, session: UserSession):
        """Clears the chat history for the current session."""
        session = self._validate_session(session)
        session.messages = []
        session.timeout_saved_to_crm = False
        self._update_activity(session)

    def end_session(self, session: UserSession):
        """
        Ends the current user session.
        """
        session = self._validate_session(session)
        
        if (session.user_type.value == UserType.REGISTERED_USER.value and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            logger.info(f"Performing CRM save for manual sign-out of session {session.session_id[:8]}.")
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        session.active = False
        try:
            self.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to mark session {session.session_id[:8]} as inactive in DB during end_session: {e}", exc_info=True)
        
        keys_to_clear = ['current_session_id', 'page', 'verification_stage', 'verification_email']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        logger.info(f"Session {session.session_id[:8]} ended. Redirecting to welcome page.")

    def manual_save_to_crm(self, session: UserSession):
        """Allows registered users to manually save their current chat transcript to Zoho CRM."""
        session = self._validate_session(session)
        
        if (session.user_type.value == UserType.REGISTERED_USER.value and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            with st.spinner("💾 Saving chat to Zoho CRM..."):
                success = self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            if success:
                st.success("✅ Chat manually saved to Zoho CRM!")
                self._update_activity(session)
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
            # Use a properly structured form with unique key and security fixes
            with st.form("login_form", clear_on_submit=True):
                st.markdown("### 🔐 Sign In to Your Account")
                username = st.text_input("Username or Email", help="Enter your WordPress username or email.")
                password = st.text_input("Password", type="password", help="Enter your WordPress password.")
                
                # Add some spacing
                st.markdown("")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submit_button = st.form_submit_button("🔐 Sign In", use_container_width=True)
                
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
            
            # Add registration link
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("👤 Start as Guest", use_container_width=True):
                st.session_state.page = "chat" # Change application page
                st.rerun() # Force a rerun to switch to the chat interface

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    """Renders the application's sidebar, displaying session information, user status, and action buttons."""
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
                st.progress(session.total_question_count / 20, text="Tier 1 (up to 20 questions)")
            else:
                st.progress((session.total_question_count - 20) / 20, text="Tier 2 (21-40 questions)")
            
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Daily questions have reset!")
            
        else: # UserType.GUEST.value
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10 questions/day.")
        
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        
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

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """
    Renders the email verification dialog for guest users who have hit their
    initial question limit (4 questions).
    """
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
            
def render_chat_interface(session_manager: SessionManager, session: UserSession):
    """Renders the main chat interface."""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting.")
    
    global_message_channel_error_handler()

    if not session.fingerprint_id or session.fingerprint_method == "temporary_fallback_python":
        fingerprint_js_code = session_manager.fingerprinting.generate_fingerprint_component(session.session_id)
        fp_result = st_javascript(fingerprint_js_code, key=f"fifi_fp_init_{session.session_id[:8]}")
        
        if fp_result:
            extracted_fp_data = session_manager.fingerprinting.extract_fingerprint_from_result(fp_result)
            if extracted_fp_data.get('fingerprint_method') not in ["fallback", "canvas_blocked", "webgl_blocked", "audio_blocked"]:
                session_manager.apply_fingerprinting(session, extracted_fp_data)
                st.rerun()
            else:
                logger.debug(f"JS Fingerprint returned a fallback/blocked result: {extracted_fp_data.get('fingerprint_method')}. Retaining Python fallback if present.")
        else:
            logger.debug(f"Fingerprinting component for session {session.session_id[:8]} did not return result on this run. Will try again.")

    if session.user_type.value == UserType.REGISTERED_USER.value:
        try:
            render_browser_close_detection_simplified(session.session_id)
        except Exception as e:
            logger.error(f"Failed to render browser close detection JS for {session.session_id[:8]}: {e}", exc_info=True)

    if session.user_type.value == UserType.REGISTERED_USER.value:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id)
        except Exception as e:
            logger.error(f"15-minute timer component execution failed: {e}", exc_info=True)
        
        if timer_result:
            if handle_timer_event(timer_result, session_manager, session):
                st.rerun()

    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed']:
        if limit_check.get('reason') == 'guest_limit':
            render_email_verification_dialog(session_manager, session)
            return
        else:
            return

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
# MAIN APPLICATION FLOW
# =============================================================================

def ensure_initialization():
    """
    Ensures all necessary application components and managers are initialized and
    stored in Streamlit's session state.
    """
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence...")
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            # DatabaseManager initialization handles its own persistence and falls back to in-memory on failure.
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager
            
            if db_manager.conn is None: # Check if connection is None after DatabaseManager.__init__
                logger.error("Database connection failed, operating in non-persistent in-memory mode. Data will not persist across reruns or browser closes.")
                st.error("⚠️ Database connection failed. Operating in limited, non-persistent mode. Please contact support.")

            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()
            
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager

            st.session_state.initialized = True
            logger.info("✅ Application initialized successfully with all features.")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup and initialization. The application cannot run.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"CRITICAL: Application initialization failed: {e}", exc_info=True)
            return False
    
    return True

def main():
    """
    Main entry point for the Streamlit application.
    """
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    global_message_channel_error_handler()

    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear_state_btn", help="Clears all session state and restarts the app. Use only for development or if the app is stuck."):
        logger.warning("User initiated 'Fresh Start (Dev)' button action.")
        st.session_state.clear()
        st.rerun()

    if not ensure_initialization():
        st.stop()

    handle_emergency_save_requests_from_query()

    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("Fatal: Session Manager failed to initialize or retrieve. The application cannot proceed.")
        logger.critical("Fatal: Session Manager not found in st.session_state after initialization attempt. App stopping.")
        st.stop()

    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session() 
        
        if session and session.active:
            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state['page']
            st.rerun()

if __name__ == "__main__":
    main()
