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
import base64
from pathlib import Path
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
from streamlit_javascript import st_javascript # Keep st_javascript for activity tracker
import asyncio # Import asyncio

# Import StreamlitSecretNotFoundError for robust secret handling
from streamlit.errors import StreamlitSecretNotFoundError

# Import production_config
from production_config import (
    DAILY_RESET_WINDOW_HOURS, SESSION_TIMEOUT_MINUTES, FINGERPRINT_TIMEOUT_SECONDS,
    TIER_1_BAN_HOURS, TIER_2_BAN_HOURS, EMAIL_VERIFIED_BAN_HOURS,
    GUEST_QUESTION_LIMIT, EMAIL_VERIFIED_QUESTION_LIMIT, REGISTERED_USER_QUESTION_LIMIT, REGISTERED_USER_TIER_1_LIMIT,
    RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
    MAX_MESSAGE_LENGTH, MAX_PDF_MESSAGES, MAX_FINGERPRINT_CACHE_SIZE, MAX_RATE_LIMIT_TRACKING, MAX_ERROR_HISTORY,
    CRM_SAVE_MIN_QUESTIONS, EVASION_BAN_HOURS,
    FASTAPI_EMERGENCY_SAVE_URL, FASTAPI_EMERGENCY_SAVE_TIMEOUT,
    DAILY_RESET_WINDOW, SESSION_TIMEOUT_DELTA,
    FASTAPI_FINGERPRINT_URL # Import FastAPI fingerprint URL
)

# =============================================================================
# AVATAR LOADING
# =============================================================================

# Helper function to load and Base64-encode images for stateless deployment
@st.cache_data
def get_image_as_base64(file_path):
    """Loads an image file and returns it as a Base64 encoded string."""
    try:
        path = Path(file_path)
        with path.open("rb") as f:
            data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        logger.error(f"Avatar image not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None

# Load images once using the helper function
# ASSUMPTION: You have an 'assets' folder with these images next to fifi.py
FIFI_AVATAR_B64 = get_image_as_base64("assets/fifi-avatar.png")
USER_AVATAR_B64 = get_image_as_base64("assets/user-avatar.png")


# =============================================================================
# SETUP
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
        
        # Don't show overlay for fingerprinting operations
        if loading_message.startswith("Setting up device recognition..."): # Changed to startsWith
            return False
        
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
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #ff6b6b;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            "></div>
            <h4 style="color: #333; margin-bottom: 0.5rem;">FiFi, AI sourcing assistant</h4>
            <p style="color: #666; margin: 0;">{loading_message}</p>
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
        # Helper function to safely get secrets, prioritizing environment variables
        def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
            # Priority 1: Check environment variables (e.g., Google Cloud Run)
            env_val = os.getenv(key)
            if env_val is not None:
                return env_val

            # Priority 2: Check Streamlit secrets (e.g., local .streamlit/secrets.toml)
            try:
                # Catch StreamlitSecretNotFoundError if secrets.toml is not found
                secrets_val = st.secrets.get(key)
                if secrets_val is not None: # st.secrets.get() returns None if key not found
                    return secrets_val
            except StreamlitSecretNotFoundError:
                logger.debug(f"StreamlitSecretNotFoundError caught for key '{key}'. This is expected in Cloud Run.")
            except Exception as e:
                logger.warning(f"Unexpected error when accessing st.secrets for key '{key}': {e}")
            
            # Priority 3: Fallback to the provided default value
            return default

        # Apply the _get_secret helper to all secret loads
        self.JWT_SECRET = _get_secret("JWT_SECRET", "default-secret")
        self.OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
        self.TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")
        self.PINECONE_API_KEY = _get_secret("PINECONE_API_KEY")
        self.PINECONE_ASSISTANT_NAME = _get_secret("PINECONE_ASSISTANT_NAME", "my-chat-assistant")
        self.WORDPRESS_URL = self._validate_url(_get_secret("WORDPRESS_URL", ""))
        self.SQLITE_CLOUD_CONNECTION = _get_secret("SQLITE_CLOUD_CONNECTION")
        self.ZOHO_CLIENT_ID = _get_secret("ZOHO_CLIENT_ID")
        self.ZOHO_CLIENT_SECRET = _get_secret("ZOHO_CLIENT_SECRET")
        self.ZOHO_REFRESH_TOKEN = _get_secret("ZOHO_REFRESH_TOKEN")
        self.ZOHO_ENABLED = all([self.ZOHO_CLIENT_ID, self.ZOHO_CLIENT_SECRET, self.ZOHO_REFRESH_TOKEN])
        self.SUPABASE_URL = _get_secret("SUPABASE_URL")
        self.SUPABASE_ANON_KEY = _get_secret("SUPABASE_ANON_KEY")
        self.SUPABASE_ENABLED = all([SUPABASE_AVAILABLE, self.SUPABASE_URL, self.SUPABASE_ANON_KEY])

        # WooCommerce Configuration
        self.WOOCOMMERCE_URL = self._validate_url(_get_secret("WOOCOMMERCE_URL", ""))
        self.WOOCOMMERCE_CONSUMER_KEY = _get_secret("WOOCOMMERCE_CONSUMER_KEY")
        self.WOOCOMMERCE_CONSUMER_SECRET = _get_secret("WOOCOMMERCE_CONSUMER_SECRET")
        self.WOOCOMMERCE_ENABLED = all([
            self.WOOCOMMERCE_URL, 
            self.WOOCOMMERCE_CONSUMER_KEY, 
            self.WOOCOMMERCE_CONSUMER_SECRET
        ])

    def _validate_url(self, url: str) -> str:
        if url and not url.startswith(('http://', 'https://')):
            logger.warning(f"Invalid URL format for {url}. Disabling feature.")
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
        self.MAX_ERROR_HISTORY_SIZE = MAX_ERROR_HISTORY

    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        error_str = str(error).lower()
        
        # Check if the error is a requests.exceptions.ReadTimeout for Zoho token
        if isinstance(error, requests.exceptions.ReadTimeout) and component == "ZohoCRMManager" and operation == "get_access_token":
             severity, message = ErrorSeverity.LOW, "token request timed out but may still be processing."
        elif "timeout" in error_str:
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
            component=component, operation=operation, error_type=type(error).__name__,
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
        if len(self.error_history) > self.MAX_ERROR_HISTORY_SIZE: 
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

    # NEW: Fields for Re-verification (to address the security concern)
    reverification_pending: bool = False
    pending_user_type: Optional[UserType] = None
    pending_email: Optional[str] = None
    pending_full_name: Optional[str] = None
    pending_zoho_contact_id: Optional[str] = None
    pending_wp_token: Optional[str] = None

    # NEW: Flag to allow guest questions after declining a recognized email
    declined_recognized_email_at: Optional[datetime] = None

    timeout_detected_at: Optional[datetime] = None
    timeout_reason: Optional[str] = None

    # NEW: Tier cycle tracking
    current_tier_cycle_id: Optional[str] = None
    tier1_completed_in_cycle: bool = False
    tier_cycle_started_at: Optional[datetime] = None

    # NEW: Login method tracking
    login_method: Optional[str] = None
    is_degraded_login: bool = False
    degraded_login_timestamp: Optional[datetime] = None


class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.conn = None
        self._connection_string = connection_string
        self._last_health_check = None
        self._health_check_interval = timedelta(minutes=5)
        self._max_reconnect_attempts = 3
        logger.info("🔄 INITIALIZING DATABASE MANAGER")
        
        if connection_string and SQLITECLOUD_AVAILABLE:
            self.conn, self.db_type = self._try_sqlite_cloud(connection_string)
        
        if not self.conn:
            self.conn, self.db_type = self._try_local_sqlite()
        
        if not self.conn:
            logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED. FALLING BACK TO NON-PERSISTENT IN-MEMORY STORAGE.")
            self.db_type = "memory"
            self.local_sessions = {}
        
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
                    display_message_offset INTEGER DEFAULT 0,
                    reverification_pending INTEGER DEFAULT 0,
                    pending_user_type TEXT,
                    pending_email TEXT,
                    pending_full_name TEXT,
                    pending_zoho_contact_id TEXT,
                    pending_wp_token TEXT,
                    declined_recognized_email_at TEXT,
                    timeout_detected_at TEXT,
                    timeout_reason TEXT,
                    current_tier_cycle_id TEXT,
                    tier1_completed_in_cycle INTEGER DEFAULT 0,
                    tier_cycle_started_at TEXT,
                    login_method TEXT,
                    is_degraded_login INTEGER DEFAULT 0,
                    degraded_login_timestamp TEXT
                )
            ''')
            
            # Add new columns if they don't exist (for existing databases)
            new_columns = [
                ("display_message_offset", "INTEGER DEFAULT 0"),
                ("reverification_pending", "INTEGER DEFAULT 0"),
                ("pending_user_type", "TEXT"),
                ("pending_email", "TEXT"),
                ("pending_full_name", "TEXT"),
                ("pending_zoho_contact_id", "TEXT"),
                ("pending_wp_token", "TEXT"),
                ("declined_recognized_email_at", "TEXT"),
                ("timeout_detected_at", "TEXT"),
                ("timeout_reason", "TEXT"),
                ("current_tier_cycle_id", "TEXT"),
                ("tier1_completed_in_cycle", "INTEGER DEFAULT 0"),
                ("tier_cycle_started_at", "TEXT"),
                ("login_method", "TEXT"),
                ("is_degraded_login", "INTEGER DEFAULT 0"),
                ("degraded_login_timestamp", "TEXT"),
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
    
    def _reconnect(self):
        """Reconnect with exponential backoff"""
        for attempt in range(self._max_reconnect_attempts):
            try:
                if self.conn:
                    try:
                        self.conn.close()
                    except:
                        pass
                
                if self._connection_string and SQLITECLOUD_AVAILABLE:
                    self.conn = sqlitecloud.connect(self._connection_string)
                    self.db_type = "cloud"
                else:
                    self.conn = sqlite3.connect("fifi_sessions_v2.db", check_same_thread=False)
                    self.db_type = "file"
                
                self._init_complete_database()
                logger.info(f"Database reconnected on attempt {attempt + 1}")
                return
            except Exception as e:
                wait_time = 2 ** attempt
                logger.error(f"Reconnect attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
                time.sleep(wait_time)
        
        logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED. FALLING BACK TO NON-PERSISTENT IN-MEMORY STORAGE.")
        self.db_type = "memory"
        self.local_sessions = {}

    def _ensure_connection_healthy(self):
        """Ensure database connection is healthy, reconnect if needed"""
        if not self._check_connection_health():
            logger.warning("Database connection unhealthy, attempting reconnection...")
            self._reconnect()
            if not self.conn:
                if not hasattr(self, 'local_sessions'):
                    self.local_sessions = {}

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with SQLite Cloud compatibility and connection health check"""
        logger.debug(f"💾 SAVING SESSION TO DB: {session.session_id[:8]} | user_type={session.user_type.value} | email={session.email} | messages={len(session.messages)} | fp_id={session.fingerprint_id[:8] if session.fingerprint_id else 'None'} | active={session.active}")
        
        self._ensure_connection_healthy()
        
        if self.db_type == "memory":
            self.local_sessions[session.session_id] = copy.deepcopy(session)
            logger.debug(f"Saved session {session.session_id[:8]} to in-memory.")
            return
        
        try:
            if hasattr(self.conn, 'row_factory'): 
                self.conn.row_factory = None
            
            if not isinstance(session.messages, list):
                logger.warning(f"Invalid messages field for session {session.session_id[:8]}, resetting to empty list")
                session.messages = []
            
            try:
                json_messages = json.dumps(session.messages)
                json_emails_used = json.dumps(session.email_addresses_used)
            except (TypeError, ValueError) as e:
                logger.error(f"Session data not JSON serializable for {session.session_id[:8]}: {e}. Resetting data to empty lists.")
                json_messages = "[]"
                json_emails_used = "[]"
                session.messages = []
                session.email_addresses_used = []
            
            last_activity_iso = session.last_activity.isoformat() if session.last_activity else None
            declined_recognized_email_at_iso = session.declined_recognized_email_at.isoformat() if session.declined_recognized_email_at else None
            timeout_detected_at_iso = session.timeout_detected_at.isoformat() if session.timeout_detected_at else None
            tier_cycle_started_at_iso = session.tier_cycle_started_at.isoformat() if session.tier_cycle_started_at else None
            degraded_login_timestamp_iso = session.degraded_login_timestamp.isoformat() if session.degraded_login_timestamp else None

            self.conn.execute(
                '''REPLACE INTO sessions (session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason, current_tier_cycle_id, tier1_completed_in_cycle, tier_cycle_started_at, login_method, is_degraded_login, degraded_login_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (session.session_id, session.user_type.value, session.email, session.full_name,
                 session.zoho_contact_id, session.created_at.isoformat(),
                 last_activity_iso, json_messages, int(session.active),
                 session.wp_token, int(session.timeout_saved_to_crm), session.fingerprint_id,
                 session.fingerprint_method, session.visitor_type,
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
                 declined_recognized_email_at_iso,
                 timeout_detected_at_iso,
                 session.timeout_reason,
                 session.current_tier_cycle_id,
                 int(session.tier1_completed_in_cycle),
                 tier_cycle_started_at_iso,
                 session.login_method,
                 int(session.is_degraded_login),
                 degraded_login_timestamp_iso
                 ))
            self.conn.commit()
            
            logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}, active={session.active}, rev_pending={session.reverification_pending}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id[:8]}: {e}", exc_info=True)
            if not hasattr(self, 'local_sessions'):
                self.local_sessions = {}
            self.local_sessions[session.session_id] = copy.deepcopy(session)
            logger.info(f"Fallback: Saved session {session.session_id[:8]} to in-memory storage")
            raise

    def _load_any_session(self, session_id: str) -> Optional[UserSession]:
            """Loads a session by its ID, regardless of its 'active' status. For internal historical lookups."""
            self._ensure_connection_healthy()
            if self.db_type == "memory":
                session = self.local_sessions.get(session_id)
                # Ensure backward compatibility for in-memory sessions
                if session:
                    if isinstance(session.user_type, str):
                        try: session.user_type = UserType(session.user_type)
                        except ValueError: session.user_type = UserType.GUEST
                    if not hasattr(session, 'display_message_offset'): session.display_message_offset = 0
                    if not hasattr(session, 'reverification_pending'):
                        session.reverification_pending = False; session.pending_user_type = None; session.pending_email = None; session.pending_full_name = None; session.pending_zoho_contact_id = None; session.pending_wp_token = None
                    if not hasattr(session, 'declined_recognized_email_at'): session.declined_recognized_email_at = None
                    if not hasattr(session, 'timeout_detected_at'): session.timeout_detected_at = None
                    if not hasattr(session, 'timeout_reason'): session.timeout_reason = None
                    if not hasattr(session, 'current_tier_cycle_id'): session.current_tier_cycle_id = None
                    if not hasattr(session, 'tier1_completed_in_cycle'): session.tier1_completed_in_cycle = False
                    if not hasattr(session, 'tier_cycle_started_at'): session.tier_cycle_started_at = None
                    if not hasattr(session, 'login_method'): session.login_method = None
                    if not hasattr(session, 'is_degraded_login'): session.is_degraded_login = False
                    if not hasattr(session, 'degraded_login_timestamp'): session.degraded_login_timestamp = None
                return copy.deepcopy(session)

            try:
                if self.db_type == "file": # Only set row_factory for local sqlite
                    self.conn.row_factory = sqlite3.Row
                else: # For sqlitecloud, use default tuple
                    self.conn.row_factory = None

                cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason, current_tier_cycle_id, tier1_completed_in_cycle, tier_cycle_started_at, login_method, is_degraded_login, degraded_login_timestamp FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.debug(f"No session (active or inactive) found for ID {session_id[:8]}.")
                    return None
                
                # Reset row factory if we set it
                if self.db_type == "file":
                    self.conn.row_factory = None

                # Safely create the UserSession object from the row data
                loaded_display_message_offset = row[31] if len(row) > 31 else 0
                loaded_reverification_pending = bool(row[32]) if len(row) > 32 else False
                loaded_pending_user_type = UserType(row[33]) if len(row) > 33 and row[33] else None
                loaded_pending_email = row[34] if len(row) > 34 else None
                loaded_pending_full_name = row[35] if len(row) > 35 else None
                loaded_pending_zoho_contact_id = row[36] if len(row) > 36 else None
                loaded_pending_wp_token = row[37] if len(row) > 37 else None
                loaded_declined_recognized_email_at = datetime.fromisoformat(row[38]) if len(row) > 38 and row[38] else None
                loaded_timeout_detected_at = datetime.fromisoformat(row[39]) if len(row) > 39 and row[39] else None
                loaded_timeout_reason = row[40] if len(row) > 40 else None
                loaded_current_tier_cycle_id = row[41] if len(row) > 41 else None
                loaded_tier1_completed_in_cycle = bool(row[42]) if len(row) > 42 else False
                loaded_tier_cycle_started_at = datetime.fromisoformat(row[43]) if len(row) > 43 and row[43] else None
                loaded_login_method = row[44] if len(row) > 44 else None
                loaded_is_degraded_login = bool(row[45]) if len(row) > 45 else False
                loaded_degraded_login_timestamp = datetime.fromisoformat(row[46]) if len(row) > 46 and row[46] else None
                loaded_last_activity = datetime.fromisoformat(row[6]) if row[6] else None

                user_session = UserSession(
                    session_id=row[0], user_type=UserType(row[1]) if row[1] else UserType.GUEST, email=row[2], full_name=row[3],
                    zoho_contact_id=row[4], created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                    last_activity=loaded_last_activity, messages=safe_json_loads(row[7], default_value=[]), active=bool(row[8]),
                    wp_token=row[9], timeout_saved_to_crm=bool(row[10]), fingerprint_id=row[11], fingerprint_method=row[12],
                    visitor_type=row[13] or 'new_visitor', daily_question_count=row[14] or 0, total_question_count=row[15] or 0,
                    last_question_time=datetime.fromisoformat(row[16]) if row[16] else None, question_limit_reached=bool(row[17]),
                    ban_status=BanStatus(row[18]) if row[18] else BanStatus.NONE,
                    ban_start_time=datetime.fromisoformat(row[19]) if row[19] else None,
                    ban_end_time=datetime.fromisoformat(row[20]) if row[20] else None, ban_reason=row[21],
                    evasion_count=row[22] or 0, current_penalty_hours=row[23] or 0, escalation_level=row[24] or 0,
                    email_addresses_used=safe_json_loads(row[25], default_value=[]), email_switches_count=row[26] or 0,
                    browser_privacy_level=row[27], registration_prompted=bool(row[28]), registration_link_clicked=bool(row[29]),
                    recognition_response=row[30], display_message_offset=loaded_display_message_offset,
                    reverification_pending=loaded_reverification_pending, pending_user_type=loaded_pending_user_type,
                    pending_email=loaded_pending_email, pending_full_name=loaded_pending_full_name,
                    pending_zoho_contact_id=loaded_pending_zoho_contact_id, pending_wp_token=loaded_pending_wp_token,
                    declined_recognized_email_at=loaded_declined_recognized_email_at,
                    timeout_detected_at=loaded_timeout_detected_at, timeout_reason=loaded_timeout_reason,
                    current_tier_cycle_id=loaded_current_tier_cycle_id, tier1_completed_in_cycle=loaded_tier1_completed_in_cycle,
                    tier_cycle_started_at=loaded_tier_cycle_started_at, login_method=loaded_login_method,
                    is_degraded_login=loaded_is_degraded_login, degraded_login_timestamp=loaded_degraded_login_timestamp
                )
                
                logger.debug(f"Successfully loaded any session {session_id[:8]}: user_type={user_session.user_type.value}, messages={len(user_session.messages)}, active={user_session.active}, rev_pending={user_session.reverification_pending}")
                return user_session
                
            except Exception as e:
                logger.error(f"Failed to _load_any_session for {session_id[:8]}: {e}", exc_info=True)
                return None


    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility and connection health check (only active sessions)"""
        self._ensure_connection_healthy()

        if self.db_type == "memory":
            session = self.local_sessions.get(session_id)
            if session and not session.active: return None # Filter inactive in memory too
            if session: # Apply backward compatibility for in-memory sessions too
                if isinstance(session.user_type, str):
                    try: session.user_type = UserType(session.user_type)
                    except ValueError: session.user_type = UserType.GUEST
                if not hasattr(session, 'display_message_offset'): session.display_message_offset = 0
                if not hasattr(session, 'reverification_pending'):
                    session.reverification_pending = False; session.pending_user_type = None; session.pending_email = None; session.pending_full_name = None; session.pending_zoho_contact_id = None; session.pending_wp_token = None
                if not hasattr(session, 'declined_recognized_email_at'): session.declined_recognized_email_at = None
                if not hasattr(session, 'timeout_detected_at'): session.timeout_detected_at = None
                if not hasattr(session, 'timeout_reason'): session.timeout_reason = None
                if not hasattr(session, 'current_tier_cycle_id'): session.current_tier_cycle_id = None
                if not hasattr(session, 'tier1_completed_in_cycle'): session.tier1_completed_in_cycle = False
                if not hasattr(session, 'tier_cycle_started_at'): session.tier_cycle_started_at = None
                if not hasattr(session, 'login_method'): session.login_method = None
                if not hasattr(session, 'is_degraded_login'): session.is_degraded_login = False
                if not hasattr(session, 'degraded_login_timestamp'): session.degraded_login_timestamp = None
            return copy.deepcopy(session)
        
        try:
            if self.db_type == "file": # Only set row_factory for local sqlite
                self.conn.row_factory = sqlite3.Row
            else: # For sqlitecloud, use default tuple
                self.conn.row_factory = None

            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason, current_tier_cycle_id, tier1_completed_in_cycle, tier_cycle_started_at, login_method, is_degraded_login, degraded_login_timestamp FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
            row = cursor.fetchone()
            
            if not row: 
                logger.debug(f"No active session found for ID {session_id[:8]}.")
                return None
            
            # Reset row factory if we set it
            if self.db_type == "file":
                self.conn.row_factory = None

            # Safely create the UserSession object from the row data
            loaded_display_message_offset = row[31] if len(row) > 31 else 0
            loaded_reverification_pending = bool(row[32]) if len(row) > 32 else False
            loaded_pending_user_type = UserType(row[33]) if len(row) > 33 and row[33] else None
            loaded_pending_email = row[34] if len(row) > 34 else None
            loaded_pending_full_name = row[35] if len(row) > 35 else None
            loaded_pending_zoho_contact_id = row[36] if len(row) > 36 else None
            loaded_pending_wp_token = row[37] if len(row) > 37 else None
            loaded_declined_recognized_email_at = datetime.fromisoformat(row[38]) if len(row) > 38 and row[38] else None
            loaded_timeout_detected_at = datetime.fromisoformat(row[39]) if len(row) > 39 and row[39] else None
            loaded_timeout_reason = row[40] if len(row) > 40 else None
            loaded_current_tier_cycle_id = row[41] if len(row) > 41 else None
            loaded_tier1_completed_in_cycle = bool(row[42]) if len(row) > 42 else False
            loaded_tier_cycle_started_at = datetime.fromisoformat(row[43]) if row[43] else None
            loaded_login_method = row[44] if len(row) > 44 else None
            loaded_is_degraded_login = bool(row[45]) if len(row) > 45 else False
            loaded_degraded_login_timestamp = datetime.fromisoformat(row[46]) if len(row) > 46 and row[46] else None
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
                display_message_offset=loaded_display_message_offset,
                reverification_pending=loaded_reverification_pending,
                pending_user_type=loaded_pending_user_type,
                pending_email=loaded_pending_email,
                pending_full_name=loaded_pending_full_name,
                pending_zoho_contact_id=loaded_pending_zoho_contact_id,
                pending_wp_token=loaded_pending_wp_token,
                declined_recognized_email_at=loaded_declined_recognized_email_at,
                timeout_detected_at=loaded_timeout_detected_at,
                timeout_reason=loaded_timeout_reason,
                current_tier_cycle_id=loaded_current_tier_cycle_id,
                tier1_completed_in_cycle=loaded_tier1_completed_in_cycle,
                tier_cycle_started_at=loaded_tier_cycle_started_at,
                login_method=loaded_login_method,
                is_degraded_login=loaded_is_degraded_login,
                degraded_login_timestamp=loaded_degraded_login_timestamp
            )
            
            logger.debug(f"Successfully loaded active session {session_id[:8]}: user_type={user_session.user_type.value}, messages={len(user_session.messages)}, active={user_session.active}, rev_pending={user_session.reverification_pending}")
            return user_session
                
        except Exception as e:
            logger.error(f"Failed to create UserSession object from row for active session {session_id[:8]}: {e}", exc_info=True)
            logger.error(f"Problematic row data (truncated): {str(row)[:200]}")
            return None
                

        except Exception as e:
            logger.error(f"Failed to load active session {session_id[:8]}: {e}", exc_info=True)
            return None

    @handle_api_errors("Database", "Find by Fingerprint")
    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint_id, including inactive ones."""
        logger.debug(f"🔍 SEARCHING FOR FINGERPRINT (including inactive): {fingerprint_id[:8]}...")
        self._ensure_connection_healthy()

        if self.db_type == "memory":
            sessions = [copy.deepcopy(s) for s in self.local_sessions.values() if s.fingerprint_id == fingerprint_id]
            for session in sessions: # Apply backward compatibility for in-memory sessions
                if not hasattr(session, 'display_message_offset'): session.display_message_offset = 0
                if not hasattr(session, 'reverification_pending'):
                    session.reverification_pending = False; session.pending_user_type = None; session.pending_email = None; session.pending_full_name = None; session.pending_zoho_contact_id = None; session.pending_wp_token = None
                if not hasattr(session, 'declined_recognized_email_at'): session.declined_recognized_email_at = None
                if not hasattr(session, 'timeout_detected_at'): session.timeout_detected_at = None
                if not hasattr(session, 'timeout_reason'): session.timeout_reason = None
                if not hasattr(session, 'current_tier_cycle_id'): session.current_tier_cycle_id = None
                if not hasattr(session, 'tier1_completed_in_cycle'): session.tier1_completed_in_cycle = False
                if not hasattr(session, 'tier_cycle_started_at'): session.tier_cycle_started_at = None
                if not hasattr(session, 'login_method'): session.login_method = None
                if not hasattr(session, 'is_degraded_login'): session.is_degraded_login = False
                if not hasattr(session, 'degraded_login_timestamp'): session.degraded_login_timestamp = None
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (MEMORY): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            return sessions
        
        try:
            if self.db_type == "file": # Only set row_factory for local sqlite
                self.conn.row_factory = sqlite3.Row
            else: # For sqlitecloud, use default tuple
                self.conn.row_factory = None

            cursor = self.conn.execute("SELECT session_id FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
            sessions = []
            for row in cursor.fetchall():
                session = self._load_any_session(row[0]) # Use _load_any_session
                if session:
                    sessions.append(session)
            
            if self.db_type == "file": # Reset row factory if we set it
                self.conn.row_factory = None
            
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (DB, incl. inactive): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            for s in sessions:
                logger.debug(f"  - {s.session_id[:8]}: type={s.user_type.value}, email={s.email}, daily_q={s.daily_question_count}, total_q={s.total_question_count}, last_activity={s.last_activity}, active={s.active}, rev_pending={s.reverification_pending}")
            return sessions
        except Exception as e:
            logger.error(f"Failed to find sessions by fingerprint '{fingerprint_id[:8]}...': {e}", exc_info=True)
            return []

    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions with the same email address, including inactive ones."""
        if not email:
            return []
            
        email_lower = email.lower()
        self._ensure_connection_healthy()
        
        if self.db_type == "memory":
            return [copy.deepcopy(s) for s in self.local_sessions.values() 
                    if s.email and s.email.lower() == email_lower]
        
        try:
            if self.db_type == "file": # Only set row_factory for local sqlite
                self.conn.row_factory = sqlite3.Row
            else: # For sqlitecloud, use default tuple
                self.conn.row_factory = None
                
            cursor = self.conn.execute(
                """SELECT session_id FROM sessions 
                   WHERE LOWER(email) = LOWER(?) 
                   ORDER BY last_activity DESC""", 
                (email,)
            )
            
            sessions = []
            for row in cursor.fetchall():
                session = self._load_any_session(row[0]) # Use _load_any_session
                if session:
                    sessions.append(session)
                    
            if self.db_type == "file": # Reset row factory if we set it
                self.conn.row_factory = None
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to find sessions by email '{email}': {e}")
            return []
    
    def cleanup_old_inactive_sessions(self):
        """Remove old inactive sessions from the database."""
        try:
            if self.db_type == "memory":
                cutoff = datetime.now() - timedelta(days=1)
                self.local_sessions = {k: v for k, v in self.local_sessions.items() if v.created_at >= cutoff}
            else:
                cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
                self.conn.execute("""
                    DELETE FROM sessions 
                    WHERE created_at < ? 
                    AND active = 0
                """, (cutoff_date,))
                self.conn.commit()
                logger.info("✅ Cleaned up old inactive sessions.")
        except Exception as e:
            logger.error(f"❌ Failed to clean up old sessions: {e}")

    # =============================================================================
    # FEATURE MANAGERS (nested within DatabaseManager as they access self.db)
    # =============================================================================

    class FingerprintingManager:
        """Manages browser fingerprinting using external HTML component file."""
        
        def __init__(self):
            self.fingerprint_cache = {}
            self.component_attempts = defaultdict(int)
            self.MAX_CACHE_SIZE = MAX_FINGERPRINT_CACHE_SIZE
            self.MAX_ATTEMPTS = MAX_RATE_LIMIT_TRACKING

        def render_fingerprint_component(self, session_id: str):
            """Renders fingerprinting component using external HTML component file."""
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                html_file_path = os.path.join(current_dir, 'fingerprint_component.html')

                logger.debug(f"🔍 Looking for fingerprint component at: {html_file_path}")

                if not os.path.exists(html_file_path):
                    logger.error(f"❌ Fingerprint component file NOT FOUND at {html_file_path}")
                    logger.info(f"📁 Current directory contents: {os.listdir(current_dir)}")
                    # No longer needs to return a fallback FP, handled by Python session object
                    return

                logger.debug(f"✅ Fingerprint component file found, reading content...")

                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                logger.debug(f"📄 Read {len(html_content)} characters from fingerprint component file")

                # Inject dynamic values into the HTML
                html_content = html_content.replace('{SESSION_ID}', json.dumps(session_id))
                html_content = html_content.replace('{FASTAPI_FINGERPRINT_URL}', json.dumps(FASTAPI_FINGERPRINT_URL))
                html_content = html_content.replace('{FINGERPRINT_TIMEOUT_SECONDS}', str(FINGERPRINT_TIMEOUT_SECONDS))

                # Render with minimal height to ensure no visual impact
                logger.debug(f"Generated fingerprint HTML (first 500 chars): {html_content[:500]}...")
                st.components.v1.html(html_content, height=1, width=1, scrolling=False) # Changed height to 1, width to 1

                logger.info(f"✅ External fingerprint component rendered for session {session_id[:8]}")

            except Exception as e:
                logger.error(f"❌ Failed to render external fingerprint component: {e}", exc_info=True)
                # No return needed here, fallback FP will be generated in Python if this fails

        def process_fingerprint_data(self, fingerprint_data: Dict[str, Any]) -> Dict[str, Any]:
            """Processes fingerprint data received from the custom component (not directly used by the new FastAPI flow, kept for compatibility)."""
            if not fingerprint_data or fingerprint_data.get('error'):
                logger.warning("Fingerprint component returned error. Using fallback.")
                return self._generate_fallback_fingerprint()
            
            fingerprint_id = fingerprint_data.get('fingerprint_id')
            fingerprint_method = fingerprint_data.get('fingerprint_method')
            privacy_level = fingerprint_data.get('privacy', 'standard')
            
            if not fingerprint_id or not fingerprint_method:
                logger.warning("Invalid fingerprint data received. Using fallback.")
                return self._generate_fallback_fingerprint()
            
            # visitor_type determination should happen during inheritance in SessionManager
            # For now, it's just a placeholder here if this method is invoked by old logic
            visitor_type = "returning_visitor" if fingerprint_id in self.fingerprint_cache else "new_visitor"
            self.fingerprint_cache[fingerprint_id] = {'last_seen': datetime.now()}
            
            if len(self.fingerprint_cache) > self.MAX_CACHE_SIZE:
                sorted_items = sorted(self.fingerprint_cache.items(), 
                                    key=lambda x: x[1].get('last_seen', datetime.min), 
                                    reverse=True)[:self.MAX_CACHE_SIZE // 2]
                self.fingerprint_cache = dict(sorted_items)

            if len(self.component_attempts) > self.MAX_ATTEMPTS:
                self.component_attempts.clear()
            
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
                error_str = str(e).lower()
                if "timeout" in error_str or "timed out" in error_str:
                    logger.warning(f"Supabase OTP request timed out for {email}, but email may have been sent: {e}")
                    st.warning("Verification code is being sent. If you don't receive it within 1 minute, please try again.")
                    return True
                else:
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
            self.question_limits = {
                UserType.GUEST.value: GUEST_QUESTION_LIMIT,
                UserType.EMAIL_VERIFIED_GUEST.value: EMAIL_VERIFIED_QUESTION_LIMIT,
                UserType.REGISTERED_USER.value: REGISTERED_USER_QUESTION_LIMIT
            }
            self.evasion_penalties = EVASION_BAN_HOURS 

        def detect_guest_email_evasion(self, session: UserSession, db_manager) -> bool:
            """
            UPDATED: Always returns False - email switching is allowed with OTP verification.
            Keeping method for compatibility and potential future use, but it's currently disabled.
            """
            logger.debug(f"Evasion detection called for session {session.session_id[:8]} but is currently disabled (always returns False).")
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
                        'message': self._get_ban_message(session, session.ban_status.value)
                    }
                else:
                    logger.info(f"Ban for session {session.session_id[:8]} expired. Resetting status and counters.")
                    
                    previous_ban_type = session.ban_status
                    
                    session.ban_status = BanStatus.NONE
                    session.ban_start_time = None
                    session.ban_end_time = None
                    session.ban_reason = None
                    session.question_limit_reached = False
                    
                    if session.user_type == UserType.REGISTERED_USER:
                        if previous_ban_type == BanStatus.TWENTY_FOUR_HOUR:
                            logger.info(f"🔄 Tier 2 ban expired for {session.session_id[:8]} - starting new tier cycle")
                            session.daily_question_count = 0
                            session.last_question_time = None
                            session.current_tier_cycle_id = str(uuid.uuid4())
                            session.tier1_completed_in_cycle = False
                            session.tier_cycle_started_at = datetime.now()
                        elif previous_ban_type == BanStatus.ONE_HOUR:
                            logger.info(f"✅ Tier 1 ban expired for {session.session_id[:8]} - can now proceed to Tier 2")
                    else:
                        session.daily_question_count = 0
                        session.last_question_time = None
            
            if session.last_question_time:
                time_since_last = datetime.now() - session.last_question_time
                if time_since_last >= DAILY_RESET_WINDOW:
                    logger.info(f"Daily question count reset for session {session.session_id[:8]} due to {DAILY_RESET_WINDOW_HOURS}-hour window expiration.")
                    session.daily_question_count = 0
                    session.question_limit_reached = False
                    if session.user_type == UserType.REGISTERED_USER:
                        session.current_tier_cycle_id = str(uuid.uuid4())
                        session.tier1_completed_in_cycle = False
                        session.tier_cycle_started_at = datetime.now()
            
            if session.user_type.value == UserType.REGISTERED_USER.value:
                if session.daily_question_count >= user_limit:
                    reason_str = 'registered_user_tier2_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str,
                        'message': self._get_ban_message(session, reason_str)
                    }
                elif session.daily_question_count >= REGISTERED_USER_TIER_1_LIMIT:
                    if session.ban_status == BanStatus.ONE_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                        time_remaining = session.ban_end_time - datetime.now()
                        return {
                            'allowed': False,
                            'reason': 'banned',
                            'ban_type': BanStatus.ONE_HOUR.value,
                            'time_remaining': time_remaining,
                            'message': self._get_ban_message(session, 'registered_user_tier1_limit')
                        }
                    elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                        return {
                            'allowed': True,
                            'tier': 1,
                            'remaining': 0,
                            'warning': f"Next question will trigger a {TIER_1_BAN_HOURS}-hour break before Tier 2."
                        }
                    else:
                        remaining = user_limit - session.daily_question_count
                        return {
                            'allowed': True,
                            'tier': 2,
                            'remaining': remaining,
                            'warning': f"Tier 2: {remaining} questions until {TIER_2_BAN_HOURS}-hour limit."
                        }
                else:
                    remaining = REGISTERED_USER_TIER_1_LIMIT - session.daily_question_count
                    return {
                        'allowed': True,
                        'tier': 1,
                        'remaining': remaining
                    }
            
            if session.user_type.value == UserType.GUEST.value:
                if session.daily_question_count >= user_limit:
                    reason_str = 'guest_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str,
                        'message': 'Please provide your email address to continue.'
                    }
            
            elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
                if session.daily_question_count >= user_limit:
                    reason_str = 'email_verified_guest_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str,
                        'message': self._get_ban_message(session, reason_str)
                    }
            
            return {'allowed': True}
        
        def record_question_and_check_ban(self, session: UserSession, session_manager: 'SessionManager') -> Dict[str, Any]:
            """Atomically check for ban trigger BEFORE recording question with proper tier cycle tracking."""
            try:
                if session.user_type == UserType.REGISTERED_USER and session.email:
                    email_sessions = session_manager.db.find_sessions_by_email(session.email)
                    now = datetime.now()
                    
                    for email_session in email_sessions:
                        if (email_session.ban_status != BanStatus.NONE and 
                            email_session.ban_end_time and 
                            email_session.ban_end_time > now):
                            session.ban_status = email_session.ban_status
                            session.ban_start_time = email_session.ban_start_time
                            session.ban_end_time = email_session.ban_end_time
                            session.ban_reason = email_session.ban_reason
                            session.question_limit_reached = True
                            
                            session.current_tier_cycle_id = email_session.current_tier_cycle_id
                            session.tier1_completed_in_cycle = email_session.tier1_completed_in_cycle
                            session.tier_cycle_started_at = email_session.tier_cycle_started_at

                            session_manager._save_session_with_retry(session)
                            
                            logger.info(f"✅ Found existing ban for email {session.email}, inherited instead of creating new: {session.ban_status.value} until {session.ban_end_time}")
                            return {"recorded": False, "ban_applied": False, "existing_ban_inherited": True, "ban_type": session.ban_status.value}
                
                if session.user_type == UserType.REGISTERED_USER:
                    if not session.current_tier_cycle_id:
                        session.current_tier_cycle_id = str(uuid.uuid4())
                        session.tier1_completed_in_cycle = False
                        session.tier_cycle_started_at = datetime.now()
                        logger.info(f"🔄 New tier cycle started for {session.session_id[:8]}: {session.current_tier_cycle_id[:8]}")
                
                ban_applied = False
                ban_type = BanStatus.NONE
                ban_reason = ""
                ban_duration_hours = 0
                
                if session.user_type == UserType.REGISTERED_USER:
                    if session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                        if not session.tier1_completed_in_cycle:
                            ban_type = BanStatus.ONE_HOUR
                            ban_reason = f"Registered user Tier 1 limit reached ({REGISTERED_USER_TIER_1_LIMIT} questions)"
                            ban_duration_hours = TIER_1_BAN_HOURS
                            ban_applied = True
                            session.tier1_completed_in_cycle = True
                            logger.info(f"🚫 Tier 1 ban triggered for {session.session_id[:8]} in cycle {session.current_tier_cycle_id[:8]}")
                        else:
                            logger.debug(f"Session {session.session_id[:8]} at question 10, but Tier 1 ban already served in cycle {session.current_tier_cycle_id[:8]}. Proceeding to Tier 2.")
                            
                    elif session.daily_question_count == REGISTERED_USER_QUESTION_LIMIT:
                        ban_type = BanStatus.TWENTY_FOUR_HOUR
                        ban_reason = f"Registered user daily limit reached ({REGISTERED_USER_QUESTION_LIMIT} questions)"
                        ban_duration_hours = TIER_2_BAN_HOURS
                        ban_applied = True
                        logger.info(f"🚫 Tier 2 ban triggered for {session.session_id[:8]} in cycle {session.current_tier_cycle_id[:8]}")
                        
                elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
                    if session.daily_question_count == EMAIL_VERIFIED_QUESTION_LIMIT:
                        ban_type = BanStatus.TWENTY_FOUR_HOUR
                        ban_reason = f"Email-verified daily limit reached ({EMAIL_VERIFIED_QUESTION_LIMIT} questions)"
                        ban_duration_hours = EMAIL_VERIFIED_BAN_HOURS
                        ban_applied = True
                
                if ban_applied:
                    self._apply_ban(session, ban_type, ban_reason)
                    logger.info(f"✅ Ban applied WITHOUT counting question: {session.session_id[:8]} -> {ban_type.value} for {ban_duration_hours}h")
                    
                    session_manager._save_session_with_retry(session)
                    
                    if session.user_type == UserType.REGISTERED_USER and session.email:
                        session_manager.sync_ban_for_registered_user(session.email, session)
                    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST and session.email:
                        session_manager.sync_email_verified_sessions(session.email, session.session_id)
                    
                    return {"recorded": False, "ban_applied": True, "ban_type": ban_type.value}
                
                session.daily_question_count += 1
                session.total_question_count += 1
                session.last_question_time = datetime.now()
                
                if (session.user_type == UserType.REGISTERED_USER or 
                    session.user_type == UserType.EMAIL_VERIFIED_GUEST) and session.email:
                    if session.user_type == UserType.REGISTERED_USER:
                        session_manager.sync_registered_user_sessions(session.email, session.session_id)
                    else:
                        session_manager.sync_email_verified_sessions(session.email, session.session_id)
                
                session_manager._save_session_with_retry(session)
                
                logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}")
                
                return {"recorded": True, "ban_applied": False}
                
            except Exception as e:
                logger.error(f"Failed to check ban and record question for {session.session_id[:8]}: {e}", exc_info=True)
                raise
        
        def _apply_ban(self, session: UserSession, ban_type: BanStatus, reason: str, start_time: Optional[datetime] = None):
            """Applies a ban to the session for a specified duration with immediate database persistence."""
            ban_hours = {
                BanStatus.ONE_HOUR.value: TIER_1_BAN_HOURS,
                BanStatus.TWENTY_FOUR_HOUR.value: TIER_2_BAN_HOURS,
                BanStatus.EVASION_BLOCK.value: session.current_penalty_hours
            }.get(ban_type.value, TIER_1_BAN_HOURS)

            session.ban_status = ban_type
            session.ban_start_time = start_time if start_time else datetime.now()
            session.ban_end_time = session.ban_start_time + timedelta(hours=ban_hours)
            session.ban_reason = reason
            session.question_limit_reached = True
        
            try:
                st.session_state.session_manager.db.save_session(session)
                logger.info(f"✅ Ban applied and saved to DB: {session.session_id[:8]} -> {ban_type.value} for {ban_hours}h")
            except Exception as e:
                logger.error(f"❌ Failed to save ban to database: {e}")

            logger.info(f"Ban applied: Type={ban_type.value}, Duration={ban_hours}h, Start={session.ban_start_time}, Reason='{reason}'.")
        
        def _get_ban_message(self, session: UserSession, ban_reason_from_limit_check: Optional[str] = None) -> str:
            """
            Provides a user-friendly message for current bans,
            now differentiating between registered user tier limits.
            """
            if session.ban_status.value == BanStatus.EVASION_BLOCK.value:
                return "Access restricted due to policy violation. Please try again later."
            elif ban_reason_from_limit_check == 'registered_user_tier1_limit' or session.ban_status.value == BanStatus.ONE_HOUR.value:
                return f"You've reached the Tier 1 limit ({REGISTERED_USER_TIER_1_LIMIT} questions). Please wait {TIER_1_BAN_HOURS} hour{'s' if TIER_1_BAN_HOURS > 1 else ''} to access Tier 2."
            elif ban_reason_from_limit_check == 'registered_user_tier2_limit' or (session.user_type.value == UserType.REGISTERED_USER.value and session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value):
                return f"Daily limit of {REGISTERED_USER_QUESTION_LIMIT} questions reached. Please retry in {TIER_2_BAN_HOURS} hours."
            elif ban_reason_from_limit_check == 'email_verified_guest_limit' or (session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value and session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value):
                return self._get_email_verified_limit_message()
            elif session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value:
                return f"Daily limit reached. Please retry in {TIER_2_BAN_HOURS} hours."
            else:
                return "Access restricted due to usage policy."
        
        def _get_email_verified_limit_message(self) -> str:
            """Specific message for email-verified guests hitting their daily limit."""
            return (f"You've reached your daily limit of {EMAIL_VERIFIED_QUESTION_LIMIT} questions. "
                    f"Your questions will reset in {EMAIL_VERIFIED_BAN_HOURS} hour{'s' if EMAIL_VERIFIED_BAN_HOURS > 1 else ''}. "
                    f"To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

# =============================================================================
# PDF EXPORTER & ZOHO CRM MANAGER (MOVED OUT OF DATABASEMANAGER)
# =============================================================================

class PDFExporter:
    """Handles generation of PDF chat transcripts."""
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        self.styles['Normal'].fontName = 'Helvetica'
        self.styles['Normal'].fontSize = 10
        self.styles['Normal'].leading = 14
        self.styles['Normal'].spaceAfter = 6
        
        self.styles.add(ParagraphStyle(
            name='ChatHeader',
            parent=self.styles['Normal'],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='UserMessage',
            parent=self.styles['Normal'],
            backColor=lightgrey,
            leftIndent=5,
            rightIndent=5,
            borderPadding=3,
            borderRadius=3,
            spaceBefore=8,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=grey,
            spaceBefore=2,
            spaceAfter=2
        ))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        """Generates a PDF of the chat transcript with size limits."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        story = []
        story.append(Paragraph("FiFi AI Chat Transcript", self.styles['ChatHeader']))
        story.append(Spacer(1, 12))

        messages_to_include = session.messages[-MAX_PDF_MESSAGES:]
        
        if len(session.messages) > MAX_PDF_MESSAGES:
            story.append(Paragraph(f"<i>[Note: Only the last {MAX_PDF_MESSAGES} messages are included. Total conversation had {len(session.messages)} messages.]</i>", self.styles['Caption']))
            story.append(Spacer(1, 8))

        for msg in messages_to_include:
            role = str(msg.get('role', 'unknown')).capitalize()
            content = str(msg.get('content', ''))
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('**', '<b>').replace('__', '<b>')
            content = content.replace('*', '<i>').replace('_', '<i>')
            content = html.escape(content)

            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
            
            if msg.get("source"):
                story.append(Paragraph(f"<i>Source: {msg['source']}</i>", self.styles['Caption']))
            
            if style.spaceAfter is None:
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
                "se_module": "Contacts"
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
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                    logger.info(f"Successfully added note to Zoho contact {contact_id}")
                    return True
                else:
                    logger.error("Failed to add note to Zoho contact.")
                    if attempt < (2 - 1):
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
        
        return False

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """Synchronously saves the chat transcript to Zoho CRM."""
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        
        if (session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] or 
            not session.email or 
            len(session.messages) < CRM_SAVE_MIN_QUESTIONS or
            not self.config.ZOHO_ENABLED):
            logger.info(f"ZOHO SAVE SKIPPED: Not eligible. (UserType: {session.user_type.value}, Email: {bool(session.email)}, Messages: {len(session.messages)}, Zoho Enabled: {self.config.ZOHO_ENABLED})")
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
                if attempt_note < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries for note addition reached. Aborting save.")
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


class WooCommerceManager:
    """Manages WooCommerce integration for order retrieval."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = f"{config.WOOCOMMERCE_URL}/wp-json/wc/v3"
        self.auth = None
        
        if self.config.WOOCOMMERCE_ENABLED:
            from requests.auth import HTTPBasicAuth
            self.auth = HTTPBasicAuth(
                self.config.WOOCOMMERCE_CONSUMER_KEY,
                self.config.WOOCOMMERCE_CONSUMER_SECRET
            )
            logger.info("✅ WooCommerce manager initialized")
        else:
            logger.warning("⚠️ WooCommerce not enabled - missing configuration")
    
    @handle_api_errors("12Taste Order Status", "Get Order", show_to_user=True)
    async def get_order(self, order_id: str, customer_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a single order by ID from WooCommerce, optionally filtered by customer_id."""
        if not self.config.WOOCOMMERCE_ENABLED:
            return None
        
        try:
            clean_order_id = str(order_id).replace('#', '').strip()
            params = {}
            if customer_id:
                params['customer'] = customer_id

            response = await asyncio.to_thread(
                requests.get,
                f"{self.base_url}/orders/{clean_order_id}",
                auth=self.auth,
                params=params,
                timeout=10,
                headers={
                    'User-Agent': 'FiFi-AI-Assistant/1.0',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 404:
                logger.info(f"Order {clean_order_id} not found")
                return {"error": "not_found", "message": f"Order #{clean_order_id} not found."}
            
            response.raise_for_status()
            order_data = response.json()

            if customer_id and order_data.get('customer_id') != customer_id:
                logger.warning(f"Order {clean_order_id} found but does not belong to customer {customer_id}.")
                return {"error": "mismatch", "message": f"Order #{clean_order_id} does not belong to your account or was not found."}
            
            logger.info(f"Successfully retrieved order #{clean_order_id}")
            return order_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"12Taste Order Status API error: {e}")
            return {"error": "api_error", "message": str(e)}

    @handle_api_errors("12Taste Order Status", "Get Customer by Email", show_to_user=False)
    async def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve a customer by email from WooCommerce."""
        if not self.config.WOOCOMMERCE_ENABLED:
            return None
        if not email:
            return None
        
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{self.base_url}/customers",
                auth=self.auth,
                params={'email': email},
                timeout=10,
                headers={
                    'User-Agent': 'FiFi-AI-Assistant/1.0',
                    'Accept': 'application/json'
                }
            )
            response.raise_for_status()
            customers = response.json()
            if customers:
                logger.info(f"Found customer for email {email}: ID {customers[0]['id']}")
                return customers[0]
            logger.info(f"No customer found for email {email}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"12Taste Order Status API error retrieving customer for {email}: {e}")
            return None
    
    def format_order_for_display(self, order: Dict[str, Any]) -> str:
        """Format order data into a user-friendly markdown string."""
        if "error" in order:
            if order["error"] == "not_found":
                return f"❌ {order['message']}"
            else:
                return f"⚠️ Error retrieving order: {order['message']}"
        
        order_id = order.get('id', 'N/A')
        order_number = order.get('number', order_id)
        status = order.get('status', 'unknown')
        date_created = order.get('date_created', '')
        total = order.get('total', '0')
        currency = order.get('currency', 'USD')
        
        billing = order.get('billing', {})
        customer_name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
        customer_email = billing.get('email', 'N/A')
        
        response = f"""## 📦 Order #{order_number}

**Status:** {self._format_status(status)}
**Date:** {self._format_date(date_created)}
**Total:** {currency} {total}

### 👤 Customer Information
- **Name:** {customer_name or 'N/A'}
- **Email:** {customer_email}
- **Phone:** {billing.get('phone', 'N/A')}

### 📍 Billing Address
{self._format_address(billing)}

"""
        
        shipping = order.get('shipping', {})
        if shipping and shipping != billing:
            response += f"""### 📍 Shipping Address
{self._format_address(shipping)}

"""
        
        line_items = order.get('line_items', [])
        if line_items:
            response += "### 🛒 Order Items\n"
            for item in line_items:
                name = item.get('name', 'Unknown')
                quantity = item.get('quantity', 0)
                price = item.get('price', '0')
                total = item.get('total', '0')
                response += f"- **{name}** × {quantity} @ {currency} {price} = {currency} {total}\n"
            response += "\n"
        
        payment_method = order.get('payment_method_title', 'N/A')
        response += f"**Payment Method:** {payment_method}\n"
        
        customer_note = order.get('customer_note', '')
        if customer_note:
            response += f"\n**Customer Note:** {customer_note}\n"
        
        return response
    
    def _format_status(self, status: str) -> str:
        """Format order status with emoji."""
        status_map = {
            'pending': '🕐 Pending',
            'processing': '⚙️ Processing',
            'on-hold': '⏸️ On Hold',
            'completed': '✅ Completed',
            'cancelled': '❌ Cancelled',
            'refunded': '💰 Refunded',
            'failed': '❌ Failed'
        }
        return status_map.get(status, status.title())
    
    def _format_date(self, date_str: str) -> str:
        """Format ISO date string to readable format."""
        try:
            if date_str:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return date_obj.strftime("%B %d, %Y at %I:%M %p")
        except:
            pass
        return date_str
    
    def _format_address(self, address_data: Dict[str, Any]) -> str:
        """Format address data into readable string."""
        parts = []
        if address_data.get('company'): parts.append(address_data['company'])
        if address_data.get('address_1'): parts.append(address_data['address_1'])
        if address_data.get('address_2'): parts.append(address_data['address_2'])
        
        city_state_zip = []
        if address_data.get('city'): city_state_zip.append(address_data['city'])
        if address_data.get('state'): city_state_zip.append(address_data['state'])
        if address_data.get('postcode'): city_state_zip.append(address_data['postcode'])
        if city_state_zip: parts.append(', '.join(city_state_zip))
        
        if address_data.get('country'): parts.append(address_data['country'])
        
        return '\n'.join(parts) if parts else 'N/A'
    
    @handle_api_errors("12Taste Order Status", "Test Connection", show_to_user=False)
    def test_connection(self) -> Dict[str, Any]:
        """Test WooCommerce API connection and return status information."""
        if not self.config.WOOCOMMERCE_ENABLED:
            return {
                "status": "disabled",
                "message": "12Taste Order Status integration is not enabled"
            }
        
        try:
            response = requests.get(
                f"{self.base_url}/system_status",
                auth=self.auth,
                timeout=5,
                headers={
                    'User-Agent': 'FiFi-AI-Assistant/1.0',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                return {
                    "status": "connected",
                    "message": "12Taste Order Status API is working correctly",
                    "api_url": self.base_url
                }
            elif response.status_code == 401:
                return {
                    "status": "authentication_failed",
                    "message": "12Taste Order Status authentication failed. Check API credentials."
                }
            else:
                return {
                    "status": "error",
                    "message": f"12Taste Order Status API returned status code: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "message": "12Taste Order Status API request timed out"
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "connection_error",
                "message": "Could not connect to 12Taste Order Status API"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"12Taste Order Status API error: {str(e)}"
            }


# =============================================================================
# RATE LIMITER & AI SYSTEM
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter to prevent abuse."""
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.MAX_TRACKED_IDS = MAX_RATE_LIMIT_TRACKING

    def is_allowed(self, identifier: str) -> Dict[str, Any]:
        """Returns detailed rate limit information including timer."""
        with self._lock:
            now = time.time()
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            
            if len(self.requests) > self.MAX_TRACKED_IDS:
                sorted_items = sorted(self.requests.items(), 
                                    key=lambda x: max(x[1]) if x[1] else 0,
                                    reverse=False)
                for old_id, _ in sorted_items[:self.MAX_TRACKED_IDS // 10]:
                    del self.requests[old_id]
            
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
                for req_time in sorted(self.requests[identifier]):
                    if req_time > now - self.window_seconds:
                        oldest_request = req_time
                        break
                else: # Fallback if list is empty or all requests are too old (should not happen with the filter above)
                    oldest_request = now
                
                time_until_next = max(0, int((oldest_request + self.window_seconds) - now))
                
                logger.warning(f"Rate limit exceeded for {identifier[:8]}... ({len(self.requests[identifier])}/{self.max_requests} within {self.window_seconds}s)")
                return {
                    'allowed': False,
                    'current_count': len(self.requests[identifier]),
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds,
                    'time_until_next': time_until_next
                }

def sanitize_input(text: str) -> str:
    """Enhanced input sanitization to prevent XSS and limit length."""
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    text = html.escape(text)
    text = text[:MAX_MESSAGE_LENGTH].strip()
    
    sql_patterns = ['DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET', 'SELECT * FROM']
    for pattern in sql_patterns:
        if pattern in text.upper():
            logger.warning(f"Potential SQL injection attempt detected: {pattern[:20]}... in '{text[:50]}'")
            
    return text

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
                "You are a document-based AI assistant for 12taste.com with STRICT business rules.\n\n"
                "**CORE IDENTITY:**\n"
                "- You are FiFi, an AI assistant for 12taste.com, a B2B food ingredients marketplace.\n"
                "- Your knowledge is LIMITED to the documents in your knowledge base.\n\n"
                "**ABSOLUTE BUSINESS RULES - NO EXCEPTIONS:**\n"
                "1. **NEVER PROVIDE PRICING OR STOCK INFO:** If asked for prices, costs, stock, or availability, you MUST respond with EXACTLY this template:\n"
                "   'Thank you for your interest in pricing information. For the most accurate and up-to-date pricing and quotes, please visit the product page directly on our website or contact our sales team at sales-eu@12taste.com for personalized assistance.'\n"
                "2. **HANDLE 'NOT FOUND' GRACEFULLY:** If you cannot find information for a specific product or topic in your documents, you MUST respond with EXACTLY: 'I don't have specific information about this topic in my knowledge base.'\n"
                "3. **NO FABRICATION:** NEVER invent information, product details, specifications, suppliers, or any data not present in your documents. NEVER create fake citations, URLs, or source references.\n"
                "4. **DISCONTINUED PRODUCTS:** If a product has been discontinued, inform the user and provide similar alternatives if available in your knowledge base.\n"
                "5. **ALWAYS INCLUDE PRODUCT URLs:** For EVERY product mentioned in your response:\n"
                "   - Include an inline link in format: [More details](https://www.12taste.com/product/[product-slug]/?utm_source=fifi-eu)\n"
                "   - End your response with a '**Sources:**' section listing all product URLs as numbered items\n"
                "   - NEVER mention a product without its corresponding URL\n\n"
                "**RESPONSE GUIDELINES:**\n"
                "- **BE CONCISE:** Provide direct answers from your documents.\n"
                "- **CITE SOURCES:** Use citations like [1], [2] when pulling information from documents.\n"
                "- **INCLUDE URLS:** Every product MUST have a clickable [More details] link.\n"
                "- **STAY ON TOPIC:** Only answer questions related to the food and beverage industry based on your documents.\n"
                "- **REDIRECT, DON'T REFUSE:** For pricing/stock questions, use the specific redirection message above. For out-of-scope topics, state you specialize in food ingredients.\n\n"
                "**PRIORITY FLOW:**\n"
                "1. Does the question ask for price or stock? -> Use the sales redirection message.\n"
                "2. Is the information in my documents? -> Answer with citations AND product URLs.\n"
                "3. Is the information NOT in my documents? -> Use the 'I don't have specific information...' message.\n"
                "4. Is the product discontinued? -> Inform and provide similar alternatives if available.\n"
                "5. Did I mention any products? -> Ensure each has a [More details] link and all are in Sources section."
            )
            
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                logger.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions=instructions
                )
                logger.info(f"✅ Created assistant '{self.assistant_name}'")
            else:
                logger.info(f"Found existing assistant: '{self.assistant_name}'")
                try:
                    result = self.pc.assistant.update_assistant(
                        assistant_name=self.assistant_name,
                        instructions=instructions
                    )
                    logger.info(f"✅ Instructions updated for '{self.assistant_name}'")
                    logger.info(f"Update result: {result}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not update instructions: {e}")
        
            assistant_obj = self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        
            try:
                status = self.pc.assistant.describe_assistant(
                    assistant_name=self.assistant_name
                )
                logger.info(f"✅ Assistant status: {status.get('status', 'Unknown')}")
                logger.info(f"✅ Assistant has {len(assistant_obj.list_files())} files")
            except Exception as e:
                logger.warning(f"Could not get assistant status: {e}")
        
            return assistant_obj
            
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
        
            import re
            def add_utm_to_url(match):
                url = match.group(2)
                if 'utm_source=fifi-eu' not in url:
                    separator = '&' if '?' in url else '?'
                    return f'{match.group(1)}({url}{separator}utm_source=fifi-eu)'
                return match.group(0)
        
            content = re.sub(r'(\[.*?\])\((https?://[^)]+)\)', add_utm_to_url, content)
        
            has_sources_section = "**Sources:**" in content or "**sources:**" in content.lower()
        
            if not has_sources_section:
                url_pattern = r'\[.*?\]\((https?://[^)]+)\)'
                found_urls = re.findall(url_pattern, content)
                if found_urls:
                    logger.info(f"Adding Sources section with {len(found_urls)} URLs")
                    citations_header = "\n\n---\n**Sources:**\n"
                    numbered_citations = []
                    seen_urls = set()
                    for url in found_urls:
                        if url not in seen_urls:
                            numbered_citations.append(f"[{len(seen_urls) + 1}] {url}")
                            seen_urls.add(url)
                
                    if numbered_citations:
                        content += citations_header + "\n".join(numbered_citations)
        
            has_citations = bool(re.findall(r'\[.*?\]\((https?://[^)]+)\)', content))
        
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi",
                "has_citations": has_citations,
                "response_length": len(content),
                "used_pinecone": True,
                "used_search": False,
                "has_inline_citations": has_citations,
                "safety_override": False
            }
        except Exception as e:
            logger.error(f"Pinecone Assistant error: {str(e)}", exc_info=True)
            return None
    
class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str, openai_api_key: str = None):
        from tavily import TavilyClient
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key, include_answer=True)

        self.openai_api_key = openai_api_key
        self.openai_client = None

        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI client initialized for query reformulation")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client for reformulation: {e}")
                self.openai_client = None

    def reformulate_query_for_search(self, current_question: str, conversation_history: List[BaseMessage]) -> str:
        """LLM-powered query reformulation for better search results, focused on F&B industry."""
        try:
            if len(current_question.split()) > 10 and not any(indicator in current_question.lower() 
                                                              for indicator in ["what about", "how about", "any", "tell me", "more about"]):
                logger.debug(f"Query is detailed enough, using as-is: {current_question}")
                return current_question
            
            if not self.openai_client:
                logger.warning("OpenAI client not available, using code-based reformulation fallback")
                return self._fallback_reformulation(current_question, conversation_history)

            context_parts = []
            if conversation_history:
                recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
                for msg in recent_messages:
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
                        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        context_parts.append(f"{role}: {content}")
            
            conversation_context = "\n".join(context_parts) if context_parts else "No prior conversation"

            reformulation_prompt = f"""Transform this question into an optimized web search query for the food and beverage industry.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT QUESTION: "{current_question}"

RULES:
1. If follow-up question, incorporate context from conversation
2. Make vague questions specific using context
3. Keep query concise but searchable (3-12 words acceptable)
4. Always include "food beverage industry" or related terms for industry focus
5. For pricing questions: add "market pricing B2B"
6. For supplier questions: include "suppliers manufacturers B2B"
7. For regulations: include "compliance standards food industry"
8. For equipment/processing: include "food processing equipment"

OUTPUT: Only the optimized search query, nothing else."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a search query optimizer. Respond only with the optimized search query."},
                    {"role": "user", "content": reformulation_prompt}
                ],
                max_tokens=30,
                temperature=0.1
            )
            
            reformulated = response.choices[0].message.content.strip()

            if not reformulated or len(reformulated) < 3:
                logger.warning(f"LLM returned invalid reformulation: '{reformulated}', using fallback")
                return self._fallback_reformulation(current_question, conversation_history)
            
            if len(reformulated.split()) > 15 or len(reformulated) > 120:
                logger.warning(f"LLM reformulation too long: '{reformulated}', using fallback")
                return self._fallback_reformulation(current_question, conversation_history)
            
            logger.info(f"✅ LLM reformulated: '{current_question}' → '{reformulated}'")
            return reformulated
            
        except Exception as e:
            logger.error(f"LLM reformulation failed: {e}, using code fallback")
            return self._fallback_reformulation(current_question, conversation_history)

    def _fallback_reformulation(self, current_question: str, conversation_history: List[BaseMessage]) -> str:
        """Fallback to code-based reformulation if LLM fails."""
        logger.debug("Using code-based reformulation fallback")

        follow_up_indicators = [
            "what about", "how about", "what are", "any", "tell me about",
            "pricing", "cost", "price", "suppliers", "where", "who", "availability"
        ]
        
        current_question_lower = current_question.lower()
        is_likely_followup = (
            len(current_question.split()) <= 4 or 
            any(indicator in current_question_lower for indicator in follow_up_indicators)
        )

        if not is_likely_followup or not conversation_history:
            if "food" not in current_question_lower and "ingredient" not in current_question_lower and "beverage" not in current_question_lower:
                return f"{current_question} food beverage industry"
            return current_question

        context_keywords = []
        if conversation_history:
            for msg in conversation_history[-3:]:
                if hasattr(msg, 'content'):
                    words = re.findall(r'\b[A-Z][a-z]+\b', msg.content)
                    context_keywords.extend(words[:2])

        if any(word in current_question_lower for word in ["pricing", "cost", "price"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} pricing costs food beverage industry"
            else:
                return f"{current_question} food beverage industry pricing market"

        elif any(word in current_question_lower for word in ["suppliers", "supplier", "source", "where"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} suppliers food beverage industry sourcing"
            else:
                return f"{current_question} food beverage industry suppliers"

        elif any(word in current_question_lower for word in ["availability", "stock", "available"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} availability food beverage industry market"
            else:
                return f"{current_question} food beverage industry availability"
        
        else:
            if context_keywords:
                return f"{current_question} {' '.join(context_keywords[:2])} food beverage industry"
            else:
                return f"{current_question} food beverage industry"

    def add_utm_to_links(self, content: str) -> str:
        """Finds all Markdown links in a string and appends the UTM parameters."""
        
        if not content:
            return ""
        
        try:
            def replacer(match):
                url = match.group(1)
                utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
                if '?' in url:
                    new_url = f"{url}&{utm_params}"
                else:
                    new_url = f"{url}?{utm_params}"
                return f"({new_url})"
            
            result = re.sub(r'\]\((https?://[^)]+)\)', replacer, content)
            return result
            
        except Exception as e:
            logger.error(f"🔍 UTM processing failed: {e}")
            return content

    def synthesize_search_results(self, results, query: str) -> str:
        """Enhanced synthesis using Tavily's answer + sources."""
        
        if isinstance(results, dict):
            tavily_answer = results.get('answer', '')
            tavily_results = results.get('results', [])

            if tavily_answer and tavily_results:
                response_parts = [tavily_answer]
                response_parts.append("\n\n**Sources:**")

                for i, res in enumerate(tavily_results, 1):
                    if isinstance(res, dict) and 'url' in res and 'title' in res:
                        response_parts.append(f"\n{i}. [{res['title']}]({res['url']})")

                return "\n".join(response_parts)
            
            elif tavily_results:
                results = tavily_results
            elif tavily_answer:
                return f"Based on web search for '{query}':\n\n{tavily_answer}"
            else:
                logger.warning(f"Unexpected dict format from Tavily: {results.keys()}")
                return f"I found some information but couldn't format it properly."

        if not isinstance(results, list):
            logger.warning(f"Tavily returned unexpected results type: {type(results)}")
            return "I couldn't process the search results properly."
            
        if not results:
            return f"I couldn't find any relevant information online for: '{query}'"
        
        response_parts = [f"Here is a summary of web search results for '{query}':\n"]
        for res in results[:3]:
            if isinstance(res, dict) and 'content' in res:
                response_parts.append(f"- {res['content']}")
        
        response_parts.append("\n\n**Sources:**")
        for i, res in enumerate(results, 1):
            if isinstance(res, dict) and 'url' in res and 'title' in res:
                response_parts.append(f"\n{i}. [{res['title']}]({res['url']})")
        
        return "\n".join(response_parts)

    def determine_search_strategy(self, question: str, pinecone_error_type: str = None) -> Dict[str, Any]:
        """Determine search strategy based on Pinecone status"""
        
        if pinecone_error_type and pinecone_error_type != "healthy" and pinecone_error_type != "recency_direct_route":
            logger.info(f"🔒 Pinecone {pinecone_error_type} - restricting Tavily to 12taste.com domain only")
            return {
                "strategy": "domain_restricted_12taste",
                "include_domains": ["12taste.com"],
                "exclude_domains": None,
                "reason": f"Pinecone {pinecone_error_type} - using 12taste.com as fallback source"
            }
        
        logger.info("🌐 Using standard Tavily search with competitor exclusions")
        return {
            "strategy": "worldwide_with_exclusions",
            "include_domains": None,
            "exclude_domains": DEFAULT_EXCLUDED_DOMAINS,
            "reason": "Standard web search with competitor exclusion"
        }

    def query(self, prompt: str, chat_history: List[BaseMessage], pinecone_error_type: str = None) -> Dict[str, Any]:
        """Query Tavily with two-step fallback when Pinecone is down"""
        try:
            reformulated_query = self.reformulate_query_for_search(prompt, chat_history)
            logger.info(f"🔍 Original query: '{prompt}' → Reformulated: '{reformulated_query}'")
            
            strategy = self.determine_search_strategy(reformulated_query, pinecone_error_type)
            
            sdk_params = {
                "query": reformulated_query,
                "max_results": 5,
                "include_answer": "advanced",
                "search_depth": "advanced",
                "include_raw_content": "text"
            }
            
            if strategy.get("include_domains"):
                sdk_params["include_domains"] = strategy["include_domains"]
                logger.info(f"🔍 Tavily domain-restricted search: {strategy['include_domains']}")
            elif strategy.get("exclude_domains"):
                sdk_params["exclude_domains"] = strategy["exclude_domains"]
                logger.info(f"🌐 Tavily worldwide search excluding {len(strategy['exclude_domains'])} competitor domains")
            
            search_results = self.tavily_client.search(**sdk_params)
            
            if (strategy["strategy"] == "domain_restricted_12taste" and 
                (not search_results or 
                 not search_results.get('results') or 
                 len(search_results.get('results', [])) == 0)):
                
                logger.warning("⚠️ No results found on 12taste.com, falling back to regular web search")
                
                sdk_params_fallback = {
                    "query": reformulated_query,
                    "max_results": 5,
                    "include_answer": "advanced",
                    "search_depth": "advanced",
                    "include_raw_content": "text",
                    "exclude_domains": DEFAULT_EXCLUDED_DOMAINS
                }
                
                sdk_params_fallback.pop("include_domains", None)
                
                logger.info(f"🌐 Retrying with worldwide search excluding {len(DEFAULT_EXCLUDED_DOMAINS)} competitor domains")
                search_results = self.tavily_client.search(**sdk_params_fallback)
                
                strategy = {
                    "strategy": "worldwide_with_exclusions_after_12taste_fallback",
                    "reason": "No results from 12taste.com, expanded to worldwide search"
                }
            
            synthesized_content = self.synthesize_search_results(search_results, reformulated_query)
            final_content = self.add_utm_to_links(synthesized_content)
            
            if strategy.get("strategy") == "worldwide_with_exclusions_after_12taste_fallback":
                final_content = f"*Note: No specific information found on 12taste.com, showing results from other industry sources.*\n\n{final_content}"
            
            return {
                "content": final_content,
                "success": True,
                "source": "FiFi Web Search",
                "used_pinecone": False,
                "used_search": True,
                "has_citations": True,
                "has_inline_citations": True,
                "safety_override": False,
                "search_strategy": strategy["strategy"],
                "search_reason": strategy.get("reason", "Standard search"),
                "reformulated_query": reformulated_query
            }
            
        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}", exc_info=True)
            return {
                "content": "I encountered an error while searching for information. Please try again.",
                "success": False,
                "source": "FiFi Web Search",
                "used_search": True,
                "used_pinecone": False,
                "has_citations": False,
                "has_inline_citations": False,
                "safety_override": False,
                "error": str(e)
            }
class EnhancedAI:
    """Enhanced AI system with improved error handling and bidirectional fallback."""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        self.pinecone_tool = None
        self.tavily_agent = None
        
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
        
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
        
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(
                    self.config.TAVILY_API_KEY, 
                    self.config.OPENAI_API_KEY
                )
                logger.info("✅ Tavily Web Search initialized successfully with LLM reformulation")
                error_handler.mark_component_healthy("Tavily")
            except Exception as e:
                logger.error(f"Tavily agent initialization failed: {e}")
                self.tavily_agent = None
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Initialize", e))

    def _detect_pinecone_error_type(self, error: Exception) -> str:
        """NEW: Enhanced Pinecone error detection with specific HTTP codes."""
        error_str = str(error).lower()
        
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
        elif any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            error_handler.component_status["Pinecone"] = "connectivity_error"
            return "connectivity_error"
        else:
            error_handler.component_status["Pinecone"] = "unknown_error"
            return "unknown_error"

    def _should_use_pinecone_first(self) -> bool:
        """Routing based on component health. Always prefers Pinecone if healthy."""
        pinecone_status = error_handler.component_status.get("Pinecone", "healthy")
        return pinecone_status == "healthy"

    def _get_current_pinecone_error_type(self) -> str:
        """Get current Pinecone error type for Tavily strategy determination."""
        return error_handler.component_status.get("Pinecone", "healthy")

    def _needs_current_information(self, prompt: str) -> bool:
        """Check if query needs current/updated information"""
        prompt_lower = prompt.lower()
        
        time_indicators = [
            "latest", "newest", "recent", "current", "today", "yesterday",
            "this week", "last week", "this month", "last month",
            "2024", "2025", "2026",
            "update", "updates", "updated",
            "now", "nowadays", "presently", "currently"
        ]
        
        market_indicators = [
            "news", "breaking", "announcement", "announced",
            "trend", "trends", "trending", "forecast",
            "market price", "market report", "market update",
            "just released"
        ]
        
        regulatory_indicators = [
            "new regulation", "updated regulation", "revised",
            "amendment", "change in law", "policy update"
        ]
        
        all_indicators = time_indicators + market_indicators + regulatory_indicators
        
        return any(indicator in prompt_lower for indicator in all_indicators)

    def should_use_web_fallback(self, pinecone_response: Dict[str, Any], original_question: str) -> bool:
        """
        Determines if a fallback to web search is needed based on strict business rules.
        """
        content = pinecone_response.get("content", "").lower()
        original_lower = original_question.lower()

        logger.warning("=" * 50)
        logger.warning("🔍 FALLBACK DEBUG START")
        logger.warning(f"   Question: '{original_question}'")
        logger.warning(f"   Content length: {len(content)}")
        logger.warning(f"   Content preview: {content[:300]}...")
        logger.warning(f"   Response source: {pinecone_response.get('source', 'Unknown')}")
        logger.warning(f"   Has citations flag: {pinecone_response.get('has_citations', False)}")
        logger.warning(f"   Success flag: {pinecone_response.get('success', False)}")
        logger.warning("=" * 50)

        has_citation_markers = "[1]" in content or "**sources:**" in content
        has_citations_flag = pinecone_response.get("has_citations", False)

        if has_citation_markers:
            logger.warning(f"🔍 CITATION DEBUG: Found markers in content, has_citations={has_citations_flag}")
            logger.warning(f"Content preview: {content[:200]}...")

        if "sales-eu@12taste.com" in content or "contact our sales team" in content:
            logger.warning("✅ Fallback SKIPPED: Pinecone provided the correct business redirect for pricing/stock.")
            return False
        logger.warning("✅ Business redirect check PASSED (no sales redirect found)")

        not_found_indicators = [
            "don't have specific information",
            "could not find specific information",
            "cannot find information",
            "no information about",
            "not available in my knowledge base",
            "don't have information about",
            "couldn't find information",
            "do not provide specific information",
            "does not provide specific information",
            "search results do not provide",
            "results do not provide",
            "do not contain specific information",
            "does not contain specific information",
            "search results do not contain",
            "results do not contain"
        ]
        for phrase in not_found_indicators:
            if phrase in content:
                logger.warning(f"🔄 Fallback TRIGGERED: Found 'not found' indicator: '{phrase}'")
                return True
        logger.warning("✅ Not found indicators check PASSED")

        if "[1]" in content or "**sources:**" in content:
            if not pinecone_response.get("has_citations", False):
                logger.warning("🚨 SAFETY OVERRIDE: Detected fake citations. Switching to web search.")
                return True
        logger.warning("✅ Citation safety check PASSED")

        regulatory_indicators = ["regulation", "directive", "compliance", "legal"]
        is_regulatory = any(indicator in original_lower for indicator in regulatory_indicators)

        if is_regulatory and not pinecone_response.get("has_citations", False):
            logger.warning("🔄 Fallback TRIGGERED: Regulatory topic without citations.")
            return True
        logger.warning(f"✅ Regulatory check PASSED (is_regulatory={is_regulatory})")

        logger.warning("✅ ALL CHECKS PASSED - Should NOT fallback.")
        return False
    
    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Gets AI response and manages session state with a corrected processing pipeline."""
        def _convert_to_langchain_format(history: List[Dict]) -> List[BaseMessage]:
            langchain_history_converted = []
            if history:
                for msg in history[-10:]:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "user":
                        langchain_history_converted.append(HumanMessage(content=content))
                    elif role == "assistant":
                        langchain_history_converted.append(AIMessage(content=content))
            return langchain_history_converted
        
        langchain_history = _convert_to_langchain_format(chat_history)
        langchain_history.append(HumanMessage(content=prompt))

        if self._needs_current_information(prompt):
            logger.info("🚀 Recency keywords detected - routing directly to Tavily")
            
            if self.tavily_agent:
                try:
                    web_response = self.tavily_agent.query(
                        prompt, 
                        langchain_history,
                        "recency_direct_route"
                    )
                    
                    if web_response and web_response.get("success"):
                        logger.info("✅ Direct Tavily response successful for recency query.")
                        error_handler.mark_component_healthy("Tavily")
                        return web_response
                        
                except Exception as e:
                    logger.error(f"Direct Tavily search for recency failed: {e}. Falling back to normal flow.")
            else:
                logger.warning("Tavily agent not initialized, cannot handle recency query directly.")

        if self.pinecone_tool:
            try:
                logger.info("🧠 Querying Pinecone knowledge base (primary)...")
                pinecone_response = self.pinecone_tool.query(langchain_history)

                if pinecone_response:
                    logger.warning("🧠 PINECONE RESPONSE RECEIVED:")
                    logger.warning(f"   Success: {pinecone_response.get('success', False)}")
                    logger.warning(f"   Content length: {len(pinecone_response.get('content', ''))}")
                    logger.warning(f"   Has citations: {pinecone_response.get('has_citations', False)}")
                else:
                    logger.error("🚨 PINECONE RETURNED NULL!")

                if pinecone_response and pinecone_response.get("success"):
                    if not self.should_use_web_fallback(pinecone_response, prompt):
                        logger.info("✅ Using Pinecone response (passed business logic checks).")
                        error_handler.mark_component_healthy("Pinecone")
                        return pinecone_response
                    else:
                        logger.warning("⤵️ Pinecone response requires web fallback. Proceeding to Tavily...")

            except Exception as e:
                error_type = self._detect_pinecone_error_type(e)
                logger.error(f"Pinecone query failed ({error_type}): {e}. Proceeding to web fallback.")

        if self.tavily_agent:
            try:
                logger.info("🌐 Falling back to FiFi web search...")
                web_response = self.tavily_agent.query(prompt, langchain_history, self._get_current_pinecone_error_type())
                
                if web_response and web_response.get("success"):
                    logger.info(f"✅ Using web search response.")
                    error_handler.mark_component_healthy("Tavily")
                    return web_response
                        
            except Exception as e:
                logger.error(f"FiFi Web search failed: {e}", exc_info=True)
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Query", e))
        
        logger.warning("⚠️ All AI tools failed. Using final system fallback response.")
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
def check_industry_context(prompt: str, chat_history: List[Dict] = None, client: Optional[openai.OpenAI] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if user question is relevant to food & beverage industry, now with conversation context.
    This version is updated to recognize and allow meta-conversation queries.
    """
    if not client or not hasattr(client, 'chat'):
        logger.debug("OpenAI client not available for industry context check. Allowing question.")
        return {"relevant": True, "reason": "context_check_unavailable"}

    conversation_context = ""
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-6:]
        context_parts = []
        for msg in recent_history:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')[:200]
            context_parts.append(f"{role}: {content}")
        conversation_context = "\n".join(context_parts)

    try:
        context_check_prompt = f"""You are an industry context validator for 12taste.com, a B2B marketplace connecting food & beverage manufacturers with ingredient suppliers.

**CONVERSATION HISTORY (for context):**
{conversation_context if conversation_context else "No previous conversation."}

**CURRENT USER QUESTION:** "{prompt}"

**TASK:** Analyze the **CURRENT USER QUESTION**. Considering the conversation history, determine if it is a relevant B2B food & beverage industry question.

**ALWAYS ALLOW:**
- Greetings and polite expressions (hello, hi, thanks, bye, how are you)
- Questions about food/beverage ingredients, additives, flavors, colors
- Supplier sourcing, B2B pricing, bulk purchasing queries  
- Food manufacturing, processing, formulation questions
- Regulatory compliance (FDA, EU, FSSAI, Halal, Kosher)
- Packaging, shelf life, technical specifications
- Market trends in food & beverage industry
- Follow-up questions that relate to previous context
- Meta-conversation queries (summarize chat, count questions, list topics)
- Order status queries (e.g., "Where is my order 123?")

**FLAG ONLY:**
- Consumer/home cooking recipes or diet advice
- Completely unrelated industries (automotive, IT, fashion)
- Political discussions, sports, entertainment

**EDGE CASES - ALLOW:**
- Equipment for food processing
- Logistics and supply chain for food industry
- Sustainability in food manufacturing

**INSTRUCTIONS:**
Respond with ONLY a JSON object in this exact format:
{{
    "relevant": true/false,
    "confidence": 0.0-1.0,
    "category": "food_ingredients" | "supplier_sourcing" | "follow_up" | "meta_conversation" | "order_query" | "off_topic" | "unrelated_industry" | "greeting_or_polite",
    "reason": "Brief explanation."
}}"""

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
        response_content = response_content.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_content)
        
        if not all(key in result for key in ['relevant', 'confidence', 'category', 'reason']):
            logger.warning("Industry context check returned incomplete JSON structure")
            return {"relevant": True, "reason": "context_check_invalid_response"}
                
        logger.info(f"Context check: relevant={result['relevant']}, category={result['category']}, confidence={result['confidence']:.2f}")
        return result
            
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
                 rate_limiter: RateLimiter, fingerprinting_manager: DatabaseManager.FingerprintingManager,
                 email_verification_manager: DatabaseManager.EmailVerificationManager,
                 question_limit_manager: DatabaseManager.QuestionLimitManager,
                 woocommerce_manager: Optional['WooCommerceManager'] = None):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.fingerprinting = fingerprinting_manager
        self.email_verification = email_verification_manager
        self.question_limits = question_limit_manager
        self.woocommerce = woocommerce_manager
        self._cleanup_interval = timedelta(hours=1)
        self._last_cleanup = datetime.now()
        
        self._ban_sync_locks = {}
        self._ban_lock_timeout = timedelta(seconds=5)

        logger.info("✅ SessionManager initialized with all component managers.")

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        return SESSION_TIMEOUT_MINUTES
    
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
            if hasattr(self.fingerprinting, 'fingerprint_cache'):
                old_entries = []
                for fp_id, data in self.fingerprinting.fingerprint_cache.items():
                    if now - data.get('last_seen', now) > timedelta(hours=24):
                        old_entries.append(fp_id)
                
                for old_fp in old_entries:
                    del self.fingerprinting.fingerprint_cache[old_fp]
                
                if old_entries:
                    logger.info(f"Cleaned up {len(old_entries)} old fingerprint cache entries")
            
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
            
            if hasattr(st.session_state, 'error_handler') and hasattr(st.session_state.error_handler, 'error_history') and len(st.session_state.error_handler.error_history) > MAX_ERROR_HISTORY:
                st.session_state.error_handler.error_history = st.session_state.error_handler.error_history[-MAX_ERROR_HISTORY // 2:]
                logger.info("Cleaned up error history")
            
            self.db.cleanup_old_inactive_sessions()

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
        error_keys = ['rate_limit_hit', 'moderation_flagged', 'context_flagged', 'pricing_stock_notice']
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
        
        frame = inspect.currentframe()
        caller = inspect.getframeinfo(frame.f_back)
        
        logger.debug("🚨 NEW SESSION BEING CREATED!")
        logger.debug(f"📍 Called from: {caller.filename}:{caller.lineno} in function {caller.function}")
        logger.debug("📋 Full stack trace:")
        for line in traceback.format_stack():
            logger.debug(line.strip())
        
        session_id = str(uuid.uuid4())
        session = UserSession(session_id=session_id, last_activity=None, login_method='guest')
        
        # For new sessions, always start with a temporary fingerprint
        session.fingerprint_id = f"temp_py_{secrets.token_hex(8)}"
        session.fingerprint_method = "temporary_fallback_python"
        
        logger.debug(f"🆔 New session created: {session_id[:8]} (NOT saved to DB yet, will be saved in get_session)")
        return session

    def _is_crm_save_eligible(self, session: UserSession, trigger_reason: str) -> bool:
        """Enhanced eligibility check for CRM saves including new user types and conditions."""
        try:
            if not session.email or not session.messages:
                logger.debug(f"CRM save not eligible - missing email ({bool(session.email)}) or messages ({bool(session.messages)}) for {session.session_id[:8]}")
                return False
            
            if session.timeout_saved_to_crm and "clear_chat" in trigger_reason.lower():
                logger.debug(f"CRM save not eligible - already saved for {session.session_id[:8]}")
                return False
            
            if session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]: 
                logger.debug(f"CRM save not eligible - user type {session.user_type.value} for {session.session_id[:8]}")
                return False
            
            if len(session.messages) < CRM_SAVE_MIN_QUESTIONS:
                logger.debug(f"CRM save not eligible - messages less than {CRM_SAVE_MIN_QUESTIONS} for {session.session_id[:8]}")
                return False

            logger.info(f"CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Messages={len(session.messages)}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    def _is_manual_crm_save_eligible(self, session: UserSession) -> bool:
        """Simple eligibility check for manual CRM saves (Sign Out, Manual Save button) - NO 15-minute requirement."""
        try:
            if not session.email or not session.messages:
                return False
            
            if session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value]:
                return False
            
            if len(session.messages) < CRM_SAVE_MIN_QUESTIONS:
                return False
            
            logger.info(f"Manual CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Messages={len(session.messages)}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking manual CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    def _attempt_fingerprint_inheritance(self, session: UserSession):
        """
        CORRECTED & COMPLETE: Attempts to inherit session data by first finding all linked identities
        (via fingerprint AND email) and then determining the absolute highest privilege level
        for the current session, setting reverification pending status accordingly.
        """
        logger.info(f"🔄 [CORRECTED LOGIC] Attempting fingerprint inheritance for session {session.session_id[:8]} (Type: {session.user_type.value}, FP: {session.fingerprint_id[:8] if session.fingerprint_id else 'None'})...")
        
        if session.user_type == UserType.REGISTERED_USER:
            logger.debug(f"Session {session.session_id[:8]} is already REGISTERED_USER. Skipping fingerprint inheritance.")
            return

        if not session.fingerprint_id or session.fingerprint_id.startswith(("temp_", "fallback_", "temp_py_")):
            session.visitor_type = "new_visitor"
            session.reverification_pending = False
            session.pending_user_type = None
            session.pending_email = None
            session.pending_full_name = None
            session.pending_zoho_contact_id = None
            session.pending_wp_token = None
            session.declined_recognized_email_at = None
            logger.debug(f"Session {session.session_id[:8]} has no stable fingerprint. No historical inheritance possible yet.")
            return

        historical_fp_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        unique_emails_from_fp = {s.email.lower() for s in historical_fp_sessions if s.email}
        all_related_sessions = list(historical_fp_sessions)
        
        for email in unique_emails_from_fp:
            email_history = self.db.find_sessions_by_email(email)
            for s in email_history:
                if s.session_id not in {sess.session_id for sess in all_related_sessions}:
                    all_related_sessions.append(s)

        all_related_sessions = [s for s in all_related_sessions if s.session_id != session.session_id]

        if not all_related_sessions:
            session.visitor_type = "new_visitor"
            session.reverification_pending = False
            session.pending_user_type = None
            session.pending_email = None
            session.pending_full_name = None
            session.pending_zoho_contact_id = None
            session.pending_wp_token = None
            session.declined_recognized_email_at = None
            logger.info(f"Inheritance complete for new visitor {session.session_id[:8]}. No historical data found.")
            return
        
        session.visitor_type = "returning_visitor"

        highest_privilege_candidate = None
        highest_level = self._get_privilege_level(session.user_type)

        now = datetime.now()

        if session.declined_recognized_email_at and (now - session.declined_recognized_email_at) < timedelta(minutes=60):
            session.reverification_pending = False
            session.pending_user_type = None
            session.pending_email = None
            session.pending_full_name = None
            session.pending_zoho_contact_id = None
            session.pending_wp_token = None
            logger.info(f"Session {session.session_id[:8]}: User recently declined recognition. Proceeding as current guest.")
            return

        for s in all_related_sessions:
            current_s_level = self._get_privilege_level(s.user_type)
            if current_s_level > highest_level:
                highest_level = current_s_level
                highest_privilege_candidate = s
            elif current_s_level == highest_level and highest_privilege_candidate:
                s_activity = s.last_activity or s.created_at
                hpc_activity = highest_privilege_candidate.last_activity or highest_privilege_candidate.created_at
                if s_activity and hpc_activity and s_activity > hpc_activity:
                    highest_privilege_candidate = s
                elif not hpc_activity and s_activity:
                     highest_privilege_candidate = s

        if highest_privilege_candidate and highest_privilege_candidate.email and \
           self._get_privilege_level(highest_privilege_candidate.user_type) > self._get_privilege_level(UserType.GUEST):
            
            logger.info(f"Session {session.session_id[:8]}: Detected higher privilege: {highest_privilege_candidate.user_type.value} for {highest_privilege_candidate.email}. Setting reverification pending.")
            session.reverification_pending = True
            session.pending_user_type = highest_privilege_candidate.user_type
            session.pending_email = highest_privilege_candidate.email
            session.pending_full_name = highest_privilege_candidate.full_name
            session.pending_zoho_contact_id = highest_privilege_candidate.zoho_contact_id
            session.pending_wp_token = highest_privilege_candidate.wp_token
            session.declined_recognized_email_at = None

            if highest_privilege_candidate.fingerprint_id and \
               not highest_privilege_candidate.fingerprint_id.startswith(("temp_", "fallback_", "not_collected_")) and \
               session.fingerprint_id.startswith(("temp_", "fallback_", "temp_py_")):
                session.fingerprint_id = highest_privilege_candidate.fingerprint_id
                session.fingerprint_method = highest_privilege_candidate.fingerprint_method
                logger.info(f"Session {session.session_id[:8]}: Updated temporary fingerprint to {session.fingerprint_id[:8]} from highest privilege candidate.")

        else:
            session.reverification_pending = False
            session.pending_user_type = None
            session.pending_email = None
            session.pending_full_name = None
            session.pending_zoho_contact_id = None
            session.pending_wp_token = None
            logger.debug(f"Session {session.session_id[:8]}: No higher privilege found for reverification. Cleared pending status.")

        # FIXED: Separate question count inheritance logic for Guest users
        best_session_for_current_counts = None
        current_max_daily_count = session.daily_question_count
        
        # For Guest users without email, inherit from ANY session with same fingerprint
        if session.user_type == UserType.GUEST and not session.email:
            logger.debug(f"Guest user {session.session_id[:8]} - checking all fingerprint sessions for question count inheritance")
            
            for s in all_related_sessions:
                # Only inherit from same or lower privilege level to avoid Guest inheriting from higher levels
                if self._get_privilege_level(s.user_type) <= self._get_privilege_level(UserType.GUEST):
                    if s.last_question_time and (now - s.last_question_time) < DAILY_RESET_WINDOW:
                        if s.daily_question_count > current_max_daily_count:
                            current_max_daily_count = s.daily_question_count
                            best_session_for_current_counts = s
                            logger.debug(f"Found better count {s.daily_question_count} from session {s.session_id[:8]}")
        
        # For users with emails (or reverification pending with target email), use email-based inheritance
        elif (session.email or (session.reverification_pending and session.pending_email)):
            target_email = session.pending_email if session.reverification_pending else session.email
            logger.debug(f"Email-based inheritance for {session.session_id[:8]} with email {target_email}")
            
            for s in all_related_sessions:
                if s.email and s.email.lower() == target_email.lower():
                    if s.last_question_time and (now - s.last_question_time) < DAILY_RESET_WINDOW:
                        if s.daily_question_count > current_max_daily_count:
                            current_max_daily_count = s.daily_question_count
                            best_session_for_current_counts = s
                    
                    # Also inherit total count and ban status for email-based users
                    session.total_question_count = max(session.total_question_count, s.total_question_count)
                    if (s.ban_status != BanStatus.NONE and s.ban_end_time and s.ban_end_time > now):
                        session.ban_status = s.ban_status
                        session.ban_start_time = s.ban_start_time
                        session.ban_end_time = s.ban_end_time
                        session.ban_reason = s.ban_reason
                        session.question_limit_reached = True
                        session.current_tier_cycle_id = s.current_tier_cycle_id
                        session.tier1_completed_in_cycle = s.tier1_completed_in_cycle
                        session.tier_cycle_started_at = s.tier_cycle_started_at
                        logger.info(f"Session {session.session_id[:8]}: Inherited active ban from session {s.session_id[:8]}.")
        
        # Apply the inherited counts
        session.daily_question_count = current_max_daily_count
        if best_session_for_current_counts:
            session.last_question_time = best_session_for_current_counts.last_question_time
            # For registered users, also inherit tier cycle info
            if session.user_type == UserType.REGISTERED_USER or (session.reverification_pending and session.pending_user_type == UserType.REGISTERED_USER):
                session.current_tier_cycle_id = best_session_for_current_counts.current_tier_cycle_id
                session.tier1_completed_in_cycle = best_session_for_current_counts.tier1_completed_in_cycle
                session.tier_cycle_started_at = best_session_for_current_counts.tier_cycle_started_at
            
            logger.info(f"Inherited question count {current_max_daily_count} from session {best_session_for_current_counts.session_id[:8]}")
        elif not session.last_question_time:
            session.daily_question_count = 0
            session.last_question_time = None
            if session.user_type == UserType.REGISTERED_USER or (session.reverification_pending and session.pending_user_type == UserType.REGISTERED_USER):
                if not session.current_tier_cycle_id:
                    session.current_tier_cycle_id = str(uuid.uuid4())
                    session.tier1_completed_in_cycle = False
                    session.tier_cycle_started_at = now
                    logger.info(f"Session {session.session_id[:8]}: New tier cycle started due to no recent inherited activity.")

        logger.info(f"✅ Final inheritance status for {session.session_id[:8]}: user_type={session.user_type.value}, daily_q={session.daily_question_count}, total_q={session.total_question_count}, ban_status={session.ban_status.value}, rev_pending={session.reverification_pending}, pending_type={session.pending_user_type.value if session.pending_user_type else 'None'}, declined_at={session.declined_recognized_email_at}")
        
    def get_session(self) -> Optional[UserSession]:
        """Gets or creates the current user session with enhanced validation."""
        logger.info(f"🔍 get_session() called - current_session_id in state: {st.session_state.get('current_session_id', 'None')}")
    
        self._periodic_cleanup()

        try:
            session_id = st.session_state.get('current_session_id')
        
            if session_id:
                if st.session_state.get(f'loading_{session_id}', False):
                    logger.warning(f"Session {session_id[:8]} already being loaded, skipping")
                    return None
    
                st.session_state[f'loading_{session_id}'] = True
                session = self.db.load_session(session_id)
                st.session_state[f'loading_{session_id}'] = False
            
                if session and session.active:
                    if session.ban_status != BanStatus.NONE:
                        if session.ban_end_time and datetime.now() >= session.ban_end_time:
                            logger.info(f"Ban expired for session {session.session_id[:8]}. Clearing ban status.")
                        
                            previous_ban_type = session.ban_status

                            session.ban_status = BanStatus.NONE
                            session.ban_start_time = None
                            session.ban_end_time = None
                            session.ban_reason = None
                            session.question_limit_reached = False
                            
                            if session.user_type == UserType.REGISTERED_USER:
                                if previous_ban_type == BanStatus.TWENTY_FOUR_HOUR:
                                    logger.info(f"🔄 Tier 2 ban expired for {session.session_id[:8]} - starting new tier cycle")
                                    session.daily_question_count = 0
                                    session.last_question_time = None
                                    session.current_tier_cycle_id = str(uuid.uuid4())
                                    session.tier1_completed_in_cycle = False
                                    session.tier_cycle_started_at = datetime.now()
                                elif previous_ban_type == BanStatus.ONE_HOUR:
                                    logger.info(f"✅ Tier 1 ban expired for {session.session_id[:8]} - can now proceed to Tier 2")
                            else:
                                session.daily_question_count = 0
                                session.last_question_time = None
                        
                            self.db.save_session(session)
                
                    fingerprint_checked_key = f'fingerprint_checked_for_inheritance_{session.session_id}'
                    
                    if session.user_type != UserType.REGISTERED_USER and \
                       session.fingerprint_id and \
                       session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")) and \
                       not st.session_state.get(fingerprint_checked_key, False):
                    
                        self._attempt_fingerprint_inheritance(session)
                        self.db.save_session(session)
                        st.session_state[fingerprint_checked_key] = True
                        logger.info(f"Fingerprint inheritance check and save completed for {session.session_id[:8]}")
                    
                    elif session.user_type == UserType.REGISTERED_USER and \
                         (session.fingerprint_id is None or session.fingerprint_id.startswith(("temp_", "fallback_", "not_collected_"))):
                        session.fingerprint_id = "not_collected_registered_user"
                        session.fingerprint_method = "email_primary"
                        self.db.save_session(session)
                        logger.info(f"Registered user {session.session_id[:8]} fingerprint marked as not collected.")


                    if (session.user_type.value == UserType.GUEST.value and 
                        session.daily_question_count == 0 and 
                        not session.reverification_pending):
                        historical_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
                        
                        email_verified_sessions = [
                            s for s in historical_sessions 
                            if s.session_id != session.session_id
                            and (s.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value or 
                                 s.user_type.value == UserType.REGISTERED_USER.value or
                                 s.email is not None)
                        ]
                        
                        if email_verified_sessions:
                            logger.info(f"Session {session.session_id[:8]} must verify email - fingerprint previously used with email verification")
                            st.session_state.must_verify_email_immediately = True
                            st.session_state.skip_email_allowed = False
                            
                            known_emails = set()
                            for sess in email_verified_sessions:
                                if sess.email:
                                    known_emails.add(sess.email.lower())
                            
                            st.session_state.known_device_emails = list(known_emails)
                
                    limit_check = self.question_limits.is_within_limits(session)
                    if not limit_check.get('allowed', True) and limit_check.get('reason') not in ['guest_limit', 'email_verified_guest_limit', 'registered_user_tier1_limit', 'registered_user_tier2_limit']:
                    
                        ban_type = limit_check.get('ban_type', 'unknown')
                        message = limit_check.get('message', 'Access restricted due to usage policy.')
                        time_remaining = limit_check.get('time_until_next') # time_until_next is an int (seconds) from RateLimiter or timedelta from QuestionLimitManager

                        st.error(f"🚫 **Access Restricted**")
                        # Handle time_remaining based on its type
                        if isinstance(time_remaining, timedelta):
                            hours = max(0, int(time_remaining.total_seconds() // 3600))
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                        elif isinstance(time_remaining, (int, float)):
                            hours = max(0, int(time_remaining // 3600))
                            minutes = int((time_remaining % 3600) // 60)
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
        
            if new_session.user_type != UserType.REGISTERED_USER:
                self._attempt_fingerprint_inheritance(new_session)
                st.session_state[f'fingerprint_checked_for_inheritance_{new_session.session_id}'] = True
            else:
                new_session.fingerprint_id = "not_collected_registered_user"
                new_session.fingerprint_method = "email_primary"
                new_session.current_tier_cycle_id = str(uuid.uuid4())
                new_session.tier1_completed_in_cycle = False
                new_session.tier_cycle_started_at = datetime.now()
                logger.info(f"New REGISTERED_USER session {new_session.session_id[:8]} fingerprint marked as not collected. New tier cycle started.")

            self.db.save_session(new_session)
            logger.info(f"Created and stored new session {new_session.session_id[:8]} (post-inheritance check), active={new_session.active}, rev_pending={new_session.reverification_pending}")
            return new_session
        
        except Exception as e:
            logger.error(f"Failed to get/create session: {e}", exc_info=True)
            fallback_session = UserSession(session_id=str(uuid.uuid4()), user_type=UserType.GUEST, last_activity=None, login_method='guest')
            fallback_session.fingerprint_id = f"emergency_fp_{fallback_session.session_id[:8]}"
            fallback_session.fingerprint_method = "emergency_fallback"
            st.session_state.current_session_id = fallback_session.session_id
            st.error("⚠️ Failed to create or load session. Operating in emergency fallback mode. Chat history may not persist.")
            logger.error(f"Emergency fallback session created {fallback_session.session_id[:8]}")
            return fallback_session
    
    def sync_registered_user_sessions(self, email: str, current_session_id: str):
        """Sync question counts and tier cycle info across all active sessions for a registered user by email"""
        try:
            email_sessions = self.db.find_sessions_by_email(email)
            active_registered_sessions = [s for s in email_sessions 
                                          if s.active and s.user_type == UserType.REGISTERED_USER]
            
            if not active_registered_sessions:
                return
            
            max_count_session = None
            now = datetime.now()
            for sess in active_registered_sessions:
                if sess.last_question_time and (now - sess.last_question_time) < DAILY_RESET_WINDOW:
                    if max_count_session is None or sess.daily_question_count > max_count_session.daily_question_count:
                        max_count_session = sess
            
            if max_count_session is None:
                logger.info(f"No recent active sessions for {email} to sync. Counts will reset for new activity.")
                return
            
            for sess in active_registered_sessions:
                if sess.session_id != max_count_session.session_id:
                    sess.daily_question_count = max_count_session.daily_question_count
                    sess.total_question_count = max_count_session.total_question_count
                    sess.last_question_time = max_count_session.last_question_time
                    
                    sess.current_tier_cycle_id = max_count_session.current_tier_cycle_id
                    sess.tier1_completed_in_cycle = max_count_session.tier1_completed_in_cycle
                    sess.tier_cycle_started_at = max_count_session.tier_cycle_started_at

                    self.db.save_session(sess)
                    logger.debug(f"Synced session {sess.session_id[:8]} to {max_count_session.daily_question_count} daily questions from {max_count_session.session_id[:8]}")
            
            logger.info(f"Synced {len(active_registered_sessions)} sessions for registered user {email}")
            
        except Exception as e:
            logger.error(f"Failed to sync registered user sessions for {email}: {e}")

    def sync_email_verified_sessions(self, email: str, current_session_id: str):
        """Sync question counts across all active email-verified sessions with the same email"""
        try:
            email_sessions = self.db.find_sessions_by_email(email)
            active_email_verified_sessions = [
                s for s in email_sessions 
                if s.active and s.user_type == UserType.EMAIL_VERIFIED_GUEST
            ]
            
            if not active_email_verified_sessions:
                return
            
            max_count_session = None
            now = datetime.now()
            for sess in active_email_verified_sessions:
                if sess.last_question_time and (now - sess.last_question_time) < DAILY_RESET_WINDOW:
                    if max_count_session is None or sess.daily_question_count > max_count_session.daily_question_count:
                        max_count_session = sess
            
            if max_count_session is None:
                logger.info(f"No recent active sessions for email-verified {email} to sync.")
                return
            
            for sess in active_email_verified_sessions:
                if sess.session_id != max_count_session.session_id:
                    sess.daily_question_count = max_count_session.daily_question_count
                    sess.total_question_count = max_count_session.total_question_count
                    sess.last_question_time = max_count_session.last_question_time
                    
                    if max_count_session.ban_status != BanStatus.NONE:
                        sess.ban_status = max_count_session.ban_status
                        sess.ban_start_time = max_count_session.ban_start_time
                        sess.ban_end_time = max_count_session.ban_end_time
                        sess.ban_reason = max_count_session.ban_reason
                        sess.question_limit_reached = max_count_session.question_limit_reached
                    
                    self.db.save_session(sess)
                    logger.debug(f"Synced session {sess.session_id[:8]} to {max_count_session.daily_question_count} daily questions from {max_count_session.session_id[:8]}")
            
            logger.info(f"Synced {len(active_email_verified_sessions)} email-verified sessions for {email}")
            
        except Exception as e:
            logger.error(f"Failed to sync email-verified sessions for {email}: {e}")

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]) -> bool:
        """Applies fingerprinting data from custom component to the session with better validation.
        This function will now be called only for non-registered users (Guests)."""
        logger.debug(f"🔍 APPLYING FINGERPRINT received from JS: {fingerprint_data.get('fingerprint_id', 'None')[:8]} to session {session.session_id[:8]} (UserType: {session.user_type.value})")
        
        if session.user_type == UserType.REGISTERED_USER:
            logger.warning(f"Attempted to apply fingerprint to REGISTERED_USER {session.session_id[:8]}. Ignoring.")
            return False

        try:
            if not fingerprint_data or not isinstance(fingerprint_data, dict):
                logger.warning("Invalid fingerprint data provided to apply_fingerprinting")
                return False
            
            old_fingerprint_id = session.fingerprint_id
            old_method = session.fingerprint_method
            
            session.fingerprint_id = fingerprint_data.get('fingerprint_id')
            session.fingerprint_method = fingerprint_data.get('fingerprint_method')
            session.browser_privacy_level = fingerprint_data.get('browser_privacy_level', 'standard')
            session.recognition_response = None
            
            if not session.fingerprint_id or not session.fingerprint_method:
                logger.error("Invalid fingerprint data: missing essential fields from JS. Reverting to old fingerprint.")
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return False
            
            self._attempt_fingerprint_inheritance(session)
            
            try:
                self.db.save_session(session)
                logger.info(f"✅ Fingerprinting applied and inheritance checked for {session.session_id[:8]}: {session.fingerprint_method} (ID: {session.fingerprint_id[:8]}...), active={session.active}, rev_pending={session.reverification_pending}")
            except Exception as e:
                logger.error(f"Failed to save session after fingerprinting (JS data received): {e}")
                session.fingerprint_id = old_fingerprint_id
                session.fingerprint_method = old_method
                return False
        except Exception as e:
            logger.error(f"Fingerprint processing failed: {e}", exc_info=True)
            return False
        return True

    def check_fingerprint_history(self, fingerprint_id: str) -> Dict[str, Any]:
        """Check if a fingerprint has historical sessions and return relevant information, now handling multiple emails."""
        
        try:
            existing_sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if not existing_sessions:
                return {'has_history': False}
            
            unique_emails = set()
            for s in existing_sessions:
                if s.email:
                    unique_emails.add(s.email.lower())
            
            if len(unique_emails) > 1:
                logger.info(f"🚨 Multiple emails ({len(unique_emails)}) detected for fingerprint {fingerprint_id[:8]}: {unique_emails}")
                return {
                    'has_history': True,
                    'multiple_emails': True,
                    'email_count': len(unique_emails),
                    'skip_recognition': True
                }
            
            most_privileged_session = None
            for s in existing_sessions:
                if s.user_type == UserType.REGISTERED_USER:
                    logger.info(f"Skipping REGISTERED_USER session {s.session_id[:8]} for fingerprint-based recognition.")
                    continue

                if s.email and (most_privileged_session is None or 
                               self._get_privilege_level(s.user_type) > self._get_privilege_level(most_privileged_session.user_type)):
                    most_privileged_session = s

            if most_privileged_session:
                return {
                    'has_history': True,
                    'multiple_emails': False,
                    'email': most_privileged_session.email,
                    'full_name': most_privileged_session.full_name,
                    'user_type': most_privileged_session.user_type.value,
                    'last_activity': most_privileged_session.last_activity,
                    'daily_question_count': most_privileged_session.daily_question_count
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
                masked_local = local[0] + '*' * (len(local) - 1)
            else:
                masked_local = local[:2] + '*' * (len(local) - 2)
            return f"{masked_local}@{domain}"
        except Exception:
            return "****@****.***"

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Email verification - allows unlimited email switches with OTP verification. No evasion penalties."""
        try:
            sanitized_email = sanitize_input(email).lower().strip()
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', sanitized_email):
                logger.debug(f"handle_guest_email_verification returning FAILURE: Invalid email format for {email}")
                return {'success': False, 'message': 'Please enter a valid email address.'}
            
            if sanitized_email not in session.email_addresses_used:
                session.email_addresses_used.append(sanitized_email)
            
            if not session.reverification_pending:
                if session.email and session.email != sanitized_email:
                    session.email_switches_count += 1
                session.email = sanitized_email
                session.login_method = 'email_verified'
            elif session.reverification_pending and sanitized_email != session.pending_email:
                session.email_switches_count += 1
                session.email = sanitized_email
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                session.login_method = 'email_verified'
                logger.warning(f"Session {session.session_id[:8]} switched email during pending re-verification. Resetting pending state.")

            session.declined_recognized_email_at = None
            
            if session.last_activity is None:
                session.last_activity = datetime.now()

            try:
                self.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session before email verification: {e}")
                final_result = {
                    'success': False, 
                    'message': 'An internal error occurred saving your session. Please try again.'
                }
                logger.debug(f"handle_guest_email_verification returning FAILURE (DB Save): {final_result}")
                return final_result
            
            code_sent = self.email_verification.send_verification_code(sanitized_email)
            
            if code_sent:
                final_result = {
                    'success': True,
                    'message': f'Verification code sent to {sanitized_email}. Please check your email.'
                }
                logger.debug(f"handle_guest_email_verification returning SUCCESS: {final_result}")
                return final_result
            else:
                final_result = {
                    'success': False, 
                    'message': 'Failed to send verification code. Please try again later.'
                }
                logger.debug(f"handle_guest_email_verification returning FAILURE (Code Not Sent): {final_result}")
                return final_result
                
        except Exception as e:
            logger.error(f"Email verification handling failed with unexpected exception: {e}", exc_info=True)
            final_result = {
                'success': False, 
                'message': 'An unexpected error occurred during verification. Please try again.'
            }
            logger.debug(f"handle_guest_email_verification returning FAILURE (Unexpected Exception): {final_result}")
            return final_result


    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """
        Verifies the email verification code and upgrades user status.
        Now correctly handles both:
        1. User declining recognition then verifying same email.
        2. Inheriting the HIGHEST question count during re-verification.
        """
        try:
            email_to_verify = session.pending_email if session.reverification_pending else session.email

            if not email_to_verify:
                logger.error(f"VERIFY EMAIL: No email_to_verify found for session {session.session_id[:8]}.")
                return {'success': False, 'message': 'No email address found for verification.'}
            
            sanitized_code = sanitize_input(code).strip()
            
            if not sanitized_code:
                return {'success': False, 'message': 'Please enter the verification code.'}
            
            verification_success = self.email_verification.verify_code(email_to_verify, sanitized_code)
            
            if verification_success:
                logger.info(f"VERIFY EMAIL: OTP successful for {email_to_verify} in session {session.session_id[:8]}.")

                user_is_reclaiming_declined_account = (
                    session.recognition_response == "no_declined_reco" and
                    session.pending_email and
                    email_to_verify.lower() == session.pending_email.lower()
                )

                if user_is_reclaiming_declined_account:
                    logger.warning(f"VERIFY EMAIL: OVERRIDE - User declined recognition for {session.pending_email} but is now verifying it. Re-enabling reverification logic.")
                    session.reverification_pending = True

                if session.reverification_pending:
                    logger.info(f"VERIFY EMAIL: Entering reverification_pending path for session {session.session_id[:8]}.")
                    
                    current_guest_daily_q = session.daily_question_count
                    current_guest_total_q = session.total_question_count
                    current_guest_last_q_time = session.last_question_time

                    all_email_sessions_for_inheritance = self.db.find_sessions_by_email(email_to_verify)
                    
                    highest_daily_count = current_guest_daily_q
                    highest_total_count = current_guest_total_q
                    most_recent_q_time = current_guest_last_q_time
                    inherited_ban_info = None
                    inherited_tier_cycle_id = None
                    inherited_tier1_completed = False
                    inherited_tier_cycle_started_at = None

                    now = datetime.now()

                    for s in all_email_sessions_for_inheritance:
                        if not session.zoho_contact_id and s.zoho_contact_id:
                            session.zoho_contact_id = s.zoho_contact_id
                            logger.debug(f"VERIFY EMAIL: Inherited Zoho Contact ID {s.zoho_contact_id} from session {s.session_id[:8]}.")

                        if s.last_question_time and (now - s.last_question_time) < DAILY_RESET_WINDOW:
                            if s.daily_question_count > highest_daily_count:
                                highest_daily_count = s.daily_question_count
                                most_recent_q_time = s.last_question_time
                            
                            highest_total_count = max(highest_total_count, s.total_question_count)

                            if s.current_tier_cycle_id:
                                inherited_tier_cycle_id = s.current_tier_cycle_id
                                inherited_tier1_completed = s.tier1_completed_in_cycle
                                inherited_tier_cycle_started_at = s.tier_cycle_started_at

                        if (s.ban_status != BanStatus.NONE and s.ban_end_time and s.ban_end_time > now):
                            inherited_ban_info = {
                                'status': s.ban_status,
                                'start_time': s.ban_start_time,
                                'end_time': s.ban_end_time,
                                'reason': s.ban_reason
                            }
                            inherited_tier_cycle_id = s.current_tier_cycle_id
                            inherited_tier1_completed = s.tier1_completed_in_cycle
                            inherited_tier_cycle_started_at = s.tier_cycle_started_at
                            logger.info(f"VERIFY EMAIL: Inherited active ban from session {s.session_id[:8]}.")
                    
                    session.user_type = session.pending_user_type
                    session.email = session.pending_email
                    session.full_name = session.pending_full_name
                    session.wp_token = session.pending_wp_token
                    
                    session.daily_question_count = highest_daily_count
                    session.total_question_count = highest_total_count
                    session.last_question_time = most_recent_q_time

                    if inherited_ban_info:
                        session.ban_status = inherited_ban_info['status']
                        session.ban_start_time = inherited_ban_info['start_time']
                        session.ban_end_time = inherited_ban_info['end_time']
                        session.ban_reason = inherited_ban_info['reason']
                        session.question_limit_reached = True
                    else:
                        session.ban_status = BanStatus.NONE # Clear any temporary ban status
                        session.question_limit_reached = False

                    # Apply tier cycle info
                    session.current_tier_cycle_id = inherited_tier_cycle_id if inherited_tier_cycle_id else str(uuid.uuid4())
                    session.tier1_completed_in_cycle = inherited_tier1_completed
                    session.tier_cycle_started_at = inherited_tier_cycle_started_at if inherited_tier_cycle_started_at else now

                    # Clean up all pending and recognition flags
                    session.reverification_pending = False
                    session.recognition_response = "reclaimed_successfully" # Track this state
                    session.pending_user_type = None
                    session.pending_email = None
                    session.pending_full_name = None
                    session.pending_zoho_contact_id = None
                    session.pending_wp_token = None
                    session.declined_recognized_email_at = None # Ensure this is cleared on successful verification
                    
                    session.login_method = 'email_verified'
                    session.is_degraded_login = False # Reclaiming an account is not degraded by definition
                    session.degraded_login_timestamp = None

                    logger.info(f"VERIFY EMAIL: ✅ User {session.session_id[:8]} reclaimed {session.user_type.value} for {session.email}. Daily_Q: {session.daily_question_count}.")

                else:
                    logger.info(f"VERIFY EMAIL: Entering standard guest-to-upgrade path for session {session.session_id[:8]}.")
                    # New email verification or upgrade from guest to a non-reclaimed account.
                    old_daily_count = session.daily_question_count # Preserve current count for potential same-session upgrade
                    
                    # This call will determine if it's a REGISTERED_USER or keeps it non-registered
                    # It will also set the login_method and is_degraded_login if applicable.
                    session = self._check_and_upgrade_to_registered(session, email_to_verify, is_fallback_from_wordpress=False)
                    
                    if session.user_type != UserType.REGISTERED_USER:
                        # If not upgraded to Registered, proceed with Email Verified Guest logic
                        session.user_type = UserType.EMAIL_VERIFIED_GUEST
                        session.email = email_to_verify # Ensure email is set on the session
                        
                        # Apply global email-based inheritance for email-verified guests (from any *other* email_verified_guest sessions)
                        should_check_global_email_history_for_guest_upgrade = (
                            session.user_type == UserType.EMAIL_VERIFIED_GUEST # Must be EVG after _check_and_upgrade_to_registered
                        )
                        
                        max_daily_count_global = 0
                        max_total_count_global = 0
                        most_recent_question_time_global = None
                        inherit_ban_global = False
                        ban_info_global = None

                        if should_check_global_email_history_for_guest_upgrade:
                            all_email_sessions_for_global_check = self.db.find_sessions_by_email(email_to_verify)
                            for email_session_check in all_email_sessions_for_global_check:
                                # Only consider other EVG sessions or current session itself for max counts
                                if email_session_check.user_type == UserType.EMAIL_VERIFIED_GUEST:
                                    if (email_session_check.last_question_time and 
                                        (now - email_session_check.last_question_time) < DAILY_RESET_WINDOW):
                                        if email_session_check.daily_question_count > max_daily_count_global:
                                            max_daily_count_global = email_session_check.daily_question_count
                                            most_recent_question_time_global = email_session_check.last_question_time
                                        
                                        if (email_session_check.ban_status == BanStatus.TWENTY_FOUR_HOUR and 
                                            email_session_check.ban_end_time and 
                                            email_session_check.ban_end_time > now):
                                            inherit_ban_global = True
                                            ban_info_global = {
                                                'status': email_session_check.ban_status,
                                                'start_time': email_session_check.ban_start_time,
                                                'end_time': email_session_check.ban_end_time,
                                                'reason': email_session_check.ban_reason
                                            }
                                    max_total_count_global = max(max_total_count_global, email_session_check.total_question_count)
                        
                        if should_check_global_email_history_for_guest_upgrade and (max_daily_count_global > 0 or inherit_ban_global):
                            logger.info(f"VERIFY EMAIL: 🌍 GLOBAL EMAIL INHERITANCE: Applying cross-device limits for {email_to_verify} as EMAIL_VERIFIED_GUEST.")
                            session.daily_question_count = max_daily_count_global
                            session.total_question_count = max(session.total_question_count, max_total_count_global)
                            session.last_question_time = most_recent_question_time_global
                            
                            if inherit_ban_global and ban_info_global:
                                session.ban_status = ban_info_global['status']
                                session.ban_start_time = ban_info_global['start_time']
                                session.ban_end_time = ban_info_global['end_time']
                                session.ban_reason = ban_info_global['reason']
                                session.question_limit_reached = True
                                logger.info(f"VERIFY EMAIL: 🚫 Inherited active ban until {ban_info_global['end_time']}")
                        else:
                            is_same_session_upgrade = (
                                session.user_type == UserType.GUEST and # Was a guest just before this upgrade
                                old_daily_count > 0 # And had asked questions in *this* guest session
                            )
                            
                            if is_same_session_upgrade:
                                logger.info(f"VERIFY EMAIL: 🔄 SAME SESSION UPGRADE: Preserving question count {old_daily_count} from GUEST to EMAIL_VERIFIED_GUEST for {session.session_id[:8]}")
                                session.daily_question_count = old_daily_count
                            else:
                                logger.info(f"VERIFY EMAIL: 🆕 FIRST TIME EMAIL: {email_to_verify} gets fresh start as EMAIL_VERIFIED_GUEST.")
                                session.daily_question_count = 0
                                session.total_question_count = 0
                                session.last_question_time = None
                                session.question_limit_reached = False
                                session.ban_status = BanStatus.NONE
                                session.ban_start_time = None
                                session.ban_end_time = None
                                session.ban_reason = None
                        
                        session.login_method = 'email_verified'
                        session.is_degraded_login = False
                        session.degraded_login_timestamp = None

                        logger.info(f"VERIFY EMAIL: ✅ User {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST: {session.email} with {session.daily_question_count} questions")
                    else:
                        logger.info(f"VERIFY EMAIL: ✅ User {session.session_id[:8]} restored to REGISTERED_USER status via email verification (from _check_and_upgrade_to_registered).")

                # Ensure these are cleared on successful verification regardless of path
                session.question_limit_reached = False
                session.declined_recognized_email_at = None 
            
                if session.last_activity is None:
                    session.last_activity = now # Use current time

                try:
                    self.db.save_session(session)
                    # Sync with other sessions based on the final user type
                    if session.user_type == UserType.REGISTERED_USER:
                        self.sync_registered_user_sessions(session.email, session.session_id)
                    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
                        self.sync_email_verified_sessions(session.email, session.session_id)
                        
                except Exception as e:
                    logger.error(f"VERIFY EMAIL: Failed to save upgraded session {session.session_id[:8]}: {e}", exc_info=True)
            
                return {
                    'success': True,
                    'message': '✅ Email verified successfully!'
                }
            else:
                logger.warning(f"VERIFY EMAIL: OTP verification failed for {email_to_verify} in session {session.session_id[:8]}.")
                return {
                    'success': False,
                    'message': 'Invalid verification code. Please check the code and try again.'
                }
            
        except Exception as e:
            logger.error(f"VERIFY EMAIL: Email code verification failed for session {session.session_id[:8]}: {e}", exc_info=True)
            return {
                'success': False,
                'message': 'Verification failed due to an unexpected error. Please try again.'
            }
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """
        CORRECTED & COMPLETE: Enhanced WordPress authentication with Email Verified fallback option.
        Now prioritizes inheriting question counts from the current in-memory guest session
        before checking the database, preventing race conditions, and fully preserving
        all ban, reset, and tier cycle inheritance logic from the original method.
        Includes extensive debug logging for question count inheritance.
        """
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
                    logger.error(f"WordPress authentication received unexpected Content-Type '{content_type}' with 200 status. Expected application/json. Response text: '{response.text[:200]}'")
                    return self._handle_wordpress_error_with_fallback(
                        "API Response Error", 
                        "WordPress returned an unexpected response. The JWT plugin might be misconfigured.",
                        username
                    )

                try:
                    data = response.json()
                except requests.exceptions.JSONDecodeError as e:
                    logger.error(f"WordPress authentication received non-JSON response with 200 status. Error: {e}")
                    return self._handle_wordpress_error_with_fallback(
                        "JSON Parse Error", 
                        "WordPress returned invalid data. Please try again or contact support.",
                        username
                    )
                
                wp_token = data.get('token')
                user_email = data.get('user_email')
                user_display_name = data.get('user_display_name')
                
                if wp_token and user_email:
                    current_session = self.get_session()
                    now = datetime.now()
                    
                    logger.debug(f"DEBUG AUTH: --- Start authenticate_with_wordpress for session {current_session.session_id[:8]} ---")
                    logger.debug(f"DEBUG AUTH: Initial current_session (from get_session) daily_q: {current_session.daily_question_count}, total_q: {current_session.total_question_count}, last_q_time: {current_session.last_question_time}")

                    inherited_daily_count_from_current_session = 0
                    inherited_total_count_from_current_session = 0
                    inherited_last_question_time_from_current_session = None
                    
                    if current_session.user_type == UserType.GUEST and current_session.daily_question_count > 0:
                        logger.debug(f"DEBUG AUTH: Capturing from in-memory guest: daily={current_session.daily_question_count}, total={current_session.total_question_count}")
                        inherited_daily_count_from_current_session = current_session.daily_question_count
                        inherited_total_count_from_current_session = current_session.total_question_count
                        inherited_last_question_time_from_current_session = current_session.last_question_time
                    else:
                        logger.debug(f"DEBUG AUTH: In-memory guest count is 0 or not a guest ({current_session.user_type.value}).")

                    current_session.fingerprint_id = "not_collected_registered_user"
                    current_session.fingerprint_method = "email_primary"
                    logger.info(f"Registered user {current_session.session_id[:8]} fingerprint marked as not collected/email primary.")

                    all_email_sessions = self.db.find_sessions_by_email(user_email)
                    
                    if all_email_sessions and any(s.session_id != current_session.session_id for s in all_email_sessions):
                        current_session.visitor_type = "returning_visitor"
                        logger.info(f"Registered user {user_email} marked as returning_visitor (found {len(all_email_sessions)} past sessions).")
                    else:
                        current_session.visitor_type = "new_visitor" 
                        logger.info(f"Registered user {user_email} marked as new_visitor (first login).")

                    most_recent_ban_session = None
                    most_recent_db_session_for_counts_and_cycle = None
                    
                    if all_email_sessions:
                        sorted_sessions = sorted(all_email_sessions, 
                                               key=lambda s: s.last_question_time or s.last_activity or s.created_at, 
                                               reverse=True)
                        
                        if sorted_sessions:
                            most_recent_db_session_for_counts_and_cycle = sorted_sessions[0]
                            logger.debug(f"DEBUG AUTH: Most recent DB session for counts: {most_recent_db_session_for_counts_and_cycle.session_id[:8]} daily_q: {most_recent_db_session_for_counts_and_cycle.daily_question_count}, last_q_time: {most_recent_db_session_for_counts_and_cycle.last_question_time}")
                        
                        for sess in sorted_sessions:
                            if (sess.ban_status != BanStatus.NONE and sess.ban_end_time and sess.ban_end_time > now):
                                most_recent_ban_session = sess
                                logger.info(f"DEBUG AUTH: Found active ban in session {sess.session_id[:8]}: {sess.ban_status.value} until {sess.ban_end_time}.")
                                break
                    else:
                        logger.debug("DEBUG AUTH: No historical email sessions found in DB.")

                    final_daily_count = inherited_daily_count_from_current_session
                    final_total_count = inherited_total_count_from_current_session
                    final_last_question_time = inherited_last_question_time_from_current_session
                    
                    logger.debug(f"DEBUG AUTH: final_daily_count initialized to in-memory guest count: {final_daily_count}")

                    if most_recent_db_session_for_counts_and_cycle:
                        db_time_check = most_recent_db_session_for_counts_and_cycle.last_question_time or \
                                        most_recent_db_session_for_counts_and_cycle.last_activity or \
                                        most_recent_db_session_for_counts_and_cycle.created_at
                        
                        time_diff = now - db_time_check if db_time_check else timedelta.max
                        logger.debug(f"DEBUG AUTH: Time difference to DB session ({most_recent_db_session_for_counts_and_cycle.session_id[:8]}): {time_diff} (DAILY_RESET_WINDOW: {DAILY_RESET_WINDOW})")

                        if db_time_check and time_diff < DAILY_RESET_WINDOW:
                            if most_recent_db_session_for_counts_and_cycle.daily_question_count > final_daily_count:
                                logger.info(f"DEBUG AUTH: Inherited higher daily count ({most_recent_db_session_for_counts_and_cycle.daily_question_count}) from DB session {most_recent_db_session_for_counts_and_cycle.session_id[:8]} (email: {user_email}).")
                                final_daily_count = most_recent_db_session_for_counts_and_cycle.daily_question_count
                                final_last_question_time = most_recent_db_session_for_counts_and_cycle.last_question_time
                                current_session.current_tier_cycle_id = most_recent_db_session_for_counts_and_cycle.current_tier_cycle_id
                                current_session.tier1_completed_in_cycle = most_recent_db_session_for_counts_and_cycle.tier1_completed_in_cycle
                                current_session.tier_cycle_started_at = most_recent_db_session_for_counts_and_cycle.tier_cycle_started_at
                            else:
                                logger.debug(f"DEBUG AUTH: DB session is recent but current session's count ({final_daily_count}) is higher/equal. No change to daily count from DB.")
                        else:
                            logger.debug("DEBUG AUTH: DB session is outside DAILY_RESET_WINDOW or has no time check. Checking in-memory.")
                            if not final_last_question_time or (now - final_last_question_time) >= DAILY_RESET_WINDOW:
                                logger.info(f"DEBUG AUTH: Resetting daily count to 0 for fresh start as no recent activity found to inherit (DB or in-memory).")
                                final_daily_count = 0
                                final_last_question_time = None
                                current_session.current_tier_cycle_id = str(uuid.uuid4()) 
                                current_session.tier1_completed_in_cycle = False
                                current_session.tier_cycle_started_at = now
                    else:
                        logger.debug("DEBUG AUTH: No most_recent_db_session_for_counts_and_cycle found.")
                        if not final_last_question_time or (now - final_last_question_time) >= DAILY_RESET_WINDOW:
                            logger.info(f"DEBUG AUTH: Resetting daily count to 0 as no historical DB or recent in-memory activity found.")
                            final_daily_count = 0
                            final_last_question_time = None
                            current_session.current_tier_cycle_id = str(uuid.uuid4())
                            current_session.tier1_completed_in_cycle = False
                            current_session.tier_cycle_started_at = now

                    current_session.daily_question_count = final_daily_count
                    current_session.total_question_count = max(current_session.total_question_count, final_total_count)
                    current_session.last_question_time = final_last_question_time
                    
                    logger.debug(f"DEBUG AUTH: Final daily_question_count after aggregation: {current_session.daily_question_count}")

                    if not current_session.current_tier_cycle_id or not current_session.tier_cycle_started_at:
                        current_session.current_tier_cycle_id = str(uuid.uuid4())
                        current_session.tier1_completed_in_cycle = False
                        current_session.tier_cycle_started_at = now
                        logger.info(f"DEBUG AUTH: Initializing new tier cycle for {user_email} as no active cycle found/inherited.")


                    if most_recent_ban_session:
                        current_session.ban_status = most_recent_ban_session.ban_status
                        current_session.ban_start_time = most_recent_ban_session.ban_start_time
                        current_session.ban_end_time = most_recent_ban_session.ban_end_time
                        current_session.ban_reason = most_recent_ban_session.ban_reason
                        current_session.question_limit_reached = True
                        current_session.current_tier_cycle_id = most_recent_ban_session.current_tier_cycle_id
                        current_session.tier1_completed_in_cycle = most_recent_ban_session.tier1_completed_in_cycle
                        current_session.tier_cycle_started_at = most_recent_ban_session.tier_cycle_started_at
                        logger.info(f"DEBUG AUTH: Inherited active ban: {current_session.ban_status.value} until {current_session.ban_end_time}.")
                    
                    if not current_session.zoho_contact_id:
                        for sess in all_email_sessions:
                            if sess.zoho_contact_id:
                                current_session.zoho_contact_id = sess.zoho_contact_id
                                logger.info(f"DEBUG AUTH: Inherited Zoho contact ID from session {sess.session_id[:8]}.")
                                break

                    current_session.evasion_count = 0
                    current_session.current_penalty_hours = 0
                    current_session.escalation_level = 0
                    
                    current_session.user_type = UserType.REGISTERED_USER
                    current_session.email = user_email
                    current_session.full_name = user_display_name
                    current_session.wp_token = wp_token
                    current_session.last_activity = now
                    current_session.login_method = 'wordpress'
                    current_session.is_degraded_login = False
                    current_session.degraded_login_timestamp = None
                    log_security_event("WORDPRESS_LOGIN_SUCCESS", current_session, {
                        "username": username,
                        "has_wp_token": bool(wp_token)
                    })

                    current_session.reverification_pending = False
                    current_session.pending_user_type = None
                    current_session.pending_email = None
                    current_session.pending_full_name = None
                    current_session.pending_zoho_contact_id = None
                    current_session.pending_wp_token = None
                    current_session.declined_recognized_email_at = None

                    st.session_state.is_chat_ready = True
                    st.session_state.fingerprint_wait_start = None

                    try:
                        self.db.save_session(current_session)
                        logger.info(f"DEBUG AUTH: Session {current_session.session_id[:8]} saved after WordPress login. Final daily_q: {current_session.daily_question_count}.")
                        logger.info(f"✅ REGISTERED_USER setup complete: {user_email}, {current_session.daily_question_count}/{REGISTERED_USER_QUESTION_LIMIT} questions. Chat enabled immediately.")
                        
                        self.sync_registered_user_sessions(user_email, current_session.session_id)

                    except Exception as e:
                        logger.error(f"DEBUG AUTH: Failed to save authenticated session {current_session.session_id[:8]}: {e}", exc_info=True)
                        st.error("Authentication succeeded but session could not be saved. Please try again.")
                        return None
                    
                    logger.debug(f"DEBUG AUTH: --- End authenticate_with_wordpress for session {current_session.session_id[:8]} ---")
                    return current_session
                else:
                    logger.error(f"WordPress authentication successful (status 200) but missing token or email in response. Response: {data}")
                    return self._handle_wordpress_error_with_fallback(
                        "Incomplete Response", 
                        "WordPress returned an incomplete response (missing token/email).",
                        username
                    )
            else:
                logger.warning(f"WordPress authentication failed with status: {response.status_code}. Response: {response.text[:200]}")
                st.error("Invalid username or password.")
                return None
            
        except requests.exceptions.SSLError as e:
            logger.error(f"WordPress SSL/Port 443 error: {e}", exc_info=True)
            return self._handle_wordpress_error_with_fallback(
                "SSL/Connection Error", 
                "Cannot establish secure connection to the authentication server (e.g., Port 443 issue).",
                username
            )
            
        except requests.exceptions.Timeout as e:
            logger.error(f"WordPress authentication timed out: {e}", exc_info=True)
            return self._handle_wordpress_error_with_fallback(
                "Timeout Error",
                "The authentication service is not responding in time. The server may be down or overloaded.",
                username
            )
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"WordPress authentication connection error: {e}", exc_info=True)
            return self._handle_wordpress_error_with_fallback(
                "Connection Error",
                "Could not connect to the authentication service. Please check your internet connection or the service may be down.",
                username
            )
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during WordPress authentication: {e}", exc_info=True)
            return self._handle_wordpress_error_with_fallback(
                "Authentication Error",
                "An unexpected error occurred during authentication. Please try again later.",
                username
            )
    def _handle_wordpress_error_with_fallback(self, error_type: str, error_message: str, username: str) -> Optional[UserSession]:
        """Handle WordPress errors with email verification fallback option."""
        
        st.session_state.wordpress_error = {
            'type': error_type,
            'message': error_message,
            'username': username,
             'show_fallback': True
        }
        
        return None

    def _check_and_upgrade_to_registered(self, session: UserSession, email: str, 
                                        is_fallback_from_wordpress: bool = False) -> UserSession:
        """
        Check if email belongs to a registered user and upgrade if found.
        NEW: Now also synchronizes the current session's fingerprint across all of the
        registered user's historical sessions to permanently merge the identities.
        """
        
        email_sessions = self.db.find_sessions_by_email(email)
        
        registered_sessions = [s for s in email_sessions 
                              if s.user_type == UserType.REGISTERED_USER
                              and s.session_id != session.session_id]
        
        if registered_sessions:
            most_recent_registered = max(registered_sessions, 
                                         key=lambda s: s.last_activity or s.created_at)
            
            logger.info(f"🎯 Found registered user history for {email} - upgrading session {session.session_id[:8]}")
            
            session.user_type = UserType.REGISTERED_USER
            session.full_name = most_recent_registered.full_name
            session.zoho_contact_id = most_recent_registered.zoho_contact_id
            
            if is_fallback_from_wordpress:
                session.is_degraded_login = True
                session.degraded_login_timestamp = datetime.now()
                session.login_method = 'email_fallback'
                session.wp_token = None
                log_security_event("DEGRADED_LOGIN", session, {"reason": "wordpress_auth_failure_fallback"})
                logger.warning(f"⚠️ Degraded login for registered user {email}")
            else:
                session.login_method = 'email_verified'
                session.is_degraded_login = False
            
            now = datetime.now()
            if (most_recent_registered.last_question_time and 
                (now - most_recent_registered.last_question_time) < DAILY_RESET_WINDOW):
                session.daily_question_count = most_recent_registered.daily_question_count
                session.total_question_count = most_recent_registered.total_question_count
                session.last_question_time = most_recent_registered.last_question_time
                session.current_tier_cycle_id = most_recent_registered.current_tier_cycle_id
                session.tier1_completed_in_cycle = most_recent_registered.tier1_completed_in_cycle
                session.tier_cycle_started_at = most_recent_registered.tier_cycle_started_at
                
                if (most_recent_registered.ban_status != BanStatus.NONE and
                    most_recent_registered.ban_end_time and
                    most_recent_registered.ban_end_time > now):
                    session.ban_status = most_recent_registered.ban_status
                    session.ban_start_time = most_recent_registered.ban_start_time
                    session.ban_end_time = most_recent_registered.ban_end_time
                    session.ban_reason = most_recent_registered.ban_reason
            
            current_fingerprint = session.fingerprint_id
            if current_fingerprint and not current_fingerprint.startswith(("temp_", "fallback_", "not_collected_")):
                logger.info(f"🔗 Merging identities: Linking fingerprint {current_fingerprint[:8]} to email {email}")
                for reg_session in registered_sessions:
                    if reg_session.fingerprint_id != current_fingerprint:
                        reg_session.fingerprint_id = current_fingerprint
                        reg_session.fingerprint_method = session.fingerprint_method
                        self.db.save_session(reg_session)
                        logger.debug(f"  - Updated fingerprint for historical session {reg_session.session_id[:8]}")

            self.db.save_session(session)
            
            self.sync_registered_user_sessions(email, session.session_id)
                
            return session
        
        session.login_method = 'email_verified'
        session.is_degraded_login = False
        session.degraded_login_timestamp = None
        session.wp_token = None
        logger.info(f"No registered user history found for {email} - proceeding as email verified guest")
        return session

    def sync_ban_for_registered_user(self, email: str, banned_session: UserSession):
        """Sync ban status with distributed lock to prevent race conditions"""
        try:
            lock_key = f"ban_sync_{email}"
            now = datetime.now()
            
            if lock_key in self._ban_sync_locks:
                lock_time = self._ban_sync_locks[lock_key]
                if now - lock_time < self._ban_lock_timeout:
                    logger.warning(f"Ban sync already in progress for {email}, skipping")
                    return
                else:
                    del self._ban_sync_locks[lock_key]
            
            self._ban_sync_locks[lock_key] = now
            
            try:
                email_sessions = self.db.find_sessions_by_email(email)
                active_registered_sessions = [s for s in email_sessions 
                                              if s.active and s.user_type == UserType.REGISTERED_USER 
                                              and s.session_id != banned_session.session_id]
                
                if not active_registered_sessions:
                    return
                
                for sess in active_registered_sessions:
                    sess.ban_status = banned_session.ban_status
                    sess.ban_start_time = banned_session.ban_start_time
                    sess.ban_end_time = banned_session.ban_end_time
                    sess.ban_reason = banned_session.ban_reason
                    sess.question_limit_reached = banned_session.question_limit_reached
                    
                    sess.current_tier_cycle_id = banned_session.current_tier_cycle_id
                    sess.tier1_completed_in_cycle = banned_session.tier1_completed_in_cycle
                    sess.tier_cycle_started_at = banned_session.tier_cycle_started_at
                    
                    self.db.save_session(sess)
                    logger.debug(f"Synced ban and cycle info to session {sess.session_id[:8]} for registered user {email}")
                
                logger.info(f"Synced ban across {len(active_registered_sessions)} sessions for registered user {email}")
                
            finally:
                if lock_key in self._ban_sync_locks:
                    del self._ban_sync_locks[lock_key]
                    
        except Exception as e:
            logger.error(f"Failed to sync ban for registered user {email}: {e}")
            if lock_key in self._ban_sync_locks:
                del self._ban_sync_locks[lock_key]

    def sync_email_verified_sessions(self, email: str, current_session_id: str):
        """Sync question counts across all active email-verified sessions with the same email"""
        try:
            email_sessions = self.db.find_sessions_by_email(email)
            active_email_verified_sessions = [
                s for s in email_sessions 
                if s.active and s.user_type == UserType.EMAIL_VERIFIED_GUEST
            ]
            
            if not active_email_verified_sessions:
                return
            
            max_count_session = None
            now = datetime.now()
            for sess in active_email_verified_sessions:
                if sess.last_question_time and (now - sess.last_question_time) < DAILY_RESET_WINDOW:
                    if max_count_session is None or sess.daily_question_count > max_count_session.daily_question_count:
                        max_count_session = sess
            
            if max_count_session is None:
                logger.info(f"No recent active sessions for email-verified {email} to sync.")
                return
            
            for sess in active_email_verified_sessions:
                if sess.session_id != max_count_session.session_id:
                    sess.daily_question_count = max_count_session.daily_question_count
                    sess.total_question_count = max_count_session.total_question_count
                    sess.last_question_time = max_count_session.last_question_time
                    
                    if max_count_session.ban_status != BanStatus.NONE:
                        sess.ban_status = max_count_session.ban_status
                        sess.ban_start_time = max_count_session.ban_start_time
                        sess.ban_end_time = max_count_session.ban_end_time
                        sess.ban_reason = max_count_session.ban_reason
                        sess.question_limit_reached = max_count_session.question_limit_reached
                    
                    self.db.save_session(sess)
                    logger.debug(f"Synced session {sess.session_id[:8]} to {max_count_session.daily_question_count} daily questions from {max_count_session.session_id[:8]}")
            
            logger.info(f"Synced {len(active_email_verified_sessions)} email-verified sessions for {email}")
            
        except Exception as e:
            logger.error(f"Failed to sync email-verified sessions for {email}: {e}")

    def detect_pricing_stock_question(self, prompt: str) -> bool:
        """Detect if question is about pricing or stock availability."""

        prompt_lower = prompt.lower()

        pricing_indicators = [
            "price", "pricing", "cost", "costs", "expensive", "cheap", "rate", "rates",
            "quote", "quotation", "budget", "fee", "charge", "tariff"
        ]

        stock_indicators = [
            "stock", "stocks", "availability", "available", "in stock", "out of stock",
            "inventory", "supply", "supplies", "quantity", "quantities", "lead time",
            "delivery time", "MOQ", "minimum order"
        ]

        has_pricing = any(indicator in prompt_lower for indicator in pricing_indicators)
        has_stock = any(indicator in prompt_lower for indicator in stock_indicators)

        return has_pricing or has_stock

    def extract_order_id_from_query(self, prompt: str) -> Optional[str]:
        """Extract order ID from natural language query using LLM."""
        if not self.ai.openai_client:
            import re
            patterns = [
                r'order\s*#?\s*(\d+)',
                r'#\s*(\d+)',
                r'order\s*number\s*(\d+)',
                r'order\s*id\s*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, prompt.lower())
                if match:
                    return match.group(1)
            return None
        
        try:
            extraction_prompt = f"""Extract the order ID/number from this query. 
If no order ID is mentioned, respond with "NONE".
Only return the numeric ID, no other text.

Query: "{prompt}"

Order ID:"""

            response = self.ai.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You extract order IDs from queries. Respond only with the numeric ID or NONE."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=20,
                temperature=0
            )
            
            extracted_id = response.choices[0].message.content.strip()
            
            if extracted_id and extracted_id != "NONE":
                cleaned_id = ''.join(filter(str.isdigit, extracted_id))
                return cleaned_id if cleaned_id else None
                
        except Exception as e:
            logger.error(f"LLM order ID extraction failed: {e}")
            
        return None
    
    def is_order_query(self, prompt: str) -> bool:
        """Check if the query is about retrieving an order."""
        order_keywords = [
            'order', 'purchase', 'invoice', 'receipt', 
            'transaction', 'order status', 'order details',
            'order number', 'order id', 'check order', 'my order'
        ]
        
        prompt_lower = prompt.lower()
        
        has_order_keyword = any(keyword in prompt_lower for keyword in order_keywords)
        
        return has_order_keyword


    def detect_meta_conversation_query_llm(self, prompt: str) -> Dict[str, Any]:
        """LLM-powered detection of meta-conversation queries."""
        if not self.ai.openai_client:
            logger.warning("OpenAI client not available for LLM meta-query detection. Falling back to keyword-based.")
            return self.detect_meta_conversation_query_keyword_fallback(prompt)

        detection_prompt = f"""Analyze if this query is asking about the conversation/chat itself rather than food & beverage industry topics.

QUERY: "{prompt}"

META-CONVERSATION INDICATORS:
- Asking for summaries of our chat
- Counting questions or messages
- Listing what was discussed
- Analyzing conversation topics/history
- Requesting chat statistics

INDUSTRY QUESTIONS (NOT META):
- Questions about food/beverage products, ingredients, suppliers
- Technical queries about processing, formulation
- Regulatory or compliance questions
- Any actual business question

Respond ONLY with JSON:
{{"is_meta": true, "type": "summarize|count|list|analyze|general|none", "confidence": 0.0-1.0}}"""

        try:
            response = self.ai.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a search query optimizer. Respond only with the optimized search query."},
                {"role": "user", "content": detection_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            response_content = response.choices[0].message.content.strip().replace('```json', '').replace('```', '').strip()
            result = json.loads(response_content)
            
            if not isinstance(result, dict) or 'is_meta' not in result or 'type' not in result:
                logger.warning(f"LLM meta-query detection returned invalid JSON: {response_content}. Falling back to keyword.")
                return self.detect_meta_conversation_query_keyword_fallback(prompt)

            logger.info(f"LLM Meta-query detection: {prompt} -> {result['is_meta']} ({result.get('type')}) Confidence: {result.get('confidence', 0.0):.2f}")
            return result

        except Exception as e:
            logger.error(f"LLM meta-query detection failed: {e}. Falling back to keyword-based detection.", exc_info=True)
            return self.detect_meta_conversation_query_keyword_fallback(prompt)

    def detect_meta_conversation_query_keyword_fallback(self, prompt: str) -> Dict[str, Any]:
        """Detect if user is asking about conversation history using static keywords (fallback)."""
        prompt_lower = prompt.lower().strip()
    
        if prompt_lower.isdigit():
            return {"is_meta": False, "type": "none", "confidence": 1.0}

        summary_patterns = [
            "summarize", "summary of", "give me a summary", "can you summarize",
            "overview of", "recap of", "sum up"
        ]

        question_patterns = [
            "what did i ask", "what all did i ask", "what have i asked", 
            "all my questions", "my previous questions", "my questions",
            "list my questions", "show my questions", "questions i asked"
        ]

        count_patterns = [
            "how many questions", "count my questions", "number of questions",
            "how many times", "total questions"
        ]

        topic_patterns = [
            "what topics", "what have we discussed", "topics we covered",
            "what did we talk about", "conversation topics", "discussion topics"
        ]

        conversation_patterns = [
            "conversation history", "chat history", "our conversation", 
            "this conversation", "our chat", "this chat", "my session"
        ]

        if any(pattern in prompt_lower for pattern in summary_patterns):
            return {"is_meta": True, "type": "summarize", "scope": "all"}
        elif any(pattern in prompt_lower for pattern in question_patterns):
            return {"is_meta": True, "type": "list", "scope": "questions"}
        elif any(pattern in prompt_lower for pattern in count_patterns):
            return {"is_meta": True, "type": "count", "scope": "questions"}
        elif any(pattern in prompt_lower for pattern in topic_patterns):
            return {"is_meta": True, "type": "analyze", "scope": "topics"}
        elif any(pattern in prompt_lower for pattern in conversation_patterns):
            return {"is_meta": True, "type": "general", "scope": "conversation"}
        
        return {"is_meta": False, "type": "none", "confidence": 1.0}

    def handle_meta_conversation_query(self, session: UserSession, query_type: str, scope: str) -> Dict[str, Any]:
        """Handle meta-conversation queries with code-based analysis (zero token cost)."""
        
        visible_messages = session.messages[session.display_message_offset:]
        user_questions = [msg['content'] for msg in visible_messages if msg.get('role') == 'user']

        if query_type == "count":
            return {
                "content": f"""📊 **Session Statistics:**

• **Questions Asked**: {len(user_questions)}
• **Total Messages**: {len(visible_messages)}
• **Session Started**: {session.created_at.strftime('%B %d, %Y at %H:%M')}
• **User Type**: {session.user_type.value.replace('_', ' ').title()}
• **Daily Usage**: {session.daily_question_count}/{self.question_limits.question_limits[session.user_type.value]} questions today""",
                "success": True,
                "source": "Session Analytics"
            }

        elif query_type == "list":
            if not user_questions:
                return {
                    "content": "You haven't asked any questions yet in this session.",
                    "success": True,
                    "source": "Session History"
                }
            
            questions_to_show = user_questions[-20:] if len(user_questions) > 20 else user_questions
            start_number = len(user_questions) - len(questions_to_show) + 1 if len(user_questions) > 20 else 1

            questions_list = []
            for i, q in enumerate(questions_to_show):
                display_q = q[:100] + "..." if len(q) > 100 else q
                questions_list.append(f"{start_number + i}. {display_q}")

            response_content = f"📋 **Your Questions in This Session:**\n\n" + \
                               "\n".join(questions_list)
            
            if len(user_questions) > 20:
                response_content += f"\n\n*Showing last 20 questions out of {len(user_questions)} total.*"

            return {
                "content": response_content,
                "success": True,
                "source": "Session History"
            }

        elif query_type in ["summarize", "general"]:
            return self._generate_conversation_summary(user_questions, session)

        elif query_type == "analyze":
            return self._analyze_conversation_topics(user_questions, session)

        return {"content": "I couldn't process that conversation query.", "success": False}

    def _generate_conversation_summary(self, user_questions: List[str], session: UserSession) -> Dict[str, Any]:
        """Generate conversation summary using code-based analysis."""

        if not user_questions:
            return {
                "content": "No questions have been asked in this session yet.",
                "success": True,
                "source": "Session Summary"
            }

        topic_words = set()
        question_categories = {
            'pricing': 0, 'suppliers': 0, 'technical': 0, 'regulatory': 0, 'applications': 0
        }

        for question in user_questions:
            q_lower = question.lower()
            words = [w for w in q_lower.split() if len(w) > 4 and w not in ['what', 'where', 'when', 'about', 'would', 'could', 'should', 'which']]
            topic_words.update(words[:3])

            if any(word in q_lower for word in ['price', 'pricing', 'cost', 'expensive']):
                question_categories['pricing'] += 1
            elif any(word in q_lower for word in ['supplier', 'source', 'vendor', 'manufacturer']):
                question_categories['suppliers'] += 1
            elif any(word in q_lower for word in ['regulation', 'compliance', 'standard', 'certification']):
                question_categories['regulatory'] += 1
            elif any(word in q_lower for word in ['application', 'use', 'formulation', 'recipe']):
                question_categories['applications'] += 1
            else:
                question_categories['technical'] += 1

        key_topics = list(topic_words)[:8]
        active_categories = {k: v for k, v in question_categories.items() if v > 0}

        summary_parts = [
            f"📈 **Conversation Summary:**\n",
            f"• **Total Questions**: {len(user_questions)}",
            f"• **Session Duration**: Started {session.created_at.strftime('%B %d at %H:%M')}",
            f"• **Key Topics Discussed**: {', '.join(key_topics) if key_topics else 'General inquiries'}"
        ]

        if active_categories:
            summary_parts.append("\n**Question Breakdown:**")
            for category, count in active_categories.items():
                summary_parts.append(f"• **{category.title()}**: {count} question{'s' if count > 1 else ''}")
        
        summary_parts.append(f"\n• **User Status**: {session.user_type.value.replace('_', ' ').title()}")

        return {
            "content": "\n".join(summary_parts),
            "success": True,
            "source": "Conversation Summary",
            "analyzed_questions": len(user_questions)
        }

    def _analyze_conversation_topics(self, user_questions: List[str], session: UserSession) -> Dict[str, Any]:
        """Analyze conversation topics using code-based extraction."""

        if not user_questions:
            return {
                "content": "No topics to analyze - no questions asked yet.",
                "success": True,
                "source": "Topic Analysis"
            }

        ingredients_mentioned = set()
        business_aspects = set()
        technical_terms = set()

        ingredient_indicators = ['extract', 'powder', 'oil', 'acid', 'syrup', 'sweetener', 'flavor', 'color']

        for question in user_questions:
            words = question.split()
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?')

                if any(indicator in question.lower() for indicator in ingredient_indicators):
                    if word.istitle() and len(word) > 3:
                        ingredients_mentioned.add(word)
                
                if word_lower in ['supplier', 'vendor', 'sourcing', 'pricing', 'cost', 'availability', 'stock']:
                    business_aspects.add(word_lower)

                if word_lower in ['formulation', 'application', 'specification', 'grade', 'purity', 'concentration']:
                    technical_terms.add(word_lower)

        temp_topic_words = set()
        for question in user_questions:
            q_lower = question.lower()
            words = [w for w in q_lower.split() if len(w) > 4 and w not in ['what', 'where', 'when', 'about', 'would', 'could', 'should', 'which']]
            temp_topic_words.update(words[:3])

        key_topics = list(temp_topic_words)[:8]
        temp_question_categories = {
            'pricing': 0, 'suppliers': 0, 'technical': 0, 'regulatory': 0, 'applications': 0
        }
        for question in user_questions:
            q_lower = question.lower()
            if any(word in q_lower for word in ['price', 'pricing', 'cost', 'expensive']):
                temp_question_categories['pricing'] += 1
            elif any(word in q_lower for word in ['supplier', 'source', 'vendor', 'manufacturer']):
                temp_question_categories['suppliers'] += 1
            elif any(word in q_lower for word in ['regulation', 'compliance', 'standard', 'certification']):
                temp_question_categories['regulatory'] += 1
            elif any(word in q_lower for word in ['application', 'use', 'formulation', 'recipe']):
                temp_question_categories['applications'] += 1
            else:
                temp_question_categories['technical'] += 1

        active_categories = {k: v for k, v in temp_question_categories.items() if v > 0}

        analysis_parts = [
            f"🔍 **Topic Analysis:**\n",
            f"• **Questions Analyzed**: {len(user_questions)}"
        ]

        if ingredients_mentioned:
            analysis_parts.append(f"• **Ingredients Discussed**: {', '.join(list(ingredients_mentioned)[:6])}")

        if business_aspects:
            analysis_parts.append(f"• **Business Aspects**: {', '.join(list(business_aspects))}")

        if technical_terms:
            analysis_parts.append(f"• **Technical Focus**: {', '.join(list(technical_terms))}")

        focus_areas = []
        combined_text = ' '.join(user_questions).lower()

        if any(term in combined_text for term in ['bakery', 'bread', 'cake', 'pastry']):
            focus_areas.append('Bakery')
        if any(term in combined_text for term in ['beverage', 'drink', 'juice', 'soda']):
            focus_areas.append('Beverages')
        if any(term in combined_text for term in ['dairy', 'milk', 'cheese', 'yogurt']):
            focus_areas.append('Dairy')
        if any(term in combined_text for term in ['confection', 'candy', 'chocolate', 'sweet']):
            focus_areas.append('Confectionery')

        if focus_areas:
            analysis_parts.append(f"• **Industry Focus**: {', '.join(focus_areas)}")


        return {
            "content": "\n".join(analysis_parts),
            "success": True,
            "source": "Topic Analysis"
        }
    
    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Gets AI response and manages session state with a corrected processing pipeline."""
        try:
            logger.info(f"Processing prompt: '{prompt[:100]}' | Session: {session.session_id[:8]} | Type: {session.user_type.value}")
            rate_limiter_id = session.fingerprint_id
            if (session.user_type in [UserType.REGISTERED_USER, UserType.EMAIL_VERIFIED_GUEST]) and session.email:
                rate_limiter_id = f"email_{session.email.lower()}"
                logger.debug(f"Rate limiter using email for {session.user_type.value}: {session.email}")
            elif rate_limiter_id is None or rate_limiter_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                rate_limiter_id = session.session_id
                logger.debug(f"Rate limiter using session ID as fallback: {session.session_id[:8]}")
            
            # 1. Rate limit check
            rate_limit_result = self.rate_limiter.is_allowed(rate_limiter_id)
            if not rate_limit_result['allowed']:
                time_until_next = rate_limit_result.get('time_until_next', 0)
                max_requests = RATE_LIMIT_REQUESTS
                window_seconds = RATE_LIMIT_WINDOW_SECONDS
                
                st.session_state.rate_limit_hit = {
                    'timestamp': datetime.now(),
                    'time_until_next': time_until_next,
                    'message': f"Rate limit exceeded. Please wait {time_until_next} seconds before asking another question. ({max_requests} questions per {window_seconds} seconds allowed)"
                }
                
                return {
                    'content': st.session_state.rate_limit_hit['message'],
                    'success': False,
                    'source': 'Rate Limiter',
                    'time_until_next': time_until_next
                }
            
            # 2. Content moderation check
            moderation_result = check_content_moderation(prompt, self.ai.openai_client)
            if moderation_result and moderation_result.get("flagged"):
                categories = moderation_result.get('categories', [])
                logger.warning(f"Input flagged by moderation for: {', '.join(categories)}")
                
                st.session_state.moderation_flagged = {
                    'timestamp': datetime.now(),
                    'categories': categories,
                    'message': moderation_result.get("message", "Your message violates our content policy. Please rephrase your question.")
                }
                
                return {
                    "content": moderation_result.get("message", "Your message violates our content policy. Please rephrase your question."),
                    "success": False, "source": "Content Moderation", "used_search": False, "used_pinecone": False,
                    "has_citations": False, "has_inline_citations": False, "safety_override": False
                }

            # 3. Pricing/Stock check
            logger.debug(f"Checking pricing/stock query for: '{prompt}'")
            if self.detect_pricing_stock_question(prompt):
                try:
                    self.question_limits.record_question_and_check_ban(session, self)
                except Exception as e:
                    logger.error(f"Failed to record pricing/stock query for {session.session_id[:8]}: {e}")
                    return {'content': 'An error occurred while tracking your question. Please try again.', 'success': False, 'source': 'Question Tracker'}

                session.messages.append({'role': 'user', 'content': prompt})
                st.session_state.pricing_stock_notice = {
                    'timestamp': datetime.now(),
                    'query_type': 'pricing' if any(word in prompt.lower() for word in ['price', 'pricing', 'cost']) else 'stock',
                    'message': """Thank you for your interest in pricing information. For the most accurate and up-to-date pricing and quotes, please visit the product page directly on our website or contact our sales team at sales-eu@12taste.com for personalized assistance."""
                }
                self._update_activity(session)
                return {'content': "", 'success': True, 'source': 'Business Rules', 'is_pricing_stock_redirect': True, 'display_only_notice': True}

            # 3.5 WooCommerce Order Check (ENHANCED)
            logger.debug(f"Checking 12Taste Order Status query for: '{prompt}'")
            if hasattr(self, 'woocommerce') and self.woocommerce and self.woocommerce.config.WOOCOMMERCE_ENABLED:
                expecting_order_id = False
                if session.messages:
                    for msg in reversed(session.messages):
                        if msg.get('role') == 'assistant' and msg.get('source') == '12Taste Order Status':
                            if 'order ID' in msg.get('content', '') or 'order number' in msg.get('content', ''):
                                expecting_order_id = True
                                break
                        if msg.get('role') == 'user':
                            break
                
                if self.is_order_query(prompt) or (expecting_order_id and prompt.strip().isdigit()):
                    if not (session.user_type == UserType.REGISTERED_USER or session.user_type == UserType.EMAIL_VERIFIED_GUEST) or not session.email:
                        message = "To check order status, please sign in as a Registered User or verify your email."
                        session.messages.append({'role': 'user', 'content': prompt})
                        session.messages.append({'role': 'assistant', 'content': message, 'source': '12Taste Order Status'})
                        self._update_activity(session)
                        if 'expecting_order_id' in st.session_state: del st.session_state['expecting_order_id']
                        return {'content': message, 'success': False, 'source': '12Taste Order Status'}

                    order_id = self.extract_order_id_from_query(prompt)
                    if not order_id and expecting_order_id and prompt.strip().isdigit():
                        order_id = prompt.strip()

                    if order_id:
                        logger.info(f"Order query detected for order ID: {order_id}. User email: {session.email}")
                        
                        try:
                            self.question_limits.record_question_and_check_ban(session, self)
                        except Exception as e:
                            logger.error(f"Failed to record order query for {session.session_id[:8]}: {e}")
                            return {'content': 'An error occurred while tracking your question. Please try again.', 'success': False, 'source': 'Question Tracker'}
                        
                        # NEW: Use asyncio.run to execute async WooCommerce methods
                        customer = asyncio.run(self.woocommerce.get_customer_by_email(session.email))
                        customer_id = customer.get('id') if customer else None

                        if not customer_id:
                            message = f"We could not find a customer account linked to your email ({session.email}). Please ensure you are logged in with the email used for your order."
                            session.messages.append({'role': 'user', 'content': prompt})
                            session.messages.append({'role': 'assistant', 'content': message, 'source': '12Taste Order Status'})
                            self._update_activity(session)
                            if 'expecting_order_id' in st.session_state: del st.session_state['expecting_order_id']
                            return {
                                'content': message,
                                'success': False,
                                'source': '12Taste Order Status'
                            }

                        order_data = asyncio.run(self.woocommerce.get_order(order_id, customer_id=customer_id))
                        
                        if order_data and not order_data.get("error"):
                            formatted_order = self.woocommerce.format_order_for_display(order_data)
                            
                            session.messages.append({'role': 'user', 'content': prompt})
                            session.messages.append({
                                'role': 'assistant', 
                                'content': formatted_order,
                                'source': '12Taste Order Status',
                                'order_id': order_id
                            })
                            self._update_activity(session)
                            if 'expecting_order_id' in st.session_state: del st.session_state['expecting_order_id']
                            return {
                                'content': formatted_order,
                                'success': True,
                                'source': '12Taste Order Status',
                                'order_id': order_id
                            }
                        else:
                            error_message = order_data.get("message", "Order not found or does not belong to your account.")
                            session.messages.append({'role': 'user', 'content': prompt})
                            session.messages.append({'role': 'assistant', 'content': f"❌ {error_message}", 'source': '12Taste Order Status'})
                            self._update_activity(session)
                            if 'expecting_order_id' in st.session_state: del st.session_state['expecting_order_id']
                            return {
                                'content': f"❌ {error_message}",
                                'success': False,
                                'source': '12Taste Order Status'
                            }
                    else:
                        try:
                            self.question_limits.record_question_and_check_ban(session, self)
                        except Exception as e:
                            logger.error(f"Failed to record order ID prompt for {session.session_id[:8]}: {e}")
                            return {'content': 'An error occurred while tracking your question. Please try again.', 'success': False, 'source': 'Question Tracker'}
                        
                        message = "I can help with order statuses, but I need the order ID/number. Could you please provide it?"
                        session.messages.append({'role': 'user', 'content': prompt})
                        session.messages.append({'role': 'assistant', 'content': message, 'source': '12Taste Order Status'})
                        self._update_activity(session)
                        st.session_state.expecting_order_id = True
                        return {
                            'content': message,
                            'success': True,
                            'source': '12Taste Order Status'
                        }
            if 'expecting_order_id' in st.session_state: del st.session_state['expecting_order_id']


            # 4. LLM-driven Industry context check (NOW RUNS BEFORE META-DETECTION)
            logger.debug(f"Checking industry context for: '{prompt}'")
            context_result = check_industry_context(prompt, session.messages, self.ai.openai_client)
            if context_result:
                category = context_result.get("category")
                is_relevant = context_result.get("relevant", True)

                # A. Handle greetings immediately and return
                if category == "greeting_or_polite":
                    try:
                        self.question_limits.record_question_and_check_ban(session, self)
                    except Exception as e:
                        logger.error(f"Failed to record greeting for {session.session_id[:8]}: {e}")
                        return {'content': 'An error occurred while tracking your question.', 'success': False, 'source': 'Question Tracker'}

                    session.messages.append({'role': 'user', 'content': prompt})
                    friendly_response = {"content": "Hello! I'm FiFi, your AI assistant for the food & beverage industry. How can I help you today?", "success": True, "source": "FiFi", "is_meta_response": True}
                    session.messages.append({'role': 'assistant', 'content': friendly_response['content'], 'source': 'Greeting', 'is_meta_response': True})
                    self._update_activity(session)
                    return friendly_response
                
                # B. Block irrelevant questions, but specifically ALLOW meta_conversation and order_queries to pass
                elif not is_relevant and category not in ["meta_conversation", "order_query"]:
                    confidence = context_result.get("confidence", 0.0)
                    reason = context_result.get("reason", "Not relevant")
                    context_message = "I'm specialized in helping food & beverage industry professionals. Please ask me about ingredients, suppliers, or market trends."
                    st.session_state.context_flagged = {'timestamp': datetime.now(), 'category': category, 'confidence': confidence, 'reason': reason, 'message': context_message}
                    return {"content": context_message, "success": False, "source": "Industry Context Filter", "used_search": False, "used_pinecone": False, "has_citations": False, "has_inline_citations": False, "safety_override": False}

            # 5. LLM-driven Meta-conversation query detection (NOW RUNS AFTER CONTEXT CHECK)
            logger.debug(f"Checking meta-conversation for: '{prompt}'")
            meta_detection = self.detect_meta_conversation_query_llm(prompt)
            if meta_detection["is_meta"]:
                logger.info(f"Meta-conversation query detected: {meta_detection['type']}")
                try:
                    self.question_limits.record_question_and_check_ban(session, self)
                except Exception as e:
                    return {'content': 'An error occurred while tracking your question.', 'success': False, 'source': 'Question Tracker'}
                
                session.messages.append({'role': 'user', 'content': prompt})
                meta_response = self.handle_meta_conversation_query(session, meta_detection["type"], meta_detection.get("scope", ""))
                session.messages.append({'role': 'assistant', 'content': meta_response.get('content'), 'source': meta_response.get('source'), 'is_meta_response': True})
                self._update_activity(session)
                return meta_response

            # --- If we've reached here, it's a valid, on-topic industry question ---
            limit_check = self.question_limits.is_within_limits(session)
            if not limit_check['allowed']:
                message = limit_check.get('message', 'Access restricted due to usage policy.')
                return {'content': message, 'success': False, 'source': 'Question Limiter', 'banned': True, 'time_remaining': limit_check.get('time_until_next')} # Ensure consistent key

            self._clear_error_notifications()
            sanitized_prompt = sanitize_input(prompt)
            if not sanitized_prompt:
                return {'content': 'Please enter a valid question.', 'success': False, 'source': 'Input Validation'}
            
            try:
                question_record_status = self.question_limits.record_question_and_check_ban(session, self)
                if question_record_status.get("ban_applied") or question_record_status.get("existing_ban_inherited"):
                    message = self.question_limits._get_ban_message(session, question_record_status.get("ban_type"))
                    time_remaining_td = session.ban_end_time - datetime.now() if session.ban_end_time else None
                    return {'content': message, 'success': False, 'source': 'Question Limiter', 'banned': True, 'time_remaining': time_remaining_td}
            except Exception as e:
                logger.error(f"❌ Critical error recording question: {e}")
                return {'content': 'A critical error occurred while recording your question.', 'success': False, 'source': 'Question Tracking Error'}
            
            ai_response = self.ai.get_response(sanitized_prompt, session.messages)
            
            user_message = {'role': 'user', 'content': sanitized_prompt}
            assistant_message = {'role': 'assistant', 'content': ai_response.get('content', 'No response.'), 'source': ai_response.get('source'), **ai_response}
            session.messages.extend([user_message, assistant_message])
            self._update_activity(session)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}", exc_info=True)
            return {'content': 'I encountered an error processing your request.', 'success': False, 'source': 'Error Handler'}
        
    def clear_chat_history(self, session: UserSession):
        """Clears chat history using soft clear mechanism."""
        try:
            if self._is_manual_crm_save_eligible(session):
                logger.info(f"CRM Save triggered by Clear Chat for session {session.session_id[:8]}")
                self.zoho.save_chat_transcript_sync(session, "Clear Chat")
            
            session.display_message_offset = len(session.messages)
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
            
            session.active = False
            session.last_activity = datetime.now()
            
            try:
                self.db.save_session(session)
            except Exception as e:
                logger.error(f"Failed to save session during end_session: {e}")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.session_state['page'] = None
            
            st.success("👋 You have been signed out successfully!")
            logger.info(f"Session {session.session_id[:8]} ended by user.")
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['page'] = None

    def check_if_attempting_to_exceed_limits(self, session: UserSession) -> bool:
        """
        Check if user is attempting to ask a question beyond their limits.
        This function displays UI messages and returns True/False to block chat input.
        Only blocks for ACTIVE bans, not for "about to exceed" scenarios.
        """
        limit_check = self.question_limits.is_within_limits(session)
        
        if not limit_check.get('allowed', True):
            reason = limit_check.get('reason')
            message = limit_check.get('message', 'Access restricted due to usage policy.')
            
            if reason == 'banned':
                ban_type = limit_check.get('ban_type', 'unknown')
                time_remaining = limit_check.get('time_until_next')
                
                st.error("🚫 **Access Restricted**")
                if time_remaining: # time_until_next is an int in seconds here
                    hours = time_remaining // 3600
                    minutes = (time_remaining % 3600) // 60
                    st.error(f"Time remaining: {hours}h {minutes}m")
                st.info(message)
                return True
                
            elif reason == 'guest_limit':
                st.error("🛑 **Guest Limit Reached**")
                st.info(f"You've used your {GUEST_QUESTION_LIMIT} guest questions. Please verify your email to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} more questions per day!")
                return True

            elif reason == 'email_verified_guest_limit':
                st.error("🛑 **Daily Limit Reached**")
                st.info(f"You've used your {EMAIL_VERIFIED_QUESTION_LIMIT} questions for today. Your questions reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!")
                return True
            
        return False

def log_security_event(event_type: str, session: UserSession, details: Dict[str, Any]):
    """Log security-relevant events for audit trail"""
    logger.info(f"SECURITY_EVENT: {event_type} | "
                f"session_id={session.session_id[:8]} | "
                f"user_type={session.user_type.value} | "
                f"email={session.email} | "
                f"login_method={session.login_method} | "
                f"is_degraded={session.is_degraded_login} | "
                f"details={json.dumps(details)}")


def render_simple_activity_tracker(session_id: str):
    """Renders a simple activity tracker that monitors user interactions."""
    if not session_id:
        logger.warning("render_simple_activity_tracker called without session_id")
        return None
    
    safe_session_id = session_id.replace('-', '_')
    component_key = f"activity_tracker_{safe_session_id}"

    simple_tracker_js = f"""
    (() => {{
        const sessionId = {json.dumps(session_id)};
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
            console.log('📍 Simple activity tracker starting for', sessionId.substring(0, 8));
            
            function updateActivity() {{
                state.lastActivity = Date.now();
            }}
            
            const events = ['mousedown', 'mousemove', 'keydown', 'click', 'scroll', 'touchstart', 'focus'];
            events.forEach(eventType => {{
                document.addEventListener(eventType, updateActivity, {{ passive: true, capture: true }});
            }});
            
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

def handle_emergency_save_requests_from_query():
    """
    Checks for and processes emergency save requests sent via URL query parameters.
    This acts as a fallback if the FastAPI beacon failed.
    """
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    reason = query_params.get("reason")
    fallback_flag = query_params.get("fallback")

    if event == "emergency_close" and session_id and reason and fallback_flag == "true":
        logger.info(f"🚨 Received emergency_close event from query params (fallback mode) for session {session_id[:8]} (Reason: {reason})")
        
        # Clear query parameters to prevent re-triggering this logic on subsequent reruns
        params_to_clear = ["event", "session_id", "reason", "fallback"]
        for param in params_to_clear:
            if param in st.query_params:
                del st.query_params[param]
        
        # Process the emergency save using the Streamlit backend logic
        processed_successfully = process_emergency_save_from_query(session_id, reason)
        if processed_successfully:
            logger.info(f"✅ Emergency save from query params processed successfully for session {session_id[:8]}.")
        else:
            logger.error(f"❌ Emergency save from query params failed to process for session {session_id[:8]}.")
        
        # Trigger a rerun if this was handled, to ensure UI state reflects the change
        st.rerun()
        return True
    return False

def check_timeout_and_trigger_reload(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]) -> bool:
    """Check if timeout has occurred and trigger browser reload using a robust, non-blocking method."""
    if not session or not session.session_id:
        logger.debug("No valid session for timeout check.")
        return False
    
    fresh_session_from_db = session_manager.db.load_session(session.session_id)
    
    if fresh_session_from_db:
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
        session.reverification_pending = fresh_session_from_db.reverification_pending
        session.pending_user_type = fresh_session_from_db.pending_user_type
        session.pending_email = fresh_session_from_db.pending_email
        session.pending_full_name = fresh_session_from_db.pending_full_name
        session.pending_zoho_contact_id = fresh_session_from_db.pending_zoho_contact_id
        session.pending_wp_token = fresh_session_from_db.pending_wp_token
        session.declined_recognized_email_at = fresh_session_from_db.declined_recognized_email_at
        session.timeout_detected_at = fresh_session_from_db.timeout_detected_at
        session.timeout_reason = fresh_session_from_db.timeout_reason
        session.current_tier_cycle_id = fresh_session_from_db.current_tier_cycle_id
        session.tier1_completed_in_cycle = fresh_session_from_db.tier1_completed_in_cycle
        session.tier_cycle_started_at = fresh_session_from_db.tier_cycle_started_at
        session.login_method = fresh_session_from_db.login_method
        session.is_degraded_login = fresh_session_from_db.is_degraded_login
        session.degraded_login_timestamp = fresh_session_from_db.degraded_login_timestamp
    else:
        logger.warning(f"Session {session.session_id[:8]} from st.session_state not found in database. Forcing reset.")
        session.active = False

    if not session.active:
        logger.info(f"Session {session.session_id[:8]} is inactive. Triggering reload to welcome page.")
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.session_state['page'] = None
        st.session_state['session_expired'] = True
        
        reload_script = "<script>parent.window.location.reload();</script>"
        components.html(reload_script, height=100, width=0)
        
        st.info("🏠 Redirecting to home page...")
        st.stop()
        return True

    if session.last_activity is None:
        logger.debug(f"Session {session.session_id[:8]}: last_activity is None, timer has not started.")
        return False
        
    if activity_result:
        try:
            js_last_activity_timestamp = activity_result.get('last_activity')
            if js_last_activity_timestamp:
                new_activity_dt = datetime.fromtimestamp(js_last_activity_timestamp / 1000)
                
                if new_activity_dt > session.last_activity:
                    session.last_activity = new_activity_dt
                    session_manager._save_session_with_retry(session) 
        except Exception as e:
            logger.error(f"Error processing client JS activity timestamp for session {session.session_id[:8]}: {e}")

    time_since_activity = datetime.now() - session.last_activity
    minutes_inactive = time_since_activity.total_seconds() / 60
    
    logger.info(f"TIMEOUT CHECK: Session {session.session_id[:8]} | Inactive: {minutes_inactive:.1f}m | last_activity: {session.last_activity.strftime('%H:%M:%S')}")
    
    if minutes_inactive >= SESSION_TIMEOUT_MINUTES:
        logger.info(f"⏰ TIMEOUT DETECTED: {session.session_id[:8]} inactive for {minutes_inactive:.1f} minutes")
        
        if session_manager._is_crm_save_eligible(session, "timeout_auto_reload"):
            logger.info(f"💾 Performing emergency save (via FastAPI beacon) before auto-reload for {session.session_id[:8]}")
            try:
                emergency_data = {
                    "session_id": session.session_id,
                    "reason": "timeout_auto_reload",
                    "timestamp": int(time.time() * 1000)
                }
                requests.post(FASTAPI_EMERGENCY_SAVE_URL, json=emergency_data, timeout=FASTAPI_EMERGENCY_SAVE_TIMEOUT)
                logger.info("✅ Emergency save beacon sent to FastAPI successfully")
            except Exception as e:
                logger.error(f"❌ Failed to send emergency save beacon to FastAPI: {e}")

        session.active = False
        session.timeout_detected_at = datetime.now()
        session.timeout_reason = f"Streamlit client detected inactivity for {minutes_inactive:.1f} minutes."

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.session_state['page'] = None
        st.session_state['session_expired'] = True
        
        st.error("⏰ **Session Timeout**")
        st.info(f"Your session has expired due to {SESSION_TIMEOUT_MINUTES} minutes of inactivity.")
        st.info("🏠 Redirecting to home page...")

        reload_script = f"""
            <script type="text/javascript">
                setTimeout(function() {{
                    parent.window.location.reload();
                }}, 2000);
            </script>
        """
        components.html(reload_script, height=0, width=0)
        
        st.stop()
        return True
    
    return False

def process_emergency_save_from_query(session_id: str, reason: str) -> bool:
    """
    Processes an emergency save request received via URL query parameters.
    This acts as a fallback if the FastAPI beacon failed.
    """
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        logger.error("❌ Session manager not available for emergency save from query.")
        return False
    
    session = session_manager.db.load_session(session_id)
    if not session:
        logger.warning(f"Emergency save from query: Session '{session_id[:8]}' not found in database. Cannot save.")
        return False
    
    logger.info(f"🚨 Processing emergency save for session '{session_id[:8]}' (Reason: {reason})")
    
    session.last_activity = datetime.now()
    session.active = False
    session.timeout_detected_at = datetime.now()
    session.timeout_reason = reason

    crm_saved = False
    try:
        if session_manager._is_crm_save_eligible(session, "emergency_fallback"):
            logger.info(f"Attempting CRM save from Streamlit query fallback for {session.session_id[:8]}")
            crm_saved = session_manager.zoho.save_chat_transcript_sync(session, reason)
            if crm_saved:
                session.timeout_saved_to_crm = True
            else:
                logger.warning(f"CRM save failed during Streamlit query fallback for {session.session_id[:8]}.")
        else:
            logger.info(f"Session {session.session_id[:8]} not eligible for CRM save (Streamlit query fallback).")
    except Exception as e:
        logger.error(f"Error during CRM save from Streamlit query fallback for {session.session_id[:8]}: {e}")
    
    try:
        session_manager.db.save_session(session)
        logger.info(f"✅ Session '{session_id[:8]}' marked inactive and saved in DB (query fallback). CRM_saved: {crm_saved}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save session '{session_id[:8]}' during emergency query fallback: {e}")
        return False

def render_simplified_browser_close_detection(session_id: str):
    """Enhanced browser close detection with eligibility check and redundancy for emergency saves."""
    if not session_id:
        return

    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        logger.debug("No session manager available for browser close detection")
        return
            
    session = session_manager.db.load_session(session_id)
    if not session:
        logger.debug(f"No session found for browser close detection: {session_id[:8]}")
        return
            
    if not session_manager._is_crm_save_eligible(session, "browser_close_check"):
        logger.info(f"🚫 Session {session.session_id[:8]} not eligible for CRM save - skipping browser close detection")
        return

    logger.info(f"✅ Setting up browser close detection for eligible session {session_id[:8]}")

    enhanced_close_js = f"""
    <script>
    (function() {{
        const sessionId = {json.dumps(session_id)};
        const FASTAPI_URL = {json.dumps(FASTAPI_EMERGENCY_SAVE_URL)};
        const FASTAPI_TIMEOUT_MS = {FASTAPI_EMERGENCY_SAVE_TIMEOUT * 1000};
        const STREAMLIT_FALLBACK_URL = window.location.origin + window.location.pathname; 
        
        if (window.fifi_close_enhanced_initialized) return;
        window.fifi_close_enhanced_initialized = true;
        
        let saveAttempted = false;
        
        console.log('🛡️ Enhanced browser close detection initialized for eligible user');
        
        function sendBeaconOrFetch(data) {{
            if (navigator.sendBeacon) {{
                try {{
                    const sent = navigator.sendBeacon(
                        FASTAPI_URL,
                        new Blob([data], {{type: 'application/json'}})
                    );
                    if (sent) {{
                        console.log('✅ Emergency save beacon sent to FastAPI');
                        return true;
                    }} else {{
                        console.warn('⚠️ Beacon send returned false, trying fetch...');
                    }}
                }} catch (e) {{
                    console.error('❌ Beacon failed:', e);
                }}
            }}
            
            try {{
                fetch(FASTAPI_URL, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: data,
                    keepalive: true,
                    signal: AbortSignal.timeout(FASTAPI_TIMEOUT_MS)
                }}).then(response => {{
                    if (response.ok) {{
                        console.log('✅ Emergency save via fetch successful');
                    }} else {{
                        console.warn('⚠️ Fetch response not OK, status:', response.status);
                    }}
                }}).catch(error => {{
                    console.error('❌ Fetch failed:', error);
                }});
                return true;
            }} catch (e) {{
                console.error('❌ Fetch setup failed:', e);
            }}
            return false;
        }}

        function triggerEmergencySave(reason) {{
            if (saveAttempted) return;
            saveAttempted = true;
            
            console.log('🚨 Triggering emergency save:', reason);
            
            const emergencyData = JSON.stringify({{
                session_id: sessionId,
                reason: reason,
                timestamp: Date.now()
            }});
            
            const sentViaNetwork = sendBeaconOrFetch(emergencyData);

            const fallbackUrl = STREAMLIT_FALLBACK_URL + 
                '?event=emergency_close' +
                '&session_id=' + sessionId + 
                '&reason=' + reason +
                '&fallback=true';
            
            const img = new Image();
            img.src = fallbackUrl;
            img.style.display = 'none';
            document.body.appendChild(img);
            
            console.log('✅ Streamlit fallback image beacon initiated');
        }}
        
        window.addEventListener('beforeunload', () => triggerEmergencySave('beforeunload'), {{ capture: true, passive: true }});
        window.addEventListener('unload', () => triggerEmergencySave('unload'), {{ capture: true, passive: true }});
        window.addEventListener('pagehide', () => triggerEmergencySave('pagehide'), {{ capture: true, passive: true }});
        
        document.addEventListener('visibilitychange', function() {{
            if (document.visibilityState === 'hidden') {{
                console.log('📱 Tab became hidden - scheduling potential save');
                setTimeout(() => {{
                    if (document.visibilityState === 'hidden') {{
                        console.log('🚨 Tab still hidden after delay - likely closed or backgrounded');
                        triggerEmergencySave('visibility_hidden_background');
                    }} else {{
                        console.log('✅ Tab became visible during delay - canceling save');
                        saveAttempted = false;
                    }}
                }}, 5000);
            }}
        }});
        
        try {{
            if (window.parent && window.parent !== window) {{
                window.parent.addEventListener('beforeunload', () => triggerEmergencySave('parent_beforeunload'), {{ capture: true, passive: true }});
                window.parent.addEventListener('unload', () => triggerEmergencySave('parent_unload'), {{ capture: true, passive: true }});
                window.parent.addEventListener('pagehide', () => triggerEmergencySave('parent_pagehide'), {{ capture: true, passive: true }});
            }}
        }} catch (e) {{
            console.debug('Cannot monitor parent events (cross-origin):', e);
        }}
        
        console.log('✅ Enhanced browser close detection ready');
    }})();
    </script>
    """
        
    try:
        st.components.v1.html(enhanced_close_js, height=1, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close detection: {e}")

# THIS FUNCTION IS REMOVED as per the new FastAPI beacon strategy
# def process_fingerprint_from_query(session_id: str, fingerprint_id: str, method: str, privacy: str, working_methods: List[str]) -> bool:
#     pass # This function is no longer used.

# THIS FUNCTION IS REMOVED as per the new FastAPI beacon strategy
# def handle_fingerprint_requests_from_query():
#     pass # This function is no longer used.

def handle_fingerprint_status_update_from_query():
    """Checks for and processes fingerprint status update requests sent via URL query parameters (image beacon)."""
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    status = query_params.get("status")
    fp_id_short = query_params.get("fingerprint_id")

    if event == "fingerprint_status_update" and session_id:
        logger.info(f"🔍 FINGERPRINT STATUS UPDATE received for session {session_id[:8]}: Status='{status}', FP_ID_Short='{fp_id_short}'")
        
        # Clear specific query parameters to prevent re-triggering this logic immediately
        params_to_clear = ["event", "session_id", "status", "fingerprint_id"]
        for param in params_to_clear:
            if param in st.query_params:
                del st.query_params[param]
        
        # FIXED: Force inheritance check after fingerprint is captured
        session_manager = st.session_state.get('session_manager')
        if session_manager and status == "success":
            # Load the session fresh from DB to get the updated fingerprint
            session = session_manager.db.load_session(session_id)
            if session and session.user_type == UserType.GUEST:
                logger.info(f"🔄 Re-attempting inheritance for Guest {session_id[:8]} after fingerprint capture")
                # Re-attempt inheritance with the real fingerprint
                session_manager._attempt_fingerprint_inheritance(session)
                session_manager.db.save_session(session)
                # Update the in-memory session in session_state if it exists
                if st.session_state.get('current_session_id') == session_id:
                    st.session_state.temp_session_update = session
        
        st.session_state.fingerprint_client_side_completed = True
        st.rerun()
        return True
    return False

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_welcome_page(session_manager: 'SessionManager'):
    """Enhanced welcome page with loading lock and WordPress fallback UI."""
    
    st.title("Welcome to :rainbow[FiFi]")
    st.subheader("Your :blue[AI sourcing] assistant")

    if show_loading_overlay():
        return
    
    # REMOVED: No session needed on welcome page
    
    st.markdown("---")
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is currently disabled because the authentication service (WordPress URL) is not configured in application secrets.")
        else:
            if st.session_state.get('wordpress_error', {}).get('show_fallback', False):
                error_info = st.session_state.wordpress_error
                
                st.error(f"🚨 **WordPress Login Issue: {error_info['type']}**")
                st.error(error_info['message'])
                
                st.info("💡 **Alternative Login Option Available**")
                st.markdown(f"""
                We can switch you to our **Email Verification** login method instead:
                - Quick verification via email OTP (one-time password).
                - If your email is associated with a registered account, your privileges will be automatically restored.
                - If not, you'll gain **Email Verified Guest** access ({EMAIL_VERIFIED_QUESTION_LIMIT} questions/day).
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📧 Switch to Email Login", use_container_width=True):
                        st.session_state.wordpress_fallback_active = True
                        st.session_state.fallback_email = error_info.get('username', '') if '@' in error_info.get('username', '') else ''
                        st.session_state.wordpress_error['show_fallback'] = False
                        st.rerun()
                        
                with col2:
                    if st.button("🔄 Try WordPress Again", use_container_width=True):
                        if 'wordpress_error' in st.session_state:
                            del st.session_state['wordpress_error']
                        st.rerun()
            
            elif st.session_state.get('wordpress_fallback_active', False):
                st.info("📧 **Email Verification Login (WordPress Fallback)**")
                st.caption("Enter your email to receive a verification code. If you're a registered user, we'll restore your account.")
                
                with st.form("fallback_email_form", clear_on_submit=False):
                    fallback_email = st.text_input(
                        "Email Address",
                        value=st.session_state.get('fallback_email', ''),
                        placeholder="your@email.com",
                        key="fallback_email_input",
                        help="We'll send an OTP to this email to verify your identity. If it's linked to a registered account, your status will be restored."
                    )
                    
                    submit_email = st.form_submit_button("📨 Send Verification Code", use_container_width=True)
                    
                    if submit_email:
                        if fallback_email:
                            # FIXED: Create session only when email is submitted
                            temp_session = session_manager._create_and_save_new_session('email_fallback_attempt')
                            st.session_state.current_session_id = temp_session.session_id
                            
                            result = session_manager.handle_guest_email_verification(temp_session, fallback_email)
                            
                            if result['success']:
                                st.success(result['message'])
                                st.session_state.fallback_verification_email = fallback_email
                                st.session_state.fallback_verification_stage = "code_entry"
                                st.session_state.fallback_session_id = temp_session.session_id
                                st.rerun()
                            else:
                                st.error(result['message'])
                        else:
                            st.error("Please enter your email address")
            
            elif st.session_state.get('fallback_verification_stage') == 'code_entry':
                email = st.session_state.get('fallback_verification_email')
                st.success(f"📧 Code sent to **{session_manager._mask_email(email)}**")
                
                st.warning("""
                ⚠️ **Important**: This is a recovery login method due to WordPress authentication issues.
                Your chat access is fully functional, but WordPress profile-specific integrations might be unavailable.
                """)
                
                with st.form("fallback_code_form", clear_on_submit=False):
                    code = st.text_input(
                        "Enter 6-Digit Code", 
                        placeholder="123456", 
                        max_chars=6,
                        key="fallback_code_input",
                        help="Enter the 6-digit code from your email. Valid for 1 minute."
                    )
                    verify_btn = st.form_submit_button("✅ Verify & Login", use_container_width=True)
                    
                    if verify_btn and code:
                        temp_session_id = st.session_state.get('fallback_session_id')
                        temp_session = session_manager.db.load_session(temp_session_id)
                        
                        if temp_session:
                            result = session_manager.verify_email_code(temp_session, code)
                            
                            if result['success']:
                                upgraded_session = session_manager._check_and_upgrade_to_registered(
                                    temp_session, 
                                    email,
                                    is_fallback_from_wordpress=True
                                )
                                
                                if upgraded_session.user_type == UserType.REGISTERED_USER:
                                    if upgraded_session.is_degraded_login:
                                        st.info("ℹ️ Logged in as Registered User via email (WordPress was unavailable)")
                                    st.success(f"✅ Welcome back! Your registered account has been restored.")
                                    st.balloons()
                                else:
                                    st.success(f"✅ Email verified! You have Email Verified Guest access.")
                                
                                for key in ['wordpress_fallback_active', 'fallback_email', 
                                           'fallback_verification_email', 'fallback_verification_stage',
                                           'fallback_session_id', 'wordpress_error']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                st.session_state.current_session_id = upgraded_session.session_id
                                st.session_state.page = "chat"
                                st.session_state.is_chat_ready = True
                                st.rerun()
                            else:
                                st.error(result['message'])
                    else:
                        st.error("Please enter a 6-digit code.")
            
                col_resend_fallback, _ = st.columns([1,2])
                with col_resend_fallback:
                    if st.button("🔄 Resend Code", use_container_width=True, key="resend_fallback_code"):
                        if email:
                            with st.spinner("Resending verification code..."):
                                verification_sent = session_manager.email_verification.send_verification_code(email)
                                if verification_sent:
                                    st.success("✅ New verification code sent! Check your email.")
                                else:
                                    st.error("❌ Failed to resend code. Please try again later.")
                        else:
                            st.error("Error: No email address found for resend. Please restart the login process.")
                        st.rerun()

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
                            st.session_state.temp_username = username
                            st.session_state.temp_password = password
                            st.session_state.loading_reason = 'authenticate'
                            set_loading_state(True, "Authenticating and preparing your session...")
                            st.rerun()
            
            st.markdown("---")
            st.info("Don't have an account? [Register here](https://www.12taste.com/in/my-account/) to unlock full features!")
    
    with tab2:
        st.markdown(f"""
        **Continue as a guest** to get a quick start and try FiFi AI Assistant without signing in.
        
        ℹ️ **What to expect as a Guest:**
        - You get an initial allowance of **{GUEST_QUESTION_LIMIT} questions** to explore FiFi AI's capabilities.
        - After these {GUEST_QUESTION_LIMIT} questions, **email verification will be required** to continue (unlocks {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day).
        - Our system utilizes **universal device fingerprinting** for security and to track usage across sessions.
        - You can always choose to **upgrade to a full registration** later for extended benefits.
        """)
        
        st.markdown("")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("👤 Start as Guest", use_container_width=True):
                # FIXED: Create session only when guest button is clicked
                new_session = session_manager._create_and_save_new_session('guest')
                st.session_state.current_session_id = new_session.session_id
                st.session_state.loading_reason = 'start_guest'
                set_loading_state(True, "Setting up your session and initializing AI assistant...")
                st.rerun()

    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown(f"• **{GUEST_QUESTION_LIMIT} questions** to try FiFi AI")
        st.markdown("• Email verification required to continue")
        st.markdown("• Quick start, no registration needed")
    
    with col2:
        st.info("📧 **Email Verified Guest**")
        st.markdown(f"• **{EMAIL_VERIFIED_QUESTION_LIMIT} questions per day** (rolling {DAILY_RESET_WINDOW_HOURS}-hour period)")
        st.markdown("• Email verification for access")
        st.markdown("• No full registration required")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown(f"• **{REGISTERED_USER_QUESTION_LIMIT} questions per day** with tier system:")
        st.markdown(f"  - **Tier 1**: Questions 1-{REGISTERED_USER_TIER_1_LIMIT} → {TIER_1_BAN_HOURS}-hour break")
        tier1_upper_bound = REGISTERED_USER_TIER_1_LIMIT
        st.markdown(f"  - **Tier 2**: Questions {tier1_upper_bound + 1}-{REGISTERED_USER_QUESTION_LIMIT} → {TIER_2_BAN_HOURS}-hour reset")
        st.markdown("• Cross-device tracking & chat saving")
        st.markdown("• • Priority access during high usage")
        
def render_sidebar(session_manager: 'SessionManager', session: UserSession, pdf_exporter: PDFExporter):
    """Enhanced sidebar with tier progression display and login status."""
    with st.sidebar:
        DEBUG_MODE = False # Set to True for debugging, False for production
        
        st.title("🎛️ Dashboard")
        
        if session.user_type.value == UserType.REGISTERED_USER.value:
            st.success("✅ **Registered User**")
            
            if DEBUG_MODE and session.is_degraded_login:
                st.info("ℹ️ Logged in via email (WordPress was unavailable)")
            # if DEBUG_MODE: # Hide "Login method:"
            #     st.caption(f"Login method: {session.login_method or 'WordPress'}")

            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/{REGISTERED_USER_QUESTION_LIMIT}")
            
            progress_bar_key = f"registered_user_progress_{session.session_id}"

            if session.daily_question_count < REGISTERED_USER_TIER_1_LIMIT:
                st.progress(min(session.daily_question_count / REGISTERED_USER_TIER_1_LIMIT, 1.0),
                           text=f"Tier 1: {session.daily_question_count}/{REGISTERED_USER_TIER_1_LIMIT} questions")
                           
                remaining_tier1 = REGISTERED_USER_TIER_1_LIMIT - session.daily_question_count
                if remaining_tier1 > 0:
                    st.caption(f"⏰ {remaining_tier1} questions until {TIER_1_BAN_HOURS}-hour break")
            elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                if session.ban_status == BanStatus.ONE_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                    st.progress(1.0, text="Tier 1 Complete")
                    time_remaining = session.ban_end_time - datetime.now()
                    hours = int(time_remaining.total_seconds() // 3600)
                    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                    st.caption(f"⏳ Tier 1 break: {hours}h {minutes}m remaining until Tier 2")
                elif session.tier1_completed_in_cycle:
                    st.progress(1.0, text="Tier 1 Complete ✅")
                    st.caption("📈 Ready to proceed to Tier 2!")
                else:
                    st.progress(min(session.daily_question_count / REGISTERED_USER_TIER_1_LIMIT, 1.0),
                               text=f"Tier 1: {session.daily_question_count}/{REGISTERED_USER_TIER_1_LIMIT} questions")
                    st.caption(f"📈 Next question will trigger a {TIER_1_BAN_HOURS}-hour break before Tier 2.")
            else:
                tier2_questions_asked = session.daily_question_count - REGISTERED_USER_TIER_1_LIMIT
                tier2_limit = REGISTERED_USER_QUESTION_LIMIT - REGISTERED_USER_TIER_1_LIMIT
                
                tier2_progress = min(tier2_questions_asked / tier2_limit, 1.0)
                st.progress(tier2_progress, text=f"Tier 2: {tier2_questions_asked}/{tier2_limit} questions")
                
                remaining_tier2 = REGISTERED_USER_QUESTION_LIMIT - session.daily_question_count
                if remaining_tier2 > 0:
                    st.caption(f"⏰ {remaining_tier2} questions until {TIER_2_BAN_HOURS}-hour reset")
                else:
                    st.caption(f"🚫 Daily limit reached - {TIER_2_BAN_HOURS}-hour reset required")
                    
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            st.info("📧 **Email Verified Guest**")
            # if DEBUG_MODE: # Hide "Login method:"
            #     st.caption(f"Login method: {session.login_method or 'Email Verified'}")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/{EMAIL_VERIFIED_QUESTION_LIMIT}")
            st.progress(min(session.daily_question_count / EMAIL_VERIFIED_QUESTION_LIMIT, 1.0))
            
            if session.ban_status == BanStatus.TWENTY_FOUR_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 Daily limit reached")
                st.caption(f"Resets in: {hours}h {minutes}m")
            elif session.last_question_time:
                expected_reset_time = session.last_question_time + timedelta(hours=DAILY_RESET_WINDOW_HOURS)
                time_to_reset = expected_reset_time - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Daily questions have reset!")
            
        else: # Guest User
            st.warning("👤 **Guest User**")
            # if DEBUG_MODE: # Hide "Login method:"
            #     st.caption(f"Login method: {session.login_method or 'Guest'}")
            st.markdown(f"**Questions:** {session.daily_question_count}/{GUEST_QUESTION_LIMIT}")
            st.progress(min(session.daily_question_count / GUEST_QUESTION_LIMIT, 1.0))
            st.caption(f"Email verification unlocks {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day.")
            if session.reverification_pending:
                st.info("💡 An account is available for this device. Re-verify email to reclaim it!")
            elif session.declined_recognized_email_at and session.daily_question_count < session_manager.question_limits.question_limits[UserType.GUEST.value]:
                st.info("💡 You are currently using guest questions. Verify your email to get more.")
                
        # Show fingerprint status
        if session.fingerprint_id:
            if session.user_type == UserType.REGISTERED_USER and session.fingerprint_id == "not_collected_registered_user":
                st.markdown("**Device ID:** Not Collected")
                st.caption("Fingerprinting not applicable for registered users (email primary)")
            elif session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                st.markdown("**Device ID:** Identifying...")
                st.caption("Fingerprinting in progress...")                
            else:
                st.markdown(f"**Device ID:** `{session.fingerprint_id[:12]}...`")
                st.caption(f"Method: {session.fingerprint_method or 'unknown'} (Privacy: {session.browser_privacy_level or 'standard'})")
        else:
            st.markdown("**Device ID:** Initializing...")
            st.caption("Starting fingerprinting...")

        # Hide "Last activity"
        # if DEBUG_MODE and session.last_activity is not None:
        #    time_since_activity = datetime.now() - session.last_activity
        #    minutes_inactive = time_since_activity.total_seconds() / 60
        #    st.caption(f"Last activity: {int(minutes_inactive)} minutes ago")
        #    
        #    timeout_duration = SESSION_TIMEOUT_MINUTES
        #    if minutes_inactive >= (timeout_duration - 1) and minutes_inactive < timeout_duration:
        #        minutes_remaining = timeout_duration - minutes_inactive
        #        st.warning(f"⚠️ Session expires in {minutes_remaining:.1f} minutes!")
        #    elif minutes_inactive >= timeout_duration:
        #        st.error(f"🚫 Session is likely expired. Type a question to check.")
        # else:
        #    if DEBUG_MODE: st.caption("Session timer will start with first interaction.")

        # Hide "🤖 AI Tools Status" text and its contents
        # if DEBUG_MODE:
        st.divider()
        st.markdown("**🤖 AI Tools Status**")
        
        ai_system = session_manager.ai
        if ai_system:
            pinecone_status = error_handler.component_status.get("Pinecone", "healthy")
            tavily_status = error_handler.component_status.get("Tavily", "healthy")
            
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
            
            if ai_system.tavily_agent:
                if tavily_status == "healthy":
                    st.success("🌐 Web Search: Ready")
                else:
                    st.warning(f"🌐 Web Search: {tavily_status.replace('_', ' ').title()}")
            elif ai_system.config.TAVILY_API_KEY:
                st.warning("🌐 Web Search: Error")
            else:
                st.info("🌐 Web Search: Not configured")
            
            # Hide "💬 OpenAI: Ready"
            # if DEBUG_MODE:
            if ai_system.openai_client:
                st.success("💬 OpenAI: Ready")
            elif ai_system.config.OPENAI_API_KEY:
                st.warning("💬 OpenAI: Error")
            else:
                st.info("💬 OpenAI: Not configured")
        else:
            st.error("🤖 AI System: Not available")
        
        # Hide "🚫 CRM Integration: Registered users & verified guests only"
        # if DEBUG_MODE:
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

        # Display 12Taste Order Status
        if session_manager.woocommerce and session_manager.woocommerce.config.WOOCOMMERCE_ENABLED:
            # Removed await asyncio.to_thread, as test_connection will be called once per render.
            # Using a simple check here is fine. If it's slow, consider caching.
            status = session_manager.woocommerce.test_connection()
            st.markdown("🛒 **12Taste Order Status**") # Changed text
            if status["status"] == "connected":
                st.success("Connected")
            else:
                st.error(f"{status['status'].replace('_', ' ').title()}")
                st.caption(status["message"])
        else:
            st.info("🛒 **12Taste Order Status**: Not configured or available") # Changed text
        
        st.divider()
        
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
                time_remaining = datetime.now() - session.ban_end_time # Corrected calculation
                if time_remaining.total_seconds() < 0: # Check if ban is still in future
                    time_remaining = -time_remaining
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
            if (session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and 
                session.email and len(session.messages) >= CRM_SAVE_MIN_QUESTIONS):
                signout_help += " Your conversation will be automatically saved to CRM before signing out."
            
            if st.button("🚪 Sign Out", use_container_width=True, help=signout_help):
                session_manager.end_session(session)
                st.rerun()

        # Hide "Save to Zoho CRM" button
        # if DEBUG_MODE and session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and len(session.messages) >= CRM_SAVE_MIN_QUESTIONS:
        if session_manager.zoho.config.ZOHO_ENABLED and session.email and len(session.messages) >= CRM_SAVE_MIN_QUESTIONS:
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
            
            # This button is hidden by default due to DEBUG_MODE = False
            # To show it, you'd need to explicitly set DEBUG_MODE = True
            if DEBUG_MODE:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, help="Manually save your current chat transcript to your linked Zoho CRM contact."):
                    session_manager.manual_save_to_crm(session)
                st.caption("💡 Chat automatically saves to CRM during Sign Out or browser/tab close.")


def display_email_prompt_if_needed(session_manager: 'SessionManager', session: UserSession) -> bool:
    """
    Renders email verification dialog if needed.
    Handles gentle prompts only when an answer was just given, otherwise shows a direct prompt.
    Controls `st.session_state.chat_blocked_by_dialog` and returns if chat input should be disabled.
    """
    
    if st.session_state.get('current_session_id') != session.session_id:
        logger.warning(f"Session mismatch in email prompt! State: {st.session_state.get('current_session_id', 'None')[:8]}, Param: {session.session_id[:8]}")
        st.session_state.current_session_id = session.session_id
    
    if 'verification_stage' not in st.session_state: st.session_state.verification_stage = None
    if 'guest_continue_active' not in st.session_state: st.session_state.guest_continue_active = False
    if 'final_answer_acknowledged' not in st.session_state: st.session_state.final_answer_acknowledged = False
    if 'gentle_prompt_shown' not in st.session_state: st.session_state.gentle_prompt_shown = False
    if 'email_verified_final_answer_acknowledged' not in st.session_state: st.session_state.email_verified_final_answer_acknowledged = False
    if 'must_verify_email_immediately' not in st.session_state: st.session_state.must_verify_email_immediately = False
    if 'skip_email_allowed' not in st.session_state: st.session_state.skip_email_allowed = True

    if st.session_state.get('send_code_now', False) and st.session_state.get('verification_email'):
        email_to_send = st.session_state.verification_email
        result = session_manager.handle_guest_email_verification(session, email_to_send)
        
        del st.session_state['send_code_now']
        
        if result['success']:
            st.success(result['message'])
            st.session_state.verification_stage = "code_entry"
        else:
            st.error(result['message'])
            st.session_state.verification_stage = st.session_state.get('verification_stage', 'initial_check')
        
        st.rerun()
        return True

    limit_check = session_manager.question_limits.is_within_limits(session)
    if not limit_check['allowed'] and limit_check.get('reason') not in ['guest_limit', 'email_verified_guest_limit', 'registered_user_tier1_limit', 'registered_user_tier2_limit']:
        st.session_state.chat_blocked_by_dialog = True
        return True

    user_is_guest = (session.user_type.value == UserType.GUEST.value)
    user_is_email_verified = (session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value)
    guest_limit_value = GUEST_QUESTION_LIMIT
    email_verified_limit_value = EMAIL_VERIFIED_QUESTION_LIMIT
    daily_q_value = session.daily_question_count
    is_guest_limit_hit = (user_is_guest and daily_q_value >= guest_limit_value)
    is_email_verified_limit_hit = (user_is_email_verified and daily_q_value >= email_verified_limit_value)
    
    user_just_hit_guest_limit = is_guest_limit_hit and st.session_state.get('just_answered', False)
    user_just_hit_email_verified_limit = is_email_verified_limit_hit and st.session_state.get('just_answered', False)
    
    must_verify_immediately = st.session_state.get('must_verify_email_immediately', False)
    skip_allowed = st.session_state.get('skip_email_allowed', True)

    should_show_prompt = False
    should_block_chat = True

    if user_is_guest and must_verify_immediately and daily_q_value == 0:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'forced_verification'

    elif session.reverification_pending:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
             st.session_state.verification_stage = 'initial_check'
             st.session_state.guest_continue_active = False

    elif user_just_hit_guest_limit:
        st.session_state.just_answered = False
        should_show_prompt = True
        should_block_chat = False
        
        st.success(f"🎯 **You've explored FiFi AI with your {GUEST_QUESTION_LIMIT} guest questions!**")
        st.info(f"Take your time to read this answer. When you're ready, verify your email to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} questions per day + chat history saving!")
        
        with st.expander("📧 Ready to Unlock More Questions?", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📧 Yes, Let's Verify My Email!", use_container_width=True, key="gentle_verify_btn"):
                    st.session_state.verification_stage = 'email_entry'
                    st.session_state.chat_blocked_by_dialog = True
                    st.session_state.final_answer_acknowledged = True
                    st.rerun()
            with col2:
                if skip_allowed:
                    if st.button("👀 Let Me Finish Reading First", use_container_width=True, key="continue_reading_btn"):
                        st.session_state.final_answer_acknowledged = True
                        st.success("Perfect! Take your time. The verification option will remain available above.")
                        st.rerun()
                else:
                    st.info("Email verification is required to continue.")
        
        st.session_state.chat_blocked_by_dialog = False
        return False

    elif user_just_hit_email_verified_limit:
        st.session_state.just_answered = False
        should_show_prompt = True
        should_block_chat = False
        
        st.success(f"🎯 **You've completed your {EMAIL_VERIFIED_QUESTION_LIMIT} daily questions!**")
        st.info(f"Take your time to read this answer. Your questions will reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!")
        
        with st.expander("🚀 Want More Questions Daily?", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔗 Go to Registration", use_container_width=True, key="register_upgrade_btn"):
                    st.link_button("Register Here", "https://www.12taste.com/in/my-account/", use_container_width=True)
                    st.session_state.email_verified_final_answer_acknowledged = True
                    st.rerun()
            with col2:
                if st.button("👀 Let Me Finish Reading First", use_container_width=True, key="email_verified_continue_reading"):
                    st.session_state.email_verified_final_answer_acknowledged = True
                    st.success(f"Perfect! Take your time reading. You'll need to wait {EMAIL_VERIFIED_BAN_HOURS} hours for more questions.")
                    st.rerun()
        
        st.session_state.chat_blocked_by_dialog = False
        return False

    elif is_guest_limit_hit:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'email_entry'
            st.session_state.guest_continue_active = False

    elif is_email_verified_limit_hit and not st.session_state.email_verified_final_answer_acknowledged:
        should_show_prompt = True
        should_block_chat = True
        st.error("🛑 **Daily Limit Reached**")
        st.info(f"You've used your {EMAIL_VERIFIED_QUESTION_LIMIT} questions for today. Your questions will reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.link_button(f"Register for {REGISTERED_USER_QUESTION_LIMIT} questions/day", "https://www.12taste.com/in/my-account/", use_container_width=True)
        with col2:
            if st.button("Return to Welcome Page", use_container_width=True):
                session_manager.end_session(session)
                app_base_url = os.getenv("APP_BASE_URL", "https://fifi-eu-121263692901.europe-west1.run.app/")
                js_redirect = f"window.top.location.href = '{app_base_url}';"
                st.components.v1.html(f"<script>{js_redirect}</script>", height=100, width=0)
                st.rerun()
        
        st.session_state.chat_blocked_by_dialog = True
        return True

    elif session.declined_recognized_email_at and not st.session_state.guest_continue_active and not is_guest_limit_hit:
        should_show_prompt = True
        should_block_chat = False
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'declined_recognized_email_prompt_only'

    if not should_show_prompt:
        st.session_state.chat_blocked_by_dialog = False
        st.session_state.verification_stage = None
        if 'just_answered' in st.session_state:
             del st.session_state.just_answered
        return False

    st.session_state.chat_blocked_by_dialog = should_block_chat
    
    if should_block_chat:
        st.error("📧 **Action Required**")

    current_stage = st.session_state.verification_stage

    if current_stage == 'forced_verification':
        st.error("📧 **Email Verification Required**")
        
        known_emails = st.session_state.get('known_device_emails', [])
        
        if known_emails:
            if len(known_emails) == 1:
                st.info(f"This device was previously verified with **{session_manager._mask_email(known_emails[0])}**. Please verify an email to continue.")
            else:
                st.info("This device was previously verified with multiple emails. Please verify one of your emails to continue:")
                for email in known_emails[:3]:
                    st.caption(f"• {session_manager._mask_email(email)}")
        else:
            st.info("This device has already been used with email verification. Please verify your email to continue using FiFi AI.")
        
        with st.form("forced_email_verification_form", clear_on_submit=False):
            st.markdown("**📧 Enter your email address to receive a verification code:**")
            current_email_input = st.text_input(
                "Email Address", 
                placeholder="your@email.com",
                key="forced_email_input",
                help="Email verification is required to continue."
            )
            submit_email = st.form_submit_button("📨 Send Verification Code", use_container_width=True)
            
            if submit_email:
                if current_email_input:
                    st.session_state.must_verify_email_immediately = False
                    st.session_state.skip_email_allowed = True
                    
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

    elif current_stage == 'initial_check':
        prompt_container = st.container()
        
        with prompt_container:
            email_to_reverify = session.pending_email
            masked_email = session_manager._mask_email(email_to_reverify) if email_to_reverify else "your registered email"
            st.info(f"🤝 **We recognize this device was previously used as a {session.pending_user_type.value.replace('_', ' ').title()} account.**")
            st.info(f"Please verify **{masked_email}** to reclaim your status and higher question limits.")
            
            button_key_suffix = session.session_id[:8]
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Verify this email", 
                            use_container_width=True, 
                            key=f"reverify_yes_{button_key_suffix}"):
                    
                    logger.info(f"Verify button clicked in session {session.session_id[:8]}")
                    
                    session.recognition_response = "yes_reverify"
                    session.declined_recognized_email_at = None
                    st.session_state.verification_email = email_to_reverify
                    st.session_state.verification_stage = "send_code_recognized"
                    
                    st.session_state.current_session_id = session.session_id
                    
                    session_manager.db.save_session(session)
                    st.rerun()
            with col2:
                if st.button("❌ No, I don't recognize the email", 
                            use_container_width=True, 
                            key=f"reverify_no_{button_key_suffix}"):
                    
                    logger.info(f"Decline button clicked in session {session.session_id[:8]}")
                    
                    session.recognition_response = "no_declined_reco"
                    session.declined_recognized_email_at = datetime.now()
                    session.user_type = UserType.GUEST 
                    session.reverification_pending = False
                    session.pending_user_type = None
                    session.pending_email = None
                    session.pending_full_name = None
                    session.pending_zoho_contact_id = None
                    session.pending_wp_token = None
                    
                    st.session_state.current_session_id = session.session_id
                    
                    session_manager.db.save_session(session)
                    st.session_state.guest_continue_active = True
                    st.session_state.chat_blocked_by_dialog = False
                    st.session_state.verification_stage = None
                    st.success("You can now continue as a Guest.")
                    st.rerun()

    elif current_stage == 'send_code_recognized':
        email_to_verify = st.session_state.get('verification_email')
        if email_to_verify:
            st.info(f"📧 **Sending verification code to {session_manager._mask_email(email_to_verify)}...**")
            
            st.session_state.send_code_now = True
            st.rerun()

    elif current_stage == 'email_entry':
        skip_allowed = st.session_state.get('skip_email_allowed', True)
        
        st.info(f"🚀 You've used your {GUEST_QUESTION_LIMIT} guest questions. Please verify your email to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} questions per day!")
        with st.form("email_verification_form", clear_on_submit=False):
            st.markdown("**📧 Enter your email address to receive a verification code:**")
            current_email_input = st.text_input(
                "Email Address", 
                placeholder="your@email.com", 
                value=st.session_state.get('verification_email', session.email or ""), 
                key="manual_email_input",
                help="We'll send you a 6-digit verification code that's valid for 1 minute."
            )
            
            if skip_allowed:
                col1, col2 = st.columns(2)
                with col1:
                    submit_email = st.form_submit_button("📨 Send Verification Code", use_container_width=True)
                with col2:
                    skip_button = st.form_submit_button("Skip for now", use_container_width=True)
                    
                if skip_button:
                    logger.info(f"User chose to skip email verification for session {session.session_id[:8]} - ending session")
                    
                    session.active = False
                    session.last_activity = datetime.now()
                    
                    try:
                        session_manager.db.save_session(session)
                    except Exception as e:
                        logger.error(f"Failed to save session during skip: {e}")
                    
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    
                    st.session_state['page'] = None
                    
                    st.info("👋 Session ended. You can start a new session anytime.")
                    st.rerun()
            else:
                submit_email = st.form_submit_button("📨 Send Verification Code", use_container_width=True)
            
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
                else:
                    st.error("Please enter an email address to receive the code.")
        
    elif current_stage == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        st.success(f"📧 **Verification code sent to** **{session_manager._mask_email(verification_email)}**")
        st.info("📱 Check your email (including spam/junk folders). The code expires in 1 minute.")
        
        with st.form("code_verification_form", clear_on_submit=False):
            code = st.text_input(
                "Enter 6-Digit Verification Code", 
                placeholder="123456", 
                max_chars=6, 
                key="verification_code_input",
                help="Enter the 6-digit code from your email"
            )
            col_verify, col_resend = st.columns(2)
            with col_verify:
                submit_code = st.form_submit_button("✅ Verify Code", use_container_width=True)
            with col_resend:
                resend_code = st.form_submit_button("🔄 Resend Code", use_container_width=True)
            
            if resend_code:
                if verification_email:
                    with st.spinner("Resending verification code..."):
                        verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                        if verification_sent:
                            st.success("✅ New verification code sent! Check your email.")
                            st.session_state.verification_stage = "code_entry"
                        else:
                            st.error("❌ Failed to resend code. Please try again later.")
                else:
                    st.error("Error: No email address found. Please go back and enter your email.")
                    st.session_state.verification_stage = "email_entry"
                st.rerun()

            if submit_code:
                if code and len(code) == 6:
                    with st.spinner("🔍 Verifying your code..."):
                        result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        st.session_state.chat_blocked_by_dialog = False
                        st.session_state.verification_stage = None
                        st.session_state.guest_continue_active = False
                        st.session_state.final_answer_acknowledged = False
                        st.session_state.gentle_prompt_shown = False
                        st.session_state.email_verified_final_answer_acknowledged = False
                        st.session_state.must_verify_email_immediately = False
                        st.session_state.skip_email_allowed = True
                        if 'just_answered' in st.session_state:
                             del st.session_state.just_answered
                        st.rerun()
                    else:
                        st.error(result['message'])
                elif code:
                    st.error("Please enter a 6-digit verification code.")
                else:
                    st.error("Please enter the verification code you received.")
        
    elif current_stage == 'declined_recognized_email_prompt_only':
        st.session_state.chat_blocked_by_dialog = False

        remaining_questions = GUEST_QUESTION_LIMIT - session.daily_question_count
        st.info(f"✅ **Continuing as Guest** - You have **{remaining_questions} questions** remaining from your guest allowance.")
        st.info(f"💡 **Pro Tip:** Verify your email anytime to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day + chat history saving.")

        with st.expander("📧 Want to Verify a Different Email?", expanded=False):
            col_opts1, col_opts2 = st.columns(2)
            with col_opts1:
                if st.button("📧 Enter My Email for Verification", use_container_width=True, key="new_email_opt_btn"):
                    st.session_state.verification_email = ""
                    st.session_state.verification_stage = "email_entry"
                    st.session_state.guest_continue_active = False
                    st.rerun()
            with col_opts2:
                if st.button("👍 Continue as Guest", use_container_width=True, key="continue_guest_btn"):
                    st.session_state.guest_continue_active = True
                    st.session_state.chat_blocked_by_dialog = False
                    st.session_state.verification_stage = None
                    st.success("Perfect! You can continue as a Guest. The email verification option will always be available.")
                    st.rerun()

    return should_block_chat

def render_chat_interface_simplified(session_manager: 'SessionManager', session: UserSession, activity_result: Optional[Dict[str, Any]]):
    """Chat interface with enhanced tier system notifications and Option 2 gentle approach."""
    
    st.title("FiFi AI Assistant")
    st.caption("Hello, I am FiFi, your AI-powered assistant, designed to support you across the sourcing and product development journey. Find the right ingredients, explore recipe ideas, technical data, and more.")

    # NEW: Show fingerprint waiting status ONLY for non-registered users
    # This block is now handled higher up in main_fixed, so it won't be displayed here
    # However, if for any reason it gets to this point and is still not ready,
    # it means the fingerprinting is taking too long or failed, and main_fixed
    # will have already triggered a rerun. This means this section will only
    # execute once is_chat_ready is True.

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

    should_disable_chat_input_by_dialog = display_email_prompt_if_needed(session_manager, session)

    if not st.session_state.get('chat_blocked_by_dialog', False):
        limit_check_for_display = session_manager.question_limits.is_within_limits(session)
        if (session.user_type.value == UserType.REGISTERED_USER.value and 
            limit_check_for_display.get('allowed') and 
            limit_check_for_display.get('tier')):
            
            tier = limit_check_for_display.get('tier')
            remaining = limit_check_for_display.get('remaining', 0)
            
            has_active_tier1_ban = (
                session.ban_status == BanStatus.ONE_HOUR and 
                session.ban_end_time and 
                datetime.now() < session.ban_end_time
            )
            
            if tier == 2 and remaining <= 3:
                st.warning(f"⚠️ **Tier 2 Alert**: Only {remaining} questions remaining until {TIER_2_BAN_HOURS}-hour reset!")
            elif tier == 1 and remaining <= 2 and remaining > 0:
                st.info(f"ℹ️ **Tier 1**: {remaining} questions remaining until {TIER_1_BAN_HOURS}-hour break.")
            elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                if has_active_tier1_ban:
                    time_remaining = session.ban_end_time - datetime.now()
                    minutes = int(time_remaining.total_seconds() / 60)
                    hours = int(time_remaining.total_seconds() / 3600)
                    if hours >= 1:
                        st.warning(f"⏳ **{TIER_1_BAN_HOURS}-hour break in progress**: {hours} hour(s) remaining")
                    else:
                        st.warning(f"⏳ **{TIER_1_BAN_HOURS}-hour break in progress**: {minutes} minutes remaining")
                elif session.tier1_completed_in_cycle:
                    st.info("✅ **Tier 1 Complete**: You can now proceed to Tier 2!")
                else:
                    st.info(f"ℹ️ **Tier 1 Complete**: Your next question will trigger a {TIER_1_BAN_HOURS}-hour break before Tier 2.")

        visible_messages = session.messages[session.display_message_offset:]
        for msg in visible_messages:
            role = msg.get("role", "user")
            avatar_icon = USER_AVATAR_B64 if role == "user" else FIFI_AVATAR_B64
            with st.chat_message(role, avatar=avatar_icon):
                if msg.get("display_only_notice", False) and msg.get("role") == "assistant":
                    pass
                else:
                    st.markdown(msg.get("content", ""))
                
                if msg.get("source"):
                    source_color = {
                        "FiFi": "🧠", "FiFi Web Search": "🌐", 
                        "Content Moderation": "🛡️", "System Fallback": "⚠️",
                        "Error Handler": "❌", "Session Analytics": "📈", 
                        "Session History": "📜", "Conversation Summary": "📝", "Topic Analysis": "🔍",
                        "Business Rules": "⚙️",
                        "12Taste Order Status": "🛒" # Changed source name
                    }.get(msg['source'], "🤖")
                    st.caption(f"{source_color} Source: {msg['source']}")
                
                # Removed "Enhanced with" block completely
                # indicators = []
                # if msg.get("used_pinecone"): indicators.append("🧠 FiFi Knowledge Base")
                # if msg.get("used_search"): indicators.append("🌐 FiFi Web Search")
                # if msg.get("is_meta_response"): indicators.append("📈 Session Analytics")
                # if msg.get("is_pricing_stock_redirect"): indicators.append("⚙️ Business Rules")
                # if msg.get("source") == "WooCommerce": indicators.append("🛒 WooCommerce")
                # if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")
                
                if msg.get("safety_override"):
                    st.warning("🛡️ Safety Override: Switched to verified sources")
                
                if msg.get("has_citations") and msg.get("has_inline_citations"):
                    st.caption("📚 Response includes verified citations")
                
    overall_chat_disabled = (
        not st.session_state.get('is_chat_ready', False) or 
        should_disable_chat_input_by_dialog or 
        session.ban_status.value != BanStatus.NONE.value or
        st.session_state.get('is_processing_question', False)
    )

    if 'rate_limit_hit' in st.session_state:
        rate_limit_info = st.session_state.rate_limit_hit
        time_until_next = rate_limit_info.get('time_until_next', 0)
        max_requests = RATE_LIMIT_REQUESTS
        window_seconds = RATE_LIMIT_WINDOW_SECONDS
        
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
            elif category == "greeting_or_polite":
                st.info(f"👋 **Greeting Detected**")
                st.markdown("Hello! I'm FiFi, your AI assistant for the food & beverage industry. How can I help you today?")
            elif category == "order_query" and session.user_type == UserType.GUEST and not session_manager.woocommerce.config.WOOCOMMERCE_ENABLED:
                st.info("🛒 **Order Inquiry** - Order lookup is available for registered users on supported e-commerce platforms. Please log in or enable WooCommerce in your configuration.")
            else:
                st.warning(f"🎯 **Off-Topic Question** - Please ask about food ingredients, suppliers, or market trends.")
            
            if category != "greeting_or_polite":
                st.info(f"💡 **Guidance**: {message}")
            st.caption(f"Confidence: {confidence:.1%} | Category: {category}")
        with col2:
            if st.button("✕", key="dismiss_context", help="Dismiss this message", use_container_width=True):
                del st.session_state.context_flagged
                st.rerun()
    
    if 'pricing_stock_notice' in st.session_state:
        notice_info = st.session_state.pricing_stock_notice
        query_type = notice_info.get('query_type', 'pricing')
        message = notice_info.get('message', '')

        col1, col2 = st.columns([5, 1])
        with col1:
            if query_type == 'pricing':
                st.info("💰 **Pricing Information Notice**")
            else:
                st.info("📦 **Stock Availability Notice**")

            st.markdown(message)
        with col2:
            if st.button("✕", key="dismiss_pricing_notice", help="Dismiss this message", use_container_width=True):
                del st.session_state.pricing_stock_notice
                st.rerun()

    if not overall_chat_disabled and not st.session_state.get('is_processing_question', False):
        user_type = session.user_type.value
        current_count = session.daily_question_count
        
        if user_type == UserType.GUEST.value and current_count == GUEST_QUESTION_LIMIT - 1:
            st.warning(f"⚠️ **Final Guest Question Coming Up!** Your next question will be your last before email verification is required.")
            
        elif user_type == UserType.EMAIL_VERIFIED_GUEST.value and current_count == EMAIL_VERIFIED_QUESTION_LIMIT - 1:
            st.warning(f"⚠️ **Final Question Today!** Your next question will be your last for the next {EMAIL_VERIFIED_BAN_HOURS} hours.")
            
        elif user_type == UserType.REGISTERED_USER.value:
            if current_count == REGISTERED_USER_TIER_1_LIMIT - 1:
                st.warning(f"⚠️ **Tier 1 Final Question Coming Up!** After your next question, you'll need a {TIER_1_BAN_HOURS}-hour break.")
            elif current_count == REGISTERED_USER_QUESTION_LIMIT - 1:
                st.warning(f"⚠️ **Final Question Today!** Your next question will be your last for {TIER_2_BAN_HOURS} hours.")

    prompt = st.chat_input(
        "Ask me about ingredients, suppliers, or market trends..." if not st.session_state.get('is_processing_question', False) 
        else "Processing your question, please wait...",
        disabled=overall_chat_disabled
    )
    
    if prompt:
        logger.info(f"🎯 Processing question from {session.session_id[:8]}")
        
        st.session_state.is_processing_question = True
        
        if session_manager.check_if_attempting_to_exceed_limits(session):
            st.session_state.is_processing_question = False
            if session.user_type.value == UserType.GUEST.value and \
               session.daily_question_count >= GUEST_QUESTION_LIMIT:
                st.session_state.verification_stage = 'email_entry'
                st.session_state.chat_blocked_by_dialog = True
                st.session_state.final_answer_acknowledged = True
            st.rerun()
            return
        
        with st.chat_message("user", avatar=USER_AVATAR_B64):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=FIFI_AVATAR_B64):
            with st.spinner("🔍 FiFi is processing your question and we request your patience..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    st.session_state.just_answered = True
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue.")
                        st.session_state.verification_stage = 'email_entry' 
                        st.session_state.chat_blocked_by_dialog = True
                    elif response.get('banned'):
                        st.error(response.get("content", 'Access restricted.'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            # Ensure time_remaining is a timedelta or convert if it's an int/float (seconds)
                            if isinstance(time_remaining, (int, float)):
                                time_remaining_td = timedelta(seconds=time_remaining)
                            else: # Assume it's already a timedelta from the BanStatus.ONE_HOUR / TWENTY_FOUR_HOUR messages
                                time_remaining_td = time_remaining
                            
                            hours = int(time_remaining_td.total_seconds() // 3600)
                            minutes = int((time_remaining_td.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    elif response.get('display_only_notice', False):
                        pass
                    else:
                        st.markdown(response.get("content", "No response generated."))
                        if response.get("source"):
                            source_color = {
                                "FiFi": "🧠", "FiFi Web Search": "🌐",
                                "Content Moderation": "🛡️", "System Fallback": "⚠️",
                                "Error Handler": "❌", "Session Analytics": "📈",
                                "Session History": "📜", "Conversation Summary": "📝", "Topic Analysis": "🔍",
                                "Business Rules": "⚙️",
                                "12Taste Order Status": "🛒" # Changed source name
                            }.get(response['source'], "🤖")
                            st.caption(f"{source_color} Source: {response['source']}")
                        
                        # Removed "Enhanced with" indicators
                        # indicators = []
                        # if response.get("used_pinecone"): indicators.append("🧠 FiFi Knowledge Base")
                        # if response.get("used_search"): indicators.append("🌐 FiFi Web Search")
                        # if response.get("is_meta_response"): indicators.append("📈 Session Analytics")
                        # if response.get("is_pricing_stock_redirect"): indicators.append("⚙️ Business Rules")
                        # if response.get("source") == "WooCommerce": indicators.append("🛒 WooCommerce")
                        # if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                        if response.get("safety_override"): st.warning("🛡️ Safety Override: Switched to verified sources")
                        if response.get("has_citations") and response.get("has_inline_citations"): st.caption("📚 Response includes verified citations")
                        
                        logger.info(f"✅ Question processed successfully")

                except Exception as e:
                    logger.error(f"❌ AI response failed: {e}", exc_info=True)
                    st.error("⚠️ I encountered an error. Please try again.")
                finally:
                    st.session_state.is_processing_question = False
        
        st.session_state.is_processing_question = False
        st.rerun()

class PersistentState:
    """Manages state that must survive reruns"""
    
    @staticmethod
    def set(key: str, value: Any, ttl_seconds: int = 300):
        """Set a value with optional TTL"""
        st.session_state[f'_persistent_{key}'] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl_seconds)
        }
    
    @staticmethod
    def get(key: str, default=None):
        """Get a value if not expired"""
        data = st.session_state.get(f'_persistent_{key}')
        if data and datetime.now() and datetime.now() < data['expires']:
            return data['value']
        return default
    
    @staticmethod
    def delete(key: str):
        """Delete a persistent state key"""
        if f'_persistent_{key}' in st.session_state:
            del st.session_state[f'_persistent_{key}']


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
                    '_load_any_session': lambda self, session_id: None, # Need for FastAPI backend
                    'find_sessions_by_fingerprint': lambda self, fingerprint_id: [],
                    'find_sessions_by_email': lambda self, email: [],
                    'cleanup_old_inactive_sessions': lambda self: None
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
                    '_needs_current_information': lambda self, prompt: False,
                    'get_response': lambda self, prompt, history=None: {
                        "content": "AI system temporarily unavailable.",
                        "success": False
                    }
                })()
            
            rate_limiter = RateLimiter(max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS)
            fingerprinting_manager = DatabaseManager.FingerprintingManager()
            
            try:
                email_verification_manager = DatabaseManager.EmailVerificationManager(config)
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
            
            question_limit_manager = DatabaseManager.QuestionLimitManager()

            try:
                if config.WOOCOMMERCE_ENABLED:
                    woocommerce_manager = WooCommerceManager(config)
                else:
                    woocommerce_manager = None
            except Exception as e:
                logger.error(f"WooCommerce manager initialization failed: {e}")
                woocommerce_manager = None

            st.session_state.session_manager = SessionManager(
                config, st.session_state.db_manager, zoho_manager, ai_system, 
                rate_limiter, fingerprinting_manager, email_verification_manager, 
                question_limit_manager, woocommerce_manager
            )
            
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager

            st.session_state.chat_blocked_by_dialog = False
            st.session_state.verification_stage = None
            st.session_state.guest_continue_active = False
            st.session_state.is_chat_ready = False
            
            st.session_state.wordpress_error = {'show_fallback': False}
            st.session_state.wordpress_fallback_active = False
            st.session_state.fallback_email = ''
            st.session_state.fallback_verification_stage = None
            st.session_state.fallback_verification_email = ''
            st.session_state.fallback_session_id = ''
            st.session_state.fingerprint_client_side_completed = False # NEW: Client-side FP status
            st.session_state.expecting_order_id = False # NEW: For WooCommerce order flow

            st.session_state.initialized = True
            logger.info("✅ Application initialized successfully")
            
        except Exception as e:
            logger.critical(f"Critical initialization failure: {e}", exc_info=True)
            st.session_state.initialized = False
            return False
    
    return True

def main_fixed():
    """Main application entry point with optimized fingerprint handling."""
    try:
        st.set_page_config(
            page_title="FiFi AI Assistant", 
            page_icon=FIFI_AVATAR_B64 if FIFI_AVATAR_B64 else "🤖", 
            layout="wide"
        )
    except Exception as e:
        logger.error(f"Failed to set page config: {e}")

    if st.session_state.get('session_expired', False):
        logger.info("Session expired flag detected - forcing welcome page")
        st.session_state['page'] = None
        if 'session_expired' in st.session_state:
            del st.session_state['session_expired']
        st.info("⏰ Your session expired. Please start a new session.")

    if 'initialized' not in st.session_state:
        defaults = {
            "initialized": False, "is_loading": False, "loading_message": "",
            "is_chat_ready": False, "fingerprint_complete": False,
            "chat_blocked_by_dialog": False, "verification_stage": None,
            "guest_continue_active": False, "final_answer_acknowledged": False,
            "gentle_prompt_shown": False, "email_verified_final_answer_acknowledged": False,
            "must_verify_email_immediately": False, "skip_email_allowed": True,
            "page": None, "fingerprint_processed_for_session": {},
            "is_processing_question": False,
            "wordpress_error": {'show_fallback': False},
            "wordpress_fallback_active": False,
            "fallback_email": '',
            "fallback_verification_stage": None,
            "fallback_verification_email": '',
            "fallback_session_id": '',
            "fingerprint_client_side_completed": False,
            "expecting_order_id": False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        init_success = ensure_initialization_fixed()
        if not init_success:
            st.error("⚠️ Application failed to initialize properly. Please refresh the page.")
            return

    # Handle URL-based events EARLY, before any other logic
    handle_emergency_save_requests_from_query()
    if handle_fingerprint_status_update_from_query():
        return

    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Session Manager not available. Please refresh the page.")
        return

    # REMOVED: No longer create sessions on initial load

    if st.session_state.get('is_loading', False):
        loading_reason = st.session_state.get('loading_reason', 'unknown')
        session_id_for_loading = st.session_state.get('current_session_id')
        session = None

        if session_id_for_loading:
            session = session_manager.db.load_session(session_id_for_loading)
            if not session:
                logger.error(f"Session {session_id_for_loading[:8]} not found during loading. Clearing state.")
                set_loading_state(False)
                st.session_state['page'] = None
                if 'current_session_id' in st.session_state:
                    del st.session_state['current_session_id']
                st.rerun()
                return

        if loading_reason == 'start_guest':
            set_loading_state(True, "Setting up device recognition...")

            # Render fingerprint component ONLY if it hasn't been rendered or fingerprinting hasn't completed
            fingerprint_rendered_key = f"fingerprint_rendered_{session_id_for_loading}"
            if not st.session_state.get(fingerprint_rendered_key, False) and \
               not st.session_state.get('fingerprint_client_side_completed', False):
                
                session_manager.fingerprinting.render_fingerprint_component(session_id_for_loading)
                st.session_state[fingerprint_rendered_key] = True
                st.session_state.fingerprint_wait_start = time.time()
                logger.info(f"✅ Fingerprint component rendered and timer started for {session_id_for_loading[:8]}. Rerunning.")
                st.rerun()

            # If fingerprint component was rendered, now wait for its completion or timeout
            if st.session_state.get(fingerprint_rendered_key, False):
                # Poll DB for fingerprint data
                latest_session_from_db = session_manager.db.load_session(session_id_for_loading)
                if latest_session_from_db:
                    session.fingerprint_id = latest_session_from_db.fingerprint_id
                    session.fingerprint_method = latest_session_from_db.fingerprint_method
                    session.browser_privacy_level = latest_session_from_db.browser_privacy_level
                    
                    # FIXED: Check if fingerprint changed from temp to real and re-attempt inheritance
                    if (session.fingerprint_id.startswith(("temp_py_", "temp_fp_")) and 
                        latest_session_from_db.fingerprint_id and
                        not latest_session_from_db.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))):
                        
                        logger.info(f"🔄 Fingerprint updated from temp to real: {latest_session_from_db.fingerprint_id[:12]}")
                        session = latest_session_from_db
                        session_manager._attempt_fingerprint_inheritance(session)
                        session_manager.db.save_session(session)
                        logger.info(f"✅ Re-attempted inheritance for Guest {session.session_id[:8]}, daily_q now: {session.daily_question_count}")
                
                fingerprint_is_stable = not (session.fingerprint_id is None or session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")))
                
                wait_start = st.session_state.get('fingerprint_wait_start', time.time())
                elapsed = time.time() - wait_start

                if fingerprint_is_stable or elapsed > FINGERPRINT_TIMEOUT_SECONDS:
                    st.session_state.is_chat_ready = True
                    set_loading_state(False)
                    if session and session.last_activity is None:
                        session.last_activity = datetime.now()
                        session_manager.db.save_session(session)
                    logger.info(f"Fingerprinting for {session_id_for_loading[:8]} completed (stable={fingerprint_is_stable}, elapsed={elapsed:.1f}s). Chat ready.")
                else:
                    # Still waiting for fingerprint
                    st.session_state.loading_message = f"Setting up device recognition... ({int(FINGERPRINT_TIMEOUT_SECONDS - elapsed)}s remaining)"
                    time.sleep(0.5)
                    st.rerun()
                    return

            st.session_state.page = "chat"
            
        elif loading_reason == 'authenticate':
            username = st.session_state.get('temp_username', '')
            password = st.session_state.get('temp_password', '')
            if username and password:
                authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                if authenticated_session:
                    session = authenticated_session
                    st.session_state.current_session_id = authenticated_session.session_id
                    st.session_state.page = "chat"
                    st.session_state.is_chat_ready = True
                    st.session_state.fingerprint_wait_start = None
                    st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                    st.balloons()
                else:
                    set_loading_state(False)
                    st.session_state.page = None
                    if 'temp_username' in st.session_state: del st.session_state['temp_username']
                    if 'temp_password' in st.session_state: del st.session_state['temp_password']
                    st.rerun()
                    return
            else:
                set_loading_state(False)
                st.error("Authentication failed: Missing username or password.")
                st.session_state.page = None
                st.rerun()
                return
            
            set_loading_state(False)

        if 'loading_reason' in st.session_state:
            del st.session_state['loading_reason']

        if session and not st.session_state.get('is_chat_ready', False):
            if session.user_type == UserType.REGISTERED_USER:
                st.session_state.is_chat_ready = True

        set_loading_state(False)
        st.rerun()
        return
            
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session(create_if_missing=False)
        
        if session is None or not session.active:
            logger.warning("Expected active session for 'chat' page but got None or inactive. Forcing welcome.")
            for key in list(st.session_state.keys()):
                if key != 'initialized':
                    del st.session_state[key]
            st.session_state['page'] = None
            st.rerun()
            return
        
        if not st.session_state.get('is_chat_ready', False):
            if 'fingerprint_wait_start' not in st.session_state:
                 st.session_state.fingerprint_wait_start = time.time()
            
            if time.time() - st.session_state.fingerprint_wait_start > FINGERPRINT_TIMEOUT_SECONDS:
                st.session_state.is_chat_ready = True
                logger.warning(f"Failsafe: Fingerprint timeout in chat page ({FINGERPRINT_TIMEOUT_SECONDS}s) - enabling chat with fallback for session {session.session_id[:8]}")
            else:
                with st.chat_message("assistant", avatar=FIFI_AVATAR_B64):
                    st.info("🔒 Initializing secure session and device recognition...")
                    st.markdown(f"*Please wait, this ensures consistent experience and usage tracking.*")
                st.rerun()
                return

        # Render activity tracker and check for session timeout
        activity_data_from_js = None
        if session and session.session_id:
            activity_data_from_js = render_simple_activity_tracker(session.session_id)
            st.session_state.latest_activity_data_from_js = activity_data_from_js
        
        timeout_triggered = check_timeout_and_trigger_reload(session_manager, session, activity_data_from_js)
        if timeout_triggered:
            return

        # Render the main UI
        render_sidebar(session_manager, session, st.session_state.pdf_exporter)
        render_chat_interface_simplified(session_manager, session, activity_data_from_js)

if __name__ == "__main__":
    main_fixed()
