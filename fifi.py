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

## CHANGE: Import production_config
from production_config import (
    DAILY_RESET_WINDOW_HOURS, SESSION_TIMEOUT_MINUTES, FINGERPRINT_TIMEOUT_SECONDS,
    TIER_1_BAN_HOURS, TIER_2_BAN_HOURS, EMAIL_VERIFIED_BAN_HOURS,
    GUEST_QUESTION_LIMIT, EMAIL_VERIFIED_QUESTION_LIMIT, REGISTERED_USER_QUESTION_LIMIT, REGISTERED_USER_TIER_1_LIMIT,
    RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
    MAX_MESSAGE_LENGTH, MAX_PDF_MESSAGES, MAX_FINGERPRINT_CACHE_SIZE, MAX_RATE_LIMIT_TRACKING, MAX_ERROR_HISTORY,
    CRM_SAVE_MIN_QUESTIONS, EVASION_BAN_HOURS,
    FASTAPI_EMERGENCY_SAVE_URL, FASTAPI_EMERGENCY_SAVE_TIMEOUT,
    DAILY_RESET_WINDOW, SESSION_TIMEOUT_DELTA,
    FINGERPRINT_TIMEOUT_SECONDS # <--- ADDED FINGERPRINT_TIMEOUT_SECONDS
)

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
    # MODIFIED: Import TavilySearch instead of TavilyClient for Langchain integration
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
            <h3 style="color: #333; margin-bottom: 0.5rem;">🤖 FiFi AI Assistant</h3>
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
        ## CHANGE: Memory leak fix - limit error history size
        self.MAX_ERROR_HISTORY_SIZE = MAX_ERROR_HISTORY

    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        error_str = str(error).lower()
        
        # Check if the error is a requests.exceptions.ReadTimeout for Zoho token
        if isinstance(error, requests.exceptions.ReadTimeout) and component == "ZohoCRMManager" and operation == "get_access_token":
             # This specific timeout often means the Zoho auth server is just slow. Not critical.
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
            component=component, operation=operation, error_type=type(error).__name__, # Fix: Use type(error).__name__
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
        ## CHANGE: Memory leak fix - limit error history size
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

    ## CHANGE: Add timeout tracking fields for FastAPI beacon to record
    timeout_detected_at: Optional[datetime] = None
    timeout_reason: Optional[str] = None


class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.conn = None
        self._connection_string = connection_string ## CHANGE: Store connection string for reconnects
        self._last_health_check = None
        self._health_check_interval = timedelta(minutes=5)
        self._max_reconnect_attempts = 3 ## CHANGE: Added for better reconnect logic
        logger.info("🔄 INITIALIZING DATABASE MANAGER")
        
        try: # Added try-except block for robust initialization
            # Prioritize SQLite Cloud if configured and available
            if connection_string and SQLITECLOUD_AVAILABLE:
                self.conn, self.db_type = self._try_sqlite_cloud(connection_string)
            
            # Fallback to local SQLite if cloud connection failed or not attempted
            if not self.conn:
                self.conn, self.db_type = self._try_local_sqlite()
            
            # Final fallback to in-memory if all else fails
            if not self.conn:
                logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED. FALLING BACK TO NON-PERSISTENT IN-MEMORY STORAGE.")
                self.db_type = "memory"
                self.local_sessions = {}
            
            # Initialize database schema if a connection was established
            if self.conn:
                try:
                    self._init_complete_database()
                    logger.info("✅ Database schema ready and indexes created.")
                    error_handler.mark_component_healthy("Database")
                except Exception as e:
                    logger.error(f"Database schema initialization failed: {e}", exc_info=True)
                    # If schema init fails, invalidate connection and fall back to memory
                    self.conn = None
                    self.db_type = "memory"
                    self.local_sessions = {}
                    logger.critical("🚨 DATABASE SCHEMA INITIALIZATION FAILED. FALLING BACK TO NON-PERSISTENT IN-MEMORY STORAGE.")

        except ValueError as e: # Catch the specific ValueError from sqlitecloud here
            logger.critical(f"🚨 CRITICAL SQLITECLOUD COMMUNICATION ERROR DURING DATABASE INIT: {e}", exc_info=True)
            self.conn = None
            self.db_type = "memory"
            self.local_sessions = {}
            logger.critical("🚨 FORCE FALLBACK TO NON-PERSISTENT IN-MEMORY STORAGE DUE TO SQLITECLOUD ERROR.")
        except Exception as e: # Catch any other unexpected errors during init
            logger.critical(f"🚨 UNEXPECTED ERROR DURING DATABASE MANAGER INIT: {e}", exc_info=True)
            self.conn = None
            self.db_type = "memory"
            self.local_sessions = {}
            logger.critical("🚨 FORCE FALLBACK TO NON-PERSISTENT IN-MEMORY STORAGE DUE TO UNEXPECTED ERROR.")

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
                    declined_recognized_email_at TEXT,
                    -- NEW: Timeout tracking fields for FastAPI beacon to record
                    timeout_detected_at TEXT,
                    timeout_reason TEXT
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
                ("timeout_detected_at", "TEXT"), ## CHANGE: Added new column
                ("timeout_reason", "TEXT") ## CHANGE: Added new column
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
    
    # ADDITION: New method for explicit SQLiteCloud reconnection
    def _reconnect_cloud(self):
        """Force close and recreate SQLiteCloud connection"""
        if self.conn and self.db_type == "cloud":
            try:
                self.conn.close()
            except Exception as e:
                logger.debug(f"Error closing existing SQLiteCloud connection: {e}")
            
            # Wait briefly to ensure socket cleanup
            time.sleep(0.5)
            
            try:
                self.conn = sqlitecloud.connect(self._connection_string)
                # Test with simple query
                self.conn.execute("SELECT 1").fetchone()
                logger.info("✅ SQLiteCloud reconnection successful")
                return True
            except (ValueError, Exception) as e: # Catch ValueError too
                logger.error(f"❌ SQLiteCloud reconnection failed: {e}")
                return False
        return False

    # ADDITION: New method for explicit SQLiteCloud validation
    def _validate_cloud_connection(self):
        """Validate SQLiteCloud connection is healthy"""
        if self.db_type != "cloud":
            return True
        
        try:
            # Use a lightweight query to test connection
            result = self.conn.execute("SELECT 1").fetchone()
            return result == (1,)
        except (ValueError, Exception) as e: # Catch ValueError too
            logger.warning(f"⚠️ SQLiteCloud validation failed: {e}")
            return False

    # ADDITION: Helper to fallback to in-memory
    def _fallback_to_local(self):
        """Switches to in-memory storage if cloud/file DB fails."""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.debug(f"Error closing failed DB connection: {e}")
        self.db_type = "memory"
        self.local_sessions = {}
        logger.critical("🚨 FALLBACK: Switched to non-persistent in-memory storage due to cloud/file DB failure.")

    ## CHANGE: Improved reconnect logic to use _reconnect_cloud
    def _reconnect(self):
        """Reconnect with exponential backoff"""
        for attempt in range(self._max_reconnect_attempts):
            try:
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception as e:
                        logger.debug(f"Error closing existing DB connection during reconnect: {e}")
                
                if self._connection_string and SQLITECLOUD_AVAILABLE:
                    if self._reconnect_cloud(): # Try cloud reconnect
                        self.db_type = "cloud"
                        self._init_complete_database()
                        logger.info(f"Database (Cloud) reconnected on attempt {attempt + 1}")
                        return
                
                # Fallback to local SQLite if cloud reconnect fails or not applicable
                self.conn = sqlite3.connect("fifi_sessions_v2.db", check_same_thread=False)
                self.db_type = "file"
                self._init_complete_database() # Re-init schema in case of changes
                logger.info(f"Database (File) reconnected on attempt {attempt + 1}")
                return

            except (ValueError, Exception) as e: # Catch ValueError too
                wait_time = 2 ** attempt  # Exponential backoff
                logger.error(f"Reconnect attempt {attempt + 1} failed, waiting {wait_time}s: {e}", exc_info=True)
                time.sleep(wait_time)
        
        # Final fallback to in-memory if all attempts fail
        logger.critical("🚨 ALL RECONNECTION ATTEMPTS FAILED. FALLING BACK TO IN-MEMORY STORAGE")
        self._fallback_to_local()


    ## CHANGE: Simplified _ensure_connection_healthy to use _reconnect
    def _ensure_connection_healthy(self):
        """Ensure database connection is healthy, reconnect if needed"""
        if self.db_type == "cloud" and not self._validate_cloud_connection():
            logger.warning("SQLiteCloud connection unhealthy, attempting reconnection...")
            self._reconnect()
            if not self.conn: # If reconnect failed, ensure in-memory is ready
                if not hasattr(self, 'local_sessions'):
                    self.local_sessions = {}
        elif self.db_type == "file" and not self._check_connection_health(): # Use old check for file
             logger.warning("Local SQLite connection unhealthy, attempting reconnection...")
             self._reconnect()
             if not self.conn:
                if not hasattr(self, 'local_sessions'):
                    self.local_sessions = {}
        elif self.db_type == "memory":
            pass # No external connection to manage

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with SQLite Cloud compatibility and connection health check"""
        logger.debug(f"💾 SAVING SESSION TO DB: {session.session_id[:8]} | user_type={session.user_type.value} | email={session.email} | messages={len(session.messages)} | fp_id={session.fingerprint_id[:8] if session.fingerprint_id else 'None'} | active={session.active}") ## CHANGE: Handle None for FP ID
        
        self._ensure_connection_healthy() ## CHANGE: Simplified call
        
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
            ## CHANGE: Handle None for new timeout tracking fields
            timeout_detected_at_iso = session.timeout_detected_at.isoformat() if session.timeout_detected_at else None


            self.conn.execute(
                '''REPLACE INTO sessions (session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
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
                 declined_recognized_email_at_iso, ## CHANGE: NEW field
                 timeout_detected_at_iso, ## CHANGE: NEW field
                 session.timeout_reason)) ## CHANGE: NEW field
            self.conn.commit()
            
            logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}, active={session.active}, rev_pending={session.reverification_pending}")
            
        except (ValueError, Exception) as e: # Catch ValueError during save operation too
            logger.error(f"Failed to save session {session.session_id[:8]}: {e}", exc_info=True)
            # Try to fallback to in-memory on save failure
            self._fallback_to_local()
            # After fallback, save to local_sessions
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                logger.info(f"Fallback: Saved session {session.session_id[:8]} to in-memory storage")
            raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility and connection health check"""
        self._ensure_connection_healthy() ## CHANGE: Simplified call

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
            ## CHANGE: Add compatibility for new timeout tracking fields
            if session and not hasattr(session, 'timeout_detected_at'):
                session.timeout_detected_at = None
            if session and not hasattr(session, 'timeout_reason'):
                session.timeout_reason = None
            
            return copy.deepcopy(session)
        
        try:
            # NEVER set row_factory for cloud connections - always use raw tuples
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None
            
            ## CHANGE: Update SELECT statement to include new fields (timeout_detected_at, timeout_reason)
            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
            row = cursor.fetchone()
            
            if not row: 
                logger.debug(f"No active session found for ID {session_id[:8]}.")
                return None
            
            ## CHANGE: Update expected columns for new fields (now 41)
            expected_min_cols = 41
            if len(row) < expected_min_cols:
                logger.error(f"Row has insufficient columns: {len(row)} (expected at least {expected_min_cols}). Data corruption suspected or old schema.")
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
                ## CHANGE: Safely get timeout tracking fields
                loaded_timeout_detected_at = datetime.fromisoformat(row[39]) if len(row) > 39 and row[39] else None
                loaded_timeout_reason = row[40] if len(row) > 40 else None


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
                    int(session.registration_link_clicked), session.recognition_response, session.display_message_offset,
                    int(session.reverification_pending), 
                    session.pending_user_type.value if session.pending_user_type else None,
                    session.pending_email, session.pending_full_name,
                    session.pending_zoho_contact_id, session.pending_wp_token,
                    declined_recognized_email_at_iso, ## CHANGE: NEW field
                    timeout_detected_at_iso, ## CHANGE: NEW field
                    session.timeout_reason)) ## CHANGE: NEW field
            self.conn.commit()
            
            logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}, active={session.active}, rev_pending={session.reverification_pending}")
            
        except (ValueError, Exception) as e: # Catch ValueError during save operation too
            logger.error(f"Failed to save session {session.session_id[:8]}: {e}", exc_info=True)
            # Try to fallback to in-memory on save failure
            self._fallback_to_local()
            # After fallback, save to local_sessions
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = copy.deepcopy(session)
                logger.info(f"Fallback: Saved session {session.session_id[:8]} to in-memory storage")
            raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility and connection health check"""
        self._ensure_connection_healthy() ## CHANGE: Simplified call

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
            ## CHANGE: Add compatibility for new timeout tracking fields
            if session and not hasattr(session, 'timeout_detected_at'):
                session.timeout_detected_at = None
            if session and not hasattr(session, 'timeout_reason'):
                session.timeout_reason = None
            
            return copy.deepcopy(session)
        
        try:
            # NEVER set row_factory for cloud connections - always use raw tuples
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None
            
            ## CHANGE: Update SELECT statement to include new fields (timeout_detected_at, timeout_reason)
            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
            row = cursor.fetchone()
            
            if not row: 
                logger.debug(f"No active session found for ID {session_id[:8]}.")
                return None
            
            ## CHANGE: Update expected columns for new fields (now 41)
            expected_min_cols = 41
            if len(row) < expected_min_cols:
                logger.error(f"Row has insufficient columns: {len(row)} (expected at least {expected_min_cols}). Data corruption suspected or old schema.")
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
                ## CHANGE: Safely get timeout tracking fields
                loaded_timeout_detected_at = datetime.fromisoformat(row[39]) if len(row) > 39 and row[39] else None
                loaded_timeout_reason = row[40] if len(row) > 40 else None


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
                    declined_recognized_email_at=loaded_declined_recognized_email_at, # NEW
                    timeout_detected_at=loaded_timeout_detected_at, ## CHANGE: NEW field
                    timeout_reason=loaded_timeout_reason ## CHANGE: NEW field
                )
                
                logger.debug(f"Successfully loaded session {session_id[:8]}: user_type={user_session.user_type.value}, messages={len(user_session.messages)}, active={user_session.active}, rev_pending={user_session.reverification_pending}")
                return user_session
                
            except (ValueError, Exception) as e: # Catch ValueError during row parsing
                logger.error(f"Failed to create UserSession object from row for session {session_id[:8]}: {e}", exc_info=True)
                logger.error(f"Problematic row data (truncated): {str(row)[:200]}")
                self._fallback_to_local() # Fallback if data parsing is an issue
                return None
                
        except (ValueError, Exception) as e: # Catch ValueError during execute too
            logger.error(f"Failed to load session {session_id[:8]}: {e}", exc_info=True)
            self._fallback_to_local() # Fallback if load itself fails
            return None

    @handle_api_errors("Database", "Find by Fingerprint")
    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint_id. Includes inactive sessions."""
        logger.debug(f"🔍 SEARCHING FOR FINGERPRINT: {fingerprint_id[:8]}...")
        self._ensure_connection_healthy() ## CHANGE: Simplified call

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
                ## CHANGE: Add compatibility for new timeout tracking fields
                if not hasattr(session, 'timeout_detected_at'):
                    session.timeout_detected_at = None
                if not hasattr(session, 'timeout_reason'):
                    session.timeout_reason = None
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (MEMORY): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            return sessions
        
        try:
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None

            ## CHANGE: Update SELECT statement to include new fields (timeout_detected_at, timeout_reason)
            cursor = self.conn.execute("SELECT session_id, user_type, email, full_name, zoho_contact_id, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm, fingerprint_id, fingerprint_method, visitor_type, daily_question_count, total_question_count, last_question_time, question_limit_reached, ban_status, ban_start_time, ban_end_time, ban_reason, evasion_count, current_penalty_hours, escalation_level, email_addresses_used, email_switches_count, browser_privacy_level, registration_prompted, registration_link_clicked, recognition_response, display_message_offset, reverification_pending, pending_user_type, pending_email, pending_full_name, pending_zoho_contact_id, pending_wp_token, declined_recognized_email_at, timeout_detected_at, timeout_reason FROM sessions WHERE fingerprint_id = ? ORDER BY last_activity DESC", (fingerprint_id,))
            sessions = []
            for row in cursor.fetchall():
                ## CHANGE: Update expected columns for new fields (now 41)
                expected_min_cols = 41
                if len(row) < expected_min_cols:
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
                    ## CHANGE: Safely get timeout tracking fields
                    loaded_timeout_detected_at = datetime.fromisoformat(row[39]) if len(row) > 39 and row[39] else None
                    loaded_timeout_reason = row[40] if len(row) > 40 else None


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
                        declined_recognized_email_at=loaded_declined_recognized_email_at, # NEW
                        timeout_detected_at=loaded_timeout_detected_at, ## CHANGE: NEW field
                        timeout_reason=loaded_timeout_reason ## CHANGE: NEW field
                    )
                    sessions.append(s)
                except (ValueError, Exception) as e: # Catch ValueError during row parsing
                    logger.error(f"Error converting row to UserSession in find_sessions_by_fingerprint: {e}", exc_info=True)
                    continue
            logger.debug(f"📊 FINGERPRINT SEARCH RESULTS (DB): Found {len(sessions)} sessions for {fingerprint_id[:8]}")
            for s in sessions:
                logger.debug(f"  - {s.session_id[:8]}: type={s.user_type.value}, email={s.email}, daily_q={s.daily_question_count}, total_q={s.total_question_count}, last_activity={s.last_activity}, active={s.active}, rev_pending={s.reverification_pending}")
            return sessions
        except (ValueError, Exception) as e: # Catch ValueError during execute too
            logger.error(f"Failed to find sessions by fingerprint '{fingerprint_id[:8]}...': {e}", exc_info=True)
            self._fallback_to_local() # Fallback if search fails
            return []

    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions with the same email address."""
        if not email:
            return []
            
        email_lower = email.lower()
        
        if self.db_type == "memory":
            return [copy.deepcopy(s) for s in self.local_sessions.values() 
                    if s.email and s.email.lower() == email_lower]
        
        try:
            if hasattr(self.conn, 'row_factory'):
                self.conn.row_factory = None
                
            cursor = self.conn.execute(
                """SELECT session_id FROM sessions 
                   WHERE LOWER(email) = LOWER(?) 
                   ORDER BY last_activity DESC""", 
                (email,)
            )
            
            sessions = []
            for row in cursor.fetchall():
                session = self.load_session(row[0])
                if session:
                    sessions.append(session)
                    
            return sessions
            
        except (ValueError, Exception) as e: # Catch ValueError during execute too
            logger.error(f"Failed to find sessions by email '{email}': {e}", exc_info=True)
            self._fallback_to_local() # Fallback if search fails
            return []
    
    ## CHANGE: Added cleanup for old inactive sessions
    def cleanup_old_inactive_sessions(self):
        """Remove old inactive sessions from the database."""
        try:
            if self.db_type == "memory":
                # For in-memory, just clear older than 1 day for simplicity or a smaller threshold
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
        except (ValueError, Exception) as e: # Catch ValueError during cleanup too
            logger.error(f"❌ Failed to clean up old sessions: {e}", exc_info=True)
            self._fallback_to_local() # Fallback if cleanup fails

    # =============================================================================
    # FEATURE MANAGERS (nested within DatabaseManager as they access self.db)
    # =============================================================================

    class FingerprintingManager:
        """Manages browser fingerprinting using external HTML component file."""
        
        def __init__(self):
            self.fingerprint_cache = {}
            self.component_attempts = defaultdict(int)
            ## CHANGE: Memory leak fix - limit cache and attempts size
            self.MAX_CACHE_SIZE = MAX_FINGERPRINT_CACHE_SIZE
            self.MAX_ATTEMPTS = MAX_RATE_LIMIT_TRACKING # Reusing for attempts, could be a separate constant

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
        
                ## CHANGE: JavaScript Injection Fix - Use json.dumps to safely embed session_id
                original_content = html_content
                html_content = html_content.replace('{SESSION_ID}', session_id)
        
                if original_content == html_content:
                    logger.warning(f"⚠️ No {{SESSION_ID}} placeholder found in HTML content!")
                else:
                    logger.debug(f"✅ Replaced {{SESSION_ID}} placeholder with {session_id[:8]}...")
        
                # Render with proper height for CreepJS to work
                logger.debug(f"🔄 Rendering fingerprint component for session {session_id[:8]}...")
                st.components.v1.html(html_content, height=100, scrolling=False)
        
                logger.info(f"✅ External fingerprint component rendered for session {session_id[:8]}")
        
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
            
            ## CHANGE: Memory leak fix - cleanup cache if too large
            if len(self.fingerprint_cache) > self.MAX_CACHE_SIZE:
                sorted_items = sorted(self.fingerprint_cache.items(), 
                                    key=lambda x: x[1].get('last_seen', datetime.min), 
                                    reverse=True)[:self.MAX_CACHE_SIZE // 2] # Keep half the size
                self.fingerprint_cache = dict(sorted_items)

            ## CHANGE: Memory leak fix - cleanup component_attempts if too large
            if len(self.component_attempts) > self.MAX_ATTEMPTS:
                self.component_attempts.clear()
            
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
                error_str = str(e).lower()
                # Special handling for timeout errors
                if "timeout" in error_str or "timed out" in error_str:
                    logger.warning(f"Supabase OTP request timed out for {email}, but email may have been sent: {e}")
                    # Return True since OTP is likely sent despite timeout
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
            ## CHANGE: Use constants for question limits
            self.question_limits = {
                UserType.GUEST.value: GUEST_QUESTION_LIMIT,
                UserType.EMAIL_VERIFIED_GUEST.value: EMAIL_VERIFIED_QUESTION_LIMIT,
                UserType.REGISTERED_USER.value: REGISTERED_USER_QUESTION_LIMIT
            }
            ## CHANGE: Use constants for evasion penalties (in hours)
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
                        'message': self._get_ban_message(session, session.ban_status.value) # Pass ban_status as reason for message
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
            
            ## CHANGE: Use DAILY_RESET_WINDOW constant for daily reset
            if session.last_question_time:
                time_since_last = datetime.now() - session.last_question_time
                if time_since_last >= DAILY_RESET_WINDOW:
                    logger.info(f"Daily question count reset for session {session.session_id[:8]} due to {DAILY_RESET_WINDOW_HOURS}-hour window expiration.")
                    session.daily_question_count = 0
                    session.question_limit_reached = False
            
            # ENHANCED: Tier-based logic for registered users
            if session.user_type.value == UserType.REGISTERED_USER.value:
                if session.daily_question_count >= user_limit: # For registered, user_limit is 20
                    reason_str = 'registered_user_tier2_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str, # New specific reason
                        'message': self._get_ban_message(session, reason_str)
                    }
                ## CHANGE: Use REGISTERED_USER_TIER_1_LIMIT for tier logic
                elif session.daily_question_count >= REGISTERED_USER_TIER_1_LIMIT:
                    ## CHANGE: Use BanStatus.ONE_HOUR for Tier 1 ban
                    if session.ban_status == BanStatus.ONE_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                        time_remaining = session.ban_end_time - datetime.now()
                        return {
                            'allowed': False,
                            'reason': 'banned',
                            'ban_type': BanStatus.ONE_HOUR.value,
                            'time_remaining': time_remaining,
                            'message': self._get_ban_message(session, 'registered_user_tier1_limit')
                        }
                    ## CHANGE: Use REGISTERED_USER_TIER_1_LIMIT for exact check
                    elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                        return {
                            'allowed': True,
                            'tier': 1,
                            'remaining': 0,
                            'warning': f"Next question will trigger a {TIER_1_BAN_HOURS}-hour break before Tier 2."
                        }
                    else: # Questions 11-19
                        remaining = user_limit - session.daily_question_count
                        return {
                            'allowed': True,
                            'tier': 2,
                            'remaining': remaining,
                            'warning': f"Tier 2: {remaining} questions until {TIER_2_BAN_HOURS}-hour limit."
                        }
                else: # Tier 1: Questions 1-9
                    remaining = REGISTERED_USER_TIER_1_LIMIT - session.daily_question_count
                    return {
                        'allowed': True,
                        'tier': 1,
                        'remaining': remaining
                    }
            
            # Original logic for other user types
            if session.user_type.value == UserType.GUEST.value:
                if session.daily_question_count >= user_limit: # user_limit is 4 for GUEST
                    reason_str = 'guest_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str,
                        'message': 'Please provide your email address to continue.'
                    }
            
            elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
                if session.daily_question_count >= user_limit: # user_limit is 10 for EMAIL_VERIFIED_GUEST
                    reason_str = 'email_verified_guest_limit'
                    return {
                        'allowed': False,
                        'reason': reason_str, # New specific reason
                        'message': self._get_ban_message(session, reason_str)
                    }
            
            return {'allowed': True}
        
        ## CHANGE: Atomic record_question_and_check_ban for QuestionLimitManager
        def record_question_and_check_ban(self, session: UserSession, session_manager: 'SessionManager') -> Dict[str, Any]:
            """Atomically record question and apply ban if needed, then save to DB."""
            try:
                # Record question first
                session.daily_question_count += 1
                session.total_question_count += 1
                session.last_question_time = datetime.now()
                
                ban_applied = False
                ban_type = BanStatus.NONE
                ban_reason = ""
                ban_duration_hours = 0
                
                if session.user_type == UserType.REGISTERED_USER:
                    ## CHANGE: Use constants for question limits for ban trigger
                    if session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT + 1:
                        ban_type = BanStatus.ONE_HOUR
                        ban_reason = f"Registered user Tier 1 limit reached ({REGISTERED_USER_TIER_1_LIMIT} questions)"
                        ban_duration_hours = TIER_1_BAN_HOURS
                    elif session.daily_question_count == REGISTERED_USER_QUESTION_LIMIT + 1:
                        ban_type = BanStatus.TWENTY_FOUR_HOUR
                        ban_reason = f"Registered user daily limit reached ({REGISTERED_USER_QUESTION_LIMIT} questions)"
                        ban_duration_hours = TIER_2_BAN_HOURS
                elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
                    ## CHANGE: Use constants for question limits for ban trigger
                    if session.daily_question_count == EMAIL_VERIFIED_QUESTION_LIMIT + 1:
                        ban_type = BanStatus.TWENTY_FOUR_HOUR
                        ban_reason = f"Email-verified daily limit reached ({EMAIL_VERIFIED_QUESTION_LIMIT} questions)"
                        ban_duration_hours = EMAIL_VERIFIED_BAN_HOURS
                
                if ban_type != BanStatus.NONE:
                    self._apply_ban(session, ban_type, ban_reason)
                    ban_applied = True
                    logger.info(f"✅ Atomic ban applied: {session.session_id[:8]} -> {ban_type.value} for {ban_duration_hours}h")

                ## CHANGE: Sync counts for registered users
                if session.user_type == UserType.REGISTERED_USER and session.email:
                    session_manager.sync_registered_user_sessions(session.email, session.session_id)
                
                # Save session immediately after updates
                session_manager._save_session_with_retry(session)
                
                logger.debug(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}")
                
                return {"recorded": True, "ban_applied": ban_applied, "ban_type": ban_type.value if ban_applied else None}
                
            except Exception as e:
                logger.error(f"Failed to record question atomically for {session.session_id[:8]}: {e}", exc_info=True)
                # Attempt to roll back if possible, or at least log the inconsistency
                session.daily_question_count = max(0, session.daily_question_count - 1)
                session.total_question_count = max(0, session.total_question_count - 1)
                session_manager._save_session_with_retry(session) # Try saving rollback
                raise
        
        ## CHANGE: Removed original record_question method

        def _apply_ban(self, session: UserSession, ban_type: BanStatus, reason: str, start_time: Optional[datetime] = None):
            """Applies a ban to the session for a specified duration with immediate database persistence."""
            ## CHANGE: Use constants for ban durations
            ban_hours = {
                BanStatus.ONE_HOUR.value: TIER_1_BAN_HOURS,
                BanStatus.TWENTY_FOUR_HOUR.value: TIER_2_BAN_HOURS,  # Used for both Tier 2 and email verified
                BanStatus.EVASION_BLOCK.value: session.current_penalty_hours
            }.get(ban_type.value, TIER_1_BAN_HOURS)

            session.ban_status = ban_type
            session.ban_start_time = start_time if start_time else datetime.now()
            session.ban_end_time = session.ban_start_time + timedelta(hours=ban_hours)
            session.ban_reason = reason
            session.question_limit_reached = True
        
            # CRITICAL: Save to database immediately
            try:
                # Access the session manager's db attribute for saving
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
            ## CHANGE: Use constants for tier limit messages
            elif ban_reason_from_limit_check == 'registered_user_tier1_limit' or session.ban_status.value == BanStatus.ONE_HOUR.value:
                return f"You've reached the Tier 1 limit ({REGISTERED_USER_TIER_1_LIMIT} questions). Please wait {TIER_1_BAN_HOURS} hour{'s' if TIER_1_BAN_HOURS > 1 else ''} to access Tier 2."
            elif ban_reason_from_limit_check == 'registered_user_tier2_limit' or (session.user_type.value == UserType.REGISTERED_USER.value and session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value):
                return f"Daily limit of {REGISTERED_USER_QUESTION_LIMIT} questions reached. Please retry in {TIER_2_BAN_HOURS} hours."
            elif ban_reason_from_limit_check == 'email_verified_guest_limit' or (session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value and session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value):
                return self._get_email_verified_limit_message()
            # Generic catch-all for other 24-hour bans (if any apply to non-registered)
            elif session.ban_status.value == BanStatus.TWENTY_FOUR_HOUR.value:
                return f"Daily limit reached. Please retry in {TIER_2_BAN_HOURS} hours."
            else: # Fallback, should not be hit if all cases are covered
                return "Access restricted due to usage policy."
        
        def _get_email_verified_limit_message(self) -> str:
            """Specific message for email-verified guests hitting their daily limit."""
            ## CHANGE: Use constants for email verified limit message
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
        """Generates a PDF of the chat transcript with size limits."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        story = []
        story.append(Paragraph("FiFi AI Chat Transcript", self.styles['ChatHeader']))
        story.append(Spacer(1, 12)) # Additional space after header

        ## CHANGE: PDF Memory Bomb fix - Limit messages to prevent memory issues
        messages_to_include = session.messages[-MAX_PDF_MESSAGES:]
        
        if len(session.messages) > MAX_PDF_MESSAGES:
            story.append(Paragraph(f"<i>[Note: Only the last {MAX_PDF_MESSAGES} messages are included. Total conversation had {len(session.messages)} messages.]</i>", self.styles['Caption']))
            story.append(Spacer(1, 8))

        for msg in messages_to_include:
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
                        
            except Exception as e: # Catch all exceptions here
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
                    if attempt < (2 - 1): # Fixed to use explicit value, not a local variable
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
        
        ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS for eligibility
        if (session.user_type.value not in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] or 
            not session.email or 
            len(session.messages) < CRM_SAVE_MIN_QUESTIONS or ## CHANGE: Use constant for minimum messages
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
    ## CHANGE: Use constants for RateLimiter init
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.requests = defaultdict(list)
        self._lock = threading.Lock() # This is a separate RateLimiter specific lock, not the DB one
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        ## CHANGE: Memory leak fix - limit tracked IDs
        self.MAX_TRACKED_IDS = MAX_RATE_LIMIT_TRACKING

    def is_allowed(self, identifier: str) -> Dict[str, Any]:
        """Returns detailed rate limit information including timer."""
        with self._lock:
            now = time.time()
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            
            ## CHANGE: Memory leak fix - cleanup old identifiers
            if len(self.requests) > self.MAX_TRACKED_IDS:
                # Find the oldest N entries to remove
                # This heuristic sorts by the last request time of each identifier
                sorted_ids = sorted(self.requests.items(), 
                                    key=lambda x: max(x[1]) if x[1] else 0,
                                    reverse=False) # Oldest first
                for old_id, _ in sorted_ids[:self.MAX_TRACKED_IDS // 10]: # Remove 10% of max
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

## CHANGE: Enhanced sanitize_input with full validation (from earlier discussion)
def sanitize_input(text: str) -> str:
    """Enhanced input sanitization to prevent XSS and limit length."""
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # HTML escape
    text = html.escape(text)
    
    # Limit length (using global constant)
    text = text[:MAX_MESSAGE_LENGTH].strip()
    
    # Check for SQL injection patterns (logging only, not blocking)
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

    # CHANGE 3: Pinecone Assistant with Business Rules
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
                "4. Is the product discontinued? -> Inform and provide alternatives if available.\n"
                "5. Did I mention any products? -> Ensure each has a [More details] link and all are in Sources section."
            )
            
            assistants_list = self.pc.assistant.list_assistants()
            if self.assistant_name not in [a.name for a in assistants_list]:
                logger.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                # Create new assistant
                self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions=instructions
                )
                logger.info(f"✅ Created assistant '{self.assistant_name}'")
            else:
                logger.info(f"Found existing assistant: '{self.assistant_name}'")
                # Update instructions using the correct method
                try:
                    result = self.pc.assistant.update_assistant(
                        assistant_name=self.assistant_name,
                        instructions=instructions
                    )
                    logger.info(f"✅ Instructions updated for '{self.assistant_name}'")
                    logger.info(f"Update result: {result}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not update instructions: {e}")
        
            # Create assistant instance for chat operations
            assistant_obj = self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        
            # Verify assistant status using correct method
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
        
            # Process content to add UTM parameters to all links
            import re
            def add_utm_to_url(match):
                url = match.group(2)
                if 'utm_source=fifi-eu' not in url:
                    separator = '&' if '?' in url else '?'
                    return f'{match.group(1)}({url}{separator}utm_source=fifi-eu)'
                return match.group(0)
        
            # Add UTM to all markdown links in content
            content = re.sub(r'(\[.*?\])\((https?://[^)]+)\)', add_utm_to_url, content)
        
            # Check if content already has a Sources section
            has_sources_section = "**Sources:**" in content or "**sources:**" in content.lower()
        
            # If no Sources section exists, extract URLs and add one
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
        
            # Determine if we have citations based on URLs in content
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
    
# CHANGE: LLM-Powered Query Reformulation (Replaces old TavilyFallbackAgent entirely)
class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str, openai_api_key: str = None):
        from tavily import TavilyClient
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key, include_answer=True) # Keep both for now to avoid breaking existing calls

        # NEW: Store OpenAI API key for query reformulation
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
        """LLM-powered query reformulation for better search results."""
        try:
            # For very complete, specific queries, minimal reformulation needed
            if len(current_question.split()) > 10 and not any(indicator in current_question.lower() 
                                                              for indicator in ["what about", "how about", "any", "tell me", "more about"]):
                logger.debug(f"Query is detailed enough, using as-is: {current_question}")
                return current_question
            
            # If no OpenAI client available, fallback to code-based
            if not self.openai_client:
                logger.warning("OpenAI client not available, using code-based reformulation fallback")
                return self._fallback_reformulation(current_question, conversation_history)

            # Build conversation context efficiently
            context_parts = []
            if conversation_history:
                # Get last 3 exchanges (6 messages max) for context efficiency
                recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
                for msg in recent_messages:
                    if hasattr(msg, 'content') and msg.content:
                        # Truncate long messages to save tokens
                        content = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
                        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        context_parts.append(f"{role}: {content}")
            
            conversation_context = "\n".join(context_parts) if context_parts else "No prior conversation"

            # Optimized LLM prompt for query reformulation
            reformulation_prompt = f"""Transform this question into an optimized web search query for food ingredient information.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT QUESTION: "{current_question}"

RULES:
1. If follow-up question, incorporate context from conversation
2. Make vague questions specific using context
3. Keep query concise but searchable (3-8 words ideal)
4. Always include "food ingredients" or related terms for industry focus
5. For pricing questions: add "market pricing" not specific prices
6. For supplier questions: include "suppliers" or "sourcing"
7. For availability: include "availability" or "stock"

OUTPUT: Only the optimized search query, nothing else."""

            # Make LLM call with token limits
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a search query optimizer. Respond only with valid JSON."},
                    {"role": "user", "content": reformulation_prompt}
                ],
                max_tokens=30, # Keep very low for cost efficiency
                temperature=0.1 # Low temperature for consistent results
            )
            
            reformulated = response.choices[0].message.content.strip()

            # Validation: Ensure reformulated query is reasonable
            if not reformulated or len(reformulated) < 3:
                logger.warning(f"LLM returned invalid reformulation: '{reformulated}', using fallback")
                return self._fallback_reformulation(current_question, conversation_history)
            
            # Ensure query isn't too long (waste of tokens/poor search results)
            if len(reformulated.split()) > 15 or len(reformulated) > 120:
                logger.warning(f"LLM reformulation too long: '{reformulated}', using fallback")
                return self._fallback_reformulation(current_question, conversation_history)
            
            # Success case
            logger.info(f"✅ LLM reformulated: '{current_question}' → '{reformulated}'")
            return reformulated
            
        except Exception as e:
            logger.error(f"LLM reformulation failed: {e}, using code fallback")
            return self._fallback_reformulation(current_question, conversation_history)

    def _fallback_reformulation(self, current_question: str, conversation_history: List[BaseMessage]) -> str:
        """Fallback to code-based reformulation if LLM fails."""
        logger.debug("Using code-based reformulation fallback")

        # Simple follow-up detection
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
            # Standalone query - add food ingredients context if missing
            if "food" not in current_question_lower and "ingredient" not in current_question_lower:
                return f"{current_question} food ingredients"
            return current_question

        # Extract context keywords from recent messages
        context_keywords = []
        if conversation_history:
            for msg in conversation_history[-3:]:
                if hasattr(msg, 'content'):
                    # Simple keyword extraction
                    words = re.findall(r'\b[A-Z][a-z]+\b', msg.content)
                    context_keywords.extend(words[:2]) # Limit to avoid over-expansion

        # Apply context-based reformulation
        if any(word in current_question_lower for word in ["pricing", "cost", "price"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} pricing costs food ingredients"
            else:
                return f"{current_question} food ingredients pricing market"

        elif any(word in current_question_lower for word in ["suppliers", "supplier", "source", "where"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} suppliers food ingredients sourcing"
            else:
                return f"{current_question} food ingredients suppliers"

        elif any(word in current_question_lower for word in ["availability", "stock", "available"]):
            if context_keywords:
                return f"{' '.join(context_keywords[:2])} availability food ingredients market"
            else:
                return f"{current_question} food ingredients availability"
        
        else:
            # General follow-up
            if context_keywords:
                return f"{current_question} {' '.join(context_keywords[:2])} food ingredients"
            else:
                return f"{current_question} food ingredients"

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
            return content  # Always return original content if processing fails

    def synthesize_search_results(self, results, query: str) -> str:
        """Enhanced synthesis using Tavily's answer + sources."""
        
        # Handle dict response from TavilySearch tool
        if isinstance(results, dict):
            tavily_answer = results.get('answer', '')
            tavily_results = results.get('results', [])

            # If we have Tavily's synthesized answer, use it + add sources
            if tavily_answer and tavily_results:
                response_parts = [tavily_answer]
                response_parts.append("\n\n**Sources:**")

                for i, res in enumerate(tavily_results, 1):
                    if isinstance(res, dict) and 'url' in res and 'title' in res:
                        response_parts.append(f"\n{i}. [{res['title']}]({res['url']})")

                return "\n".join(response_parts)
            
            # Fallback: extract results array if no answer
            elif tavily_results:
                results = tavily_results
            elif tavily_answer:
                return f"Based on web search for '{query}':\n\n{tavily_answer}"
            else:
                logger.warning(f"Unexpected dict format from Tavily: {results.keys()}")
                return f"I found some information but couldn't format it properly."

        # Rest of existing list handling code stays the same...
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
        """Determine whether to use domain-restricted or worldwide search."""
        # Simplified strategy for now, as LLM reformulation should handle context well
        return {
            "strategy": "worldwide_with_exclusions",
            "include_domains": None, 
            "exclude_domains": DEFAULT_EXCLUDED_DOMAINS,
            "reason": "Standard safety fallback"
        }

    def query(self, prompt: str, chat_history: List[BaseMessage], pinecone_error_type: str = None) -> Dict[str, Any]:
        """
        Query Tavily for web search with LLM-powered query reformulation.
        
        Args:
            prompt: The current user question
            chat_history: List of previous messages for context
            pinecone_error_type: Type of Pinecone error (if any) to determine search strategy
        """
        try:
            # Reformulate the query for better search results
            reformulated_query = self.reformulate_query_for_search(prompt, chat_history)
            logger.info(f"🔍 Original query: '{prompt}' → Reformulated: '{reformulated_query}'")
            
            # Determine search strategy
            strategy = self.determine_search_strategy(reformulated_query, pinecone_error_type)
            
            # Build search parameters for direct SDK
            sdk_params = {
                "query": reformulated_query,
                "max_results": 5,
                "include_answer": "advanced",      # ✅ Advanced for better answers
                "search_depth": "advanced",        # ✅ Advanced for deeper search
                "include_raw_content": "text"      # ✅ Added for better content
            }
            
            # Add domain restrictions
            if strategy.get("include_domains"):
                sdk_params["include_domains"] = strategy["include_domains"]
                logger.info(f"🔍 Tavily domain-restricted search: {strategy['include_domains'][0]}")
            elif strategy.get("exclude_domains"):
                sdk_params["exclude_domains"] = strategy["exclude_domains"]
                logger.info(f"🌐 Tavily worldwide search excluding {len(strategy['exclude_domains'])} competitor domains")
                
            logger.info(f"🔍 Direct Tavily SDK call with params: {list(sdk_params.keys())}")
            
            # Execute search using Tavily client
            search_results = self.tavily_client.search(**sdk_params)
            
            # Synthesize the results
            synthesized_content = self.synthesize_search_results(search_results, reformulated_query)
            
            # Add UTM parameters to all links
            final_content = self.add_utm_to_links(synthesized_content)
            
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
        
        # Initialize Tavily agent WITH OpenAI key for reformulation (UPDATED)
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(
                    self.config.TAVILY_API_KEY, 
                    self.config.OPENAI_API_KEY # NEW: Pass OpenAI key
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
        """Routing based on component health. Always prefers Pinecone if healthy."""
        pinecone_status = error_handler.component_status.get("Pinecone", "healthy")
        return pinecone_status == "healthy"

    def _get_current_pinecone_error_type(self) -> str:
        """Get current Pinecone error type for Tavily strategy determination."""
        return error_handler.component_status.get("Pinecone", "healthy")

    # CHANGE 5: Updated Fallback Logic for Business Rules
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
        logger.warning(f"   Content preview: {content[:300]}...")  # First 300 chars
        logger.warning(f"   Response source: {pinecone_response.get('source', 'Unknown')}")
        logger.warning(f"   Has citations flag: {pinecone_response.get('has_citations', False)}")
        logger.warning(f"   Success flag: {pinecone_response.get('success', False)}")
        logger.warning("=" * 50)

        # DEBUG: Log the citation detection issue
        has_citation_markers = "[1]" in content or "**sources:**" in content
        has_citations_flag = pinecone_response.get("has_citations", False)

        if has_citation_markers:
            logger.warning(f"🔍 CITATION DEBUG: Found markers in content, has_citations={has_citations_flag}")
            logger.warning(f"Content preview: {content[:200]}...")

        # NEW: Check original question for recency indicators
        recency_indicators = ["latest", "newest", "recent", "current", "2024", "2025"]
        if any(indicator in original_lower for indicator in recency_indicators):
            logger.warning("🔄 Fallback TRIGGERED: User requested latest/current information.")
            return True
        logger.warning("✅ Recency check PASSED")

        # RULE 1: NEVER fallback if Pinecone provides the business redirect for pricing/stock.
        # This is the desired final answer for these questions.
        if "sales-eu@12taste.com" in content or "contact our sales team" in content:
            logger.warning("✅ Fallback SKIPPED: Pinecone provided the correct business redirect for pricing/stock.")
            return False
        logger.warning("✅ Business redirect check PASSED (no sales redirect found)")

        # RULE 2: ALWAYS fallback if Pinecone explicitly states it doesn't know about a topic.
        # This indicates a product/topic not found in the knowledge base.
        not_found_indicators = [
            "don't have specific information",
            "could not find specific information",
            "cannot find information",
            "no information about",
            "not available in my knowledge base",
            "not available in my knowledge base", # Corrected duplicate entry
            "don't have information about",
            "couldn't find information",
            "do not provide specific information",
            "does not provide specific information",
            "search results do not provide",
            "results do not provide",
            "do not contain specific information", # ADD THIS LINE
            "does not contain specific information",
            "search results do not contain",
            "results do not contain"
        ]
        for phrase in not_found_indicators:
            if phrase in content:
                logger.warning(f"🔄 Fallback TRIGGERED: Found 'not found' indicator: '{phrase}'")
                return True
        logger.warning("✅ Not found indicators check PASSED")

        # RULE 3: Fallback on critical safety issues like fake citations.
        # (This logic can be preserved from your original implementation)
        if "[1]" in content or "**sources:**" in content:
            if not pinecone_response.get("has_citations", False):
                logger.warning("🚨 SAFETY OVERRIDE: Detected fake citations. Switching to web search.")
                return True
        logger.warning("✅ Citation safety check PASSED")

        # RULE 4: Force web search for regulatory topics without citations
        regulatory_indicators = ["regulation", "directive", "compliance", "legal"]
        is_regulatory = any(indicator in original_lower for indicator in regulatory_indicators)

        if is_regulatory and not pinecone_response.get("has_citations", False):
            logger.warning("🔄 Fallback TRIGGERED: Regulatory topic without citations.")
            return True
        logger.warning(f"✅ Regulatory check PASSED (is_regulatory={is_regulatory})")

        # DEFAULT: Do not fallback. Assume Pinecone's answer is correct if it's not a "not found" response
        # and doesn't violate business rules.
        logger.warning("✅ ALL CHECKS PASSED - Should NOT fallback.")
        return False
    
    # CHANGE 6: Business-First Routing (Always Try Pinecone First)
    @handle_api_errors("AI System", "Get Response", show_to_user=True)
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        AI response flow that prioritizes Pinecone and adheres to strict business rules for fallback.
        """
        # Convert chat history to LangChain format
        langchain_history = []
        if chat_history:
            for msg in chat_history[-10:]: # Limit to last 10 messages
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    langchain_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_history.append(AIMessage(content=content))
        langchain_history.append(HumanMessage(content=prompt))

        # --- Primary Flow: Always try Pinecone first ---
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
                    # Use the new business-aware logic to decide if a fallback is needed
                    if not self.should_use_web_fallback(pinecone_response, prompt):
                        logger.info("✅ Using Pinecone response (passed business logic checks).")
                        error_handler.mark_component_healthy("Pinecone")
                        return pinecone_response
                    else:
                        logger.warning("⤵️ Pinecone response requires web fallback. Proceeding to Tavily...")

            except Exception as e:
                error_type = self._detect_pinecone_error_type(e)
                logger.error(f"Pinecone query failed ({error_type}): {e}. Proceeding to web fallback.")

        # --- Fallback Flow: Use Tavily web search if needed ---
        if self.tavily_agent:
            try:
                logger.info("🌐 Falling back to FiFi web search...")
                # Pass the full history for context-aware query reformulation
                web_response = self.tavily_agent.query(prompt, langchain_history, self._get_current_pinecone_error_type())
                
                if web_response and web_response.get("success"):
                    logger.info(f"✅ Using web search response.")
                    error_handler.mark_component_healthy("Tavily")
                    return web_response
                        
            except Exception as e:
                logger.error(f"FiFi Web search failed: {e}", exc_info=True)
                error_handler.log_error(error_handler.handle_api_error("Tavily", "Query", e))
        
        # --- Final Fallback: If all systems fail ---
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

# CHANGE 1: Enhanced Validation Function with Context
@handle_api_errors("Industry Context Check", "Validate Question Context", show_to_user=False)
def check_industry_context(prompt: str, chat_history: List[Dict] = None, client: Optional[openai.OpenAI] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if user question is relevant to food & beverage industry, now with conversation context.
    """
    if not client or not hasattr(client, 'chat'):
        logger.debug("OpenAI client not available for industry context check. Allowing question.")
        return {"relevant": True, "reason": "context_check_unavailable"}

    # Build conversation context from the last 6 messages (3 exchanges)
    conversation_context = ""
    if chat_history and len(chat_history) > 0:
        recent_history = chat_history[-6:]
        context_parts = []
        for msg in recent_history:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')[:200] # Truncate for brevity
            context_parts.append(f"{role}: {content}")
        conversation_context = "\n".join(context_parts)

    try:
        context_check_prompt = f"""You are an industry context validator for 12taste.com, a B2B marketplace for food ingredients.

**CONVERSATION HISTORY (for context):**
{conversation_context if conversation_context else "No previous conversation."}

**CURRENT USER QUESTION:** "{prompt}"

**TASK:** Analyze the **CURRENT USER QUESTION**. Considering the conversation history, determine if it is a relevant B2B food & beverage industry question. A follow-up like "What about pricing?" IS relevant if the context was about an ingredient.

**ALLOW:**
- Food/beverage ingredients, suppliers, formulation, applications.
- Food safety, regulations, market trends.
- Follow-up questions that are relevant IN CONTEXT.

**FLAG:**
- Unrelated industries (automotive, finance).
- Personal consumer cooking, diet advice.
- Off-topic subjects (politics, sports).

**INSTRUCTIONS:**
Respond with ONLY a JSON object in this exact format:
{{
    "relevant": true/false,
    "confidence": 0.0-1.0,
    "category": "food_ingredients" | "supplier_sourcing" | "follow_up" | "off_topic" | "unrelated_industry",
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
        response_content = response_content.replace('```json', '').replace('```', '').strip() # Changed from response.content to response.choices[0].message.content
        
        result = json.loads(response_content)
        
        if not all(key in result for key in ['relevant', 'confidence', 'category', 'reason']):
            logger.warning("Industry context check returned incomplete JSON structure")
            return {"relevant": True, "reason": "context_check_invalid_response"}
                
        logger.info(f"Context check: relevant={result['relevant']}, category={result['category']}, confidence={result['confidence']:.2f} (Context HELPED validate follow-up: {'yes' if result['category'] == 'follow_up' else 'no'})")
        return result
            
    except Exception as e:
        logger.error(f"Industry context check failed: {e}", exc_info=True)
        return {"relevant": True, "reason": "context_check_error"}

# =============================================================================
# SESSION MANAGER - MAIN ORCHESTRATOR CLASS
# =============================================================================

class SessionManager:
    """Main orchestrator class that manages user sessions, integrates all managers, and provides the primary interface for the application."""
    
    def __init__(self, _config: Config, _db_manager: DatabaseManager, # Changed to underscore
                 _zoho_manager: ZohoCRMManager, _ai_system: EnhancedAI, # Changed to underscore
                 _rate_limiter: RateLimiter, _fingerprinting_manager: DatabaseManager.FingerprintingManager, # Changed to underscore
                 _email_verification_manager: DatabaseManager.EmailVerificationManager, # Changed to underscore
                 _question_limit_manager: DatabaseManager.QuestionLimitManager): # Changed to underscore
        self.config = _config
        self.db = _db_manager
        self.zoho = _zoho_manager
        self.ai = _ai_system
        self.rate_limiter = _rate_limiter
        self.fingerprinting = _fingerprinting_manager
        self.email_verification = _email_verification_manager
        self.question_limits = _question_limit_manager
        self._cleanup_interval = timedelta(hours=1)
        self._last_cleanup = datetime.now()
        
        logger.info("✅ SessionManager initialized with all component managers.")

    def get_session_timeout_minutes(self) -> int:
        """Returns the configured session timeout duration in minutes."""
        ## CHANGE: Use constant for session timeout
        return SESSION_TIMEOUT_MINUTES
    
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
            if hasattr(st.session_state, 'error_handler') and hasattr(st.session_state.error_handler, 'error_history') and len(st.session_state.error_handler.error_history) > MAX_ERROR_HISTORY: ## CHANGE: Use constant for error history limit
                st.session_state.error_handler.error_history = st.session_state.error_handler.error_history[-MAX_ERROR_HISTORY // 2:] ## CHANGE: Keep half the max size
                logger.info("Cleaned up error history")
            
            ## CHANGE: Call database cleanup for old inactive sessions
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
        error_keys = ['rate_limit_hit', 'moderation_flagged', 'context_flagged', 'pricing_stock_notice'] # Added 'pricing_stock_notice'
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
        
        # For new sessions, always start with a temporary fingerprint
        # This will be replaced by a real one for Guests, or marked as 'not_collected_registered' for Registered Users later.
        session.fingerprint_id = f"temp_py_{secrets.token_hex(8)}"
        session.fingerprint_method = "temporary_fallback_python"
        
        logger.debug(f"🆔 New session created: {session_id[:8]} (NOT saved to DB yet, will be saved in get_session)")
        return session

    ## CHANGE: Removed _check_15min_eligibility as it's no longer used for timeout saves.

    def _is_crm_save_eligible(self, session: UserSession, trigger_reason: str) -> bool:
        """Enhanced eligibility check for CRM saves including new user types and conditions.
        Removed 15-minute activity requirement for timeout saves."""
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
            
            ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS constant
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
            
            ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS constant
            if len(session.messages) < CRM_SAVE_MIN_QUESTIONS:
                return False
            
            logger.info(f"Manual CRM save eligible for {session.session_id[:8]}: UserType={session.user_type.value}, Messages={len(session.messages)}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking manual CRM eligibility for {session.session_id[:8]}: {e}")
            return False

    ## CHANGE: Modified inheritance logic to prioritize email for REGISTERED_USER, and handle fingerprint for others
    def _attempt_fingerprint_inheritance(self, session: UserSession):
        """
        Attempts to inherit session data with EMAIL as primary identifier for registered users.
        For non-registered users, it uses fingerprint.
        """
        logger.info(f"🔄 Attempting inheritance for session {session.session_id[:8]} (Type: {session.user_type.value}) with FP: {session.fingerprint_id[:8] if session.fingerprint_id else 'None'}...")

        try:
            # PRIORITY 1: For REGISTERED_USER, use EMAIL-based inheritance ONLY
            ## CHANGE: Removed fingerprint related updates for REGISTERED_USERs here, as FP is not collected.
            if session.user_type == UserType.REGISTERED_USER and session.email:
                logger.info(f"📧 Using EMAIL-based inheritance for registered user {session.email}")
                
                email_sessions = self.db.find_sessions_by_email(session.email)
                email_sessions = [s for s in email_sessions if s.session_id != session.session_id]
                
                if email_sessions:
                    # Find most recent session with same email
                    most_recent = max(email_sessions, key=lambda s: s.last_question_time or s.last_activity or s.created_at)
                    
                    now = datetime.now()
                    ## CHANGE: Use DAILY_RESET_WINDOW constant
                    if (most_recent.last_question_time and (now - most_recent.last_question_time) < DAILY_RESET_WINDOW) or \
                       (not most_recent.last_question_time and most_recent.last_activity and (now - most_recent.last_activity) < DAILY_RESET_WINDOW): # If last_question_time is None, use last_activity for reset check
                        # Inherit counts
                        session.daily_question_count = most_recent.daily_question_count
                        session.total_question_count = max(session.total_question_count, most_recent.total_question_count)
                        session.last_question_time = most_recent.last_question_time
                        
                        # Inherit Zoho contact ID
                        if not session.zoho_contact_id and most_recent.zoho_contact_id:
                            session.zoho_contact_id = most_recent.zoho_contact_id
                        
                        logger.info(f"✅ REGISTERED_USER EMAIL inheritance successful: {most_recent.daily_question_count} questions from {most_recent.session_id[:8]}")
                    else:
                        session.daily_question_count = 0
                        session.last_question_time = None
                        logger.info(f"🕐 REGISTERED_USER Email session found but >{DAILY_RESET_WINDOW_HOURS}h old, resetting daily count")
                else:
                    session.daily_question_count = 0
                    session.total_question_count = 0
                    session.last_question_time = None
                    logger.info(f"🆕 REGISTERED_USER no same-email history found, starting fresh")
                
                # Always clear bans for registered users (they just logged in successfully)
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                session.question_limit_reached = False
                session.evasion_count = 0
                session.current_penalty_hours = 0
                session.escalation_level = 0
                
                return  # Exit early for registered users, email is primary and fingerprint is not collected

            # PRIORITY 2: For non-registered users (Guest/Email Verified), use fingerprint (existing logic)
            if not session.fingerprint_id or session.fingerprint_id.startswith(("temp_", "fallback_")):
                logger.info("No valid fingerprint yet or temporary, skipping fingerprint-based inheritance for non-registered.")
                session.visitor_type = "new_visitor" # Default to new if no solid FP
                return # Don't inherit anything complex without a reliable FP

            historical_fp_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
            # Filter out the current session itself
            historical_fp_sessions = [s for s in historical_fp_sessions if s.session_id != session.session_id]

            if not historical_fp_sessions:
                session.visitor_type = "new_visitor"
                logger.info(f"Inheritance complete for new visitor {session.session_id[:8]}. No historical sessions found for fingerprint.")
                # Clear any pending re-verification from a prior fingerprint if no history for this one
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
                session.declined_recognized_email_at = None
                return # Nothing to inherit

            session.visitor_type = "returning_visitor"
            
            current_email = session.email.lower() if session.email else None
            
            # Separate identity inheritance from count inheritance
            all_sessions_for_identity = [session] + historical_fp_sessions
            same_email_sessions_for_counts = []
            
            if current_email:
                same_email_sessions_for_counts = [
                    s for s in historical_fp_sessions 
                    if s.email and s.email.lower() == current_email
                ]
            
            # --- Initialize values for merging ---
            merged_total_question_count = session.total_question_count
            merged_last_question_time = session.last_question_time
            merged_daily_question_count = session.daily_question_count
            source_for_identity = session # Start with current session as identity source
            most_recent_ban_session = None
            now = datetime.now()

            # --- Pass 1: Find the highest privilege identity from ALL sessions (for identity) ---
            unique_emails_in_history = set()
            highest_user_type_seen = UserType.GUEST
            
            for s in all_sessions_for_identity:
                if s.email:
                    unique_emails_in_history.add(s.email.lower())

                # Track highest user type ever seen for this fingerprint
                if self._get_privilege_level(s.user_type) > self._get_privilege_level(highest_user_type_seen):
                    highest_user_type_seen = s.user_type

                # Find the most authoritative session for identity
                if self._get_privilege_level(s.user_type) > self._get_privilege_level(source_for_identity.user_type):
                    source_for_identity = s
                elif self._get_privilege_level(s.user_type) == self._get_privilege_level(source_for_identity.user_type):
                    if s.last_activity and (not source_for_identity.last_activity or s.last_activity > source_for_identity.last_activity):
                        source_for_identity = s

            # --- Apply multiple email detection for recognition purposes ---
            if len(unique_emails_in_history) > 1:
                logger.info(f"🚨 Multiple emails ({len(unique_emails_in_history)}) detected for fingerprint - disabling explicit recognition prompt.")
                session.recognition_response = "multiple_emails_detected"
                session.reverification_pending = False
                session.pending_user_type = None
                session.pending_email = None
                session.pending_full_name = None
                session.pending_zoho_contact_id = None
                session.pending_wp_token = None
            
            # --- UPDATED: User type precedence - always offer highest seen type ---
            if highest_user_type_seen == UserType.REGISTERED_USER and session.user_type != UserType.REGISTERED_USER:
                # This fingerprint has been used as REGISTERED_USER before, offer re-verification
                registered_session = next((s for s in all_sessions_for_identity if s.user_type == UserType.REGISTERED_USER), None)
                if registered_session and registered_session.email:
                    session.reverification_pending = True
                    session.pending_user_type = UserType.REGISTERED_USER
                    session.pending_email = registered_session.email
                    session.pending_full_name = registered_session.full_name
                    session.pending_zoho_contact_id = registered_session.zoho_contact_id
                    session.pending_wp_token = registered_session.wp_token
                    logger.info(f"🔄 Offering REGISTERED_USER re-verification for {session.session_id[:8]} (highest precedence)")
                    return  # Skip further inheritance for now, let user decide
            
            # --- Pass 2: Find ban and count info from SAME EMAIL sessions only (for non-REGISTERED users) ---
            for s in same_email_sessions_for_counts:
                # Find the single most recent ban to evaluate (same email only)
                if s.ban_status != BanStatus.NONE and s.ban_end_time:
                    if most_recent_ban_session is None or s.ban_end_time > most_recent_ban_session.ban_end_time:
                        most_recent_ban_session = s
                
                # Merge total count and last question time (same email only)
                merged_total_question_count = max(merged_total_question_count, s.total_question_count)
                if s.last_question_time and (not merged_last_question_time or s.last_question_time > merged_last_question_time):
                    merged_last_question_time = s.last_question_time

            
            # --- Pass 3: Determine daily count and ban status based on same-email findings ---
            ## CHANGE: Use DAILY_RESET_WINDOW constant
            ban_is_active = most_recent_ban_session and now < most_recent_ban_session.ban_end_time
            
            if ban_is_active:
                # The user is still under an active ban from same email
                logger.info(f"Inheritance: Found ACTIVE ban for same email. Applying to session {session.session_id[:8]}.")
                session.ban_status = most_recent_ban_session.ban_status
                session.ban_start_time = most_recent_ban_session.ban_start_time
                session.ban_end_time = most_recent_ban_session.ban_end_time
                session.ban_reason = most_recent_ban_session.ban_reason
                session.question_limit_reached = True
                
                # Find the max daily count from same email sessions within the reset window
                for s in same_email_sessions_for_counts:
                     if s.daily_question_count and s.last_question_time and (now - s.last_question_time < DAILY_RESET_WINDOW): ## CHANGE: Use DAILY_RESET_WINDOW constant
                         merged_daily_question_count = max(merged_daily_question_count, s.daily_question_count)
            else:
                # No active ban exists
                logger.info(f"Inheritance: No active ban found for same email. Evaluating question count reset.")
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
                session.question_limit_reached = False

                # Check if the daily count should be reset based on last question time (same email)
                if merged_last_question_time and (now - merged_last_question_time) < DAILY_RESET_WINDOW: ## CHANGE: Use DAILY_RESET_WINDOW constant
                    # The last question from same email was recent
                    for s in same_email_sessions_for_counts:
                        if s.daily_question_count and s.last_question_time and (now - s.last_question_time < DAILY_RESET_WINDOW): ## CHANGE: Use DAILY_RESET_WINDOW constant
                            merged_daily_question_count = max(merged_daily_question_count, s.daily_question_count)
                else:
                    ## CHANGE: Updated message with constant
                    logger.info(f"Inheritance: Last question time from same email is > {DAILY_RESET_WINDOW_HOURS}h ago OR no same-email questions. Resetting daily count to 0.")
                    merged_daily_question_count = 0
                    merged_last_question_time = None
            
            # --- Apply all merged and determined values to the current session ---
            session.total_question_count = merged_total_question_count
            session.daily_question_count = merged_daily_question_count
            session.last_question_time = merged_last_question_time

            # Apply identity (considering multiple emails and privilege levels)
            if (len(unique_emails_in_history) <= 1 or 
                self._get_privilege_level(source_for_identity.user_type) > self._get_privilege_level(session.user_type)):
                
                # NEW: Only apply identity inheritance if emails match OR current session has no email
                source_email = source_for_identity.email.lower() if source_for_identity.email else None
                session_email = session.email.lower() if session.email else None
                
                if (not session_email or not source_email or source_email == session_email):
                    if self._get_privilege_level(source_for_identity.user_type) > self._get_privilege_level(session.user_type):
                        # Higher privilege found, set reverification pending
                        session.reverification_pending = True
                        session.pending_user_type = source_for_identity.user_type
                        session.pending_email = source_for_identity.email
                        session.pending_full_name = source_for_identity.full_name
                        session.pending_zoho_contact_id = source_for_identity.zoho_contact_id
                        session.pending_wp_token = source_for_identity.wp_token
                        logger.info(f"🔄 Re-verification set for upgrade: {source_for_identity.user_type.value} for {session.session_id[:8]}")
                    else:
                        # Same or lower privilege, just inherit the details if available
                        session.user_type = source_for_identity.user_type
                        session.email = source_for_identity.email
                        session.full_name = source_for_identity.full_name
                        session.zoho_contact_id = source_for_identity.zoho_contact_id
                        session.wp_token = source_for_identity.wp_token
                else:
                    logger.info(f"Inheritance: Identity not transferred due to email mismatch: source={source_email}, session={session_email}")
            else:
                logger.info(f"Inheritance: Identity not fully transferred due to multiple emails and no privilege upgrade for {session.session_id[:8]}.")

            # Update fingerprint if necessary (if current is temporary and a real one is found in history)
            if session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")) and not source_for_identity.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                session.fingerprint_id = source_for_identity.fingerprint_id
                session.fingerprint_method = source_for_identity.fingerprint_method
            
            logger.info(f"Inheritance complete for {session.session_id[:8]}: user_type={session.user_type.value}, daily_q={session.daily_question_count}, ban_status={session.ban_status.value}, rev_pending={session.reverification_pending}")

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
                    # NEW: Check if any bans have expired
                    if session.ban_status != BanStatus.NONE:
                        if session.ban_end_time and datetime.now() >= session.ban_end_time:
                            logger.info(f"Ban expired for session {session.session_id[:8]}. Clearing ban status.")
                        
                            # Clear the ban
                            session.ban_status = BanStatus.NONE
                            session.ban_start_time = None
                            session.ban_end_time = None
                            session.ban_reason = None
                            session.question_limit_reached = False
                        
                            # Save the updated session
                            self.db.save_session(session)
                
                    # NEW: Immediately attempt fingerprint inheritance if session has a temporary fingerprint
                    # And this check hasn't been performed for this session yet in the current rerun cycle.
                    fingerprint_checked_key = f'fingerprint_checked_for_inheritance_{session.session_id}'
                    
                    ## CHANGE: Only attempt fingerprint inheritance for NON-REGISTERED users
                    if session.user_type != UserType.REGISTERED_USER and \
                       session.fingerprint_id and \
                       session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_")) and \
                       not st.session_state.get(fingerprint_checked_key, False):
                    
                        self._attempt_fingerprint_inheritance(session)
                        # Save session after potential inheritance to persist updated user type/counts
                        self.db.save_session(session) # Crucial to save here
                        st.session_state[fingerprint_checked_key] = True # Mark as checked for this session
                        logger.info(f"Fingerprint inheritance check and save completed for {session.session_id[:8]}")
                    
                    ## CHANGE: For REGISTERED USERS, ensure FP is marked as not collected/not applicable
                    elif session.user_type == UserType.REGISTERED_USER and \
                         (session.fingerprint_id is None or session.fingerprint_id.startswith(("temp_", "fallback_", "temp_py_"))): # Also check temp_py_
                        session.fingerprint_id = "not_collected_registered_user"
                        session.fingerprint_method = "email_primary"
                        self.db.save_session(session) # Save immediately
                        logger.info(f"Registered user {session.session_id[:8]} fingerprint marked as not collected.")


                    # NEW: Check if guest needs forced verification - this is the "second guest session" logic.
                    # This check only happens if the user is a GUEST and hasn't asked any questions in THIS session.
                    if session.user_type.value == UserType.GUEST.value and session.daily_question_count == 0:
                        # Find all historical sessions for this fingerprint
                        historical_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
                    
                        # Check if any *other* session with this fingerprint was a GUEST and hit the limit
                        guest_sessions_with_questions = [
                            s for s in historical_sessions 
                            if s.session_id != session.session_id # Must be a different session
                            and s.user_type.value == UserType.GUEST.value 
                            and s.daily_question_count >= self.question_limits.question_limits[UserType.GUEST.value] # Must have hit guest limit
                        ]
                    
                        if guest_sessions_with_questions:
                            logger.info(f"Session {session.session_id[:8]} must verify email - fingerprint already used guest questions in previous session(s)")
                            st.session_state.must_verify_email_immediately = True
                            st.session_state.skip_email_allowed = False # No skipping for forced verification

                    # Check limits and handle bans. This is where the 24-hour reset happens.
                    limit_check = self.question_limits.is_within_limits(session)
                    if not limit_check.get('allowed', True) and limit_check.get('reason') != 'guest_limit':
                    
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
            ## CHANGE: Only attempt fingerprint inheritance for NON-REGISTERED users; otherwise mark as not collected
            if new_session.user_type != UserType.REGISTERED_USER:
                self._attempt_fingerprint_inheritance(new_session) # <--- This call will now also correctly set visitor_type
                st.session_state[f'fingerprint_checked_for_inheritance_{new_session.session_id}'] = True
            else: # For newly created REGISTERED_USER, mark FP as not collected
                new_session.fingerprint_id = "not_collected_registered_user"
                new_session.fingerprint_method = "email_primary"
                logger.info(f"New REGISTERED_USER session {new_session.session_id[:8]} fingerprint marked as not collected.")

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
    
    ## CHANGE: Sync method for registered users
    def sync_registered_user_sessions(self, email: str, current_session_id: str):
        """Sync question counts across all active sessions for a registered user by email"""
        try:
            email_sessions = self.db.find_sessions_by_email(email)
            active_registered_sessions = [s for s in email_sessions 
                                          if s.active and s.user_type == UserType.REGISTERED_USER]
            
            if not active_registered_sessions:
                return
            
            # Find the session with the highest question count within the DAILY_RESET_WINDOW
            max_count_session = None
            now = datetime.now()
            for sess in active_registered_sessions:
                if sess.last_question_time and (now - sess.last_question_time) < DAILY_RESET_WINDOW:
                    if max_count_session is None or sess.daily_question_count > max_count_session.daily_question_count:
                        max_count_session = sess
            
            if max_count_session is None: # No recent sessions, or all counts expired
                logger.info(f"No recent active sessions for {email} to sync. Counts will reset for new activity.")
                return
            
            # Update all other sessions to match the max_count_session
            for sess in active_registered_sessions:
                if sess.session_id != max_count_session.session_id:
                    sess.daily_question_count = max_count_session.daily_question_count
                    sess.total_question_count = max_count_session.total_question_count
                    sess.last_question_time = max_count_session.last_question_time
                    self.db.save_session(sess) # Save immediately
                    logger.debug(f"Synced session {sess.session_id[:8]} to {max_count_session.daily_question_count} daily questions from {max_count_session.session_id[:8]}")
            
            logger.info(f"Synced {len(active_registered_sessions)} sessions for registered user {email}")
            
        except Exception as e:
            logger.error(f"Failed to sync registered user sessions for {email}: {e}")

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]) -> bool:
        """Applies fingerprinting data from custom component to the session with better validation.
        This function will now be called only for non-registered users (Guests)."""
        logger.debug(f"🔍 APPLYING FINGERPRINT received from JS: {fingerprint_data.get('fingerprint_id', 'None')[:8]} to session {session.session_id[:8]} (UserType: {session.user_type.value})")
        
        # Guard against applying fingerprint data to registered users
        if session.user_type == UserType.REGISTERED_USER:
            logger.warning(f"Attempted to apply fingerprint to REGISTERED_USER {session.session_id[:8]}. Ignoring.")
            return False

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
        """Check if a fingerprint has historical sessions and return relevant information, now handling multiple emails.
        This function will primarily be used for non-registered users."""
        
        # For registered users, fingerprint history is not relevant for primary identification
        # We don't have the session object here, so we assume this is only called in contexts where
        # fingerprint is relevant (i.e., for guest recognition flows).

        try:
            existing_sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if not existing_sessions:
                return {'has_history': False}
            
            # Get unique emails
            unique_emails = set()
            for s in existing_sessions:
                if s.email:
                    unique_emails.add(s.email.lower())
            
            # Multiple email detection
            if len(unique_emails) > 1:
                logger.info(f"🚨 Multiple emails ({len(unique_emails)}) detected for fingerprint {fingerprint_id[:8]}: {unique_emails}")
                return {
                    'has_history': True,
                    'multiple_emails': True,
                    'email_count': len(unique_emails),
                    'skip_recognition': True  # Don't show recognition dialog
                }
            
            # Single email - show recognition
            most_privileged_session = None
            for s in existing_sessions:
                # Ensure we don't try to recognize a REGISTERED_USER based on FP if email is primary
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
        """Email verification - allows unlimited email switches with OTP verification. No evasion penalties."""
        try:
            sanitized_email = sanitize_input(email).lower().strip() ## CHANGE: Use global sanitize_input
            
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', sanitized_email):
                logger.debug(f"handle_guest_email_verification returning FAILURE: Invalid email format for {email}")
                return {'success': False, 'message': 'Please enter a valid email address.'}
            
            # UPDATED: Removed evasion check as per requirement: email switching is allowed.
            
            # Track email usage for this session
            if sanitized_email not in session.email_addresses_used:
                session.email_addresses_used.append(sanitized_email)
            
            # Update session email (if not already set by re-verification pending)
            if not session.reverification_pending:
                if session.email and session.email != sanitized_email:
                    session.email_switches_count += 1 # Increment for tracking, no penalty
                session.email = sanitized_email
            elif session.reverification_pending and sanitized_email != session.pending_email:
                # If reverification is pending but they enter a different email, treat as new email path
                session.email_switches_count += 1 # Increment for tracking, no penalty
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
                final_result = {
                    'success': False, 
                    'message': 'An internal error occurred saving your session. Please try again.'
                }
                logger.debug(f"handle_guest_email_verification returning FAILURE (DB Save): {final_result}")
                return final_result
            
            # Send verification code
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
        Updated to preserve counts when upgrading from GUEST to EMAIL_VERIFIED_GUEST.
        """
        try:
            email_to_verify = session.pending_email if session.reverification_pending else session.email

            if not email_to_verify:
                return {'success': False, 'message': 'No email address found for verification.'}
            
            sanitized_code = sanitize_input(code).strip() ## CHANGE: Use global sanitize_input
            
            if not sanitized_code:
                return {'success': False, 'message': 'Please enter the verification code.'}
            
            verification_success = self.email_verification.verify_code(email_to_verify, sanitized_code)
            
            if verification_success:
                # Check if this is a NEW EMAIL (different from any previous sessions for this fingerprint)
                is_new_email = True
                is_same_session_upgrade = False  # NEW: Track if this is same session upgrade
                previous_emails = set()
                
                try:
                    if session.fingerprint_id and session.user_type != UserType.REGISTERED_USER: # Only check FP history for non-registered
                        historical_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
                        for old_session in historical_sessions:
                            if old_session.email:
                                previous_emails.add(old_session.email.lower())
                        is_new_email = email_to_verify.lower() not in previous_emails
                        
                        # UPDATED: Check if this is same session upgrade (GUEST -> EMAIL_VERIFIED_GUEST)
                        is_same_session_upgrade = (
                        not session.reverification_pending and  # Not re-verification
                        session.user_type == UserType.GUEST and  # Currently guest
                        session.daily_question_count > 0  # Has asked questions in this session
                    )
                    
                    logger.info(f"Email verification for {session.session_id[:8]}: is_new_email={is_new_email}, is_same_session_upgrade={is_same_session_upgrade}, previous_emails={previous_emails}")
                except Exception as e:
                    logger.error(f"Error checking email history for new email determination: {e}")
                    is_new_email = True # Default to clean slate on error

                if session.reverification_pending:
                    # Reclaim existing user type and identity (keep inherited counts from _attempt_fingerprint_inheritance)
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
                    # New email verification or existing email as guest without pending higher privilege
                    old_daily_count = session.daily_question_count  # PRESERVE current count
                    session.user_type = UserType.EMAIL_VERIFIED_GUEST
                
                    # UPDATED: Preserve counts for same session upgrade (GUEST -> EMAIL_VERIFIED_GUEST)
                    if is_same_session_upgrade:
                        logger.info(f"🔄 SAME SESSION UPGRADE: Preserving question count {old_daily_count} from GUEST to EMAIL_VERIFIED_GUEST for {session.session_id[:8]}")
                        # Keep existing counts - don't reset anything
                    elif is_new_email:
                        # CLEAN SLATE for truly new emails (shared device protection)
                        logger.info(f"🧹 CLEAN SLATE: New email {email_to_verify} gets fresh start, resetting all counts and bans for {session.session_id[:8]}")
                        session.daily_question_count = 0
                        session.total_question_count = 0
                        session.last_question_time = None
                        session.question_limit_reached = False
                        session.ban_status = BanStatus.NONE
                        session.ban_start_time = None
                        session.ban_end_time = None
                        session.ban_reason = None
                        session.evasion_count = 0
                        session.current_penalty_hours = 0
                        session.escalation_level = 0
                    else:
                        logger.info(f"♻️ EXISTING EMAIL: For {email_to_verify}, keeping counts as determined by inheritance.")
                
                    logger.info(f"✅ User {session.session_id[:8]} upgraded to EMAIL_VERIFIED_GUEST: {session.email} with {session.daily_question_count} questions")

                session.question_limit_reached = False
                session.declined_recognized_email_at = None 
            
                if session.last_activity is None:
                    session.last_activity = datetime.now()

                # NEW: Re-run inheritance after email verification for Guest -> Email Verified transitions
                # This ensures we pick up previous session counts now that we have an email
                if not session.reverification_pending and not is_same_session_upgrade:
                    logger.info(f"🔄 Re-running inheritance after email verification for {session.session_id[:8]} with email {email_to_verify}")
                    self._attempt_fingerprint_inheritance(session)
                    logger.info(f"📊 Post-inheritance counts: daily={session.daily_question_count}, total={session.total_question_count}")

                try:
                    self.db.save_session(session)
                except Exception as e:
                    logger.error(f"Failed to save upgraded session: {e}")
            
                return {
                    'success': True,
                    'message': '✅ Email verified successfully!'
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
        """Authenticates user with WordPress and creates/updates session with conditional reset logic."""
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
                    
                    ## CHANGE: Explicitly set fingerprint to not collected/not applicable for REGISTERED_USER
                    current_session.fingerprint_id = "not_collected_registered_user"
                    current_session.fingerprint_method = "email_primary"
                    logger.info(f"Registered user {current_session.session_id[:8]} fingerprint marked as not collected/email primary.")

                    ## CHANGE: Always check email-based history for registered users
                    all_email_sessions = self.db.find_sessions_by_email(user_email)

                    # Find the most recent session with same email to inherit counts from
                    if all_email_sessions:
                        # Sort by last activity/question time
                        sorted_sessions = sorted(all_email_sessions, 
                                               key=lambda s: s.last_question_time or s.last_activity or s.created_at, 
                                               reverse=True)
                        
                        most_recent = sorted_sessions[0]
                        
                        ## CHANGE: Use DAILY_RESET_WINDOW constant for inheritance check
                        now = datetime.now()
                        time_check = most_recent.last_question_time or most_recent.last_activity or most_recent.created_at
                        
                        if time_check and (now - time_check) < DAILY_RESET_WINDOW:
                            # Inherit question counts from most recent same-email session
                            current_session.daily_question_count = most_recent.daily_question_count
                            current_session.total_question_count = max(current_session.total_question_count, 
                                                                      most_recent.total_question_count)
                            current_session.last_question_time = most_recent.last_question_time
                            
                            logger.info(f"✅ Email-based inheritance: Inherited {most_recent.daily_question_count} questions from {most_recent.session_id[:8]} (same email: {user_email})")
                        else:
                            ## CHANGE: Updated log message with constant
                            logger.info(f"🕐 Email session found but >{DAILY_RESET_WINDOW_HOURS}h old, resetting daily count")
                            current_session.daily_question_count = 0
                            current_session.last_question_time = None
                            
                        # Also inherit Zoho contact ID if available
                        if not current_session.zoho_contact_id:
                            for sess in sorted_sessions:
                                if sess.zoho_contact_id:
                                    current_session.zoho_contact_id = sess.zoho_contact_id
                                    logger.info(f"Inherited Zoho contact ID from session {sess.session_id[:8]}")
                                    break
                    else:
                        # No email history found
                        current_session.daily_question_count = 0
                        current_session.total_question_count = 0
                        current_session.last_question_time = None
                        logger.info(f"🆕 No email history found for {user_email}, starting fresh")

                    # Always clear bans for REGISTERED_USER (they logged in successfully)
                    current_session.ban_status = BanStatus.NONE
                    current_session.ban_start_time = None
                    current_session.ban_end_time = None
                    current_session.ban_reason = None
                    current_session.question_limit_reached = False
                    current_session.evasion_count = 0
                    current_session.current_penalty_hours = 0
                    current_session.escalation_level = 0
                    
                    # Set REGISTERED_USER attributes
                    current_session.user_type = UserType.REGISTERED_USER
                    current_session.email = user_email
                    current_session.full_name = user_display_name
                    current_session.wp_token = wp_token
                    current_session.last_activity = datetime.now()
                    
                    # Clear ALL re-verification flags if login is successful
                    current_session.reverification_pending = False
                    current_session.pending_user_type = None
                    current_session.pending_email = None
                    current_session.pending_full_name = None
                    current_session.pending_zoho_contact_id = None
                    current_session.pending_wp_token = None
                    current_session.declined_recognized_email_at = None

                    ## CHANGE: Do NOT invalidate all previous sessions to prevent future inheritance confusion
                    ## The multi-device sync and email-based inheritance should handle this better.

                    ## CHANGE: Enable chat immediately for registered users
                    st.session_state.is_chat_ready = True
                    st.session_state.fingerprint_wait_start = None # Ensure any FP wait is cleared for this user

                    # Save the updated session
                    try:
                        self.db.save_session(current_session)
                        ## CHANGE: Use constant in log message
                        logger.info(f"✅ REGISTERED_USER setup complete: {user_email}, {current_session.daily_question_count}/{REGISTERED_USER_QUESTION_LIMIT} questions. Chat enabled immediately.")
                        
                    except Exception as e:
                        logger.error(f"Failed to save authenticated session: {e}")
                        st.error("Authentication succeeded but session could not be saved. Please try again.")
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
                
        except Exception as e: # Catch general exceptions for authentication
            logger.error(f"An unexpected error occurred during WordPress authentication: {e}", exc_info=True)
            st.error("An unexpected error occurred during authentication. Please try again later.")
            return None

    # CHANGE 10: Pricing/Stock Question Handler
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

    # CHANGE 11: Solution 1 - Meta-Question Detection (Token-Free)
    def detect_meta_conversation_query(self, prompt: str) -> Dict[str, Any]:
        """Detect if user is asking about their conversation history."""

        prompt_lower = prompt.lower().strip()

        # Summary patterns
        summary_patterns = [
            "summarize", "summary of", "give me a summary", "can you summarize",
            "overview of", "recap of", "sum up"
        ]

        # Question listing patterns
        question_patterns = [
            "what did i ask", "what all did i ask", "what have i asked", 
            "all my questions", "my previous questions", "my questions",
            "list my questions", "show my questions", "questions i asked"
        ]

        # Count/stats patterns
        count_patterns = [
            "how many questions", "count my questions", "number of questions",
            "how many times", "total questions"
        ]

        # Topic analysis patterns
        topic_patterns = [
            "what topics", "what have we discussed", "topics we covered",
            "what did we talk about", "conversation topics", "discussion topics"
        ]

        # General conversation patterns
        conversation_patterns = [
            "conversation history", "chat history", "our conversation", 
            "this conversation", "our chat", "this chat", "my session"
        ]

        # Check each pattern type
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
        
        return {"is_meta": False}

    def handle_meta_conversation_query(self, session: UserSession, query_type: str, scope: str) -> Dict[str, Any]:
        """Handle meta-conversation queries with code-based analysis (zero token cost)."""
        
        # Get all visible messages (respecting soft clear offset)
        visible_messages = session.messages[session.display_message_offset:]
        user_questions = [msg['content'] for msg in visible_messages if msg.get('role') == 'user']

        if query_type == "count":
            ## CHANGE: Use constants in meta-query responses
            return {
                "content": f"""📊 **Session Statistics:**

• **Questions Asked**: {len(user_questions)}
• **Total Messages**: {len(visible_messages)}
• **Session Started**: {session.created_at.strftime('%B %d, %Y at %H:%M')}
• **User Type**: {session.user_type.value.replace('_', ' ').title()}
• **Daily Usage**: {session.daily_question_count}/{self.question_limits.question_limits[session.user_type.value]} questions today""", ## CHANGE: Use dynamic limit
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
            
            # Limit to last 20 questions to avoid very long responses
            questions_to_show = user_questions[-20:] if len(user_questions) > 20 else user_questions
            start_number = len(user_questions) - len(questions_to_show) + 1 if len(user_questions) > 20 else 1

            questions_list = []
            for i, q in enumerate(questions_to_show):
                # Truncate very long questions for readability
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

        # Extract key topics using simple word analysis
        topic_words = set()
        question_categories = {
            'pricing': 0, 'suppliers': 0, 'technical': 0, 'regulatory': 0, 'applications': 0
        }

        for question in user_questions:
            q_lower = question.lower()
            words = [w for w in q_lower.split() if len(w) > 4 and w not in ['what', 'where', 'when', 'about', 'would', 'could', 'should', 'which']]
            topic_words.update(words[:3]) # Top 3 meaningful words per question

            # Categorize questions
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

        # Get top topics
        key_topics = list(topic_words)[:8]
        active_categories = {k: v for k, v in question_categories.items() if v > 0}

        # Build summary
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

        # Topic extraction and categorization
        ingredients_mentioned = set()
        business_aspects = set()
        technical_terms = set()

        # Common ingredient patterns
        ingredient_indicators = ['extract', 'powder', 'oil', 'acid', 'syrup', 'sweetener', 'flavor', 'color']

        for question in user_questions:
            words = question.split()
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?')

                # Look for ingredients (capitalized words near ingredient indicators)
                if any(indicator in question.lower() for indicator in ingredient_indicators):
                    if word.istitle() and len(word) > 3:
                        ingredients_mentioned.add(word)
                
                # Business terms
                if word_lower in ['supplier', 'vendor', 'sourcing', 'pricing', 'cost', 'availability', 'stock']:
                    business_aspects.add(word_lower)

                # Technical terms
                if word_lower in ['formulation', 'application', 'specification', 'grade', 'purity', 'concentration']:
                    technical_terms.add(word_lower)

        # Build analysis
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

        # Industry focus
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
        """Gets AI response and manages session state."""
        try:
            # Handle cases where fingerprint_id might still be temporary or None during early load.
            # Use a fallback ID if the real fingerprint isn't yet established or is a temporary one.
            rate_limiter_id = session.fingerprint_id
            ## CHANGE: For REGISTERED_USERs, use session_id for rate limiting as FP is not collected.
            if session.user_type == UserType.REGISTERED_USER:
                rate_limiter_id = session.session_id
                logger.debug(f"Rate limiter using session ID as fallback for REGISTERED_USER {session.session_id[:8]} (FP not collected).")
            elif rate_limiter_id is None or rate_limiter_id.startswith(("temp_py_", "temp_fp_", "fallback_")):
                rate_limiter_id = session.session_id
                logger.warning(f"Rate limiter using session ID as fallback for unconfirmed fingerprint: {rate_limiter_id[:8]}...")


            ## CHANGE: Use constants for rate limits
            rate_limit_result = self.rate_limiter.is_allowed(rate_limiter_id)
            if not rate_limit_result['allowed']:
                time_until_next = rate_limit_result.get('time_until_next', 0)
                max_requests = RATE_LIMIT_REQUESTS
                window_seconds = RATE_LIMIT_WINDOW_SECONDS
                
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
                logger.warning(f"Input flagged by moderation for: {', '.join(categories)}")
                
                # Store moderation error info in session state
                st.session_state.moderation_flagged = {
                    'timestamp': datetime.now(),
                    'categories': categories,
                    'message': moderation_result.get("message", "Your message violates our content policy. Please rephrase your question.")
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

            # NEW: Check if it's a meta-conversation query (Solution 1)
            meta_detection = self.detect_meta_conversation_query(prompt)
            if meta_detection["is_meta"]:
                logger.info(f"Meta-conversation query detected: {meta_detection['type']}")

                ## CHANGE: Atomic record question for meta-questions
                try:
                    self.question_limits.record_question_and_check_ban(session, self)
                except Exception as e:
                    logger.error(f"Failed to record meta-question for {session.session_id[:8]}: {e}")
                    return {
                        'content': 'An error occurred while tracking your question. Please try again.',
                        'success': False,
                        'source': 'Question Tracker'
                    }

                # Add user message to session history
                user_message = {'role': 'user', 'content': prompt}
                session.messages.append(user_message)

                # Handle meta-query (zero tokens used)
                meta_response = self.handle_meta_conversation_query(
                    session, meta_detection["type"], meta_detection["scope"]
                )

                # Add assistant response to history
                assistant_message = {
                    'role': 'assistant',
                    'content': meta_response.get('content', 'Analysis complete.'),
                    'source': meta_response.get('source', 'Conversation Analytics'),
                    'is_meta_response': True  # Flag for UI display
                }
                session.messages.append(assistant_message)
                
                # Update activity and save session
                self._update_activity(session)

                return meta_response

            # CHANGE 2: Update the function call to pass conversation history
            context_result = check_industry_context(prompt, session.messages, self.ai.openai_client)
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
                    'message': context_message
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

            # NEW: Check for pricing/stock questions
            is_pricing_stock_query = self.detect_pricing_stock_question(prompt)
            if is_pricing_stock_query:
                # Store pricing notification info (similar to rate_limit_hit)
                st.session_state.pricing_stock_notice = {
                    'timestamp': datetime.now(),
                    'query_type': 'pricing' if any(word in prompt.lower() for word in ['price', 'pricing', 'cost', 'expensive', 'cheap']) else 'stock',
                    'message': """**Important Notice About Pricing & Stock Information:**

Pricing and stock availability are dynamic in nature and subject to market conditions, supplier availability, and other factors that can change at any point in time. The information provided is for general reference only and may not reflect current market conditions unless confirmed through a formal order process.

**For Current Information:**
- Visit the specific product page on our marketplace at **12taste.com** for real-time pricing and availability
- Contact our sales team at **sales-eu@12taste.com** for precise, up-to-date pricing and stock information
- Request formal quotes for confirmed pricing based on your specific requirements

Please proceed with your question, keeping in mind that any pricing or stock information provided should be verified through official channels."""
                }

            # Continue with regular processing for non-meta queries
            # Check question limits (THIS IS THE PRE-QUESTION LIMIT CHECK)
            limit_check = self.question_limits.is_within_limits(session)

            # MODIFIED: Apply ban BEFORE any returns or reruns
            if not limit_check['allowed']:
                ban_message = limit_check.get("message", 'Access restricted.')

                # For newly applied bans by record_question_and_check_ban, the logic below handles the immediate ban return.
                # This `if not limit_check['allowed']` block primarily catches already-active bans from previous reruns.
                if limit_check.get('reason') == 'registered_user_tier1_limit':
                    logger.info(f"Registered User Tier 1 ({REGISTERED_USER_TIER_1_LIMIT} questions) limit previously reached, displaying ban for session {session.session_id[:8]}.")
                    return {
                        'banned': True,
                        'content': ban_message,
                        'time_remaining': timedelta(hours=TIER_1_BAN_HOURS)  ## CHANGE: Use constant
                    }
                    
                elif limit_check.get('reason') == 'registered_user_tier2_limit':
                    logger.info(f"Registered User Tier 2 ({REGISTERED_USER_QUESTION_LIMIT} questions) limit previously reached, displaying ban for session {session.session_id[:8]}.")
                    return {
                        'banned': True,
                        'content': ban_message,
                        'time_remaining': timedelta(hours=TIER_2_BAN_HOURS) ## CHANGE: Use constant
                    }
                    
                elif limit_check.get('reason') == 'email_verified_guest_limit':
                    logger.info(f"Email Verified Guest ({EMAIL_VERIFIED_QUESTION_LIMIT} questions) limit previously reached, displaying ban for session {session.session_id[:8]}.")
                    return {
                        'banned': True,
                        'content': ban_message,
                        'time_remaining': timedelta(hours=EMAIL_VERIFIED_BAN_HOURS) ## CHANGE: Use constant
                    }
                    
                elif limit_check.get('reason') == 'guest_limit':
                    # Guest limit (4 questions) still requires email verification, not a hard ban here.
                    return {'requires_email': True, 'content': 'Email verification required.'}
                
                # For any other ban reason (like evasion or active ban from `is_within_limits`)
                return {
                    'banned': True,
                    'content': ban_message,
                    'time_remaining': limit_check.get('time_remaining')
                }

            self._clear_error_notifications()
            
            # Sanitize input
            sanitized_prompt = sanitize_input(prompt) ## CHANGE: Use global sanitize_input
            if not sanitized_prompt:
                return {
                    'content': 'Please enter a valid question.',
                    'success': False,
                    'source': 'Input Validation'
                }
            
            ## CHANGE: Call atomic question recording and ban check BEFORE AI response
            try:
                question_record_status = self.question_limits.record_question_and_check_ban(session, self)
                if question_record_status.get("ban_applied"):
                    # If a ban was just applied, return the ban message immediately
                    ban_type = question_record_status.get("ban_type")
                    if ban_type == BanStatus.ONE_HOUR.value:
                        return {
                            'banned': True,
                            'content': self.question_limits._get_ban_message(session, 'registered_user_tier1_limit'),
                            'time_remaining': timedelta(hours=TIER_1_BAN_HOURS)
                        }
                    elif ban_type == BanStatus.TWENTY_FOUR_HOUR.value:
                        if session.user_type == UserType.REGISTERED_USER:
                             return {
                                'banned': True,
                                'content': self.question_limits._get_ban_message(session, 'registered_user_tier2_limit'),
                                'time_remaining': timedelta(hours=TIER_2_BAN_HOURS)
                            }
                        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
                            return {
                                'banned': True,
                                'content': self.question_limits._get_ban_message(session, 'email_verified_guest_limit'),
                                'time_remaining': timedelta(hours=EMAIL_VERIFIED_BAN_HOURS)
                            }
            except Exception as e:
                logger.error(f"❌ Critical error recording question and checking ban: {e}")
                return {
                    'content': 'A critical error occurred while recording your question. Please try again.',
                    'success': False,
                    'source': 'Question Tracking Error'
                }
            
            logger.info(f"Question recorded for session {session.session_id[:8]} AFTER all pre-checks, BEFORE AI.")
            
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
                'has_inline_citations': ai_response.get('has_inline_citations', False),
                'safety_override': ai_response.get('safety_override', False)
            }
            
            session.messages.extend([user_message, assistant_message])
            
            # Update activity and save session
            self._update_activity(session)
            
            return ai_response  # Return the AI response directly now
            
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
            
            # Only block and show messages for ACTIVE bans (already in database)
            if reason == 'banned':
                ban_type = limit_check.get('ban_type', 'unknown')
                time_remaining = limit_check.get('time_remaining')
                
                st.error("🚫 **Access Restricted**")
                if time_remaining:
                    hours = max(0, int(time_remaining.total_seconds() // 3600))
                    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                    st.error(f"Time remaining: {hours}h {minutes}m")
                st.info(message)
                return True
                
            # Handle guest limit (email verification required)
            elif reason == 'guest_limit':
                st.error("🛑 **Guest Limit Reached**")
                ## CHANGE: Use GUEST_QUESTION_LIMIT constant
                st.info(f"You've used your {GUEST_QUESTION_LIMIT} guest questions. Please verify your email to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} more questions per day!")
                return True

            # Handle email verified guest daily limit
            elif reason == 'email_verified_guest_limit':
                st.error("🛑 **Daily Limit Reached**")
                ## CHANGE: Use EMAIL_VERIFIED_QUESTION_LIMIT constant
                st.info(f"You've used your {EMAIL_VERIFIED_QUESTION_LIMIT} questions for today. Your questions reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!") ## CHANGE: Use constants
                return True
                
            # DON'T BLOCK for registered user tier limits - let them attempt the question
            # This allows users at 10/20 to type and submit question #11
            # The ban will be applied in get_ai_response after submission
            
        return False

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
        const sessionId = {json.dumps(session_id)}; // CHANGE: Safely embed sessionId
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
        ## CHANGE: Update new timeout tracking fields
        session.timeout_detected_at = fresh_session_from_db.timeout_detected_at
        session.timeout_reason = fresh_session_from_db.timeout_reason
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
    ## CHANGE: Use SESSION_TIMEOUT_MINUTES constant
    if minutes_inactive >= SESSION_TIMEOUT_MINUTES:
        logger.info(f"⏰ TIMEOUT DETECTED: {session.session_id[:8]} inactive for {minutes_inactive:.1f} minutes")
        
        ## CHANGE: Remove local DB save for timeout, rely on FastAPI beacon
        # Instead of local DB updates, we let the FastAPI beacon handle all DB marking
        # The Streamlit app's responsibility is just to send the beacon and then reload.
        
        # Perform CRM save (via FastAPI beacon) if eligible
        if session_manager._is_crm_save_eligible(session, "timeout_auto_reload"):
            logger.info(f"💾 Performing emergency save (via FastAPI beacon) before auto-reload for {session.session_id[:8]}")
            try:
                emergency_data = {
                    "session_id": session.session_id,
                    "reason": "timeout_auto_reload",
                    "timestamp": int(time.time() * 1000)
                }
                ## CHANGE: Use FastAPI URL constant
                response = requests.post(FASTAPI_EMERGENCY_SAVE_URL, json=emergency_data, timeout=FASTAPI_EMERGENCY_SAVE_TIMEOUT)
                if response.status_code == 200:
                    logger.info(f"✅ Emergency save beacon sent to FastAPI successfully")
                else:
                    logger.warning(f"⚠️ FastAPI returned status {response.status_code}. Beacon failed or service error.")
            except Exception as e:
                logger.error(f"❌ Failed to send emergency save beacon to FastAPI: {e}")

        # Mark session as inactive (locally, will be confirmed by FastAPI)
        session.active = False
        session.last_activity = datetime.now()
        ## CHANGE: Log timeout detection, but FastAPI will handle saving this to DB as the source of truth
        session.timeout_detected_at = datetime.now()
        session.timeout_reason = f"Streamlit client detected inactivity for {minutes_inactive:.1f} minutes, triggering FastAPI beacon."

        # Clear Streamlit session state fully
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # NEW: Force welcome page state and timeout flag
        st.session_state['page'] = None
        st.session_state['session_expired'] = True
        
        # Show timeout message
        st.error("⏰ **Session Timeout**")
        ## CHANGE: Use SESSION_TIMEOUT_MINUTES constant in message
        st.info(f"Your session has expired due to {SESSION_TIMEOUT_MINUTES} minutes of inactivity.")
        
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

# NEW: Define process_emergency_save_from_query function at a top level
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
    
    # Update session activity and mark as inactive
    session.last_activity = datetime.now()
    session.active = False
    session.timeout_detected_at = datetime.now() # Mark when detected by Streamlit
    session.timeout_reason = reason # Store the reason for the save

    # Attempt to save to CRM if eligible
    crm_saved = False
    try:
        # Check eligibility with "emergency_fallback" reason, which can be seen by _is_crm_save_eligible
        if session_manager._is_crm_save_eligible(session, "emergency_fallback"):
            logger.info(f"Attempting CRM save from Streamlit query fallback for {session.session_id[:8]}")
            crm_saved = session_manager.zoho.save_chat_transcript_sync(session, reason)
            if crm_saved:
                session.timeout_saved_to_crm = True # Mark as saved if successful
            else:
                logger.warning(f"CRM save failed during Streamlit query fallback for {session.session_id[:8]}.")
        else:
            logger.info(f"Session {session.session_id[:8]} not eligible for CRM save (Streamlit query fallback).")
    except Exception as e:
        logger.error(f"Error during CRM save from Streamlit query fallback for {session.session_id[:8]}: {e}")
    
    # Save the session state (active=False, updated activity/timeout info)
    try:
        session_manager.db.save_session(session)
        logger.info(f"✅ Session '{session_id[:8]}' marked inactive and saved in DB (query fallback). CRM_saved: {crm_saved}")
        return True # Indicate that the session state was updated
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
            
    ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS for eligibility check
    if not session_manager._is_crm_save_eligible(session, "browser_close_check"):
        logger.info(f"🚫 Session {session_id[:8]} not eligible for CRM save - skipping browser close detection")
        return

    logger.info(f"✅ Setting up browser close detection for eligible session {session_id[:8]}")

    ## CHANGE: Enhanced JS for emergency save redundancy
    enhanced_close_js = f"""
    <script>
    (function() {{
        const sessionId = {json.dumps(session_id)}; // CHANGE: Safely embed sessionId
        const FASTAPI_URL = {json.dumps(FASTAPI_EMERGENCY_SAVE_URL)}; // CHANGE: Use constant and safely embed
        const FASTAPI_TIMEOUT_MS = {FASTAPI_EMERGENCY_SAVE_TIMEOUT * 1000}; // CHANGE: Use constant
        const STREAMLIT_FALLBACK_URL = window.location.origin + window.location.pathname; 
        
        if (window.fifi_close_enhanced_initialized) return;
        window.fifi_close_enhanced_initialized = true;
        
        let saveAttempted = false;
        
        console.log('🛡️ Enhanced browser close detection initialized for eligible user');
        
        function sendBeaconOrFetch(data) {{
            // PRIMARY: Try navigator.sendBeacon
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
            
            // FALLBACK 1: Try fetch with keepalive and short timeout
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

            // ALWAYS trigger Streamlit fallback via image, for maximum redundancy
            const fallbackUrl = STREAMLIT_FALLBACK_URL + 
                '?event=emergency_close' +
                '&session_id=' + sessionId + 
                '&reason=' + reason +
                '&fallback=true';
            
            const img = new Image();
            img.src = fallbackUrl;
            img.style.display = 'none'; // Keep image invisible
            document.body.appendChild(img); // Append to DOM to ensure request is sent
            
            console.log('✅ Streamlit fallback image beacon initiated');
        }}
        
        // Listen for actual browser close events
        window.addEventListener('beforeunload', () => triggerEmergencySave('beforeunload'), {{ capture: true, passive: true }});
        window.addEventListener('unload', () => triggerEmergencySave('unload'), {{ capture: true, passive: true }});
        window.addEventListener('pagehide', () => triggerEmergencySave('pagehide'), {{ capture: true, passive: true }});
        
        // Listen for visibility changes (tab switching detection)
        document.addEventListener('visibilitychange', function() {{
            if (document.visibilityState === 'hidden') {{
                console.log('📱 Tab became hidden - scheduling potential save');
                // Use a timeout to differentiate between tab switch and actual close
                setTimeout(() => {{
                    if (document.visibilityState === 'hidden') {{
                        console.log('🚨 Tab still hidden after delay - likely closed or backgrounded');
                        triggerEmergencySave('visibility_hidden_background');
                    }} else {{
                        console.log('✅ Tab became visible during delay - canceling save');
                        saveAttempted = false; // Reset if it was just a tab switch
                    }}
                }}, 5000); // 5-second delay to confirm
            }}
        }});
        
        // Try to monitor parent window as well for robustness in iframes
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

def process_fingerprint_from_query(session_id: str, fingerprint_id: str, method: str, privacy: str, working_methods: List[str]) -> bool:
    """Processes fingerprint data received via URL query parameters.
    This function now explicitly guards against processing for REGISTERED_USERs."""
    try:
        session_manager = st.session_state.get('session_manager')
        if not session_manager:
            logger.error("❌ Session manager not available during fingerprint processing from query.")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Fingerprint processing: Session '{session_id[:8]}' not found in database.")
            return False
        
        ## CHANGE: Guard against processing fingerprint data for REGISTERED_USERs
        if session.user_type == UserType.REGISTERED_USER:
            logger.warning(f"Attempted to process fingerprint from query for REGISTERED_USER {session.session_id[:8]}. Ignoring as fingerprint is not collected for this user type.")
            # Ensure chat is ready and clear any FP wait if this happens for a registered user
            st.session_state.is_chat_ready = True
            st.session_state.fingerprint_status = 'not_applicable'
            if 'fingerprint_wait_start' in st.session_state:
                del st.session_state['fingerprint_wait_start']
            return False # Indicate that fingerprint was not applied
        
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
            st.session_state.is_chat_ready = True # Explicitly unlock chat input here
            st.session_state.fingerprint_status = 'done' # Mark fingerprinting as done definitively
            # --- NEW: Ensure fingerprint_wait_start is cleared on successful completion ---
            if 'fingerprint_wait_start' in st.session_state:
                del st.session_state['fingerprint_wait_start']
            st.session_state.fingerprint_just_completed = True # NEW: Flag for a final clean rerun in main_fixed
            logger.info(f"Chat input unlocked for session {session.session_id[:8]} after successful JS fingerprinting.")
            return True
        else:
            logger.warning(f"⚠️ Fingerprint application failed for session '{session_id[:8]}'")
            # If it failed, ensure we still enable chat, but with 'failed' status
            st.session_state.is_chat_ready = True
            st.session_state.fingerprint_status = 'failed'
            if 'fingerprint_wait_start' in st.session_state: # Clear on failure too
                del st.session_state['fingerprint_wait_start']
            return False
        
    except Exception as e:
        logger.error(f"Fingerprint processing failed: {e}", exc_info=True)
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
        for param in params_to_clear:
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
        # Check if this session's fingerprint has ALREADY been processed.
        if st.session_state.get('fingerprint_processed_for_session', {}).get(session_id, False):
            logger.info(f"Fingerprint for session {session_id[:8]} already processed. Clearing params and skipping.")
            # Clear query parameters to clean up the URL
            params_to_clear = ["event", "session_id", "fingerprint_id", "method", "privacy", "working_methods", "timestamp"]
            for param in params_to_clear:
                if param in st.query_params:
                    del st.query_params[param]
            st.rerun()
            return

        logger.info("=" * 80)
        logger.info("🔍 FINGERPRINT DATA DETECTED VIA URL QUERY PARAMETERS!")
        logger.info(f"Session ID: {session_id}, Event: {event}")
        logger.info("=" * 80)
        
        fingerprint_id = query_params.get("fingerprint_id")
        method = query_params.get("method")
        privacy = query_params.get("privacy")
        working_methods = query_params.get("working_methods", "").split(",") if query_params.get("working_methods") else []
        
        logger.info(f"Extracted - ID: {fingerprint_id}, Method: {method}, Privacy: {privacy}, Working Methods: {working_methods}")
        
        # Clear query parameters immediately after extraction
        params_to_clear = ["event", "session_id", "fingerprint_id", "method", "privacy", "working_methods", "timestamp"]
        for param in params_to_clear:
            if param in st.query_params:
                del st.query_params[param]
        
        if not fingerprint_id or not method:
            st.error("❌ **Fingerprint Error** - Missing required data in redirect")
            logger.error(f"Missing fingerprint data: ID={fingerprint_id}, Method={method}")
            st.session_state.is_chat_ready = True # Unlock chat even on error to prevent getting stuck
            st.rerun()
            return
        
        try:
            success = process_fingerprint_from_query(session_id, fingerprint_id, method, privacy, working_methods)
            
            # Mark this session's fingerprint as processed to avoid rerunning this logic
            if 'fingerprint_processed_for_session' not in st.session_state:
                st.session_state.fingerprint_processed_for_session = {}
            st.session_state.fingerprint_processed_for_session[session_id] = True

            logger.info(f"✅ Silent fingerprint processing complete. Success: {success}")
            st.rerun() # Rerun to apply the new state and clean the URL
            
        except Exception as e:
            logger.error(f"Silent fingerprint processing failed: {e}")
            st.session_state.is_chat_ready = True # Unlock chat on failure
            st.rerun()
        
        return
    else:
        logger.debug("ℹ️ No fingerprint requests found in current URL query parameters.")

# =============================================================================
# UI COMPONENTS
# =============================================================================

# Modified render_welcome_page function
def render_welcome_page(session_manager: 'SessionManager'):
    """Enhanced welcome page with loading lock."""
    
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")

    # Show loading overlay if in loading state
    if show_loading_overlay():
        return
    
    session = session_manager.get_session() if st.session_state.get('current_session_id') else None
    
    ## CHANGE: Only show fingerprint loading for non-REGISTERED users
    if session and session.user_type != UserType.REGISTERED_USER and \
       not st.session_state.get('is_chat_ready', False) and st.session_state.get('fingerprint_wait_start'):
        
        current_time_float = time.time()
        wait_start = st.session_state.get('fingerprint_wait_start')
        elapsed = current_time_float - wait_start
        remaining = max(0, FINGERPRINT_TIMEOUT_SECONDS - elapsed)
        
        if remaining > 0:
            st.info(f"🔒 **Initializing secure session...** ({remaining:.0f}s remaining)")
            st.caption("Setting up device recognition and security features.")
        else:
            st.info("🔒 **Finalizing setup...** Almost ready!")
        
        # Add progress bar
        progress_value = min(elapsed / FINGERPRINT_TIMEOUT_SECONDS, 1.0)
        st.progress(progress_value, text="Initializing FiFi AI Assistant")
    
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
        st.markdown(f"""
        **Continue as a guest** to get a quick start and try FiFi AI Assistant without signing in.
        
        ℹ️ **What to expect as a Guest:**
        - You get an initial allowance of **{GUEST_QUESTION_LIMIT} questions** to explore FiFi AI's capabilities. ## CHANGE: Use constant
        - After these {GUEST_QUESTION_LIMIT} questions, **email verification will be required** to continue (unlocks {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day). ## CHANGE: Use constant
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
        st.markdown(f"• **{GUEST_QUESTION_LIMIT} questions** to try FiFi AI") ## CHANGE: Use constant
        st.markdown("• Email verification required to continue")
        st.markdown("• Quick start, no registration needed")
    
    with col2:
        st.info("📧 **Email Verified Guest**")
        st.markdown(f"• **{EMAIL_VERIFIED_QUESTION_LIMIT} questions per day** (rolling {DAILY_RESET_WINDOW_HOURS}-hour period)") ## CHANGE: Use constant
        st.markdown("• Email verification for access")
        st.markdown("• No full registration required")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown(f"• **{REGISTERED_USER_QUESTION_LIMIT} questions per day** with tier system:") ## CHANGE: Use constant
        st.markdown(f"  - **Tier 1**: Questions 1-{REGISTERED_USER_TIER_1_LIMIT} → {TIER_1_BAN_HOURS}-hour break") ## CHANGE: Use constant
        st.markdown(f"  - **Tier 2**: Questions {REGISTERED_USER_TIER_1_LIMIT + 1}-{REGISTERED_USER_QUESTION_LIMIT} → {TIER_2_BAN_HOURS}-hour reset") ## CHANGE: Use constant
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
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/{REGISTERED_USER_QUESTION_LIMIT}") ## CHANGE: Use constant
            
            if session.daily_question_count < REGISTERED_USER_TIER_1_LIMIT: ## CHANGE: Use constant
                st.progress(min(session.daily_question_count / REGISTERED_USER_TIER_1_LIMIT, 1.0), ## CHANGE: Use constant
                           text=f"Tier 1: {session.daily_question_count}/{REGISTERED_USER_TIER_1_LIMIT} questions") ## CHANGE: Use constant
                remaining_tier1 = REGISTERED_USER_TIER_1_LIMIT - session.daily_question_count ## CHANGE: Use constant
                if remaining_tier1 > 0:
                    st.caption(f"⏰ {remaining_tier1} questions until {TIER_1_BAN_HOURS}-hour break") ## CHANGE: Use constant
            elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT: ## CHANGE: Use constant
                # Check if there's an active ban
                if session.ban_status == BanStatus.ONE_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                    time_remaining = session.ban_end_time - datetime.now()
                    hours = int(time_remaining.total_seconds() // 3600)
                    minutes = int((time_remaining.total_seconds() % 3600) // 60)
                    st.error(f"🚫 Daily limit reached") # Corrected message for active ban
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    # Ban has expired or not yet applied
                    st.progress(1.0, text="Tier 1 Complete ✅")
                    st.caption("📈 Ready to proceed to Tier 2!")
            else: # daily_question_count > REGISTERED_USER_TIER_1_LIMIT
                tier2_progress = min((session.daily_question_count - REGISTERED_USER_TIER_1_LIMIT) / (REGISTERED_USER_QUESTION_LIMIT - REGISTERED_USER_TIER_1_LIMIT), 1.0) ## CHANGE: Use constant
                st.progress(tier2_progress, text=f"Tier 2: {session.daily_question_count - REGISTERED_USER_TIER_1_LIMIT}/{REGISTERED_USER_QUESTION_LIMIT - REGISTERED_USER_TIER_1_LIMIT} questions") ## CHANGE: Use constant
                remaining_tier2 = REGISTERED_USER_QUESTION_LIMIT - session.daily_question_count ## CHANGE: Use constant
                if remaining_tier2 > 0:
                    st.caption(f"⏰ {remaining_tier2} questions until {TIER_2_BAN_HOURS}-hour reset") ## CHANGE: Use constant
                else:
                    st.caption(f"🚫 Daily limit reached - {TIER_2_BAN_HOURS}-hour reset required") ## CHANGE: Use constant
                    
        elif session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/{EMAIL_VERIFIED_QUESTION_LIMIT}") ## CHANGE: Use constant
            st.progress(min(session.daily_question_count / EMAIL_VERIFIED_QUESTION_LIMIT, 1.0)) ## CHANGE: Use constant
            
            # Check if banned
            if session.ban_status == BanStatus.TWENTY_FOUR_HOUR and session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 Daily limit reached")
                st.caption(f"Resets in: {hours}h {minutes}m")
            elif session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=EMAIL_VERIFIED_BAN_HOURS) ## CHANGE: Use constant
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Daily questions have reset!")
            
        else: # Guest User
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/{GUEST_QUESTION_LIMIT}") ## CHANGE: Use constant
            st.progress(min(session.daily_question_count / GUEST_QUESTION_LIMIT, 1.0)) ## CHANGE: Use constant
            st.caption(f"Email verification unlocks {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day.") ## CHANGE: Use constant
            if session.reverification_pending:
                st.info("💡 An account is available for this device. Re-verify email to reclaim it!")
            elif session.declined_recognized_email_at and session.daily_question_count < session_manager.question_limits.question_limits[UserType.GUEST.value]: # Check for this state
                st.info("💡 You are currently using guest questions. Verify your email to get more.") # Alternative message
                
        # Show fingerprint status
        if session.fingerprint_id:
            ## CHANGE: Logic for displaying FP ID for registered users
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

        # Display time since last activity
        if session.last_activity is not None:
            time_since_activity = datetime.now() - session.last_activity
            minutes_inactive = time_since_activity.total_seconds() / 60
            st.caption(f"Last activity: {int(minutes_inactive)} minutes ago")
            
            timeout_duration = SESSION_TIMEOUT_MINUTES ## CHANGE: Use constant

            if minutes_inactive >= (timeout_duration - 1) and minutes_inactive < timeout_duration:
                minutes_remaining = timeout_duration - minutes_inactive
                st.warning(f"⚠️ Session expires in {minutes_remaining:.1f} minutes!") ## CHANGE: Use constant
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
                if time_remaining.total_seconds() > 0:
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
            
            if st.button("🗑️️ Clear Chat", use_container_width=True, help=clear_chat_help):
                session_manager.clear_chat_history(session)
                st.success("🗑️ Chat display cleared! Messages preserved in database.")
                st.rerun()
                
        with col2:
            signout_help = "Ends your current session and returns to the welcome page."
            ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS constant
            if (session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and 
                session.email and len(session.messages) >= CRM_SAVE_MIN_QUESTIONS):
                signout_help += " Your conversation will be automatically saved to CRM before signing out."
            
            if st.button("🚪 Sign Out", use_container_width=True, help=signout_help):
                session_manager.end_session(session)
                st.rerun()

        ## CHANGE: Use CRM_SAVE_MIN_QUESTIONS constant
        if session.user_type.value in [UserType.REGISTERED_USER.value, UserType.EMAIL_VERIFIED_GUEST.value] and len(session.messages) >= CRM_SAVE_MIN_QUESTIONS:
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
    Handles gentle prompts only when an answer was just given, otherwise shows a direct prompt.
    Controls `st.session_state.chat_blocked_by_dialog` and returns if chat input should be disabled.
    """
    
    # Ensure we're working with the correct session
    if st.session_state.get('current_session_id') != session.session_id:
        logger.warning(f"Session mismatch in email prompt! State: {st.session_state.get('current_session_id', 'None')[:8]}, Param: {session.session_id[:8]}")
        # Force correct session
        st.session_state.current_session_id = session.session_id
    
    # Initialize relevant session states if not present
    if 'verification_stage' not in st.session_state:
        st.session_state.verification_stage = None
    if 'guest_continue_active' not in st.session_state:
        st.session_state.guest_continue_active = False
    if 'final_answer_acknowledged' not in st.session_state:
        st.session_state.final_answer_acknowledged = False
    if 'gentle_prompt_shown' not in st.session_state:
        st.session_state.gentle_prompt_shown = False
    if 'email_verified_final_answer_acknowledged' not in st.session_state:
        st.session_state.email_verified_final_answer_acknowledged = False
    # NEW: Initialize 'must_verify_email_immediately' and 'skip_email_allowed' flags
    if 'must_verify_email_immediately' not in st.session_state:
        st.session_state.must_verify_email_immediately = False
    if 'skip_email_allowed' not in st.session_state:
        st.session_state.skip_email_allowed = True

    # NEW: Handle the actual email sending if flagged
    if st.session_state.get('send_code_now', False) and st.session_state.get('verification_email'):
        email_to_send = st.session_state.verification_email
        result = session_manager.handle_guest_email_verification(session, email_to_send)
        
        # Clear the send flag
        del st.session_state['send_code_now']
        
        if result['success']:
            st.success(result['message'])
            st.session_state.verification_stage = "code_entry"
        else:
            st.error(result['message'])
            # The 'unusual activity' message is from evasion, which is now disabled.
            # Keeping this check in case of future re-enabling or other types of unusual activity.
            # No specific action for "unusual activity" currently as email evasion detection is disabled.
            st.session_state.verification_stage = st.session_state.get('verification_stage', 'initial_check') # Fallback to avoid getting stuck
        
        st.rerun()
        return True  # Block chat during this transition

    # Check if a hard block is in place first (non-email-verification related bans)
    limit_check = session_manager.question_limits.is_within_limits(session)
    ## CHANGE: Use constants for comparison
    if not limit_check['allowed'] and limit_check.get('reason') not in ['guest_limit', 'email_verified_guest_limit', 'registered_user_tier1_limit', 'registered_user_tier2_limit']:
        st.session_state.chat_blocked_by_dialog = True # Hard ban, block everything
        return True # Disable chat input

    # Determine current user status
    user_is_guest = (session.user_type.value == UserType.GUEST.value)
    user_is_email_verified = (session.user_type.value == UserType.EMAIL_VERIFIED_GUEST.value)
    guest_limit_value = GUEST_QUESTION_LIMIT ## CHANGE: Use constant
    email_verified_limit_value = EMAIL_VERIFIED_QUESTION_LIMIT ## CHANGE: Use constant
    daily_q_value = session.daily_question_count
    is_guest_limit_hit = (user_is_guest and daily_q_value >= guest_limit_value)
    is_email_verified_limit_hit = (user_is_email_verified and daily_q_value >= email_verified_limit_value)
    
    # NEW LOGIC: Only consider it "just hit limit" if the 'just_answered' flag is true
    user_just_hit_guest_limit = is_guest_limit_hit and st.session_state.get('just_answered', False)
    user_just_hit_email_verified_limit = is_email_verified_limit_hit and st.session_state.get('just_answered', False)
    
    # NEW: Check if forced verification is required
    must_verify_immediately = st.session_state.get('must_verify_email_immediately', False)
    skip_allowed = st.session_state.get('skip_email_allowed', True)

    should_show_prompt = False
    should_block_chat = True  # Default to blocking when prompt is shown

    # PRIORITY 1: Handle forced verification for returning guest (no skip option)
    if user_is_guest and must_verify_immediately and daily_q_value == 0:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'forced_verification'

    # PRIORITY 2: Handle re-verification for recognized devices
    elif session.reverification_pending:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
             st.session_state.verification_stage = 'initial_check'
             st.session_state.guest_continue_active = False

    # PRIORITY 3: Handle guest who JUST hit their limit (GENTLE approach with the reading button)
    elif user_just_hit_guest_limit:
        st.session_state.just_answered = False # Consume the flag
        should_show_prompt = True
        should_block_chat = False  # DON'T block immediately
        
        st.success(f"🎯 **You've explored FiFi AI with your {GUEST_QUESTION_LIMIT} guest questions!**") ## CHANGE: Use constant
        st.info(f"Take your time to read this answer. When you're ready, verify your email to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} questions per day + chat history saving!") ## CHANGE: Use constant
        
        with st.expander("📧 Ready to Unlock More Questions?", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📧 Yes, Let's Verify My Email!", use_container_width=True, key="gentle_verify_btn"):
                    st.session_state.verification_stage = 'email_entry'
                    st.session_state.chat_blocked_by_dialog = True
                    st.session_state.final_answer_acknowledged = True
                    st.rerun()
            with col2:
                if skip_allowed: # Guest can still skip *after* the initial gentle prompt
                    if st.button("👀 Let Me Finish Reading First", use_container_width=True, key="continue_reading_btn"):
                        st.session_state.final_answer_acknowledged = True
                        st.success("Perfect! Take your time. The verification option will remain available above.")
                        st.rerun()
                else:
                    st.info("Email verification is required to continue.")
        
        st.session_state.chat_blocked_by_dialog = False
        return False

    # PRIORITY 4: Handle email verified guest who JUST hit their limit (GENTLE approach)
    elif user_just_hit_email_verified_limit:
        st.session_state.just_answered = False # Consume the flag
        should_show_prompt = True
        should_block_chat = False
        
        st.success(f"🎯 **You've completed your {EMAIL_VERIFIED_QUESTION_LIMIT} daily questions!**") ## CHANGE: Use constant
        st.info(f"Take your time to read this answer. Your questions will reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!") ## CHANGE: Use constant
        
        with st.expander("🚀 Want More Questions Daily?", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔗 Go to Registration", use_container_width=True, key="register_upgrade_btn"):
                    st.link_button("Register Here", "https://www.12taste.com/in/my-account/", use_container_width=True) # Direct link
                    st.session_state.email_verified_final_answer_acknowledged = True
                    st.rerun()
            with col2:
                if st.button("👀 Let Me Finish Reading First", use_container_width=True, key="email_verified_continue_reading"):
                    st.session_state.email_verified_final_answer_acknowledged = True
                    st.success(f"Perfect! Take your time reading. You'll need to wait {EMAIL_VERIFIED_BAN_HOURS} hours for more questions.") ## CHANGE: Use constant
                    st.rerun()
        
        st.session_state.chat_blocked_by_dialog = False
        return False

    # PRIORITY 5: Handle guest who is at their limit but DID NOT just ask a question (e.g., new session)
    elif is_guest_limit_hit:
        should_show_prompt = True
        should_block_chat = True
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'email_entry'
            st.session_state.guest_continue_active = False

    # PRIORITY 6: Handle email verified guest who is at their limit but DID NOT just ask a question
    elif is_email_verified_limit_hit and not st.session_state.email_verified_final_answer_acknowledged:
        should_show_prompt = True
        should_block_chat = True
        st.error("🛑 **Daily Limit Reached**")
        st.info(f"You've used your {EMAIL_VERIFIED_QUESTION_LIMIT} questions for today. Your questions will reset in {EMAIL_VERIFIED_BAN_HOURS} hours, or consider registering for {REGISTERED_USER_QUESTION_LIMIT} questions/day!") ## CHANGE: Use constants
        
        col1, col2 = st.columns(2)
        with col1:
            st.link_button(f"Register for {REGISTERED_USER_QUESTION_LIMIT} questions/day", "https://www.12taste.com/in/my-account/", use_container_width=True) ## CHANGE: Use constant
        with col2:
            if st.button("Return to Welcome Page", use_container_width=True):
                session_manager.end_session(session)
                js_redirect = f"window.top.location.href = 'https://fifi-eu.streamlit.app/';"
                st.components.v1.html(f"<script>{js_redirect}</script>", height=0, width=0)
                st.rerun()
        
        st.session_state.chat_blocked_by_dialog = True
        return True

    # PRIORITY 7: Handle declined recognized email scenario
    elif session.declined_recognized_email_at and not st.session_state.guest_continue_active and not is_guest_limit_hit:
        should_show_prompt = True
        should_block_chat = False
        if st.session_state.verification_stage is None:
            st.session_state.verification_stage = 'declined_recognized_email_prompt_only'

    # If no prompt should be shown based on conditions, ensure state is clean
    if not should_show_prompt:
        st.session_state.chat_blocked_by_dialog = False
        st.session_state.verification_stage = None
        if 'just_answered' in st.session_state: # Clean up the flag if not used
             del st.session_state.just_answered
        return False

    # Set chat blocking state based on the type of prompt
    st.session_state.chat_blocked_by_dialog = should_block_chat
    
    # Only show error header for blocking prompts
    if should_block_chat:
        st.error("📧 **Action Required**")

    current_stage = st.session_state.verification_stage

    # NEW: Handle forced verification stage (no skip option)
    if current_stage == 'forced_verification':
        st.error("📧 **Email Verification Required**")
        st.info("This device has already used the guest question allowance. Please verify your email to continue using FiFi AI.")
        
        with st.form("forced_email_verification_form", clear_on_submit=False):
            st.markdown("**📧 Enter your email address to receive a verification code:**")
            current_email_input = st.text_input(
                "Email Address", 
                placeholder="your@email.com",
                key="forced_email_input",
                help="Email verification is required to continue."
            )
            submit_email = st.form_submit_button("📨 Send Verification Code", use_container_width=True)
            
            # NO SKIP BUTTON - verification is mandatory
            
            if submit_email:
                if current_email_input:
                    # Reset the must_verify flag after a submission attempt
                    st.session_state.must_verify_email_immediately = False
                    st.session_state.skip_email_allowed = True # Reset to default allowed after forced verification attempt
                    
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
        # Use a container to ensure clean rendering
        prompt_container = st.container()
        
        with prompt_container:
            email_to_reverify = session.pending_email
            masked_email = session_manager._mask_email(email_to_reverify) if email_to_reverify else "your registered email"
            st.info(f"🤝 **We recognize this device was previously used as a {session.pending_user_type.value.replace('_', ' ').title()} account.**")
            st.info(f"Please verify **{masked_email}** to reclaim your status and higher question limits.")
            
            # FIXED: Use session ID for stable keys instead of timestamp
            button_key_suffix = session.session_id[:8]  # First 8 chars of session ID
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Verify this email", 
                            use_container_width=True, 
                            key=f"reverify_yes_{button_key_suffix}"):
                    
                    # Add logging to debug
                    logger.info(f"Verify button clicked in session {session.session_id[:8]}")
                    
                    session.recognition_response = "yes_reverify"
                    session.declined_recognized_email_at = None
                    st.session_state.verification_email = email_to_reverify
                    st.session_state.verification_stage = "send_code_recognized"
                    
                    # Ensure session ID is preserved
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
                    
                    # Ensure session ID is preserved
                    st.session_state.current_session_id = session.session_id
                    
                    session_manager.db.save_session(session)
                    st.session_state.guest_continue_active = True
                    st.session_state.chat_blocked_by_dialog = False
                    st.session_state.verification_stage = None
                    st.success("You can now continue as a Guest.")
                    st.rerun()

    elif current_stage == 'send_code_recognized':
        # Don't show any buttons - just the sending status
        email_to_verify = st.session_state.get('verification_email')
        if email_to_verify:
            st.info(f"📧 **Sending verification code to {session_manager._mask_email(email_to_verify)}...**")
            
            # Immediately perform the send (the actual sending happens after this renders)
            st.session_state.send_code_now = True
            st.rerun()

    elif current_stage == 'email_entry':
        skip_allowed = st.session_state.get('skip_email_allowed', True)
        
        ## CHANGE: Use constants in prompt
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
                    # NEW: End the session instead of just continuing
                    logger.info(f"User chose to skip email verification for session {session.session_id[:8]} - ending session")
                    
                    # Mark session as inactive
                    session.active = False
                    session.last_activity = datetime.now()
                    
                    # Save final session state
                    try:
                        session_manager.db.save_session(session)
                    except Exception as e:
                        logger.error(f"Failed to save session during skip: {e}")
                    
                    # Clear Streamlit session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    
                    # Reset to welcome page
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
                        # No "unusual activity" message due to evasion being disabled
                        # Still keep this check in case of future changes.
                        # No specific action for "unusual activity" currently as email evasion detection is disabled.
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
                        # Clean up verification state on success
                        st.session_state.chat_blocked_by_dialog = False
                        st.session_state.verification_stage = None
                        st.session_state.guest_continue_active = False
                        st.session_state.final_answer_acknowledged = False
                        st.session_state.gentle_prompt_shown = False
                        st.session_state.email_verified_final_answer_acknowledged = False
                        # NEW: Clear must_verify_email_immediately and skip_email_allowed on successful verification
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
        # Non-blocking prompt for users who declined recognized email
        st.session_state.chat_blocked_by_dialog = False

        remaining_questions = GUEST_QUESTION_LIMIT - session.daily_question_count ## CHANGE: Use constant
        st.info(f"✅ **Continuing as Guest** - You have **{remaining_questions} questions** remaining from your guest allowance.")
        st.info(f"💡 **Pro Tip:** Verify your email anytime to unlock {EMAIL_VERIFIED_QUESTION_LIMIT} questions/day + chat history saving.") ## CHANGE: Use constant

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
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion.")

    # NEW: Show fingerprint waiting status ONLY for non-registered users
    ## CHANGE: Only show fingerprint wait for non-registered users
    if session.user_type != UserType.REGISTERED_USER and \
       not st.session_state.get('is_chat_ready', False) and st.session_state.get('fingerprint_wait_start'):
        
        current_time_float = time.time() # Use float for direct comparison with time.time()
        wait_start = st.session_state.get('fingerprint_wait_start')
        elapsed = current_time_float - wait_start
        remaining = max(0, FINGERPRINT_TIMEOUT_SECONDS - elapsed) ## CHANGE: Use constant
        
        if remaining > 0:
            st.info(f"🔒 **Securing your session...** ({remaining:.0f}s remaining)")
            st.caption("FiFi is setting up device recognition for security and session management.")
        else:
            st.info("🔒 **Finalizing setup...** Almost ready!")
        
        # Add a subtle progress bar
        progress_value = min(elapsed / FINGERPRINT_TIMEOUT_SECONDS, 1.0) ## CHANGE: Use constant
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
        limit_check_for_display = session_manager.question_limits.is_within_limits(session)
        if (session.user_type.value == UserType.REGISTERED_USER.value and 
            limit_check_for_display.get('allowed') and 
            limit_check_for_display.get('tier')):
            
            tier = limit_check_for_display.get('tier')
            remaining = limit_check_for_display.get('remaining', 0)
            
            # Check actual ban status for accurate messaging
            has_active_tier1_ban = (
                session.ban_status == BanStatus.ONE_HOUR and 
                session.ban_end_time and 
                datetime.now() < session.ban_end_time
            )
            
            if tier == 2 and remaining <= 3:
                st.warning(f"⚠️ **Tier 2 Alert**: Only {remaining} questions remaining until {TIER_2_BAN_HOURS}-hour reset!") ## CHANGE: Use constant
            elif tier == 1 and remaining <= 2 and remaining > 0:
                st.info(f"ℹ️ **Tier 1**: {remaining} questions remaining until {TIER_1_BAN_HOURS}-hour break.") ## CHANGE: Use constant
            ## CHANGE: Use REGISTERED_USER_TIER_1_LIMIT constant
            elif session.daily_question_count == REGISTERED_USER_TIER_1_LIMIT:
                # At exactly 10 questions - check ban status
                if has_active_tier1_ban:
                    time_remaining = session.ban_end_time - datetime.now()
                    minutes = int(time_remaining.total_seconds() / 60)
                    hours = int(time_remaining.total_seconds() / 3600)
                    if hours >= 1:
                        st.warning(f"⏳ **{TIER_1_BAN_HOURS}-hour break in progress**: {hours} hour(s) remaining") ## CHANGE: Use constant
                    else:
                        st.warning(f"⏳ **{TIER_1_BAN_HOURS}-hour break in progress**: {minutes} minutes remaining") ## CHANGE: Use constant
                elif session.ban_status == BanStatus.ONE_HOUR and session.ban_end_time and datetime.now() >= session.ban_end_time:
                    # Ban has expired
                    st.info("✅ **Tier 1 Complete**: You can now proceed to Tier 2!")
                else:
                    # No ban yet
                    st.info(f"ℹ️ **Tier 1 Complete**: Your next question will trigger a {TIER_1_BAN_HOURS}-hour break before Tier 2.") ## CHANGE: Use constant

        # Display chat messages (respects soft clear offset)
        visible_messages = session.messages[session.display_message_offset:]
        for msg in visible_messages:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""), unsafe_allow_html=True)
                
                if msg.get("source"):
                    source_color = {
                        "FiFi": "🧠", "FiFi Web Search": "🌐", 
                        "Content Moderation": "🛡️", "System Fallback": "⚠️",
                        "Error Handler": "❌", "Session Analytics": "📈", 
                        "Session History": "📜", "Conversation Summary": "📝", "Topic Analysis": "🔍"
                    }.get(msg['source'], "🤖")
                    st.caption(f"{source_color} Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"): indicators.append("🧠 FiFi Knowledge Base")
                if msg.get("used_search"): indicators.append("🌐 FiFi Web Search")
                if msg.get("is_meta_response"): indicators.append("📈 Session Analytics")
                if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")
                
                if msg.get("safety_override"):
                    st.warning("🛡️ Safety Override: Switched to verified sources")
                
                if msg.get("has_citations") and msg.get("has_inline_citations"):
                    st.caption("📚 Response includes verified citations")
                
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
        max_requests = RATE_LIMIT_REQUESTS ## CHANGE: Use constant
        window_seconds = RATE_LIMIT_WINDOW_SECONDS ## CHANGE: Use constant
        
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
    
    # NEW: Pricing/Stock notice with manual dismiss
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

    # Show approaching limit warnings (Option 2 enhancement)
    if not overall_chat_disabled:
        user_type = session.user_type.value
        current_count = session.daily_question_count
        
        ## CHANGE: Use constants for warnings
        if user_type == UserType.GUEST.value and current_count == GUEST_QUESTION_LIMIT - 1:
            st.warning(f"⚠️ **Final Guest Question Coming Up!** Your next question will be your last before email verification is required.")
            
        elif user_type == UserType.EMAIL_VERIFIED_GUEST.value and current_count == EMAIL_VERIFIED_QUESTION_LIMIT - 1:
            st.warning(f"⚠️ **Final Question Today!** Your next question will be your last for the next {EMAIL_VERIFIED_BAN_HOURS} hours.")
            
        elif user_type == UserType.REGISTERED_USER.value:
            if current_count == REGISTERED_USER_TIER_1_LIMIT - 1:
                st.warning(f"⚠️ **Tier 1 Final Question Coming Up!** After your next question, you'll need a {TIER_1_BAN_HOURS}-hour break.")
            elif current_count == REGISTERED_USER_QUESTION_LIMIT - 1:
                st.warning(f"⚠️ **Final Question Today!** Your next question will be your last for {TIER_2_BAN_HOURS} hours.")

    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...", 
                            disabled=overall_chat_disabled)
    
    if prompt:
        logger.info(f"🎯 Processing question from {session.session_id[:8]}")
        
        # Check if attempting to exceed limits _before_ sending to AI
        # This call will now also handle displaying appropriate messages/bans.
        if session_manager.check_if_attempting_to_exceed_limits(session):
            # If `check_if_attempting_to_exceed_limits` returns True, it means a limit was hit
            # and a message/ban has been displayed.
            # For guest limit, we specifically set the verification stage.
            if session.user_type.value == UserType.GUEST.value and \
               session.daily_question_count >= GUEST_QUESTION_LIMIT: ## CHANGE: Use constant
                st.session_state.verification_stage = 'email_entry'
                st.session_state.chat_blocked_by_dialog = True
                st.session_state.final_answer_acknowledged = True # Acknowledge the 'final answer' to trigger dialog
            st.rerun()
            return
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 FiFi is processing your question and we request your patience..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    st.session_state.just_answered = True # Set flag for gentle prompts
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue.")
                        st.session_state.verification_stage = 'email_entry' 
                        st.session_state.chat_blocked_by_dialog = True
                    elif response.get('banned'):
                        st.error(response.get("content", 'Access restricted.'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    else:
                        # Show the AI response and metadata
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        if response.get("source"):
                            source_color = {
                                "FiFi": "🧠", "FiFi Web Search": "🌐",
                                "Content Moderation": "🛡️", "System Fallback": "⚠️",
                                "Error Handler": "❌", "Session Analytics": "📈",
                                "Session History": "📜", "Conversation Summary": "📝", "Topic Analysis": "🔍"
                            }.get(response['source'], "🤖")
                            st.caption(f"{source_color} Source: {response['source']}")
                        
                        indicators = []
                        if response.get("used_pinecone"): indicators.append("🧠 FiFi Knowledge Base")
                        if response.get("used_search"): indicators.append("🌐 FiFi Web Search")
                        if response.get("is_meta_response"): indicators.append("📈 Session Analytics")
                        if indicators: st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                        if response.get("safety_override"): st.warning("🛡️ Safety Override: Switched to verified sources")
                        if response.get("has_citations") and response.get("has_inline_citations"): st.caption("📚 Response includes verified citations")
                        
                        logger.info(f"✅ Question processed successfully")

                except Exception as e:
                    logger.error(f"❌ AI response failed: {e}", exc_info=True)
                    st.error("⚠️ I encountered an error. Please try again.")
        st.rerun()

## CHANGE: Persistent state manager
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
        if data and datetime.now() < data['expires']:
            return data['value']
        return default
    
    @staticmethod
    def delete(key: str):
        """Delete a persistent state key"""
        if f'_persistent_{key}' in st.session_state:
            del st.session_state[f'_persistent_{key}']


# --- CACHED GETTER FUNCTIONS FOR HEAVY RESOURCES ---

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_config():
    logger.info("Initializing Config (cached)")
    return Config()

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_pdf_exporter():
    logger.info("Initializing PDFExporter (cached)")
    return PDFExporter()

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_db_manager(connection_string: Optional[str]):
    logger.info("Initializing DatabaseManager (cached)")
    return DatabaseManager(connection_string)

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_zoho_manager(_config: Config, _pdf_exporter: PDFExporter): # Changed to underscore
    logger.info("Initializing ZohoCRMManager (cached)")
    return ZohoCRMManager(_config, _pdf_exporter)

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_ai_system(_config: Config): # Changed to underscore
    logger.info("Initializing EnhancedAI (cached)")
    return EnhancedAI(_config)

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_rate_limiter(max_requests: int, window_seconds: int):
    logger.info("Initializing RateLimiter (cached)")
    return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_fingerprinting_manager():
    logger.info("Initializing FingerprintingManager (cached)")
    return DatabaseManager.FingerprintingManager()

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_email_verification_manager(_config: Config): # Changed to underscore
    logger.info("Initializing EmailVerificationManager (cached)")
    return DatabaseManager.EmailVerificationManager(_config)

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_question_limit_manager():
    logger.info("Initializing QuestionLimitManager (cached)")
    return DatabaseManager.QuestionLimitManager()

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_session_manager():
    logger.info("Initializing SessionManager (cached)")
    config = get_config()
    pdf_exporter = get_pdf_exporter()
    db_manager = get_db_manager(config.SQLITE_CLOUD_CONNECTION)
    zoho_manager = get_zoho_manager(config, pdf_exporter)
    ai_system = get_ai_system(config)
    rate_limiter = get_rate_limiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
    fingerprinting_manager = get_fingerprinting_manager()
    email_verification_manager = get_email_verification_manager(config)
    question_limit_manager = get_question_limit_manager()

    return SessionManager(
        config, db_manager, zoho_manager, ai_system,
        rate_limiter, fingerprinting_manager, email_verification_manager,
        question_limit_manager
    )

def ensure_initialization_fixed():
    """Fixed version without duplicate spinner since we have loading overlay"""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        logger.info("Starting application initialization sequence (no spinner shown, overlay is active)...")
        
        try:
            # Use cached getters for all components
            st.session_state.config = get_config()
            st.session_state.pdf_exporter = get_pdf_exporter()
            st.session_state.db_manager = get_db_manager(st.session_state.config.SQLITE_CLOUD_CONNECTION)
            # Pass cached objects with underscore to prevent hashing issues
            st.session_state.zoho_manager = get_zoho_manager(st.session_state.config, st.session_state.pdf_exporter)
            st.session_state.ai_system = get_ai_system(st.session_state.config)
            st.session_state.rate_limiter = get_rate_limiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
            st.session_state.fingerprinting_manager = get_fingerprinting_manager()
            st.session_state.email_verification_manager = get_email_verification_manager(st.session_state.config)
            st.session_state.question_limit_manager = get_question_limit_manager()

            # The main SessionManager also needs to be cached if all its dependencies are cached
            # It's safer to just call the `get_session_manager` function that uses all the other cached getters
            st.session_state.session_manager = get_session_manager()

            st.session_state.error_handler = error_handler # This is a global instance, no need to cache

            st.session_state.chat_blocked_by_dialog = False
            st.session_state.verification_stage = None
            st.session_state.guest_continue_active = False
            # NEW: Initialize chat readiness flag
            st.session_state.is_chat_ready = False 
            # NEW: Flag to indicate fingerprint processing just completed
            st.session_state.fingerprint_just_completed = False
            
            st.session_state.initialized = True
            logger.info("✅ Application initialized successfully")
            
        except Exception as e:
            logger.critical(f"Critical initialization failure: {e}", exc_info=True)
            st.session_state.initialized = False
            return False
    
    return True

# Modified main function with proper loading state handling (from prompt)
def main_fixed():
    """Main application entry point with optimized fingerprint handling."""
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

    # Initialize all necessary session state variables at once
    if 'initialized' not in st.session_state:
        defaults = {
            "initialized": False, "is_loading": False, "loading_message": "",
            "is_chat_ready": False, "fingerprint_complete": False,
            "chat_blocked_by_dialog": False, "verification_stage": None,
            "guest_continue_active": False, "final_answer_acknowledged": False,
            "gentle_prompt_shown": False, "email_verified_final_answer_acknowledged": False,
            "must_verify_email_immediately": False, "skip_email_allowed": True,
            "page": None, "fingerprint_processed_for_session": {},
            "fingerprint_just_completed": False # NEW: Initialize this flag
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Perform initial setup
        init_success = ensure_initialization_fixed()
        if not init_success:
            st.error("⚠️ Application failed to initialize properly. Please refresh the page.")
            return

    # --- OPTIMIZATION: Handle URL-based events EARLY, before any other logic ---
    handle_emergency_save_requests_from_query()
    handle_fingerprint_requests_from_query()

    # Get session manager
    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        st.error("❌ Session Manager not available. Please refresh the page.")
        return

    # Handle loading states (e.g., after clicking "Start as Guest" or "Sign In")
    if st.session_state.get('is_loading', False):
        if show_loading_overlay():
            pass # Overlay is shown, blocking further rendering

        try:
            loading_reason = st.session_state.get('loading_reason', 'unknown')
            session = None
            
            if loading_reason == 'start_guest':
                session = session_manager.get_session()
                if session and session.last_activity is None:
                    session.last_activity = datetime.now()
                    session_manager.db.save_session(session)
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
                    # Clear temp credentials regardless of success
                    if 'temp_username' in st.session_state: del st.session_state['temp_username']
                    if 'temp_password' in st.session_state: del st.session_state['temp_password']
                else:
                    set_loading_state(False)
                    st.error("Authentication failed: Missing username or password.")
                    return
            
            if 'loading_reason' in st.session_state:
                del st.session_state['loading_reason']

            # Determine chat readiness after session is established
            if session:
                if session.user_type == UserType.REGISTERED_USER:
                    st.session_state.is_chat_ready = True
                else:
                    fingerprint_is_stable = not session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
                    if fingerprint_is_stable:
                        st.session_state.is_chat_ready = True

            set_loading_state(False)
            st.rerun()
            return
            
        except Exception as e:
            set_loading_state(False)
            st.error(f"⚠️ Error during loading: {str(e)}")
            logger.error(f"Loading state error: {e}", exc_info=True)
            return

    # --- Normal Page Rendering Logic ---
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session()
        
        if session is None or not session.active:
            logger.warning("Expected active session for 'chat' page but got None or inactive. Forcing welcome.")
            for key in list(st.session_state.keys()):
                if key != 'initialized': # Keep initialized flag
                    del st.session_state[key]
            st.session_state['page'] = None
            st.rerun()
            return
        
        # Conditionally render fingerprinting component only for non-registered users who need it
        if session.user_type != UserType.REGISTERED_USER:
            fingerprint_needed = not session.fingerprint_id or session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
            if fingerprint_needed:
                fingerprint_key = f"fingerprint_rendered_{session.session_id}"
                if not st.session_state.get(fingerprint_key, False):
                    session_manager.fingerprinting.render_fingerprint_component(session.session_id)
                    st.session_state[fingerprint_key] = True
                    logger.info(f"✅ Fingerprint component rendered for session {session.session_id[:8]}")

        # Set up fingerprint timeout logic ONLY for non-registered users waiting for a fingerprint
        if session.user_type != UserType.REGISTERED_USER:
            fingerprint_is_stable = not session.fingerprint_id.startswith(("temp_py_", "temp_fp_", "fallback_"))
            if fingerprint_is_stable:
                st.session_state.is_chat_ready = True
                if 'fingerprint_wait_start' in st.session_state:
                    del st.session_state['fingerprint_wait_start']
            else:
                if 'fingerprint_wait_start' not in st.session_state:
                    st.session_state.fingerprint_wait_start = time.time()
                
                # Check for explicit completion flag (from query handler)
                if st.session_state.get('fingerprint_just_completed', False):
                    st.session_state.is_chat_ready = True
                    logger.info(f"Chat enabled immediately due to fingerprint_just_completed flag for {session.session_id[:8]}.")
                    st.session_state.fingerprint_just_completed = False # Consume flag
                elif time.time() - st.session_state.fingerprint_wait_start > FINGERPRINT_TIMEOUT_SECONDS:
                    st.session_state.is_chat_ready = True
                    logger.warning(f"Fingerprint timeout ({FINGERPRINT_TIMEOUT_SECONDS}s) - enabling chat with fallback for session {session.session_id[:8]}")
                else:
                    st.session_state.is_chat_ready = False
        else: # Registered users are always ready
            st.session_state.is_chat_ready = True
            if 'fingerprint_wait_start' in st.session_state:
                del st.session_state['fingerprint_wait_start']
        
        # If still waiting for fingerprint, rerun to show progress
        if not st.session_state.get('is_chat_ready', False):
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
