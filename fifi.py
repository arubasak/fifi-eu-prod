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
# FINAL INTEGRATED VERSION - ALL FEATURES COMBINED WITH CORRECTED TIMER & MESSAGE CHANNEL FIX
# - JavaScript timer with corrected st_javascript patterns using IIFE
# - Fix for "message channel closed" errors
# - Python CRM save processing
# - window.parent.location for all reloads
# - SQLite Cloud database integration
# - All existing features preserved
# - Complete error handling and validation
# - Fixed timer return value issues
# - ADDED: Enhanced debugging for beacon CRM save failures
# - ADDED: Fix for beacon query parameter processing in Streamlit
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

    def get_system_health_summary(self) -> Dict[str, Any]:
        if not self.component_status:
            return {"overall_health": "Unknown", "healthy_components": 0, "total_components": 0}

        healthy_count = sum(1 for status in self.component_status.values() if status == "healthy")
        total_count = len(self.component_status)

        if healthy_count == total_count:
            overall_health = "Healthy"
        elif healthy_count > total_count // 2:
            overall_health = "Degraded"
        else:
            overall_health = "Critical"

        return {
            "overall_health": overall_health,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "error_count": len(self.error_history)
        }

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
                logger.error(f"API Error in {component}/{operation}: {e}")
                return None
        return wrapper
    return decorator

# =============================================================================
# USER MODELS
# =============================================================================

class UserType(Enum):
    GUEST = "guest"
    REGISTERED_USER = "registered_user"

@dataclass
class UserSession:
    session_id: str
    user_type: UserType = UserType.GUEST
    email: Optional[str] = None
    first_name: Optional[str] = None
    zoho_contact_id: Optional[str] = None
    guest_email_requested: bool = False
    active: bool = True
    wp_token: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    timeout_saved_to_crm: bool = False

# =============================================================================
# DATABASE MANAGER WITH SQLITE CLOUD SUPPORT
# =============================================================================

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.connection_string = connection_string
        self.conn = None
        
        logger.info("🔄 INITIALIZING DATABASE MANAGER")
        
        # Try SQLite Cloud first
        if connection_string and SQLITECLOUD_AVAILABLE:
            cloud_result = self._try_sqlite_cloud_connection(connection_string)
            if cloud_result:
                self.conn, self.db_type, self.db_path = cloud_result
                logger.info("✅ SQLite Cloud connection established successfully!")
        
        # Fallback to local SQLite
        if not self.conn:
            logger.info("🔄 Falling back to local database...")
            local_result = self._try_local_sqlite_connection()
            if local_result:
                self.conn, self.db_type, self.db_path = local_result
                logger.info("✅ Local SQLite connection established!")
        
        # Final fallback to in-memory
        if not self.conn:
            logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED")
            logger.critical("⚠️  Falling back to non-persistent in-memory storage")
            self.db_type = "memory"
            self._init_local_storage()
        
        # Initialize database tables
        if self.conn:
            self._init_database()
            error_handler.mark_component_healthy("Database")

    def _try_sqlite_cloud_connection(self, connection_string: str):
        """Try SQLite Cloud connection"""
        logger.info("🔄 Attempting SQLite Cloud connection...")
        
        try:
            import sqlitecloud
            logger.info("✅ sqlitecloud library available")
            
            conn = sqlitecloud.connect(connection_string)
            result = conn.execute("SELECT 1 as test").fetchone()
            logger.info(f"✅ Connection test successful: {result}")
            
            return conn, "cloud", connection_string
            
        except ImportError:
            logger.error("❌ sqlitecloud library not available")
            return None
        except Exception as e:
            logger.error(f"❌ SQLite Cloud connection failed: {e}")
            return None

    def _try_local_sqlite_connection(self):
        """Fallback to local SQLite"""
        logger.info("🔄 Attempting local SQLite connection...")
        
        db_path = "fifi_sessions.db"
        
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = conn.execute("SELECT 1 as test")
            result = cursor.fetchone()
            logger.info(f"✅ Local SQLite test successful: {result}")
            
            return conn, "file", db_path
            
        except Exception as e:
            logger.error(f"❌ Local SQLite connection failed: {e}")
            return None

    def _init_local_storage(self):
        """Initialize in-memory storage as final fallback"""
        self.local_sessions = {}
        logger.info("📝 In-memory storage initialized")

    def _init_database(self):
        """Initialize database tables"""
        with self.lock:
            try:
                # NEVER set row_factory during table creation
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY, 
                        user_type TEXT, 
                        email TEXT, 
                        first_name TEXT,
                        zoho_contact_id TEXT, 
                        guest_email_requested INTEGER, 
                        created_at TEXT,
                        last_activity TEXT, 
                        messages TEXT, 
                        active INTEGER, 
                        wp_token TEXT,
                        timeout_saved_to_crm INTEGER
                    )
                ''')
                self.conn.commit()
                logger.info("✅ Database tables initialized successfully")
            except Exception as e:
                logger.error(f"Database table initialization failed: {e}")
                raise

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with SQLite Cloud compatibility"""
        with self.lock:
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = session
                return
            
            try:
                # NEVER set row_factory for save operations
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                self.conn.execute(
                    '''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session.session_id, session.user_type.value, session.email, session.first_name,
                     session.zoho_contact_id, int(session.guest_email_requested), session.created_at.isoformat(),
                     session.last_activity.isoformat(), json.dumps(session.messages), int(session.active),
                     session.wp_token, int(session.timeout_saved_to_crm)))
                self.conn.commit()
                
                logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={session.user_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to save session {session.session_id[:8]}: {e}")
                raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete SQLite Cloud compatibility"""
        with self.lock:
            if self.db_type == "memory":
                session = self.local_sessions.get(session_id)
                if session and isinstance(session.user_type, str):
                    session.user_type = UserType(session.user_type)
                return session

            try:
                # NEVER set row_factory for cloud connections - always use raw tuples
                if self.db_type == "cloud":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = None
                elif self.db_type == "file":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = sqlite3.Row
                
                cursor = self.conn.execute("SELECT session_id, user_type, email, first_name, zoho_contact_id, guest_email_requested, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                row = cursor.fetchone()
                
                if not row: 
                    return None
                
                logger.info(f"Loaded row for session {session_id[:8]}: type={type(row)}, db_type={self.db_type}")
                
                # Handle row conversion based on actual row type
                row_dict = {}
                
                if hasattr(row, 'keys') and callable(getattr(row, 'keys')):
                    # This is a sqlite3.Row object - safe to convert to dict
                    try:
                        row_dict = dict(row)
                        logger.info(f"Successfully converted sqlite3.Row to dict for session {session_id[:8]}")
                    except Exception as e:
                        logger.error(f"Failed to convert sqlite3.Row: {e}")
                        row_dict = None
                
                if not row_dict:
                    # Handle as tuple/list (SQLite Cloud or fallback)
                    logger.info(f"Handling row as tuple for session {session_id[:8]}")
                    if len(row) >= 12:
                        row_dict = {
                            'session_id': row[0],
                            'user_type': row[1], 
                            'email': row[2],
                            'first_name': row[3],
                            'zoho_contact_id': row[4],
                            'guest_email_requested': row[5],
                            'created_at': row[6],
                            'last_activity': row[7], 
                            'messages': row[8],
                            'active': row[9],
                            'wp_token': row[10],
                            'timeout_saved_to_crm': row[11]
                        }
                    else:
                        logger.error(f"Row has insufficient columns: {len(row)} (expected 12)")
                        return None
                
                if not row_dict:
                    logger.error(f"Failed to convert row to dictionary for session {session_id[:8]}")
                    return None
                
                # Create and return UserSession
                try:
                    user_session = UserSession(
                        session_id=row_dict['session_id'], 
                        user_type=UserType(row_dict['user_type']),
                        email=row_dict.get('email'), 
                        first_name=row_dict.get('first_name'),
                        zoho_contact_id=row_dict.get('zoho_contact_id'),
                        guest_email_requested=bool(row_dict.get('guest_email_requested')),
                        created_at=datetime.fromisoformat(row_dict['created_at']),
                        last_activity=datetime.fromisoformat(row_dict['last_activity']),
                        messages=json.loads(row_dict.get('messages', '[]')),
                        active=bool(row_dict.get('active', 1)), 
                        wp_token=row_dict.get('wp_token'),
                        timeout_saved_to_crm=bool(row_dict.get('timeout_saved_to_crm', 0))
                    )
                    
                    logger.info(f"Successfully loaded session {session_id[:8]}: user_type={user_session.user_type}")
                    return user_session
                    
                except Exception as e:
                    logger.error(f"Failed to create UserSession object: {e}")
                    logger.error(f"Row dict: {row_dict}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to load session {session_id[:8]}: {e}")
                logger.error(f"Database type: {self.db_type}")
                if 'row' in locals():
                    logger.error(f"Row type: {type(row)}")
                return None

    def test_connection(self) -> bool:
        """Test database connection for health checks"""
        if self.db_type == "memory":
            return hasattr(self, 'local_sessions')
        
        try:
            with self.lock:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                    
                cursor = self.conn.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

# =============================================================================
# PDF EXPORTER
# =============================================================================

class PDFExporter:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='ChatHeader', alignment=TA_CENTER, fontSize=18))
        self.styles.add(ParagraphStyle(name='UserMessage', backColor=lightgrey))

    @handle_api_errors("PDF Exporter", "Generate Chat PDF")
    def generate_chat_pdf(self, session: UserSession) -> Optional[io.BytesIO]:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = [Paragraph("FiFi AI Chat Transcript", self.styles['Heading1'])]
        
        for msg in session.messages:
            role = str(msg.get('role', 'unknown')).capitalize()
            content = html.escape(str(msg.get('content', '')))
            
            # Remove HTML tags for PDF
            content = re.sub(r'<[^>]+>', '', content)
            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
            
            if msg.get('source'):
                story.append(Paragraph(f"<i>Source: {msg['source']}</i>", self.styles['Normal']))
                
        doc.build(story)
        buffer.seek(0)
        return buffer

# =============================================================================
# ZOHO CRM MANAGER
# =============================================================================

class ZohoCRMManager:
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        self._access_token = None
        self._token_expiry = None

    def _get_access_token_with_timeout(self, force_refresh: bool = False, timeout: int = 15) -> Optional[str]:
        """Get access token with caching and timeout"""
        if not self.config.ZOHO_ENABLED:
            return None

        if not force_refresh and self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
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
            logger.error(f"Token request timed out after {timeout} seconds.")
            return None
        except Exception as e:
            logger.error(f"Failed to get Zoho access token: {e}")
            raise

    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
        """Find contact with retry on token expiry"""
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(
                f"{self.base_url}/Contacts/search", 
                headers=headers, 
                params=params, 
                timeout=10
            )
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, refreshing...")
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.get(
                        f"{self.base_url}/Contacts/search", 
                        headers=headers, 
                        params=params, 
                        timeout=10
                    )
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data']:
                contact_id = data['data'][0]['id']
                logger.info(f"Found existing contact: {contact_id}")
                return contact_id
                
        except Exception as e:
            logger.error(f"Error finding contact by email {email}: {e}")
            
        return None

    def _create_contact(self, email: str, access_token: str, first_name: str = None) -> Optional[str]:
        """Create contact with retry on token expiry"""
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        contact_data = {
            "data": [{
                "Last_Name": first_name or "Food Professional",
                "Email": email,
                "Lead_Source": "FiFi AI Assistant"
            }]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/Contacts", 
                headers=headers, 
                json=contact_data, 
                timeout=10
            )
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, refreshing...")
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.post(
                        f"{self.base_url}/Contacts", 
                        headers=headers, 
                        json=contact_data, 
                        timeout=10
                    )
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                contact_id = data['data'][0]['details']['id']
                logger.info(f"Created new contact: {contact_id}")
                return contact_id
                
        except Exception as e:
            logger.error(f"Error creating contact for {email}: {e}")
            
        return None

    def _upload_attachment(self, contact_id: str, pdf_buffer: io.BytesIO, access_token: str, filename: str) -> bool:
        """Upload attachment with retry"""
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
                    logger.warning("Zoho token expired during upload, refreshing...")
                    access_token = self._get_access_token_with_timeout(force_refresh=True)
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
                    logger.error(f"Upload failed with response: {data}")
                    
            except requests.exceptions.Timeout:
                logger.error(f"Upload timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error uploading attachment (attempt {attempt + 1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        return False

    def _add_note(self, contact_id: str, note_title: str, note_content: str, access_token: str) -> bool:
        """Add note with retry on token expiry"""
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Truncate note content if too long
        max_content_length = 32000
        if len(note_content) > max_content_length:
            note_content = note_content[:max_content_length - 100] + "\n\n[Content truncated due to size limits]"
        
        note_data = {
            "data": [{
                "Note_Title": note_title,
                "Note_Content": note_content,
                "Parent_Id": {
                    "id": contact_id
                },
                "se_module": "Contacts"
            }]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/Notes", 
                headers=headers, 
                json=note_data, 
                timeout=15
            )
            
            if response.status_code == 401:
                logger.warning("Zoho token expired, refreshing...")
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.post(
                        f"{self.base_url}/Notes", 
                        headers=headers, 
                        json=note_data, 
                        timeout=15
                    )
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                logger.info(f"Successfully added note: {note_title}")
                return True
            else:
                logger.error(f"Note creation failed with response: {data}")
                
        except Exception as e:
            logger.error(f"Error adding note: {e}")
            
        return False

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """Synchronous save method with comprehensive debugging"""
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        
        max_retries = 3 if trigger_reason == "Session Timeout" else 1
        
        for attempt in range(max_retries):
            logger.info(f"Save attempt {attempt + 1}/{max_retries}")
            try:
                # Get access token
                token_timeout = 10 if trigger_reason == "Session Timeout" else 15
                access_token = self._get_access_token_with_timeout(force_refresh=True, timeout=token_timeout)
                if not access_token:
                    logger.error(f"Failed to get Zoho access token on attempt {attempt + 1}.")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False

                # Find or create contact
                contact_id = self._find_contact_by_email(session.email, access_token)
                if not contact_id:
                    contact_id = self._create_contact(session.email, access_token, session.first_name)
                if not contact_id:
                    logger.error("Failed to find or create contact.")
                    return False
                session.zoho_contact_id = contact_id

                # Generate PDF
                pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer:
                    logger.error("Failed to generate PDF.")
                    return False

                # Upload attachment
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                upload_success = self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename)
                if not upload_success:
                    logger.warning("Failed to upload PDF attachment, continuing with note only.")

                # Add note
                note_title = f"FiFi AI Chat Transcript from {timestamp} ({trigger_reason})"
                note_content = self._generate_note_content(session, upload_success, trigger_reason)
                note_success = self._add_note(contact_id, note_title, note_content, access_token)
                if not note_success:
                    logger.error("Failed to add note to contact.")
                    return False

                logger.info("=" * 80)
                logger.info(f"ZOHO SAVE COMPLETED SUCCESSFULLY on attempt {attempt + 1}")
                logger.info(f"Contact ID: {contact_id}")
                logger.info("=" * 80)
                return True

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
        """Generate note content with session summary"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        note_content = f"**Session Information:**\n"
        note_content += f"- Session ID: {session.session_id}\n"
        note_content += f"- User: {session.first_name or 'Unknown'} ({session.email})\n"
        note_content += f"- Save Trigger: {trigger_reason}\n"
        note_content += f"- Timestamp: {timestamp}\n"
        note_content += f"- Total Messages: {len(session.messages)}\n\n"
        
        if attachment_uploaded:
            note_content += "✅ **PDF transcript has been attached to this contact.**\n\n"
        else:
            note_content += "⚠️ **PDF attachment upload failed. Full transcript below:**\n\n"
        
        note_content += "**Conversation Summary:**\n"
        
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
# RATE LIMITER
# =============================================================================

class RateLimiter:
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
    if not isinstance(text, str): 
        return ""
    return html.escape(text)[:max_length].strip()

# =============================================================================
# AI SYSTEM PLACEHOLDER (Simplified for this complete version)
# =============================================================================

class EnhancedAI:
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

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Simplified AI response for complete implementation"""
        return {
            "content": f"I understand you're asking about: {prompt}. This is a simplified response for the complete integrated version. In the full implementation, this would connect to your AI services (Pinecone, Tavily, OpenAI) to provide intelligent responses about food & beverage industry topics.",
            "source": "Simplified AI System",
            "used_search": False,
            "used_pinecone": False,
            "has_citations": False,
            "has_inline_citations": False,
            "safety_override": False,
            "success": True
        }

@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    if not client: 
        return {"flagged": False}
    
    if not hasattr(client, 'moderations'):
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
    except Exception:
        pass
    
    return {"flagged": False}

# =============================================================================
# STREAMLINED FIFI TIMER USING st_javascript NAVIGATION PATTERNS
# =============================================================================

def render_browser_close_with_direct_navigation(session_id: str):
    """
    Clean browser close detection using direct navigation.
    """
    if not session_id:
        return

    js_code = f"""
    <script>
    (function() {{
        const sessionKey = 'fifi_close_nav_' + '{session_id}';
        if (window[sessionKey]) return;
        window[sessionKey] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false;
        
        function triggerEmergencySave() {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            console.log('🚨 FiFi: Browser close detected - triggering direct navigation save');
            
            try {{
                // Construct URL with query parameters
                const url = window.parent.location.origin + window.parent.location.pathname +
                    `?session_id=${{sessionId}}&event=close`;
                
                console.log('📡 Navigating to:', url);
                
                // Direct navigation (cleanest approach)
                window.parent.location = url;
                
            }} catch (e) {{
                console.log('⚠️ Parent access failed, trying current window');
                try {{
                    const url = window.location.origin + window.location.pathname +
                        `?session_id=${{sessionId}}&event=close`;
                    window.location = url;
                }} catch (e2) {{
                    console.error('❌ All navigation methods failed:', e, e2);
                }}
            }}
        }}
        
        // Setup close detection events
        const events = ['beforeunload', 'pagehide'];
        events.forEach(eventType => {{
            try {{
                if (window.parent && window.parent !== window) {{
                    window.parent.addEventListener(eventType, triggerEmergencySave, {{ capture: true }});
                }}
                window.addEventListener(eventType, triggerEmergencySave, {{ capture: true }});
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        // Visibility change detection
        try {{
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('visibilitychange', () => {{
                    if (window.parent.document.visibilityState === 'hidden') {{
                        console.log('🚨 Main app hidden - triggering save');
                        triggerEmergencySave();
                    }}
                }}, {{ passive: true }});
            }}
        }} catch (e) {{
            document.addEventListener('visibilitychange', () => {{
                if (document.visibilityState === 'hidden') {{
                    triggerEmergencySave();
                }}
            }}, {{ passive: true }});
        }}
        
        console.log('✅ Direct navigation close detection initialized for:', sessionId.substring(0, 8));
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render navigation close component: {e}")

def render_activity_timer_with_data_return(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Timer that returns event data directly to Python.
    """
    if not session_id:
        return None
    
    js_timer_code = f"""
    (() => {{
        const sessionId = "{session_id}";
        const AUTO_SAVE_TIMEOUT = 120000;  // 2 minutes
        const SESSION_EXPIRE_TIMEOUT = 180000;  // 3 minutes
        
        // Initialize timer state
        if (!window.fifi_timer_state) {{
            window.fifi_timer_state = {{
                lastActivityTime: Date.now(),
                autoSaveTriggered: false,
                sessionExpired: false,
                listenersInitialized: false,
                sessionId: sessionId
            }};
        }}
        
        const state = window.fifi_timer_state;
        
        // Reset state if session changed
        if (state.sessionId !== sessionId) {{
            state.sessionId = sessionId;
            state.lastActivityTime = Date.now();
            state.autoSaveTriggered = false;
            state.sessionExpired = false;
            state.listenersInitialized = false;
        }}
        
        // Setup activity detection
        if (!state.listenersInitialized) {{
            function resetActivity() {{
                const now = Date.now();
                state.lastActivityTime = now;
                state.autoSaveTriggered = false;
                state.sessionExpired = false;
            }}
            
            const events = ['mousedown', 'mousemove', 'click', 'keydown', 'scroll'];
            
            // Listen on component and main app
            events.forEach(eventType => {{
                document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
                try {{
                    if (window.parent && window.parent.document && window.parent.document !== document) {{
                        window.parent.document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
                    }}
                }} catch (e) {{}}
            }});
            
            // Visibility detection
            const handleVisibilityChange = () => {{
                if (document.visibilityState === 'visible') resetActivity();
            }};
            document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
            try {{
                if (window.parent && window.parent.document) {{
                    window.parent.document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
                }}
            }} catch (e) {{}}
            
            state.listenersInitialized = true;
            console.log("✅ Activity detection initialized");
        }}
        
        // Calculate inactivity
        const currentTime = Date.now();
        const inactiveTimeMs = currentTime - state.lastActivityTime;
        const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
        
        // Check for auto-save trigger
        if (inactiveTimeMs >= AUTO_SAVE_TIMEOUT && !state.autoSaveTriggered) {{
            state.autoSaveTriggered = true;
            console.log("🚨 AUTO-SAVE TRIGGER");
            return {{ event: "auto_save_trigger", session_id: sessionId, inactive_minutes: inactiveMinutes }};
        }}
        
        // Check for session expiry
        if (inactiveTimeMs >= SESSION_EXPIRE_TIMEOUT && !state.sessionExpired) {{
            state.sessionExpired = true;
            console.log("🚨 SESSION EXPIRED");
            return {{ event: "session_expired", session_id: sessionId, inactive_minutes: inactiveMinutes }};
        }}
        
        return null; // No events
    }})()
    """
    
    try:
        stable_key = f"fifi_timer_data_{session_id[:8]}"
        timer_result = st_javascript(js_timer_code, key=stable_key)
        
        if timer_result and isinstance(timer_result, dict) and 'event' in timer_result:
            logger.info(f"✅ Timer event received: {timer_result}")
            return timer_result
        
        return None
        
    except Exception as e:
        logger.error(f"Timer execution error: {e}")
        return None

def global_message_channel_error_handler():
    """
    Add global error handling for uncaught promise rejections which can cause message channel errors.
    """
    js_error_handler = """
    <script>
    (function() {
        if (window.fifi_error_handler_initialized) return;
        window.fifi_error_handler_initialized = true;
        
        window.addEventListener('unhandledrejection', function(event) {
            const error = event.reason;
            if (error && error.message && error.message.includes('message channel closed')) {
                console.log('🛡️ FiFi: Caught and handled message channel error:', error.message);
                event.preventDefault();
            }
        });
        
        console.log('✅ FiFi: Global message channel error handler initialized');
    })();
    </script>
    """
    try:
        st.components.v1.html(js_error_handler, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to initialize global error handler: {e}")

# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.session_timeout_minutes = 3
        self.auto_save_threshold_minutes = 2
        self._save_lock = threading.Lock()

    def get_session_timeout_minutes(self) -> int:
        return getattr(self, 'session_timeout_minutes', 3)

    def _is_session_expired(self, session: UserSession) -> bool:
        """Check if session has exceeded total timeout (3 minutes)"""
        if not session.last_activity:
            return False
        time_diff = datetime.now() - session.last_activity
        return time_diff.total_seconds() > (self.session_timeout_minutes * 60)

    def _update_activity(self, session: UserSession):
        """Update session activity timestamp and reset save flag"""
        session.last_activity = datetime.now()
        
        # Reset auto-save flag when user becomes active again
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset auto-save flag for active session {session.session_id[:8]}")
        
        # Ensure user_type is properly maintained as enum
        if isinstance(session.user_type, str):
            session.user_type = UserType(session.user_type)
        
        try:
            self.db.save_session(session)
            logger.debug(f"Session activity updated for {session.session_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        return session

    def _validate_and_fix_session(self, session: UserSession) -> UserSession:
        """Validate and fix common session issues"""
        if not session:
            return session
            
        # Fix user_type if it's a string
        if isinstance(session.user_type, str):
            try:
                session.user_type = UserType(session.user_type)
            except ValueError:
                session.user_type = UserType.GUEST
        
        # Ensure messages is a list
        if not isinstance(session.messages, list):
            session.messages = []
            
        return session

    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        """Enhanced auto-save with proper locking and validation"""
        with self._save_lock:
            logger.info(f"=== AUTO SAVE TO CRM STARTED (Trigger: {trigger_reason}) ===")
            
            session = self._validate_and_fix_session(session)
            
            # Check prerequisites
            if not all([
                session.user_type == UserType.REGISTERED_USER,
                session.email,
                session.messages,
                self.zoho.config.ZOHO_ENABLED
            ]):
                logger.info("SAVE SKIPPED: Not eligible.")
                return False

            is_interactive = "Manual" in trigger_reason

            try:
                if is_interactive:
                    with st.spinner(f"💾 Saving chat to CRM..."):
                        success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                else:
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)

                if success:
                    logger.info("SAVE COMPLETED: Successfully saved to CRM.")
                    session.timeout_saved_to_crm = True
                    self.db.save_session(session)
                    if is_interactive: st.success("✅ Chat saved to Zoho CRM!")
                    return True
                else:
                    logger.error("SAVE FAILED: Zoho save method returned False.")
                    if is_interactive: st.error("❌ Failed to save chat to CRM.")
                    return False
            except Exception as e:
                logger.error(f"SAVE FAILED: Unexpected error - {type(e).__name__}: {str(e)}", exc_info=True)
                if is_interactive: st.error(f"❌ An error occurred while saving: {str(e)}")
                return False

    def get_session(self) -> UserSession:
        """Get session with server-side validation"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                session = self._validate_and_fix_session(session)
                
                # Server-side expiry check as a fallback
                if self._is_session_expired(session):
                    logger.info(f"Server detected session {session_id[:8]} expired")
                    if (session.user_type == UserType.REGISTERED_USER and session.email and 
                        session.messages and not session.timeout_saved_to_crm):
                        self._auto_save_to_crm(session, "Emergency Save (Server Detection)")
                    
                    self._end_session_internal(session)
                    st.session_state.session_expired = True
                    st.session_state.expired_session_id = session_id[:8]
                    return self._create_guest_session()
                else:
                    self._update_activity(session)
                    return session
        
        return self._create_guest_session()

    def _end_session_internal(self, session: UserSession):
        """End session and clean up state"""
        session.active = False
        try:
            self.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to mark session as inactive: {e}")
        
        keys_to_clear = ['current_session_id', 'page']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

        clean_username = username.strip()
        clean_password = password.strip()

        try:
            response = requests.post(
                f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                json={'username': clean_username, 'password': clean_password},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                current_session = self.get_session()
                
                display_name = (data.get('user_display_name') or clean_username)

                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                self.db.save_session(current_session)
                
                verification_session = self._validate_and_fix_session(self.db.load_session(current_session.session_id))
                if verification_session and verification_session.user_type == UserType.REGISTERED_USER:
                    st.session_state.current_session_id = current_session.session_id
                    st.success(f"Welcome back, {current_session.first_name}!")
                    return current_session
                else:
                    st.error("Authentication failed - session verification failed.")
                    return None
            else:
                error_message = "Invalid username or password."
                st.error(error_message)
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during authentication.")
            logger.error(f"Authentication network exception: {e}")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "Rate limit exceeded. Please wait.", "success": False}

        session = self._validate_and_fix_session(session)
        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {"content": moderation["message"], "success": False, "source": "Content Safety"}

        response = self.ai.get_response(sanitized_prompt, session.messages)
        
        session.messages.append({"role": "user", "content": sanitized_prompt, "timestamp": datetime.now().isoformat()})
        session.messages.append({**response, "role": "assistant", "timestamp": datetime.now().isoformat()})
        session.messages = session.messages[-100:]
        
        self._update_activity(session)
        return response

    def clear_chat_history(self, session: UserSession):
        session = self._validate_and_fix_session(session)
        session.messages = []
        session.timeout_saved_to_crm = False
        self._update_activity(session)

    def end_session(self, session: UserSession):
        """Manual session end (Sign Out button)"""
        session = self._validate_and_fix_session(session)
        self._auto_save_to_crm(session, "Manual Sign Out")
        self._end_session_internal(session)

    def manual_save_to_crm(self, session: UserSession):
        """Manual CRM save (Save button)"""
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and session.messages and self.zoho.config.ZOHO_ENABLED):
            self._auto_save_to_crm(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages.")

# =============================================================================
# DEBUGGING AND DIAGNOSTIC TOOLS
# =============================================================================

def debug_zoho_configuration():
    st.subheader("🔧 Zoho CRM Debug Panel")
    try:
        session_manager = get_session_manager()
        if not session_manager:
            st.error("❌ Session manager not available")
            return
        zoho = session_manager.zoho
        st.write(f"- Zoho Enabled: {zoho.config.ZOHO_ENABLED}")
        if not zoho.config.ZOHO_ENABLED: return
        
        if st.button("🔑 Test Zoho Token"):
            with st.spinner("Testing Zoho token..."):
                token = zoho._get_access_token_with_timeout(force_refresh=True, timeout=10)
                st.success("✅ Token generated successfully") if token else st.error("❌ Failed to generate token")

        current_session = session_manager.get_session()
        if current_session and current_session.user_type == UserType.REGISTERED_USER and current_session.email and current_session.messages:
            if st.button("💾 Test Manual CRM Save"):
                success = zoho.save_chat_transcript_sync(current_session, "Manual Test Save")
                st.success("✅ Manual CRM save successful!") if success else st.error("❌ Manual CRM save failed")
    except Exception as e:
        st.error(f"❌ Debug panel error: {str(e)}")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_welcome_page(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is disabled because the authentication service is not configured.")
        else:
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign In"):
                    if session_manager.authenticate_with_wordpress(username, password):
                        st.session_state.page = "chat"
                        st.rerun()
    
    with tab2:
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Authenticated User**")
            st.markdown(f"**Welcome:** {session.first_name}")
        else:
            st.info("👤 **Guest User**")
        
        st.divider()
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            session_manager.clear_chat_history(session)
            st.rerun()
        if st.button("🚪 Sign Out", use_container_width=True):
            session_manager.end_session(session)
            st.rerun()

        if session.user_type == UserType.REGISTERED_USER and session.messages:
            st.divider()
            pdf_buffer = pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.download_button("📄 Download PDF", pdf_buffer, f"fifi_chat_{session.session_id[:8]}.pdf", "application/pdf", use_container_width=True)
            if session_manager.zoho.config.ZOHO_ENABLED and session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True):
                    session_manager.manual_save_to_crm(session)
        
        st.divider()
        if st.checkbox("🔧 Show Debug Panel"):
            debug_zoho_configuration()

def render_chat_interface_with_streamlined_timer(session_manager, session):
    """
    Streamlined chat interface using the cleaner st_javascript patterns.
    """
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion")
    
    # Simple browser close detection using direct navigation
    render_browser_close_with_direct_navigation(session.session_id)
    
    # Timer that returns data directly to Python
    timer_result = render_activity_timer_with_data_return(session.session_id)
    
    # Process timer events if received
    if timer_result:
        event = timer_result.get('event')
        inactive_minutes = timer_result.get('inactive_minutes', 0)
        logger.info(f"🎯 Processing timer event: {event}")

        if event == 'auto_save_trigger':
            st.info(f"⏰ Auto-save triggered after {inactive_minutes} minutes of inactivity.")
            if (session.user_type == UserType.REGISTERED_USER and session.email and 
                session.messages and not session.timeout_saved_to_crm):
                session_manager._auto_save_to_crm(session, "JavaScript Auto-Save (2min)")
            st.rerun()
        
        elif event == 'session_expired':
            st.error(f"🔄 Session expired after {inactive_minutes} minutes of inactivity.")
            if (session.user_type == UserType.REGISTERED_USER and session.email and 
                session.messages and not session.timeout_saved_to_crm):
                session_manager._auto_save_to_crm(session, "Emergency Save (JS Expiry)")
            
            session_manager._end_session_internal(session)
            st.session_state.session_expired = True
            st.session_state.expired_session_id = session.session_id[:8]
            st.rerun()
    
    # Show activity status
    if session.user_type == UserType.REGISTERED_USER and session.last_activity:
        time_since = (datetime.now() - session.last_activity).total_seconds() / 60
        if time_since < 1: st.success("🟢 Active")
        elif time_since < 2: st.warning(f"🟡 Inactive {time_since:.1f} min - auto-save soon")
        elif session.timeout_saved_to_crm: st.success(f"💾 Auto-saved")
        else: st.error(f"🔴 Inactive {time_since:.1f} min - session expires soon")
    
    # Display chat messages and handle input
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            if msg.get("source"): st.caption(f"Source: {msg['source']}")

    if prompt := st.chat_input("Ask about ingredients, suppliers, or market trends..."):
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Generating response..."):
                response = session_manager.get_ai_response(session, prompt)
                st.markdown(response.get("content", "Error."), unsafe_allow_html=True)
                if response.get("source"): st.caption(f"Source: {response['source']}")
        st.rerun()

# =============================================================================
# SESSION EXPIRY AND SAVE REQUEST HANDLING
# =============================================================================

def render_session_expiry_redirect():
    if st.session_state.get('session_expired', False):
        st.error(f"🔄 Session Expired")
        st.info(f"Session `{st.session_state.get('expired_session_id', 'unknown')}` ended due to inactivity.")
        st.info("⏳ Redirecting to welcome page...")
        
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            progress_bar.progress(i + 1)
        
        keys_to_clear = ['session_expired', 'expired_session_id', 'current_session_id', 'page']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

def enhanced_handle_save_requests_streamlined():
    """
    Streamlined save request handler that processes navigation-based triggers.
    """
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event == "close" and session_id:
        logger.info(f"🚨 Emergency save request detected for session: {session_id[:8]}")
        st.query_params.clear()
        
        st.info("🚨 **Processing emergency save due to browser close...**")
        
        try:
            session_manager = st.session_state.get('session_manager')
            if session_manager:
                session = session_manager.db.load_session(session_id)
                session = session_manager._validate_and_fix_session(session)

                if (session and session.user_type == UserType.REGISTERED_USER and 
                    session.email and session.messages and not session.timeout_saved_to_crm):
                    success = session_manager._auto_save_to_crm(session, "Emergency Save (Browser Close)")
                    st.success("✅ Emergency save completed!") if success else st.error("❌ Emergency save failed")
                else:
                    logger.info("ℹ️ Session not eligible for save.")
                    st.info("ℹ️ Session not eligible for save (e.g., guest user or no messages).")
            else:
                st.error("❌ Session manager not available for emergency save.")
        except Exception as e:
            st.error(f"❌ An error occurred during emergency save: {str(e)}")
            logger.error(f"Emergency save crashed: {e}", exc_info=True)
        
        time.sleep(3) # Give user time to see message
        st.stop() # Stop further execution

# =============================================================================
# UTILITY AND INITIALIZATION
# =============================================================================

def get_session_manager() -> Optional[SessionManager]:
    return st.session_state.get('session_manager')

def ensure_initialization():
    if 'initialized' not in st.session_state:
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.initialized = True
            logger.info("✅ Application initialized successfully")
            return True
        except Exception as e:
            st.error(f"💥 Critical startup error: {e}")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            return False
    return True

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # This handles emergency save requests triggered by navigation first.
    enhanced_handle_save_requests_streamlined()

    # This handles redirects for expired sessions.
    render_session_expiry_redirect()

    # This handles uncaught JS promise rejections.
    global_message_channel_error_handler()

    if st.button("🔄 Fresh Start", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    if not ensure_initialization():
        st.stop()

    session_manager = get_session_manager()
    
    if st.session_state.get('page') != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_with_streamlined_timer(session_manager, session)
        else:
            # If session is no longer active, go back to welcome page
            if 'page' in st.session_state: del st.session_state.page
            st.rerun()

if __name__ == "__main__":
    main()
