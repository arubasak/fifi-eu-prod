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
# FINAL INTEGRATED VERSION - ALL FEATURES COMBINED WITH CORRECTED TIMER
# - JavaScript timer with corrected st_javascript patterns using IIFE
# - Python CRM save processing
# - window.parent.location for all reloads
# - SQLite Cloud database integration
# - All existing features preserved
# - Complete error handling and validation
# - Fixed timer return value issues
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
# FIXED TIMER COMPONENT - ADDRESSES st_javascript RETURN VALUE ISSUES
# =============================================================================
# The timer functions in the existing code have been replaced with these corrected versions

def render_activity_timer_component(session_id: str, session_manager) -> Optional[Dict[str, Any]]:
    """
    FIXED JavaScript timer - handles st_javascript return value issues properly
    
    Fixes:
    - Returns explicit null instead of 0/false/undefined
    - Better validation for falsy values from st_javascript  
    - More stable component key to prevent unregistration
    - Enhanced error handling
    """
    
    if not session_id:
        return None
    
    # More explicit JavaScript with guaranteed null return
    js_timer_code = f"""
    (() => {{
        const sessionId = "{session_id}";
        const AUTO_SAVE_TIMEOUT = 120000;  // 2 minutes
        const SESSION_EXPIRE_TIMEOUT = 180000;  // 3 minutes
        
        console.log("🕐 FiFi Timer checking session:", sessionId.substring(0, 8));
        
        // Initialize timer state with more defensive programming
        if (typeof window.fifi_timer_state === 'undefined' || window.fifi_timer_state === null) {{
            window.fifi_timer_state = {{
                lastActivityTime: Date.now(),
                autoSaveTriggered: false,
                sessionExpired: false,
                listenersInitialized: false,
                sessionId: sessionId
            }};
            console.log("🆕 FiFi Timer state initialized");
        }}
        
        const state = window.fifi_timer_state;
        
        // Reset state if session changed
        if (state.sessionId !== sessionId) {{
            console.log("🔄 Session changed, resetting timer state");
            state.sessionId = sessionId;
            state.lastActivityTime = Date.now();
            state.autoSaveTriggered = false;
            state.sessionExpired = false;
            state.listenersInitialized = false;
        }}
        
        // Initialize activity listeners once per session
        if (!state.listenersInitialized) {{
            console.log("👂 Setting up FiFi activity listeners");
            
            function resetActivity() {{
                const now = Date.now();
                state.lastActivityTime = now;
                state.autoSaveTriggered = false;
                state.sessionExpired = false;
                console.log("🔄 Activity detected:", new Date(now).toLocaleTimeString());
            }}
            
            // Comprehensive activity events
            const events = [
                'mousedown', 'mousemove', 'mouseup', 'click', 'dblclick',
                'keydown', 'keyup', 'keypress',
                'scroll', 'wheel',
                'touchstart', 'touchmove', 'touchend',
                'focus', 'blur'
            ];
            
            // Add to current document
            let listenersAdded = 0;
            events.forEach(eventType => {{
                try {{
                    document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
                    listenersAdded++;
                }} catch (e) {{
                    console.debug("Failed to add listener:", eventType, e);
                }}
            }});
            
            // Add to parent document (Streamlit app) if accessible
            try {{
                if (window.parent && 
                    window.parent.document && 
                    window.parent.document !== document) {{
                    
                    events.forEach(eventType => {{
                        try {{
                            window.parent.document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
                            listenersAdded++;
                        }} catch (e) {{
                            console.debug("Failed to add parent listener:", eventType, e);
                        }}
                    }});
                    console.log("👂 Parent document listeners added");
                }}
            }} catch (e) {{
                console.debug("Cannot access parent document for activity detection:", e);
            }}
            
            // Visibility change detection
            const handleVisibilityChange = () => {{
                try {{
                    if (document.visibilityState === 'visible') {{
                        resetActivity();
                    }}
                }} catch (e) {{
                    console.debug("Visibility change error:", e);
                }}
            }};
            
            try {{
                document.addEventListener('visibilitychange', handleVisibilityChange);
                if (window.parent && window.parent.document) {{
                    window.parent.document.addEventListener('visibilitychange', handleVisibilityChange);
                }}
            }} catch (e) {{
                console.debug("Cannot setup visibility detection:", e);
            }}
            
            state.listenersInitialized = true;
            console.log(`✅ FiFi activity listeners initialized (${{listenersAdded}} total)`);
        }}
        
        // Calculate current inactivity
        const currentTime = Date.now();
        const inactiveTimeMs = currentTime - state.lastActivityTime;
        const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
        const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
        
        console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
        
        // Check for auto-save trigger (2 minutes)
        if (inactiveTimeMs >= AUTO_SAVE_TIMEOUT && !state.autoSaveTriggered) {{
            state.autoSaveTriggered = true;
            console.log("🚨 AUTO-SAVE TRIGGER ACTIVATED");
            
            const result = {{
                event: "auto_save_trigger",
                session_id: sessionId,
                inactive_time_ms: inactiveTimeMs,
                inactive_minutes: inactiveMinutes,
                inactive_seconds: inactiveSeconds,
                timestamp: currentTime
            }};
            console.log("📤 Returning auto-save result:", result);
            return result;
        }}
        
        // Check for session expiry (3 minutes)
        if (inactiveTimeMs >= SESSION_EXPIRE_TIMEOUT && !state.sessionExpired) {{
            state.sessionExpired = true;
            console.log("🚨 SESSION EXPIRED");
            
            const result = {{
                event: "session_expired",
                session_id: sessionId,
                inactive_time_ms: inactiveTimeMs,
                inactive_minutes: inactiveMinutes,
                inactive_seconds: inactiveSeconds,
                timestamp: currentTime
            }};
            console.log("📤 Returning expiry result:", result);
            return result;
        }}
        
        // EXPLICITLY return null - not 0, false, or undefined
        console.log("📤 No events triggered, returning null");
        return null;
    }})()
    """
    
    try:
        # More stable key that includes app session info
        app_session_hash = hash(str(st.session_state.get('current_session_id', 'default'))) % 10000
        stable_key = f"fifi_timer_{session_id[:8]}_{app_session_hash}"
        
        # Execute JavaScript and get result
        timer_result = st_javascript(js_timer_code, key=stable_key)
        
        # Enhanced validation for various falsy values from st_javascript
        logger.info(f"⏰ Raw timer result: {timer_result} (type: {type(timer_result)})")
        
        # Handle various falsy values that st_javascript might return
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            logger.debug("No timer events triggered (falsy value returned)")
            return None
        
        # Validate result structure
        if isinstance(timer_result, dict) and 'event' in timer_result:
            event = timer_result.get('event')
            received_session_id = timer_result.get('session_id')
            
            # Verify session ID matches
            if received_session_id == session_id:
                logger.info(f"✅ Valid timer event: {event} for session {session_id[:8]}")
                return timer_result
            else:
                logger.warning(f"⚠️ Session ID mismatch: expected {session_id[:8]}, got {received_session_id[:8] if received_session_id else 'None'}")
                return None
        else:
            logger.warning(f"⚠️ Invalid timer result structure: {timer_result} (type: {type(timer_result)})")
            return None
        
    except Exception as e:
        logger.error(f"❌ JavaScript timer execution error: {e}")
        # Don't spam user with timer errors unless critical
        return None

def handle_timer_event(timer_result: Dict[str, Any], session_manager, session) -> bool:
    """
    FIXED timer event handler with better error handling and validation
    """
    
    if not timer_result or not isinstance(timer_result, dict):
        return False
    
    event = timer_result.get('event')
    session_id = timer_result.get('session_id')
    inactive_minutes = timer_result.get('inactive_minutes', 0)
    
    logger.info(f"🎯 Processing timer event: {event} for session {session_id[:8] if session_id else 'unknown'}")
    
    try:
        # Import UserType to ensure it's available
        from enum import Enum
        
        class UserType(Enum):
            GUEST = "guest"
            REGISTERED_USER = "registered_user"
        
        # Ensure session has proper UserType
        if hasattr(session, 'user_type'):
            if isinstance(session.user_type, str):
                session.user_type = UserType(session.user_type)
        else:
            logger.error("Session missing user_type attribute")
            return False
        
        if event == 'auto_save_trigger':
            # Handle auto-save
            st.info(f"⏰ **Auto-save triggered** after {inactive_minutes} minutes of inactivity")
            
            # Check if session is eligible for auto-save
            if (session.user_type == UserType.REGISTERED_USER and 
                hasattr(session, 'email') and session.email and 
                hasattr(session, 'messages') and session.messages and
                hasattr(session, 'timeout_saved_to_crm') and not session.timeout_saved_to_crm):
                
                with st.spinner("💾 Auto-saving chat to CRM..."):
                    try:
                        save_success = session_manager._auto_save_to_crm(session, "JavaScript Auto-Save (2min)")
                    except Exception as e:
                        logger.error(f"Auto-save failed: {e}")
                        save_success = False
                
                if save_success:
                    st.success("✅ Chat automatically saved to CRM!")
                    # Update session activity to prevent immediate re-trigger
                    session.last_activity = datetime.now()
                    try:
                        session_manager.db.save_session(session)
                    except Exception as e:
                        logger.error(f"Failed to update session after auto-save: {e}")
                else:
                    st.warning("⚠️ Auto-save failed, but session continues")
                
                return True  # Rerun to refresh timer state
            else:
                st.info("ℹ️ Auto-save skipped (not eligible)")
                logger.info(f"Auto-save eligibility check failed: user_type={session.user_type}, email={getattr(session, 'email', None)}, messages={len(getattr(session, 'messages', []))}, saved={getattr(session, 'timeout_saved_to_crm', None)}")
                return False
                
        elif event == 'session_expired':
            # Handle session expiry
            st.error(f"🔄 **Session expired** after {inactive_minutes} minutes of inactivity")
            
            # Emergency save if eligible
            if (session.user_type == UserType.REGISTERED_USER and 
                hasattr(session, 'email') and session.email and 
                hasattr(session, 'messages') and session.messages and
                hasattr(session, 'timeout_saved_to_crm') and not session.timeout_saved_to_crm):
                
                st.info("💾 Performing emergency save before expiry...")
                try:
                    session_manager._auto_save_to_crm(session, "Emergency Save (JavaScript Expiry)")
                    st.success("✅ Emergency save completed")
                except Exception as e:
                    st.error(f"❌ Emergency save failed: {str(e)}")
                    logger.error(f"Emergency save failed: {e}")
            
            # End the session
            try:
                session_manager._end_session_internal(session)
            except Exception as e:
                logger.error(f"Failed to end session: {e}")
            
            # Set expiry flags for UI
            st.session_state.session_expired = True
            st.session_state.expired_session_id = session_id[:8] if session_id else 'unknown'
            st.session_state.expiry_trigger = "javascript_timer"
            
            st.info("⏳ Redirecting to welcome page in 3 seconds...")
            time.sleep(3)
            return True  # Rerun to redirect
            
        else:
            logger.warning(f"⚠️ Unknown timer event: {event}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing timer event {event}: {e}", exc_info=True)
        st.error(f"⚠️ Timer event processing error: {str(e)}")
        return False

def render_browser_close_component(session_id: str):
    """
    FIXED browser close detection with better error handling
    """
    if not session_id:
        return

    # More defensive JavaScript for browser close detection
    js_code = f"""
    <script>
    (function() {{
        // Prevent multiple initializations
        if (window.fifi_close_listener_added_for_session === '{session_id}') return;
        window.fifi_close_listener_added_for_session = '{session_id}';
        
        const sessionId = '{session_id}';
        let parentUrl = '';
        
        try {{
            parentUrl = window.parent.location.origin + window.parent.location.pathname;
        }} catch (e) {{
            try {{
                parentUrl = window.location.origin + window.location.pathname;
            }} catch (e2) {{
                console.warn('FiFi: Cannot determine URL for emergency save');
                return;
            }}
        }}
        
        let saveTriggered = false;

        function sendEmergencySave() {{
            if (!saveTriggered) {{
                saveTriggered = true;
                console.log('🚨 FiFi: Browser close detected - emergency save');
                
                const url = `${{parentUrl}}?event=close&session_id=${{sessionId}}`; 
                
                try {{
                    if (navigator.sendBeacon) {{
                        const success = navigator.sendBeacon(url);
                        console.log('📡 FiFi: Emergency save beacon sent:', success);
                    }} else {{
                        const xhr = new XMLHttpRequest();
                        xhr.open('GET', url, false);
                        xhr.send();
                        console.log('📡 FiFi: Emergency save XHR sent');
                    }}
                }} catch (e) {{
                    console.error('❌ FiFi: Emergency save failed:', e);
                }}
            }}
        }}
        
        // Monitor visibility changes with error handling
        function setupVisibilityListeners() {{
            try {{
                if (window.parent && window.parent.document && window.parent.document !== document) {{
                    window.parent.document.addEventListener('visibilitychange', () => {{
                        try {{
                            if (window.parent.document.visibilityState === 'hidden') {{
                                sendEmergencySave();
                            }}
                        }} catch (e) {{
                            console.debug('FiFi: Parent visibility check failed:', e);
                        }}
                    }});
                }}
            }} catch (e) {{
                console.debug('FiFi: Cannot access parent for visibility detection:', e);
                // Fallback to current document
                document.addEventListener('visibilitychange', () => {{
                    try {{
                        if (document.visibilityState === 'hidden') {{
                            sendEmergencySave();
                        }}
                    }} catch (e) {{
                        console.debug('FiFi: Visibility check failed:', e);
                    }}
                }});
            }}
        }}
        
        // Monitor page unload events with error handling
        function setupUnloadListeners() {{
            const events = ['beforeunload', 'pagehide', 'unload'];
            
            events.forEach(eventType => {{
                try {{
                    if (window.parent && window.parent !== window) {{
                        window.parent.addEventListener(eventType, sendEmergencySave, {{capture: true}});
                    }}
                }} catch (e) {{
                    console.debug(`FiFi: Cannot add parent ${{eventType}} listener:`, e);
                }}
                
                try {{
                    window.addEventListener(eventType, sendEmergencySave, {{capture: true}});
                }} catch (e) {{
                    console.debug(`FiFi: Cannot add ${{eventType}} listener:`, e);
                }}
            }});
        }}
        
        // Initialize all listeners
        setupVisibilityListeners();
        setupUnloadListeners();
        
        console.log('✅ FiFi: Browser close detection initialized for session:', sessionId.substring(0, 8));
    }})();
    </script>
    """
    
    try:
        components.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render browser close component: {e}")

def render_activity_status_indicator(session, session_manager):
    """
    FIXED activity status indicator with better error handling
    """
    try:
        # Import UserType to ensure it's available
        from enum import Enum
        
        class UserType(Enum):
            GUEST = "guest"
            REGISTERED_USER = "registered_user"
        
        # Ensure session has proper UserType
        if hasattr(session, 'user_type'):
            if isinstance(session.user_type, str):
                session.user_type = UserType(session.user_type)
        else:
            return  # Can't show status without user type
        
        if (session.user_type == UserType.REGISTERED_USER and 
            hasattr(session, 'last_activity') and session.last_activity):
            
            time_since_activity = datetime.now() - session.last_activity
            minutes_since = time_since_activity.total_seconds() / 60
            
            # Activity status with color-coded indicators
            if minutes_since < 0.5:
                st.success("🟢 **Active** - Auto-save in 2 minutes if inactive")
            elif minutes_since < 1:
                remaining = 2 - minutes_since
                st.info(f"🟡 **Inactive** for {minutes_since:.1f} min - Auto-save in {remaining:.1f} min")
            elif minutes_since < 2:
                remaining = 2 - minutes_since
                if remaining > 0:
                    st.warning(f"🟠 **Inactive** for {minutes_since:.1f} min - Auto-save in {remaining:.1f} min")
                else:
                    st.warning(f"🟠 **Inactive** for {minutes_since:.1f} min - Auto-save will trigger soon")
            elif hasattr(session, 'timeout_saved_to_crm') and session.timeout_saved_to_crm:
                remaining_to_expiry = 3 - minutes_since
                if remaining_to_expiry > 0:
                    st.success(f"💾 **Auto-saved** after {minutes_since:.1f} min - Session expires in {remaining_to_expiry:.1f} min")
                else:
                    st.success(f"💾 **Auto-saved** after {minutes_since:.1f} min")
            else:
                remaining_to_expiry = 3 - minutes_since
                if remaining_to_expiry > 0:
                    st.error(f"🔴 **Inactive** for {minutes_since:.1f} min - Session expires in {remaining_to_expiry:.1f} min")
                else:
                    st.error(f"🔴 **Inactive** for {minutes_since:.1f} min - Session will expire soon")
    
    except Exception as e:
        logger.error(f"Error rendering activity status: {e}")
        # Don't show error to user, just log it

# =============================================================================
# SESSION MANAGER WITH CORRECTED TIMER INTEGRATION
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
                logger.info(f"Fixed user_type conversion for session {session.session_id[:8]}")
            except ValueError:
                logger.error(f"Invalid user_type string: {session.user_type}")
                session.user_type = UserType.GUEST
        
        # Ensure messages is a list
        if not isinstance(session.messages, list):
            session.messages = []
            
        return session

    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        """Enhanced auto-save with proper locking and validation"""
        with self._save_lock:
            logger.info(f"=== AUTO SAVE TO CRM STARTED ===")
            logger.info(f"Trigger: {trigger_reason}")
            logger.info(f"Session ID: {session.session_id[:8] if session.session_id else 'None'}")
            
            # Validate and fix session before proceeding
            session = self._validate_and_fix_session(session)
            
            # Check prerequisites
            if session.user_type != UserType.REGISTERED_USER:
                logger.info(f"SAVE SKIPPED: Not a registered user (current: {session.user_type})")
                return False
            if not session.email:
                logger.info("SAVE SKIPPED: No email address")
                return False
            if not session.messages:
                logger.info("SAVE SKIPPED: No messages to save")
                return False
            if not self.zoho.config.ZOHO_ENABLED:
                logger.info("SAVE SKIPPED: Zoho is not enabled")
                return False

            is_interactive = trigger_reason in ["Manual Sign Out", "Manual Save to Zoho CRM", "Manual Test"]

            try:
                if is_interactive:
                    with st.spinner(f"💾 Saving chat to CRM ({trigger_reason.lower()})..."):
                        success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    
                    if success:
                        st.success("✅ Chat saved to Zoho CRM!")
                        logger.info("SAVE COMPLETED: Interactive save successful")
                        return True
                    else:
                        st.error("❌ Failed to save chat to CRM. Please try again.")
                        logger.error("SAVE FAILED: Interactive save failed")
                        return False
                else:
                    # Non-interactive save (from JavaScript timer)
                    logger.info(f"Starting non-interactive auto-save...")
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    if success:
                        logger.info("SAVE COMPLETED: JavaScript-triggered auto-save successful")
                        # Mark as saved to prevent duplicate saves
                        session.timeout_saved_to_crm = True
                        self.db.save_session(session)
                        return True
                    else:
                        logger.error("SAVE FAILED: JavaScript-triggered auto-save failed")
                        return False

            except Exception as e:
                logger.error(f"SAVE FAILED: Unexpected error - {type(e).__name__}: {str(e)}", exc_info=True)
                if is_interactive:
                    st.error(f"❌ An error occurred while saving: {str(e)}")
                return False
            finally:
                logger.info(f"=== AUTO SAVE TO CRM ENDED ===\n")

    def get_session(self) -> UserSession:
        """Get session with server-side validation"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Validate and fix session data
                session = self._validate_and_fix_session(session)
                
                # Check if session has completely expired server-side (fallback)
                if self._is_session_expired(session):
                    logger.info(f"Server detected session {session_id[:8]} expired")
                    
                    # Emergency save if not already saved
                    if (session.user_type == UserType.REGISTERED_USER and 
                        session.email and 
                        session.messages and
                        not session.timeout_saved_to_crm):
                        
                        logger.info("Emergency save during server-side expiry detection")
                        try:
                            self._auto_save_to_crm(session, "Emergency Save (Server Detection)")
                        except Exception as e:
                            logger.error(f"Emergency save failed: {e}", exc_info=True)
                    
                    # End session and show expiry message
                    self._end_session_internal(session)
                    st.session_state.session_expired = True
                    st.session_state.expired_session_id = session_id[:8]
                    return self._create_guest_session()
                else:
                    # Session is active - update activity and continue
                    self._update_activity(session)
                    return session
        
        # No session or inactive
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
                
                try:
                    current_session = self.get_session()
                except Exception as e:
                    logger.error(f"Error getting session during auth: {e}")
                    current_session = self._create_guest_session()
                
                display_name = (
                    data.get('user_display_name') or 
                    data.get('displayName') or 
                    data.get('name') or 
                    data.get('user_nicename') or 
                    data.get('first_name') or
                    data.get('nickname') or
                    clean_username
                )

                # Ensure proper enum assignment
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False  # Reset save flag
                
                try:
                    self.db.save_session(current_session)
                    logger.info(f"Saved authenticated session: user_type={current_session.user_type}")
                except Exception as e:
                    logger.error(f"Failed to save authenticated session: {e}")
                    st.error("Authentication failed - could not save session.")
                    return None
                
                verification_session = self.db.load_session(current_session.session_id)
                if verification_session:
                    verification_session = self._validate_and_fix_session(verification_session)
                    if verification_session.user_type == UserType.REGISTERED_USER:
                        st.session_state.current_session_id = current_session.session_id
                        st.success(f"Welcome back, {current_session.first_name}!")
                        return current_session
                    else:
                        logger.error(f"Session verification failed: expected REGISTERED_USER, got {verification_session.user_type}")
                        st.error("Authentication failed - session verification failed.")
                        return None
                else:
                    st.error("Authentication failed - session could not be verified.")
                    return None
                
            else:
                error_message = f"Invalid username or password (Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except json.JSONDecodeError:
                    pass
                
                st.error(error_message)
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during authentication. Please check your connection.")
            logger.error(f"Authentication network exception: {e}")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "Rate limit exceeded. Please wait.", "success": False}

        # Validate session before using
        session = self._validate_and_fix_session(session)
        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {
                "content": moderation["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        response = self.ai.get_response(sanitized_prompt, session.messages)
        
        session.messages.append({
            "role": "user", 
            "content": sanitized_prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        response_message = {
            "role": "assistant",
            "content": response.get("content", "No response generated."),
            "source": response.get("source", "Unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add metadata flags
        for flag in ["used_search", "used_pinecone", "has_citations", "has_inline_citations", "safety_override"]:
            if response.get(flag):
                response_message[flag] = True
            
        session.messages.append(response_message)
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
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._auto_save_to_crm(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_welcome_page(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Knowledge Base**\nAccess curated F&B industry information")
    with col2:
        st.info("🌐 **Web Search**\nReal-time market data and trends") 
    with col3:
        st.info("📚 **Smart Citations**\nClickable inline source references")
    
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    
    with tab1:
        if not session_manager.config.WORDPRESS_URL:
            st.warning("Sign-in is disabled because the authentication service is not configured.")
        else:
            with st.form("login_form", clear_on_submit=False):
                # ✅ Secure login form with masked password
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")  # ✅ Masked input
                submit_button = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                            
                        if authenticated_session:
                            st.balloons()
                            st.success(f"🎉 Welcome back, {authenticated_session.first_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest limitations:**
        - Chat history is not saved across sessions
        - No PDF export capability  
        - Limited personalization features
        - No automatic CRM integration
        
        ✨ **Sign in benefits:**
        - Chat history saved and exportable as PDF
        - Automatic integration with Zoho CRM
        - Chat transcripts auto-saved to CRM after 2 minutes of inactivity
        - Personalized experience with your profile
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # User status section
        if session.user_type == UserType.REGISTERED_USER or session.user_type.value == "registered_user":
            st.success("✅ **Authenticated User**")
            if session.first_name: 
                st.markdown(f"**Welcome:** {session.first_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # CRM Status
            if session_manager.zoho.config.ZOHO_ENABLED:
                if session.zoho_contact_id: 
                    st.success("🔗 **CRM Linked**")
                else: 
                    st.info("📋 **CRM Ready**")
                if session.email: 
                    if session.timeout_saved_to_crm:
                        st.caption("💾 Auto-saved to CRM")
                    else:
                        st.caption("💾 Auto-save enabled")
            else: 
                st.caption("🚫 CRM Disabled")
            
            # Simple activity indicator
            if session.last_activity:
                time_since_activity = datetime.now() - session.last_activity
                minutes_since = time_since_activity.total_seconds() / 60
                
                if minutes_since < 1:
                    st.caption("🟢 Active")
                elif minutes_since < 2:
                    st.caption(f"🟡 Inactive {minutes_since:.1f} min")
                elif session.timeout_saved_to_crm:
                    st.caption(f"🟠 Auto-saved ({minutes_since:.1f} min)")
                else:
                    st.caption("🔴 Will expire soon")
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for auto-save features*")
        
        st.divider()
        
        # Session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(session)
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(session)
                st.rerun()

        # Download & save section (authenticated users)
        if (session.user_type == UserType.REGISTERED_USER or session.user_type.value == "registered_user") and session.messages:
            st.divider()
            
            # PDF Download
            pdf_buffer = pdf_exporter.generate_chat_pdf(session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer,
                    file_name=f"fifi_chat_transcript_{session.session_id[:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            # CRM Save (if enabled)
            if session_manager.zoho.config.ZOHO_ENABLED and session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True):
                    session_manager.manual_save_to_crm(session)
                st.caption("💡 Chat auto-saves after 2 min inactivity")

def render_chat_interface_with_timer(session_manager, session):
    """
    FIXED chat interface with improved timer integration and error handling
    """
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion")
    
    # Add browser close detection with error handling
    try:
        render_browser_close_component(session.session_id)
    except Exception as e:
        logger.error(f"Failed to render browser close component: {e}")
    
    # Show activity status for registered users
    try:
        render_activity_status_indicator(session, session_manager)
    except Exception as e:
        logger.error(f"Failed to render activity status: {e}")
    
    # Execute the timer component and handle results with better error handling
    timer_result = None
    try:
        timer_result = render_activity_timer_component(session.session_id, session_manager)
    except Exception as e:
        logger.error(f"Timer component error: {e}")
        # Continue without timer - don't break the whole interface
    
    # Process timer events if any were triggered
    if timer_result:
        try:
            should_rerun = handle_timer_event(timer_result, session_manager, session)
            
            if should_rerun:
                # Small delay to let user see the message, then rerun
                time.sleep(1)
                st.rerun()
        except Exception as e:
            logger.error(f"Timer event handling error: {e}")
            st.warning("⚠️ Timer event processing encountered an error, but continuing...")
    
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

    # Handle pending queries from sidebar
    pending_query = st.session_state.get('pending_query')
    if pending_query:
        prompt = pending_query
        del st.session_state.pending_query
    else:
        prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Content moderation check
        try:
            # Check if session_manager has the required methods
            if hasattr(session_manager, 'ai') and hasattr(session_manager.ai, 'openai_client'):
                # Assuming check_content_moderation is available in the global scope
                moderation_result = check_content_moderation(prompt, session_manager.ai.openai_client)
            else:
                moderation_result = {"flagged": False}
        except Exception as e:
            logger.error(f"Content moderation check failed: {e}")
            moderation_result = {"flagged": False}
        
        if moderation_result.get("flagged"):
            with st.chat_message("assistant"):
                st.error(f"🚨 {moderation_result['message']}")
            
            session.messages.append({
                "role": "assistant",
                "content": moderation_result['message'],
                "source": "Content Safety Policy",
                "timestamp": datetime.now().isoformat()
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base and web..."):
                    try:
                        response = session_manager.get_ai_response(session, prompt)
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                    except Exception as e:
                        logger.error(f"AI response generation failed: {e}")
                        st.error("⚠️ Sorry, I encountered an error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# CORRECTED SESSION EXPIRY HANDLING
# =============================================================================

def render_session_expiry_redirect():
    """
    CORRECTED session expiry redirect with better UX
    """
    if st.session_state.get('session_expired', False):
        expired_session_id = st.session_state.get('expired_session_id', 'unknown')
        trigger = st.session_state.get('expiry_trigger', 'unknown')
        
        # Show comprehensive expiry message
        st.error(f"🔄 **Session Expired**")
        st.info(f"Session `{expired_session_id}` was automatically ended due to 3 minutes of inactivity")
        
        if trigger == "javascript_timer":
            st.success("⏰ Detected by JavaScript activity timer")
            st.success("💾 Your chat was automatically saved to CRM")
        
        st.info("⏳ Redirecting to welcome page...")
        
        # Progress bar for better UX
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)  # 3 second total delay
            progress_bar.progress(i + 1)
        
        # Clean up session state
        keys_to_clear = [
            'session_expired', 'expired_session_id', 'expiry_trigger', 
            'current_session_id', 'page'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_session_manager() -> Optional[SessionManager]:
    """Safely get the session manager from session state"""
    if 'session_manager' not in st.session_state:
        return None
    
    manager = st.session_state.session_manager
    if not hasattr(manager, 'get_session'):
        logger.error("Invalid SessionManager instance in session state")
        return None
    
    return manager

def ensure_initialization():
    """Ensure the application is properly initialized"""
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.initialized = True
            
            logger.info("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            return False
    
    return True

def handle_save_requests():
    """Handle save requests from browser close detection"""
    query_params = st.query_params
    
    if query_params.get("event") == "close":
        session_id = query_params.get("session_id")
        if session_id:
            logger.info(f"🚪 BROWSER CLOSE EVENT for session {session_id[:8]}...")
            
            # Clear query params
            st.query_params.clear()
            
            try:
                session_manager = get_session_manager()
                if session_manager:
                    session = session_manager.db.load_session(session_id)
                    if (session and 
                        session.user_type == UserType.REGISTERED_USER and 
                        session.email and 
                        session.messages):
                        
                        # Extend session life during save
                        session.last_activity = datetime.now()
                        session_manager.db.save_session(session)
                        
                        # Attempt emergency save
                        success = session_manager.zoho.save_chat_transcript_sync(session, "Browser Close")
                        logger.info(f"Browser close save result: {success}")
                        
            except Exception as e:
                logger.error(f"Browser close save error: {e}", exc_info=True)
            
            # Return empty response for beacon requests
            st.stop()
            
# =============================================================================
# DEBUG FUNCTION TO TEST st_javascript BEHAVIOR
# =============================================================================

def debug_st_javascript_return_values():
    """
    Add this function to test what st_javascript actually returns in your environment
    Call this in your sidebar or main interface to debug the issue
    """
    
    st.subheader("🔧 st_javascript Debug Tests")
    
    # Test 1: Explicit null return
    test1 = st_javascript("(() => { return null; })()", key="debug_null")
    st.write(f"**Test 1 (null):** Value=`{test1}`, Type=`{type(test1)}`")
    
    # Test 2: Explicit 0 return  
    test2 = st_javascript("(() => { return 0; })()", key="debug_zero")
    st.write(f"**Test 2 (0):** Value=`{test2}`, Type=`{type(test2)}`")
    
    # Test 3: Explicit false return
    test3 = st_javascript("(() => { return false; })()", key="debug_false")
    st.write(f"**Test 3 (false):** Value=`{test3}`, Type=`{type(test3)}`")
    
    # Test 4: Explicit undefined return
    test4 = st_javascript("(() => { return undefined; })()", key="debug_undefined")
    st.write(f"**Test 4 (undefined):** Value=`{test4}`, Type=`{type(test4)}`")
    
    # Test 5: No explicit return (should be undefined)
    test5 = st_javascript("(() => { const x = 1; })()", key="debug_no_return")
    st.write(f"**Test 5 (no return):** Value=`{test5}`, Type=`{type(test5)}`")
    
    # Test 6: Object return
    test6 = st_javascript("(() => { return {test: 'value'}; })()", key="debug_object")
    st.write(f"**Test 6 (object):** Value=`{test6}`, Type=`{type(test6)}`")
    
    st.markdown("---")
    st.write("**Analysis:** Check which values are returned as `0` vs `None` vs other types")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # Handle session expiry redirect first
    render_session_expiry_redirect()

    # Clear state button
    if st.button("🔄 Fresh Start", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Initialize application
    if not ensure_initialization():
        st.stop()

    # Handle save requests (browser close, etc.)
    handle_save_requests()

    # Get session manager
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get session manager.")
        st.stop()
    
    # Main application flow
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_with_timer(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state.page
            st.rerun()

if __name__ == "__main__":
    main()
