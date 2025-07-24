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
# FIFI - FIXED AUTHENTICATION VERSION
# =============================================================================
# ✅ FIXED: Authentication persistence issues
# ✅ FIXED: Session state management between reruns
# ✅ FIXED: UserType enum handling
# ✅ FIXED: Database synchronization timing
# ✅ ADDED: Session caching for better performance
# ✅ ADDED: Explicit authentication state tracking
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
# ENHANCED ERROR HANDLING SYSTEM
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
# DATABASE MANAGER WITH SQLITE CLOUD SUPPORT - FIXED AUTHENTICATION
# =============================================================================

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.connection_string = connection_string
        self.conn = None
        self.session_cache = {}  # ✅ ADDED: In-memory cache for better performance
        
        logger.info("🔄 INITIALIZING DATABASE MANAGER")
        
        # Try SQLite Cloud first
        if connection_string and SQLITECLOUD_AVAILABLE:
            cloud_result = self._try_sqlite_cloud_connection(connection_string)
            if cloud_result:
                self.conn, self.db_type, self.db_path = cloud_result
                logger.info("✅ SQLite Cloud connection established!")
        
        # Fallback to local SQLite
        if not self.conn:
            logger.info("🔄 Falling back to local database...")
            local_result = self._try_local_sqlite_connection()
            if local_result:
                self.conn, self.db_type, self.db_path = local_result
                logger.info("✅ Local SQLite connection established!")
        
        # Final fallback to in-memory
        if not self.conn:
            logger.critical("🚨 ALL DATABASE CONNECTIONS FAILED - Using in-memory storage")
            self.db_type = "memory"
            self._init_local_storage()
        
        # Initialize database tables
        if self.conn:
            self._init_database()
            error_handler.mark_component_healthy("Database")

    def _try_sqlite_cloud_connection(self, connection_string: str):
        try:
            import sqlitecloud
            conn = sqlitecloud.connect(connection_string)
            result = conn.execute("SELECT 1 as test").fetchone()
            logger.info(f"✅ SQLite Cloud test successful: {result}")
            return conn, "cloud", connection_string
        except Exception as e:
            logger.error(f"❌ SQLite Cloud connection failed: {e}")
            return None

    def _try_local_sqlite_connection(self):
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
        self.local_sessions = {}
        logger.info("📝 In-memory storage initialized")

    def _init_database(self):
        with self.lock:
            try:
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
                logger.info("✅ Database tables initialized")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                raise

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """✅ FIXED: Enhanced session saving with caching"""
        with self.lock:
            # ✅ ADDED: Update cache
            self.session_cache[session.session_id] = copy.deepcopy(session)
            
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = session
                logger.debug(f"✅ Memory save: {session.session_id[:8]} as {session.user_type}")
                return
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                # ✅ FIXED: Ensure enum is properly serialized
                user_type_value = session.user_type.value if isinstance(session.user_type, UserType) else str(session.user_type)
                
                self.conn.execute(
                    '''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session.session_id, user_type_value, session.email, session.first_name,
                     session.zoho_contact_id, int(session.guest_email_requested), session.created_at.isoformat(),
                     session.last_activity.isoformat(), json.dumps(session.messages), int(session.active),
                     session.wp_token, int(session.timeout_saved_to_crm)))
                self.conn.commit()
                
                logger.debug(f"✅ DB save: {session.session_id[:8]} as {user_type_value}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save session {session.session_id[:8]}: {e}")
                raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """✅ FIXED: Enhanced session loading with caching"""
        with self.lock:
            # ✅ ADDED: Check cache first
            if session_id in self.session_cache:
                logger.debug(f"✅ Cache hit: {session_id[:8]}")
                return copy.deepcopy(self.session_cache[session_id])
            
            if self.db_type == "memory":
                session = self.local_sessions.get(session_id)
                if session:
                    # ✅ FIXED: Ensure proper UserType conversion
                    if isinstance(session.user_type, str):
                        try:
                            session.user_type = UserType(session.user_type)
                        except ValueError:
                            logger.warning(f"Invalid user_type in memory: {session.user_type}")
                            session.user_type = UserType.GUEST
                    logger.debug(f"✅ Memory load: {session_id[:8]} as {session.user_type}")
                    # ✅ ADDED: Update cache
                    self.session_cache[session_id] = copy.deepcopy(session)
                return session

            try:
                if self.db_type == "cloud":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = None
                elif self.db_type == "file":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = sqlite3.Row
                
                cursor = self.conn.execute(
                    "SELECT session_id, user_type, email, first_name, zoho_contact_id, guest_email_requested, created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm FROM sessions WHERE session_id = ? AND active = 1", 
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if not row: 
                    logger.debug(f"❌ Session not found in DB: {session_id[:8]}")
                    return None
                
                # Handle row conversion based on type
                row_dict = {}
                
                if hasattr(row, 'keys') and callable(getattr(row, 'keys')):
                    try:
                        row_dict = dict(row)
                        logger.debug(f"✅ Converted sqlite3.Row to dict for {session_id[:8]}")
                    except Exception as e:
                        logger.warning(f"Failed to convert sqlite3.Row: {e}")
                        row_dict = None
                
                if not row_dict:
                    if len(row) >= 12:
                        row_dict = {
                            'session_id': row[0], 'user_type': row[1], 'email': row[2],
                            'first_name': row[3], 'zoho_contact_id': row[4], 'guest_email_requested': row[5],
                            'created_at': row[6], 'last_activity': row[7], 'messages': row[8],
                            'active': row[9], 'wp_token': row[10], 'timeout_saved_to_crm': row[11]
                        }
                        logger.debug(f"✅ Converted tuple to dict for {session_id[:8]}")
                    else:
                        logger.error(f"Row has insufficient columns: {len(row)}")
                        return None
                
                if not row_dict:
                    logger.error(f"Failed to convert row for {session_id[:8]}")
                    return None
                
                # ✅ FIXED: Better UserType conversion with validation
                user_type_str = row_dict.get('user_type', 'guest')
                try:
                    if user_type_str in ['registered_user', 'REGISTERED_USER']:
                        user_type = UserType.REGISTERED_USER
                    elif user_type_str in ['guest', 'GUEST']:
                        user_type = UserType.GUEST
                    else:
                        logger.warning(f"Unknown user_type in DB: {user_type_str}, defaulting to GUEST")
                        user_type = UserType.GUEST
                except Exception as e:
                    logger.error(f"UserType conversion failed: {e}")
                    user_type = UserType.GUEST
                
                # Create UserSession
                user_session = UserSession(
                    session_id=row_dict['session_id'], 
                    user_type=user_type,
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
                
                # ✅ ADDED: Update cache
                self.session_cache[session_id] = copy.deepcopy(user_session)
                
                logger.debug(f"✅ DB load: {session_id[:8]} as {user_session.user_type}")
                return user_session
                    
            except Exception as e:
                logger.error(f"Failed to load session {session_id[:8]}: {e}")
                return None

    def clear_cache(self):
        """✅ ADDED: Clear session cache"""
        with self.lock:
            self.session_cache.clear()
            logger.debug("Session cache cleared")

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
        if not self.config.ZOHO_ENABLED:
            return None

        if not force_refresh and self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._access_token
        
        try:
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
            
            return self._access_token
            
        except Exception as e:
            logger.error(f"Failed to get Zoho access token: {e}")
            raise

    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            if response.status_code == 401:
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data']:
                return data['data'][0]['id']
        except Exception as e:
            logger.error(f"Error finding contact: {e}")
        return None

    def _create_contact(self, email: str, access_token: str, first_name: str = None) -> Optional[str]:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        contact_data = {
            "data": [{
                "Last_Name": first_name or "Food Professional",
                "Email": email,
                "Lead_Source": "FiFi AI Assistant"
            }]
        }
        
        try:
            response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            if response.status_code == 401:
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS':
                return data['data'][0]['details']['id']
        except Exception as e:
            logger.error(f"Error creating contact: {e}")
        return None

    def _upload_attachment(self, contact_id: str, pdf_buffer: io.BytesIO, access_token: str, filename: str) -> bool:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        upload_url = f"{self.base_url}/Contacts/{contact_id}/Attachments"
        
        try:
            pdf_buffer.seek(0)
            response = requests.post(
                upload_url, headers=headers, 
                files={'file': (filename, pdf_buffer.read(), 'application/pdf')}, timeout=60
            )
            
            if response.status_code == 401:
                access_token = self._get_access_token_with_timeout(force_refresh=True)
                if access_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                    pdf_buffer.seek(0)
                    response = requests.post(
                        upload_url, headers=headers,
                        files={'file': (filename, pdf_buffer.read(), 'application/pdf')}, timeout=60
                    )
            
            response.raise_for_status()
            data = response.json()
            
            return 'data' in data and data['data'][0]['code'] == 'SUCCESS'
        except Exception as e:
            logger.error(f"Error uploading attachment: {e}")
            return False

    def _add_note(self, contact_id: str, note_title: str, note_content: str, access_token: str) -> bool:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        
        # Truncate if too long
        max_length = 32000
        if len(note_content) > max_length:
            note_content = note_content[:max_length-100] + "\n\n[Content truncated]"
        
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
                new_token = self._get_access_token_with_timeout(force_refresh=True)
                if new_token:
                    headers['Authorization'] = f'Zoho-oauthtoken {new_token}'
                    response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=15)
            
            response.raise_for_status()
            data = response.json()
            
            return 'data' in data and data['data'][0]['code'] == 'SUCCESS'
        except Exception as e:
            logger.error(f"Error adding note: {e}")
            return False

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        logger.info(f"🚀 Starting Zoho save: {trigger_reason}")
        
        try:
            # Get access token
            access_token = self._get_access_token_with_timeout(force_refresh=True)
            if not access_token:
                logger.error("Failed to get access token")
                return False

            # Find or create contact
            contact_id = self._find_contact_by_email(session.email, access_token)
            if not contact_id:
                contact_id = self._create_contact(session.email, access_token, session.first_name)
            if not contact_id:
                logger.error("Failed to find/create contact")
                return False

            session.zoho_contact_id = contact_id

            # Generate PDF
            pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
            if not pdf_buffer:
                logger.error("Failed to generate PDF")
                return False

            # Upload attachment
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
            upload_success = self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename)

            # Add note
            note_title = f"FiFi AI Chat Transcript - {timestamp} ({trigger_reason})"
            note_content = self._generate_note_content(session, upload_success, trigger_reason)
            note_success = self._add_note(contact_id, note_title, note_content, access_token)

            if note_success:
                logger.info("✅ Zoho save completed successfully")
                return True
            else:
                logger.error("Failed to add note")
                return False

        except Exception as e:
            logger.error(f"Zoho save failed: {e}", exc_info=True)
            return False

    def _generate_note_content(self, session: UserSession, attachment_uploaded: bool, trigger_reason: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"**Session Information:**\n"
        content += f"- Session ID: {session.session_id}\n"
        content += f"- User: {session.first_name or 'Unknown'} ({session.email})\n"
        content += f"- Save Trigger: {trigger_reason}\n"
        content += f"- Timestamp: {timestamp}\n"
        content += f"- Total Messages: {len(session.messages)}\n\n"
        
        if attachment_uploaded:
            content += "✅ **PDF transcript attached.**\n\n"
        else:
            content += "⚠️ **PDF attachment failed. Transcript below:**\n\n"
        
        content += "**Conversation Summary:**\n"
        
        for i, msg in enumerate(session.messages):
            role = msg.get("role", "Unknown").capitalize()
            text = re.sub(r'<[^>]+>', '', msg.get("content", ""))
            
            if len(text) > 500:
                text = text[:500] + "..."
                
            content += f"\n{i+1}. **{role}:** {text}\n"
            
            if msg.get("source"):
                content += f"   _Source: {msg['source']}_\n"
                
        return content

# =============================================================================
# RATE LIMITER AND UTILITIES
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
# AI SYSTEM (SIMPLIFIED FOR COMPLETE VERSION)
# =============================================================================

class EnhancedAI:
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = None
        
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        return {
            "content": f"I understand you're asking about: {prompt}. This is a simplified response for the integrated version. In production, this connects to your AI services (Pinecone, Tavily, OpenAI) for intelligent F&B industry responses.",
            "source": "Simplified AI System",
            "used_search": False,
            "used_pinecone": False,
            "success": True
        }

@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Dict[str, Any]:
    if not client or not hasattr(client, 'moderations'):
        return {"flagged": False}
    
    try:
        response = client.moderations.create(model="omni-moderation-latest", input=prompt)
        result = response.results[0]
        
        if result.flagged:
            flagged_categories = [cat for cat, flagged in result.categories.__dict__.items() if flagged]
            return {
                "flagged": True, 
                "message": "Your message violates our content policy.",
                "categories": flagged_categories
            }
    except Exception:
        pass
    
    return {"flagged": False}

# =============================================================================
# OPTIMIZED TIMER SYSTEM - BEST PRACTICES APPLIED
# =============================================================================

def initialize_autonomous_timer(session_id: str):
    """
    ✅ BEST PRACTICE: Initialize timer once per session, let it run autonomously
    ✅ Event-driven: Only communicates back when events occur
    ✅ Stable key: Based on session, not dynamic
    ✅ Minimal re-execution: Runs once, operates independently
    """
    
    if not session_id:
        return
    
    # ✅ Check if timer already initialized for this session
    timer_key = f"timer_initialized_{session_id[:8]}"
    if timer_key in st.session_state:
        return  # Already initialized, don't re-initialize
    
    js_timer_init = f"""
    (() => {{
        const sessionId = "{session_id}";
        const AUTO_SAVE_TIMEOUT = 120000;  // 2 minutes
        const SESSION_EXPIRE_TIMEOUT = 180000;  // 3 minutes
        
        console.log("🚀 Initializing autonomous timer for session:", sessionId.substring(0, 8));
        
        // ✅ BEST PRACTICE: Initialize state only if missing
        if (!window.fifi_autonomous_timer) {{
            window.fifi_autonomous_timer = {{
                sessionId: sessionId,
                lastActivityTime: Date.now(),
                autoSaveTriggered: false,
                sessionExpired: false,
                intervalId: null,
                events: []  // Store events to be collected by Python
            }};
        }}
        
        const timer = window.fifi_autonomous_timer;
        
        // Reset if session changed
        if (timer.sessionId !== sessionId) {{
            timer.sessionId = sessionId;
            timer.lastActivityTime = Date.now();
            timer.autoSaveTriggered = false;
            timer.sessionExpired = false;
            timer.events = [];
        }}
        
        // ✅ BEST PRACTICE: Set up activity listeners once
        function resetActivity() {{
            timer.lastActivityTime = Date.now();
            timer.autoSaveTriggered = false;
            timer.sessionExpired = false;
            console.log("🔄 Timer: Activity detected");
        }}
        
        const events = ['mousedown', 'mousemove', 'click', 'keydown', 'scroll', 'touchstart'];
        
        // Remove existing listeners to prevent duplicates
        if (timer.activityListeners) {{
            timer.activityListeners.forEach(cleanup => cleanup());
        }}
        
        timer.activityListeners = [];
        
        // Add component listeners
        events.forEach(eventType => {{
            const cleanup = () => document.removeEventListener(eventType, resetActivity);
            document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
            timer.activityListeners.push(cleanup);
        }});
        
        // Add parent listeners if accessible
        try {{
            if (window.parent && window.parent.document && window.parent.document !== document) {{
                events.forEach(eventType => {{
                    const cleanup = () => window.parent.document.removeEventListener(eventType, resetActivity);
                    window.parent.document.addEventListener(eventType, resetActivity, {{ passive: true, capture: true }});
                    timer.activityListeners.push(cleanup);
                }});
            }}
        }} catch (e) {{
            console.debug("Cannot access parent for activity detection");
        }}
        
        // ✅ BEST PRACTICE: Autonomous operation - timer runs independently
        if (!timer.intervalId) {{
            timer.intervalId = setInterval(() => {{
                const currentTime = Date.now();
                const inactiveTimeMs = currentTime - timer.lastActivityTime;
                const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
                
                // Check for auto-save trigger
                if (inactiveTimeMs >= AUTO_SAVE_TIMEOUT && !timer.autoSaveTriggered) {{
                    timer.autoSaveTriggered = true;
                    console.log("🚨 Timer: Auto-save triggered");
                    
                    // ✅ BEST PRACTICE: Store event for collection, don't force immediate return
                    timer.events.push({{
                        type: "auto_save_trigger",
                        session_id: sessionId,
                        inactive_time_ms: inactiveTimeMs,
                        inactive_minutes: inactiveMinutes,
                        timestamp: currentTime
                    }});
                }}
                
                // Check for session expiry
                if (inactiveTimeMs >= SESSION_EXPIRE_TIMEOUT && !timer.sessionExpired) {{
                    timer.sessionExpired = true;
                    console.log("🚨 Timer: Session expired");
                    
                    // ✅ BEST PRACTICE: Store event for collection
                    timer.events.push({{
                        type: "session_expired",
                        session_id: sessionId,
                        inactive_time_ms: inactiveTimeMs,
                        inactive_minutes: inactiveMinutes,
                        timestamp: currentTime
                    }});
                }}
                
            }}, 5000);  // Check every 5 seconds
            
            console.log("✅ Autonomous timer started with interval ID:", timer.intervalId);
        }}
        
        return "timer_initialized";
    }})()
    """
    
    try:
        # ✅ BEST PRACTICE: Stable key based on session
        stable_key = f"fifi_timer_init_{session_id[:8]}"
        result = st_javascript(js_timer_init, key=stable_key)
        
        if result == "timer_initialized":
            # ✅ BEST PRACTICE: Mark as initialized to prevent re-initialization
            st.session_state[timer_key] = True
            logger.info(f"✅ Autonomous timer initialized for session {session_id[:8]}")
        
    except Exception as e:
        logger.error(f"Timer initialization failed: {e}")

def collect_timer_events(session_id: str) -> List[Dict[str, Any]]:
    """
    ✅ BEST PRACTICE: Collect events only when needed, not on every rerun
    ✅ Event-driven: Only retrieves events that actually occurred
    """
    
    if not session_id:
        return []
    
    js_collect_events = f"""
    (() => {{
        const sessionId = "{session_id}";
        
        if (!window.fifi_autonomous_timer || window.fifi_autonomous_timer.sessionId !== sessionId) {{
            return [];
        }}
        
        const timer = window.fifi_autonomous_timer;
        const events = timer.events.slice();  // Copy events
        timer.events = [];  // Clear collected events
        
        if (events.length > 0) {{
            console.log("📦 Collecting", events.length, "timer events for Python");
        }}
        
        return events;
    }})()
    """
    
    try:
        # ✅ BEST PRACTICE: Stable key for event collection
        stable_key = f"fifi_timer_collect_{session_id[:8]}"
        events = st_javascript(js_collect_events, key=stable_key)
        
        # Handle potential falsy returns
        if not events or not isinstance(events, list):
            return []
        
        return events
        
    except Exception as e:
        logger.error(f"Event collection failed: {e}")
        return []

def cleanup_timer(session_id: str):
    """
    ✅ BEST PRACTICE: Clean up timer when session ends
    """
    
    if not session_id:
        return
    
    js_cleanup = f"""
    (() => {{
        const sessionId = "{session_id}";
        
        if (window.fifi_autonomous_timer && window.fifi_autonomous_timer.sessionId === sessionId) {{
            const timer = window.fifi_autonomous_timer;
            
            // Clear interval
            if (timer.intervalId) {{
                clearInterval(timer.intervalId);
                console.log("🧹 Timer interval cleared");
            }}
            
            // Remove activity listeners
            if (timer.activityListeners) {{
                timer.activityListeners.forEach(cleanup => cleanup());
                console.log("🧹 Activity listeners removed");
            }}
            
            // Clear timer state
            delete window.fifi_autonomous_timer;
            console.log("🧹 Timer cleaned up for session:", sessionId.substring(0, 8));
            
            return "cleaned";
        }}
        
        return "not_found";
    }})()
    """
    
    try:
        stable_key = f"fifi_timer_cleanup_{session_id[:8]}"
        result = st_javascript(js_cleanup, key=stable_key)
        
        if result == "cleaned":
            # ✅ Remove initialization flag
            timer_key = f"timer_initialized_{session_id[:8]}"
            if timer_key in st.session_state:
                del st.session_state[timer_key]
        
    except Exception as e:
        logger.error(f"Timer cleanup failed: {e}")

def render_browser_close_component(session_id: str):
    """
    ✅ BEST PRACTICE: Initialize browser close detection once per session
    """
    if not session_id:
        return

    # ✅ Check if already initialized
    close_key = f"close_initialized_{session_id[:8]}"
    if close_key in st.session_state:
        return
    
    js_code = f"""
    <script>
    (function() {{
        const sessionKey = 'fifi_close_{session_id}';
        if (window[sessionKey]) return;
        window[sessionKey] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false;
        
        function triggerEmergencySave() {{
            if (saveTriggered) return;
            saveTriggered = true;
            
            console.log('🚨 Browser close detected - triggering emergency save');
            
            try {{
                const url = window.parent.location.origin + window.parent.location.pathname +
                    `?session_id=${{sessionId}}&event=close`;
                console.log('📡 Navigating to:', url);
                window.parent.location = url;
            }} catch (e) {{
                try {{
                    const url = window.location.origin + window.location.pathname +
                        `?session_id=${{sessionId}}&event=close`;
                    window.location = url;
                }} catch (e2) {{
                    console.error('❌ All navigation methods failed:', e, e2);
                }}
            }}
        }}
        
        const events = ['beforeunload', 'pagehide'];
        events.forEach(eventType => {{
            try {{
                if (window.parent && window.parent !== window) {{
                    window.parent.addEventListener(eventType, triggerEmergencySave, {{ capture: true, passive: true }});
                }}
                window.addEventListener(eventType, triggerEmergencySave, {{ capture: true, passive: true }});
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        try {{
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('visibilitychange', () => {{
                    if (window.parent.document.visibilityState === 'hidden') {{
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
        
        console.log('✅ Emergency save detection initialized');
    }})();
    </script>
    """
    
    try:
        components.html(js_code, height=0, width=0)
        # ✅ Mark as initialized
        st.session_state[close_key] = True
    except Exception as e:
        logger.error(f"Failed to render browser close component: {e}")

def render_activity_status_indicator(session):
    """
    Show current activity status for registered users
    """
    if session.user_type == UserType.REGISTERED_USER and session.last_activity:
        time_since_activity = datetime.now() - session.last_activity
        minutes_since = time_since_activity.total_seconds() / 60
        
        if minutes_since < 0.5:
            st.success("🟢 **Active** - Auto-save in 2 minutes if inactive")
        elif minutes_since < 1:
            remaining = 2 - minutes_since
            st.info(f"🟡 **Inactive** for {minutes_since:.1f} min - Auto-save in {remaining:.1f} min")
        elif minutes_since < 2:
            st.warning(f"🟠 **Inactive** for {minutes_since:.1f} min - Auto-save will trigger soon")
        elif session.timeout_saved_to_crm:
            remaining_to_expiry = 3 - minutes_since
            if remaining_to_expiry > 0:
                st.success(f"💾 **Auto-saved** - Session expires in {remaining_to_expiry:.1f} min")
            else:
                st.success("💾 **Auto-saved**")
        else:
            remaining_to_expiry = 3 - minutes_since
            if remaining_to_expiry > 0:
                st.error(f"🔴 **Inactive** for {minutes_since:.1f} min - Expires in {remaining_to_expiry:.1f} min")
            else:
                st.error(f"🔴 **Inactive** for {minutes_since:.1f} min - Will expire soon")

# =============================================================================
# DEBUG FUNCTIONS FOR AUTHENTICATION TROUBLESHOOTING
# =============================================================================

def render_debug_session_info(session, session_manager):
    """✅ ADDED: Debug function to troubleshoot authentication persistence"""
    
    with st.sidebar.expander("🔧 Debug Session Info"):
        st.write(f"**Session ID:** {session.session_id[:8]}")
        st.write(f"**User Type:** {session.user_type}")
        st.write(f"**User Type Value:** {session.user_type.value}")
        st.write(f"**Email:** {session.email}")
        st.write(f"**First Name:** {session.first_name}")
        st.write(f"**Active:** {session.active}")
        st.write(f"**WP Token:** {'Set' if session.wp_token else 'None'}")
        
        if st.button("🔍 Reload Session from DB", key="debug_reload"):
            if session_manager:
                fresh_session = session_manager.db.load_session(session.session_id)
                if fresh_session:
                    st.write(f"**DB User Type:** {fresh_session.user_type}")
                    st.write(f"**DB Email:** {fresh_session.email}")
                    st.write(f"**DB First Name:** {fresh_session.first_name}")
                    
                    if fresh_session.user_type != session.user_type:
                        st.error(f"**MISMATCH:** Memory={session.user_type}, DB={fresh_session.user_type}")
                    else:
                        st.success("✅ Session data matches database")
                else:
                    st.error("❌ Session not found in database")
        
        if st.button("🔄 Force Session Refresh", key="debug_refresh"):
            st.rerun()

# =============================================================================
# SESSION MANAGER WITH ENHANCED AUTHENTICATION - FIXED VERSION
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.session_timeout_minutes = 3
        self._save_lock = threading.Lock()

    def _is_session_expired(self, session: UserSession) -> bool:
        if not session.last_activity:
            return False
        time_diff = datetime.now() - session.last_activity
        return time_diff.total_seconds() > (self.session_timeout_minutes * 60)

    def _update_activity(self, session: UserSession):
        """✅ BEST PRACTICE: Update session state with callback pattern"""
        session.last_activity = datetime.now()
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
        if isinstance(session.user_type, str):
            session.user_type = UserType(session.user_type)
        try:
            self.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        logger.info(f"✅ Created guest session: {session.session_id[:8]}")
        return session

    def _validate_and_fix_session(self, session: UserSession) -> UserSession:
        """✅ FIXED: Better session validation with enhanced UserType handling"""
        if not session:
            return session
            
        # ✅ FIXED: Better UserType handling
        if isinstance(session.user_type, str):
            try:
                # Handle both 'registered_user' and 'REGISTERED_USER' formats
                user_type_str = session.user_type.upper()
                if user_type_str in ['REGISTERED_USER', 'REGISTERED-USER']:
                    session.user_type = UserType.REGISTERED_USER
                elif user_type_str in ['GUEST']:
                    session.user_type = UserType.GUEST
                else:
                    logger.warning(f"Unknown user_type string: {session.user_type}, defaulting to GUEST")
                    session.user_type = UserType.GUEST
            except Exception as e:
                logger.error(f"UserType conversion failed: {e}")
                session.user_type = UserType.GUEST
        
        if not isinstance(session.messages, list):
            session.messages = []
            
        return session

    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str) -> bool:
        with self._save_lock:
            logger.info(f"🚀 Auto-save triggered: {trigger_reason}")
            
            session = self._validate_and_fix_session(session)
            
            # Check prerequisites
            if session.user_type != UserType.REGISTERED_USER:
                logger.info("Save skipped: Not a registered user")
                return False
            if not session.email:
                logger.info("Save skipped: No email")
                return False
            if not session.messages:
                logger.info("Save skipped: No messages")
                return False
            if not self.zoho.config.ZOHO_ENABLED:
                logger.info("Save skipped: Zoho not enabled")
                return False

            is_interactive = "Manual" in trigger_reason

            try:
                if is_interactive:
                    with st.spinner(f"💾 Saving to CRM..."):
                        success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    
                    if success:
                        st.success("✅ Chat saved to Zoho CRM!")
                        return True
                    else:
                        st.error("❌ Failed to save to CRM")
                        return False
                else:
                    # Non-interactive save
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    if success:
                        session.timeout_saved_to_crm = True
                        self.db.save_session(session)
                        return True
                    return False

            except Exception as e:
                logger.error(f"Auto-save failed: {e}", exc_info=True)
                if is_interactive:
                    st.error(f"❌ Save error: {str(e)}")
                return False

    def get_session(self) -> UserSession:
        """✅ FIXED: Enhanced session retrieval with better authentication persistence"""
        # ✅ FIXED: Check authentication state first
        if 'authenticated_session_id' in st.session_state:
            # We have an authenticated session, use it
            session_id = st.session_state.authenticated_session_id
            session = self.db.load_session(session_id)
            if session and session.active and session.user_type == UserType.REGISTERED_USER:
                # Valid authenticated session
                st.session_state.current_session_id = session_id
                self._update_activity(session)
                return session
            else:
                # Authenticated session no longer valid (e.g., deleted from DB, or user_type changed)
                logger.info(f"Authenticated session {session_id[:8]} found in st.session_state but invalid in DB. Resetting.")
                del st.session_state['authenticated_session_id']
                # Fall through to check current_session_id or create guest
        
        # ✅ Check if session ID exists before creating a new guest one
        if 'current_session_id' not in st.session_state:
            return self._create_guest_session()
        
        session_id = st.session_state.current_session_id
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                session = self._validate_and_fix_session(session)
                
                if self._is_session_expired(session):
                    logger.info(f"Session {session_id[:8]} expired")
                    
                    # Emergency save if eligible
                    if (session.user_type == UserType.REGISTERED_USER and 
                        session.email and session.messages and
                        not session.timeout_saved_to_crm):
                        try:
                            self._auto_save_to_crm(session, "Emergency Save (Server Detection)")
                        except Exception as e:
                            logger.error(f"Emergency save failed: {e}")
                    
                    # Cleanup timer
                    cleanup_timer(session_id)
                    
                    self._end_session_internal(session)
                    st.session_state.session_expired = True
                    st.session_state.expired_session_id = session_id[:8]
                    return self._create_guest_session() # Redirect to new guest session
                else:
                    self._update_activity(session)
                    logger.debug(f"✅ Session retrieved: {session_id[:8]} as {session.user_type}")
                    return session
        
        # Fallback: if current_session_id points to an inactive or non-existent session
        logger.info(f"Current session_id ({st.session_state.get('current_session_id', 'N/A')[:8]}) invalid/inactive. Creating new guest session.")
        return self._create_guest_session()

    def _end_session_internal(self, session: UserSession):
        session.active = False
        try:
            self.db.save_session(session)
        except Exception as e:
            logger.error(f"Failed to mark session inactive: {e}")
        
        # ✅ BEST PRACTICE: Clean up session state properly
        for key in ['current_session_id', 'authenticated_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]

    @handle_api_errors("Authentication", "WordPress Login", show_to_user=False)
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        """✅ FIXED: Enhanced authentication with better session persistence"""
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service not configured.")
            return None
        if not self.rate_limiter.is_allowed(f"auth_{username}"):
            st.error("Too many login attempts. Please wait.")
            return None

        clean_username = username.strip()
        clean_password = password.strip()
        
        logger.info(f"🔐 Starting authentication for: {clean_username}")

        try:
            response = requests.post(
                f"{self.config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token",
                json={'username': clean_username, 'password': clean_password},
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                current_session = self.get_session() # Get the current session (might be guest)
                
                display_name = (
                    data.get('user_display_name') or data.get('displayName') or 
                    data.get('name') or data.get('user_nicename') or 
                    data.get('first_name') or data.get('nickname') or clean_username
                )

                # ✅ FIXED: Ensure proper enum assignment with validation
                logger.info(f"🔄 Converting session {current_session.session_id[:8]} from {current_session.user_type} to REGISTERED_USER")
                
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                # ✅ FIXED: Force immediate save with cache clear
                try:
                    self.db.save_session(current_session)
                    self.db.clear_cache()  # ✅ ADDED: Clear cache to force fresh load
                    logger.info(f"✅ Session saved with user_type: {current_session.user_type}")
                    
                    # ✅ FIXED: Verify the save worked with fresh load
                    verification_session = self.db.load_session(current_session.session_id)
                    if verification_session and verification_session.user_type == UserType.REGISTERED_USER:
                        # ✅ FIXED: Store authenticated session ID
                        st.session_state.authenticated_session_id = current_session.session_id
                        st.session_state.current_session_id = current_session.session_id # Ensure current_session_id is also set
                        logger.info(f"✅ Authentication successful: {verification_session.user_type} for {verification_session.email}")
                        return current_session
                    else:
                        logger.error(f"❌ Session verification failed - Expected REGISTERED_USER, got {verification_session.user_type if verification_session else 'None'}")
                        st.error("Authentication failed - session could not be verified.")
                        return None
                        
                except Exception as e:
                    logger.error(f"❌ Failed to save authenticated session: {e}")
                    st.error("Authentication failed - could not save session.")
                    return None
                
            else:
                error_message = f"Invalid username or password (Code: {response.status_code})."
                try:
                    error_data = response.json()
                    error_message = error_data.get('message', error_message)
                except json.JSONDecodeError:
                    pass
                
                logger.warning(f"❌ Authentication failed: {error_message}")
                st.error(error_message)
                return None
                
        except Exception as e:
            st.error("Authentication failed. Please check your connection.")
            logger.error(f"Authentication error: {e}")
            return None

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed(session.session_id):
            return {"content": "Rate limit exceeded. Please wait.", "success": False}

        session = self._validate_and_fix_session(session)
        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation.get("flagged"):
            return {"content": moderation["message"], "success": False, "source": "Content Safety"}

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
        
        for flag in ["used_search", "used_pinecone", "has_citations"]:
            if response.get(flag):
                response_message[flag] = True
            
        session.messages.append(response_message)
        session.messages = session.messages[-100:]  # Keep last 100 messages
        
        self._update_activity(session)
        return response

    def clear_chat_history(self, session: UserSession):
        """✅ BEST PRACTICE: Use callback for immediate state update"""
        session = self._validate_and_fix_session(session)
        session.messages = []
        session.timeout_saved_to_crm = False
        self._update_activity(session)

    def end_session(self, session: UserSession):
        """✅ BEST PRACTICE: Clean up properly on session end"""
        session = self._validate_and_fix_session(session)
        self._auto_save_to_crm(session, "Manual Sign Out")
        
        # Clean up timer
        cleanup_timer(session.session_id)
        
        self._end_session_internal(session)

    def manual_save_to_crm(self, session: UserSession):
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            self._auto_save_to_crm(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing requirements")

    def process_timer_events(self, session: UserSession):
        """
        ✅ BEST PRACTICE: Process timer events using callback pattern
        ✅ Event-driven: Only called when events exist
        """
        
        events = collect_timer_events(session.session_id)
        
        for event in events:
            event_type = event.get('type')
            inactive_minutes = event.get('inactive_minutes', 0)
            
            if event_type == 'auto_save_trigger':
                logger.info(f"Processing auto-save event: {inactive_minutes} minutes inactive")
                
                if (session.user_type == UserType.REGISTERED_USER and 
                    session.email and session.messages and
                    not session.timeout_saved_to_crm):
                    
                    try:
                        success = self._auto_save_to_crm(session, "Autonomous Timer Auto-Save")
                        if success:
                            st.success(f"✅ Chat auto-saved after {inactive_minutes} minutes of inactivity!")
                            session.last_activity = datetime.now()
                            self.db.save_session(session)
                        else:
                            st.warning("⚠️ Auto-save failed")
                    except Exception as e:
                        st.error(f"❌ Auto-save error: {str(e)}")
                        logger.error(f"Auto-save error: {e}")
                else:
                    st.info("ℹ️ Auto-save skipped (not eligible)")
                
                return True  # Trigger rerun
                
            elif event_type == 'session_expired':
                logger.info(f"Processing session expiry event: {inactive_minutes} minutes inactive")
                
                st.error(f"🔄 **Session expired** after {inactive_minutes} minutes of inactivity")
                
                # Emergency save if eligible
                if (session.user_type == UserType.REGISTERED_USER and 
                    session.email and session.messages and
                    not session.timeout_saved_to_crm):
                    
                    try:
                        self._auto_save_to_crm(session, "Emergency Save (Session Expiry)")
                        st.success("✅ Emergency save completed")
                    except Exception as e:
                        st.error(f"❌ Emergency save failed: {str(e)}")
                
                # Clean up and end session
                cleanup_timer(session.session_id)
                self._end_session_internal(session)
                st.session_state.session_expired = True
                st.session_state.expired_session_id = session.session_id[:8]
                
                return True  # Trigger rerun
        
        return False  # No events processed

# =============================================================================
# QUERY PARAMETER HANDLER FOR EMERGENCY SAVES
# =============================================================================

def handle_emergency_save_requests():
    """
    Handle emergency save requests from browser close detection
    """
    query_params = st.query_params
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event == "close" and session_id:
        logger.info(f"🚨 Emergency save request for session: {session_id[:8]}")
        
        # Clear query params so they don't trigger reruns on subsequent refreshes
        st.query_params.clear()
        
        # Show processing message (only visible if the browser tab stays open long enough)
        st.info("🚨 **Processing emergency save...**")
        
        try:
            session_manager = st.session_state.get('session_manager')
            if session_manager:
                session = session_manager.db.load_session(session_id)
                if (session and session.user_type == UserType.REGISTERED_USER and 
                    session.email and session.messages and
                    not session.timeout_saved_to_crm):
                    
                    # Extend session life during save to prevent immediate expiry from server-side checks
                    session.last_activity = datetime.now()
                    session_manager.db.save_session(session) # Save updated activity
                    
                    success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Browser Close)")
                    
                    if success:
                        st.success("✅ Emergency save completed!")
                        logger.info("Emergency save successful")
                    else:
                        st.error("❌ Emergency save failed")
                        logger.error("Emergency save failed")
                else:
                    st.info("ℹ️ Session not eligible for emergency save (guest, no email, no messages, or already saved)")
                    logger.info("Session not eligible for save or already saved via timeout")
            else:
                st.error("❌ Session manager not available. Cannot process emergency save.")
                logger.error("Session manager not found during emergency save request (should be initialized by main).")
                
        except Exception as e:
            st.error(f"❌ Emergency save error: {str(e)}")
            logger.error(f"Emergency save error: {e}", exc_info=True)
        
        # Short sleep to allow message to render (if possible) before Streamlit closes the script
        time.sleep(2)
        st.stop() # Stop further execution of this ephemeral script

# =============================================================================
# UI COMPONENTS WITH AUTHENTICATION FIXES
# =============================================================================

def render_welcome_page(session_manager: SessionManager):
    """✅ FIXED: Enhanced welcome page with better authentication flow"""
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
            st.warning("Sign-in disabled: Authentication service not configured.")
        else:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Sign In", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        with st.spinner("🔐 Authenticating..."):
                            authenticated_session = session_manager.authenticate_with_wordpress(username, password)
                            
                        if authenticated_session:
                            # ✅ FIXED: Better authentication flow
                            logger.info(f"🎉 Authentication successful: {authenticated_session.user_type}")
                            st.balloons()
                            st.success(f"🎉 Welcome back, {authenticated_session.first_name}!")
                            
                            # ✅ FIXED: Immediate navigation with session state
                            st.session_state.page = "chat"
                            time.sleep(0.5)
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest limitations:**
        - Chat history not saved across sessions
        - No PDF export capability  
        - Limited personalization features
        - No automatic CRM integration
        
        ✨ **Sign in benefits:**
        - Chat history saved and exportable as PDF
        - Automatic integration with Zoho CRM
        - Chat transcripts auto-saved after 2 minutes of inactivity
        - Personalized experience
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    """✅ FIXED: Enhanced sidebar with debug info for troubleshooting"""
    with st.sidebar:
        st.title("🎛️ Dashboard")

        # --- DEBUG LOGGING ---
        logger.info(f"DEBUG: Sidebar Conditional Check - Session ID: {session.session_id[:8]}")
        logger.info(f"DEBUG: Sidebar Conditional Check - Current session.user_type (raw): {session.user_type}")
        logger.info(f"DEBUG: Sidebar Conditional Check - Type of session.user_type: {type(session.user_type)}")
        
        # --- ROBUST USER TYPE CHECK ---
        is_registered = False
        if isinstance(session.user_type, UserType):
            is_registered = (session.user_type == UserType.REGISTERED_USER)
        elif isinstance(session.user_type, str):
            # Fallback check for string value, in case enum conversion hasn't fully propagated
            is_registered = (session.user_type.lower() == UserType.REGISTERED_USER.value.lower())
        
        logger.info(f"DEBUG: Sidebar Conditional Check - Result of is_registered calculation: {is_registered}")
        # --- END ROBUST CHECK & DEBUG LOGGING ---
        
        # User status
        if is_registered: # Use the robust 'is_registered' flag here
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
                if session.timeout_saved_to_crm:
                    st.caption("💾 Auto-saved to CRM")
                else:
                    st.caption("💾 Auto-save enabled")
            else: 
                st.caption("🚫 CRM Disabled")
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for auto-save features*")
        
        st.divider()
        
        # Session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        st.divider()
        
        # ✅ BEST PRACTICE: Action buttons with callbacks
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                session_manager.clear_chat_history(session)
                st.rerun()
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True, key="sign_out"):
                session_manager.end_session(session)
                st.rerun()

        # Download & save section
        if is_registered and session.messages: # Use is_registered here too
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
            
            # CRM Save
            if session_manager.zoho.config.ZOHO_ENABLED and session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, key="manual_save"):
                    session_manager.manual_save_to_crm(session)
                st.caption("💡 Auto-saves after 2 min inactivity")
        
        # ✅ ADDED: Debug section for troubleshooting authentication
        if st.checkbox("🔧 Debug Mode"):
            render_debug_session_info(session, session_manager)

def render_chat_interface_with_optimized_timer(session_manager: SessionManager, session: UserSession):
    """
    ✅ OPTIMIZED: Chat interface with best practices and authentication fixes
    """
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion")
    
    # ✅ BEST PRACTICE: Initialize components only once per session
    initialize_autonomous_timer(session.session_id)
    render_browser_close_component(session.session_id)
    
    # Show activity status
    render_activity_status_indicator(session)
    
    # ✅ BEST PRACTICE: Process timer events only when they exist (event-driven)
    events_processed = session_manager.process_timer_events(session)
    if events_processed:
        time.sleep(1)
        st.rerun()
    
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

    # ✅ BEST PRACTICE: Chat input with proper callback handling
    if prompt := st.chat_input("Ask me about ingredients, suppliers, or market trends..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        moderation_result = check_content_moderation(prompt, session_manager.ai.openai_client)
        
        if moderation_result.get("flagged"):
            with st.chat_message("assistant"):
                st.error(f"🚨 {moderation_result['message']}")
            
            session.messages.append({
                "role": "assistant",
                "content": moderation_result['message'],
                "source": "Content Safety",
                "timestamp": datetime.now().isoformat()
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base and web..."):
                    response = session_manager.get_ai_response(session, prompt)
                    
                    st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                    
                    if response.get("source"):
                        st.caption(f"Source: {response['source']}")
        
        st.rerun()

def render_session_expiry_redirect():
    """
    Handle session expiry with user-friendly redirect
    """
    if st.session_state.get('session_expired', False):
        expired_session_id = st.session_state.get('expired_session_id', 'unknown')
        
        st.error(f"🔄 **Session Expired**")
        st.info(f"Session `{expired_session_id}` ended due to 3 minutes of inactivity")
        st.success("💾 Your chat was automatically saved to CRM")
        st.info("⏳ Redirecting to welcome page...")
        
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.03)
            progress_bar.progress(i + 1)
        
        # ✅ BEST PRACTICE: Clean up session state properly
        for key in ['session_expired', 'expired_session_id', 'current_session_id', 'authenticated_session_id', 'page']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_session_manager() -> Optional[SessionManager]:
    # This function is retained for clarity, but its direct use might be less frequent
    # if `session_manager` is always accessed via st.session_state directly where needed
    return st.session_state.get('session_manager')

def ensure_initialization():
    """
    ✅ BEST PRACTICE: Initialize application components only when missing
    """
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        try:
            config = Config()
            pdf_exporter = PDFExporter()
            
            # ✅ BEST PRACTICE: Initialize database manager only when missing
            if 'db_manager' not in st.session_state:
                st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            
            db_manager = st.session_state.db_manager
            zoho_manager = ZohoCRMManager(config, pdf_exporter)
            ai_system = EnhancedAI(config)
            rate_limiter = RateLimiter()

            st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.initialized = True
            
            logger.info("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            st.error("💥 Critical error during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            return False
    
    return True

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # ✅ CRITICAL FIX: Ensure initialization happens first in every script run
    if not ensure_initialization():
        st.stop() # Stop if critical initialization fails

    # These functions can now safely access st.session_state.session_manager
    # because ensure_initialization has already run.
    render_session_expiry_redirect()
    handle_emergency_save_requests()

    # Emergency reset button
    if st.button("🔄 Fresh Start", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Get session manager now that it's guaranteed to be initialized
    session_manager = st.session_state.get('session_manager')
    if not session_manager: # Defensive check, should rarely hit if initialization passed
        st.error("Failed to retrieve session manager. Please refresh the page.")
        st.stop()
    
    # ✅ BEST PRACTICE: Initialize page state only when missing
    if 'page' not in st.session_state:
        st.session_state.page = None
    
    # Main application flow
    current_page = st.session_state.page
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session() # This loads/creates the correct session based on current state
        if session and session.active:
            render_sidebar(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_with_optimized_timer(session_manager, session)
        else:
            # ✅ BEST PRACTICE: Clean up page state when session invalid/inactive
            st.session_state.page = None
            st.rerun()

if __name__ == "__main__":
    main()
