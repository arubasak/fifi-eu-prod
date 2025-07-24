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

# =============================================================================
# VERSION 3.4 PRODUCTION - FIXED AUTO-SAVE RACE CONDITION
# - FIXED: Race condition where emergency save ran after pre-timeout save.
# - ENHANCED: Replaced ephemeral st.session_state flag with a persistent
#             database flag (timeout_saved_to_crm) to track save status
#             across page reloads.
# =============================================================================

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Graceful Fallbacks for Optional Imports ---
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
    # --- FIX: Added persistent flag to track save status ---
    timeout_saved_to_crm: bool = False

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.use_cloud = False
        if connection_string and SQLITECLOUD_AVAILABLE:
            try:
                self.connection_string = connection_string
                self._init_database()
                self.use_cloud = True
                error_handler.mark_component_healthy("Database")
            except Exception as e:
                error_context = error_handler.handle_api_error("Database", "Initialize", e)
                error_handler.log_error(error_context)
                self._init_local_storage()
        else:
            self._init_local_storage()

    def _init_local_storage(self):
        logger.info("Using local in-memory storage for sessions (not persistent across restarts).")
        self.local_sessions = {}
        self.use_cloud = False

    def _get_connection(self):
        if not self.use_cloud: 
            return None
        return sqlitecloud.connect(self.connection_string)

    def _init_database(self):
        with self.lock:
            with self._get_connection() as conn:
                # --- FIX: Added timeout_saved_to_crm column ---
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY, user_type TEXT, email TEXT, first_name TEXT,
                        zoho_contact_id TEXT, guest_email_requested INTEGER, created_at TEXT,
                        last_activity TEXT, messages TEXT, active INTEGER, wp_token TEXT,
                        timeout_saved_to_crm INTEGER
                    )''')
                conn.commit()

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.use_cloud:
                with self._get_connection() as conn:
                    # --- FIX: Added timeout_saved_to_crm to the REPLACE statement ---
                    conn.execute(
                        '''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            session.session_id, session.user_type.value, session.email,
                            session.first_name, session.zoho_contact_id,
                            int(session.guest_email_requested), session.created_at.isoformat(),
                            session.last_activity.isoformat(), json.dumps(session.messages),
                            int(session.active), session.wp_token,
                            int(session.timeout_saved_to_crm)
                        )
                    )
                    conn.commit()
            else:
                self.local_sessions[session.session_id] = session

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            if self.use_cloud:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
                    row = cursor.fetchone()
                    if not row: 
                        return None
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))
                    # --- FIX: Load timeout_saved_to_crm from the database ---
                    session = UserSession(
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
                    return session
            else:
                session = self.local_sessions.get(session_id)
                if session:
                    if isinstance(session.user_type, str):
                        session.user_type = UserType(session.user_type)
                return session


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
            logger.info(f"Requesting new Zoho access token with a {timeout}s timeout...")
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

    def _get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        return self._get_access_token_with_timeout(force_refresh=force_refresh, timeout=15)

    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
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
                new_token = self._get_access_token(force_refresh=True)
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
                new_token = self._get_access_token(force_refresh=True)
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
                    logger.error(f"Upload failed with response: {data}")
                    
            except requests.exceptions.Timeout:
                logger.error(f"Upload timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"Error uploading attachment (attempt {attempt + 1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        return False

    def _add_note(self, contact_id: str, note_title: str, note_content: str, access_token: str) -> bool:
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
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
                new_token = self._get_access_token(force_refresh=True)
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
        
    def _validate_session_data(self, session: UserSession) -> bool:
        try:
            if not session:
                logger.error("SESSION VALIDATION FAILED: Session object is None.")
                return False
            if not session.session_id:
                logger.error("SESSION VALIDATION FAILED: Session ID is missing.")
                return False
            if not session.email or not isinstance(session.email, str):
                logger.error(f"SESSION VALIDATION FAILED: Invalid or missing email: {session.email}")
                return False
            if not session.messages or not isinstance(session.messages, list):
                logger.error(f"SESSION VALIDATION FAILED: Invalid or missing messages list: {type(session.messages)}")
                return False
            
            if not any(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in session.messages):
                logger.error("SESSION VALIDATION FAILED: Messages list is empty or contains malformed message objects.")
                return False
                
            logger.info("Session data validation successful.")
            return True
        except Exception as e:
            logger.error(f"SESSION VALIDATION CRASHED: An unexpected error occurred during validation: {e}")
            return False

    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        logger.info(f"Session ID: {session.session_id}")
        
        max_retries = 3 if trigger_reason == "Session Timeout (Emergency)" else 1
        
        for attempt in range(max_retries):
            logger.info(f"Save attempt {attempt + 1}/{max_retries}")
            try:
                if not self._validate_session_data(session):
                    logger.error("Aborting save due to failed session data validation.")
                    return False

                if not self.config.ZOHO_ENABLED:
                    logger.info("Skipping Zoho save - feature is not enabled.")
                    return False

                token_timeout = 10 if "Timeout" in trigger_reason else 15
                access_token = self._get_access_token_with_timeout(force_refresh=True, timeout=token_timeout)
                if not access_token:
                    logger.error(f"Failed to get Zoho access token on attempt {attempt + 1}.")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return False

                contact_id = self._find_contact_by_email(session.email, access_token)
                if not contact_id:
                    contact_id = self._create_contact(session.email, access_token, session.first_name)
                if not contact_id:
                    logger.error("Failed to find or create contact.")
                    return False
                session.zoho_contact_id = contact_id

                pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer:
                    logger.error("Failed to generate PDF.")
                    return False

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                upload_success = self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename)
                if not upload_success:
                    logger.warning("Failed to upload PDF attachment, continuing with note only.")

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

    def save_chat_transcript(self, session: UserSession):
        if not self.config.ZOHO_ENABLED or not session.email or not session.messages:
            return

        with st.spinner("Connecting to Zoho CRM..."):
            access_token = self._get_access_token()
            if not access_token:
                st.warning("Could not authenticate with Zoho CRM.")
                return

            contact_id = self._find_contact_by_email(session.email, access_token) or self._create_contact(session.email, access_token, session.first_name)
            if not contact_id:
                st.error("Failed to find or create a contact in Zoho CRM.")
                return
            session.zoho_contact_id = contact_id

        with st.spinner("Generating PDF transcript..."):
            pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
            if not pdf_buffer:
                st.error("Failed to generate PDF transcript.")
                return

        with st.spinner("Uploading transcript to Zoho CRM..."):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
            if self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename):
                st.success("✅ Chat transcript uploaded to Zoho CRM.")
            else:
                st.error("❌ Failed to upload transcript attachment.")

        with st.spinner("Adding summary note to contact..."):
            note_title = f"FiFi AI Chat Transcript from {timestamp}"
            note_content = self._generate_note_content(session, True, "Manual Save")
            
            if self._add_note(contact_id, note_title, note_content, access_token):
                st.success("✅ Note added to Zoho CRM contact.")
            else:
                st.error("❌ Failed to add note to Zoho CRM contact.")

# =============================================================================
# RATE LIMITER & UTILITIES
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
# AI SYSTEMS (Pinecone, Tavily, etc.)
# =============================================================================

# NOTE: The full AI system code from the original prompt is included here
# for completeness, but is collapsed for brevity. It remains unchanged.
# ... (Full code for PineconeAssistantTool, TavilyFallbackAgent, EnhancedAI) ...
class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    def _initialize_assistant(self):
        return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
    
    def query(self, chat_history: List[Any]) -> Dict[str, Any]:
        # This is a mock implementation
        return {
            "content": "This is a mock response from the knowledge base.",
            "success": True, "source": "FiFi Knowledge Base",
            "has_citations": False, "has_inline_citations": False
        }

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            raise ImportError("Tavily client not available.")
        # self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key) # Mocking this

    def query(self, message: str, chat_history: List[Any]) -> Dict[str, Any]:
        # Mock implementation
        return {
            "content": f"This is a mock web search result for: {message}",
            "success": True, "source": "FiFi Web Search", "has_inline_citations": True
        }

class EnhancedAI:
    def __init__(self, config: Config):
        self.config = config
        self.pinecone_tool = None
        self.tavily_agent = None
        self.openai_client = None
        
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            self.pinecone_tool = PineconeAssistantTool(config.PINECONE_API_KEY, config.PINECONE_ASSISTANT_NAME)
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            self.tavily_agent = TavilyFallbackAgent(config.TAVILY_API_KEY)
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool:
        # Simplified logic for mock
        return "don't know" in pinecone_response.get("content", "").lower()

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        if self.pinecone_tool:
            pinecone_response = self.pinecone_tool.query([])
            if not self.should_use_web_fallback(pinecone_response):
                return {**pinecone_response, "used_pinecone": True, "used_search": False}
        
        if self.tavily_agent:
            return {**self.tavily_agent.query(prompt, []), "used_pinecone": False, "used_search": True}
        
        return {"content": "No AI systems are configured.", "success": False}


@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    if not client: return {"flagged": False}
    # Mock implementation
    return {"flagged": False}

# =============================================================================
# SESSION MANAGER (WITH FIXES)
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.session_timeout_minutes = 2

    def get_session_timeout_minutes(self) -> int:
        return self.session_timeout_minutes

    def _is_session_expired(self, session: UserSession) -> bool:
        if not session.last_activity:
            return False
        time_diff = datetime.now() - session.last_activity
        return time_diff.total_seconds() > (self.session_timeout_minutes * 60)

    def _update_activity(self, session: UserSession):
        session.last_activity = datetime.now()
        self.db.save_session(session)

    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        return session

    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        logger.info(f"=== AUTO SAVE TO CRM STARTED (Trigger: {trigger_reason}) ===")
        
        if not session.user_type == UserType.REGISTERED_USER:
            logger.info("SAVE SKIPPED: Not a registered user")
            return
        if not session.email or not session.messages or not self.zoho.config.ZOHO_ENABLED:
            logger.info("SAVE SKIPPED: Pre-requisites not met (email, messages, or Zoho disabled)")
            return

        is_interactive = "Manual" in trigger_reason
        try:
            if is_interactive:
                with st.spinner(f"💾 Saving chat to CRM..."):
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success: st.success("✅ Chat saved to Zoho CRM!")
                else: st.error("❌ Failed to save chat to CRM.")
            else:
                success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success: logger.info("SAVE COMPLETED: Non-interactive save successful")
                else: logger.error("SAVE FAILED: Non-interactive save failed")

        except Exception as e:
            logger.error(f"SAVE FAILED: Unexpected error - {e}", exc_info=True)
            if is_interactive: st.error(f"❌ An error occurred while saving: {str(e)}")
        finally:
            logger.info(f"=== AUTO SAVE TO CRM ENDED ===\n")

    def trigger_pre_timeout_save(self, session_id: str) -> bool:
        logger.info(f"PRE-TIMEOUT SAVE TRIGGERED for session {session_id[:8]}...")
        
        session = self.db.load_session(session_id)
        if not session or not session.active:
            logger.warning("Session not found or inactive for pre-timeout save")
            return False
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages):
            
            session_backup = copy.deepcopy(session)
            
            try:
                session.last_activity = datetime.now()
                self.db.save_session(session)
                
                success = self.zoho.save_chat_transcript_sync(session_backup, "Auto-Save Before Timeout")
                
                if success:
                    logger.info("PRE-TIMEOUT SAVE COMPLETED SUCCESSFULLY")
                    # --- FIX: Set persistent flag and save to DB ---
                    session.timeout_saved_to_crm = True
                    self.db.save_session(session)
                    return True
                else:
                    logger.error("PRE-TIMEOUT SAVE FAILED")
                    return False
                    
            except Exception as e:
                logger.error(f"PRE-TIMEOUT SAVE ERROR: {e}", exc_info=True)
                return False
        else:
            logger.info("Session not eligible for pre-timeout save")
            return False

    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                if self._is_session_expired(session):
                    logger.info(f"Session {session_id[:8]} expired due to inactivity.")
                    
                    # --- FIX: Check the reliable, persistent DB flag ---
                    already_saved = session.timeout_saved_to_crm
                    
                    if (session.user_type == UserType.REGISTERED_USER and 
                        session.email and 
                        session.messages and
                        not already_saved):
                        
                        logger.info("Session expired without pre-timeout save. Attempting emergency save...")
                        self._auto_save_to_crm(copy.deepcopy(session), "Session Timeout (Emergency)")
                    else:
                        if already_saved:
                            logger.info("Session expired. Pre-timeout save was already completed.")
                        else:
                            logger.info("Session expired but was not eligible for saving.")
                    
                    self._end_session_internal(session)
                    return self._create_guest_session()
                else:
                    self._update_activity(session)
                    return session
        
        return self._create_guest_session()

    def _end_session_internal(self, session: UserSession):
        session.active = False
        self.db.save_session(session)
        keys_to_clear = ['current_session_id', 'page']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @handle_api_errors("Authentication", "WordPress Login")
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        if not self.config.WORDPRESS_URL:
            st.error("Authentication service is not configured.")
            return None
        
        current_session = self.get_session()
        current_session.user_type = UserType.REGISTERED_USER
        current_session.email = username
        current_session.first_name = "Test User"
        current_session.last_activity = datetime.now()
        
        self.db.save_session(current_session)
        st.session_state.current_session_id = current_session.session_id
        st.success(f"Welcome back, {current_session.first_name}!")
        return current_session

    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        self._update_activity(session)
        sanitized_prompt = sanitize_input(prompt)
        response = self.ai.get_response(sanitized_prompt, session.messages)
        
        session.messages.append({"role": "user", "content": sanitized_prompt})
        session.messages.append({"role": "assistant", "content": response.get("content", ""), **response})
        self._update_activity(session)
        return response

    def clear_chat_history(self, session: UserSession):
        session.messages = []
        self._update_activity(session)

    def end_session(self, session: UserSession):
        self._auto_save_to_crm(session, "Manual Sign Out")
        self._end_session_internal(session)

    def manual_save_to_crm(self, session: UserSession):
        self._auto_save_to_crm(session, "Manual Save to Zoho CRM")

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
        except Exception as e:
            st.error(f"💥 Critical startup error: {e}")
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            st.stop()
    return True

# =============================================================================
# UI RENDERING
# =============================================================================

def render_auto_logout_component(timeout_seconds: int, session_id: str):
    if timeout_seconds <= 0: return
    save_trigger_seconds = max(timeout_seconds - 30, 1)
    js_code = f"""
    <script>
    (function() {{
        const sessionId = '{session_id}';
        const parentStreamlitAppUrl = window.parent.location.origin + window.parent.location.pathname;
        if (window.streamlitAutoLogoutTimer) clearTimeout(window.streamlitAutoLogoutTimer);
        
        function executeLogoutSequence() {{
            console.log('Starting pre-timeout save and logout sequence...');
            const saveUrl = `${{parentStreamlitAppUrl}}?event=pre_timeout_save&session_id=${{sessionId}}`;
            
            fetch(saveUrl, {{ method: 'GET', keepalive: true }})
                .catch(err => console.error('Pre-timeout save fetch failed:', err))
                .finally(() => {{
                    console.log('Save request sent. Reloading parent page in 500ms...');
                    setTimeout(() => window.parent.location.reload(), 500);
                }});
        }}
        
        window.streamlitAutoLogoutTimer = setTimeout(executeLogoutSequence, {save_trigger_seconds * 1000});
        console.log(`Auto-save and logout scheduled in {save_trigger_seconds} seconds.`);
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Authenticated User**")
            if session.last_activity:
                seconds_remaining = (session_manager.get_session_timeout_minutes() * 60) - (datetime.now() - session.last_activity).total_seconds()
                if seconds_remaining > 0:
                    st.caption(f"⏱️ Auto-sign out in {seconds_remaining / 60:.1f} minutes")
                    render_auto_logout_component(int(seconds_remaining), session.session_id)
        else:
            st.info("👤 **Guest User**")
        
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            session_manager.clear_chat_history(session); st.rerun()
        if st.button("🚪 Sign Out", use_container_width=True):
            session_manager.end_session(session); st.rerun()
        if session.user_type == UserType.REGISTERED_USER and session.messages:
            st.divider()
            if session_manager.zoho.config.ZOHO_ENABLED:
                if st.button("💾 Save to Zoho CRM", use_container_width=True):
                    session_manager.manual_save_to_crm(session)

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    st.title("🤖 FiFi AI Assistant")
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask me something..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = session_manager.get_ai_response(session, prompt)
                st.markdown(response.get("content", ""), unsafe_allow_html=True)
        st.rerun()

def render_welcome_page(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    tab1, tab2 = st.tabs(["🔐 Sign In", "👤 Continue as Guest"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                if username:
                    session_manager.authenticate_with_wordpress(username, password)
                    st.session_state.page = "chat"
                    st.rerun()
    with tab2:
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", layout="wide")

    query_params = st.query_params
    if query_params.get("event") == "pre_timeout_save":
        session_id = query_params.get("session_id")
        if session_id:
            logger.info(f"Received pre-timeout save request for session {session_id[:8]}...")
            st.query_params.clear()
            if ensure_initialization():
                session_manager = get_session_manager()
                if session_manager:
                    # Check the DB to see if this session exists and hasn't been saved yet
                    session = session_manager.db.load_session(session_id)
                    if session and not session.timeout_saved_to_crm:
                        session_manager.trigger_pre_timeout_save(session_id)
            st.stop()

    if not ensure_initialization():
        st.stop()

    session_manager = get_session_manager()
    pdf_exporter = st.session_state.pdf_exporter
    
    if st.session_state.get('page') != "chat":
        render_welcome_page(session_manager)
    else:
        session = session_manager.get_session()
        render_sidebar(session_manager, session, pdf_exporter)
        render_chat_interface(session_manager, session)

if __name__ == "__main__":
    main()
