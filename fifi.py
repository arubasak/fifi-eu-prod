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
# VERSION 3.3 PRODUCTION - FIXED PRE-TIMEOUT SAVE
# - FIXED: Pre-timeout save mechanism triggers 30 seconds before timeout
# - FIXED: JavaScript coordination ensures save completes before reload
# - FIXED: Query parameter handling for save requests
# - ENHANCED: Save tracking prevents duplicate saves
# - ENHANCED: Visual feedback during auto-save
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
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY, user_type TEXT, email TEXT, first_name TEXT,
                        zoho_contact_id TEXT, guest_email_requested INTEGER, created_at TEXT,
                        last_activity TEXT, messages TEXT, active INTEGER, wp_token TEXT
                    )''')
                conn.commit()

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.use_cloud:
                with self._get_connection() as conn:
                    conn.execute(
                        '''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            session.session_id, session.user_type.value, session.email,
                            session.first_name, session.zoho_contact_id,
                            int(session.guest_email_requested), session.created_at.isoformat(),
                            session.last_activity.isoformat(), json.dumps(session.messages),
                            int(session.active), session.wp_token
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
                        wp_token=row_dict.get('wp_token')
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
            
            # Remove HTML tags for PDF
            content = re.sub(r'<[^>]+>', '', content)
            
            style = self.styles['UserMessage'] if role == 'User' else self.styles['Normal']
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{role}:</b> {content}", style))
            
            # Add source information if available
            if msg.get('source'):
                story.append(Paragraph(f"<i>Source: {msg['source']}</i>", self.styles['Normal']))
                
        doc.build(story)
        buffer.seek(0)
        return buffer

# =============================================================================
# FIXED ZOHO CRM MANAGER
# =============================================================================

class ZohoCRMManager:
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        self._access_token = None
        self._token_expiry = None

    # SOLUTION: New method to get token with a configurable timeout
    def _get_access_token_with_timeout(self, force_refresh: bool = False, timeout: int = 15) -> Optional[str]:
        """Get access token with caching, retry logic, and a configurable timeout."""
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
                timeout=timeout # Using configurable timeout
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
        """Legacy get access token method, now calls the new one with a default timeout."""
        return self._get_access_token_with_timeout(force_refresh=force_refresh, timeout=15)


    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
        """Find contact with retry on token expiry."""
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        params = {'criteria': f'(Email:equals:{email})'}
        
        try:
            response = requests.get(
                f"{self.base_url}/Contacts/search", 
                headers=headers, 
                params=params, 
                timeout=10
            )
            
            if response.status_code == 401:  # Token expired
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
        """Create contact with retry on token expiry."""
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
            
            if response.status_code == 401:  # Token expired
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
        """Upload attachment with retry and better error handling."""
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        upload_url = f"{self.base_url}/Contacts/{contact_id}/Attachments"
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Reset buffer position
                pdf_buffer.seek(0)
                
                response = requests.post(
                    upload_url, 
                    headers=headers, 
                    files={'file': (filename, pdf_buffer.read(), 'application/pdf')},
                    timeout=60  # Increased timeout for large files
                )
                
                if response.status_code == 401:  # Token expired
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
        """Add note with retry on token expiry."""
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Truncate note content if too long (Zoho has limits)
        max_content_length = 32000  # Zoho's limit is typically 32KB
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
            
            if response.status_code == 401:  # Token expired
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
        
    # SOLUTION: New method to validate session data before saving
    def _validate_session_data(self, session: UserSession) -> bool:
        """Validate critical session data before attempting a save to CRM."""
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
            
            # Check for at least one valid message
            if not any(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in session.messages):
                logger.error("SESSION VALIDATION FAILED: Messages list is empty or contains malformed message objects.")
                return False
                
            logger.info("Session data validation successful.")
            return True
        except Exception as e:
            logger.error(f"SESSION VALIDATION CRASHED: An unexpected error occurred during validation: {e}")
            return False


    # SOLUTION: Rewritten save method with retries, timeouts, and validation
    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        """
        Synchronous save method with retries, timeouts, and validation.
        Returns True on success, False on failure.
        """
        logger.info("=" * 80)
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}")
        logger.info(f"Session ID: {session.session_id}")
        
        # Determine retry strategy based on trigger
        max_retries = 3 if trigger_reason == "Session Timeout" else 1
        
        for attempt in range(max_retries):
            logger.info(f"Save attempt {attempt + 1}/{max_retries}")
            try:
                # STEP 0: Validate session data on each attempt
                if not self._validate_session_data(session):
                    logger.error("Aborting save due to failed session data validation.")
                    return False # No point in retrying if data is bad

                if not self.config.ZOHO_ENABLED:
                    logger.info("Skipping Zoho save - feature is not enabled.")
                    return False

                # STEP 1: Get access token with a shorter timeout for background saves
                token_timeout = 10 if trigger_reason == "Session Timeout" else 15
                access_token = self._get_access_token_with_timeout(force_refresh=True, timeout=token_timeout)
                if not access_token:
                    logger.error(f"Failed to get Zoho access token on attempt {attempt + 1}.")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt) # Exponential backoff
                        continue
                    return False

                # STEP 2: Find or create contact
                contact_id = self._find_contact_by_email(session.email, access_token)
                if not contact_id:
                    contact_id = self._create_contact(session.email, access_token, session.first_name)
                if not contact_id:
                    logger.error("Failed to find or create contact.")
                    # This is a critical failure, likely not transient, so we don't retry
                    return False
                session.zoho_contact_id = contact_id

                # STEP 3: Generate PDF
                pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer:
                    logger.error("Failed to generate PDF.")
                    return False

                # STEP 4: Upload attachment
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                upload_success = self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename)
                if not upload_success:
                    logger.warning("Failed to upload PDF attachment, continuing with note only.")

                # STEP 5: Add note
                note_title = f"FiFi AI Chat Transcript from {timestamp} ({trigger_reason})"
                note_content = self._generate_note_content(session, upload_success, trigger_reason)
                note_success = self._add_note(contact_id, note_title, note_content, access_token)
                if not note_success:
                    logger.error("Failed to add note to contact.")
                    return False # A failure here means the save was incomplete

                logger.info("=" * 80)
                logger.info(f"ZOHO SAVE COMPLETED SUCCESSFULLY on attempt {attempt + 1}")
                logger.info(f"Contact ID: {contact_id}")
                logger.info("=" * 80)
                return True # Success!

            except Exception as e:
                logger.error("=" * 80)
                logger.error(f"ZOHO SAVE FAILED on attempt {attempt + 1} with an exception.")
                logger.error(f"Error: {type(e).__name__}: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff before next retry
                else:
                    logger.error("Max retries reached. Aborting save.")
                    return False
        
        return False # Should not be reached, but as a fallback


    def _generate_note_content(self, session: UserSession, attachment_uploaded: bool, trigger_reason: str) -> str:
        """Generate note content with session summary."""
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
            
            # Limit each message to reasonable length
            max_msg_length = 500
            if len(content) > max_msg_length:
                content = content[:max_msg_length] + "..."
                
            note_content += f"\n{i+1}. **{role}:** {content}\n"
            
            if msg.get("source"):
                note_content += f"   _Source: {msg['source']}_\n"
                
        return note_content

    def save_chat_transcript(self, session: UserSession):
        """Legacy method for manual saves with UI feedback."""
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
# AI SYSTEM - PINECONE ASSISTANT
# =============================================================================

def insert_citations(response) -> str:
    """
    Insert clickable citation markers. If a source URL is available, the marker
    links directly to it in a new tab. Otherwise, it links to the citation list at the bottom.
    """
    if not hasattr(response, 'citations') or not response.citations:
        return response.message.content

    result = response.message.content
    citations = response.citations
    offset = 0

    sorted_citations = sorted(enumerate(citations, start=1), key=lambda x: x[1].position)

    for i, cite in sorted_citations:
        link_url = None
        if hasattr(cite, 'references') and cite.references:
            reference = cite.references[0]
            if hasattr(reference, 'file') and reference.file:
                if hasattr(reference.file, 'metadata') and reference.file.metadata:
                    link_url = reference.file.metadata.get('source_url')
                if not link_url and hasattr(reference.file, 'signed_url') and reference.file.signed_url:
                    link_url = reference.file.signed_url

                if link_url:
                    if '?' in link_url:
                        link_url += '&utm_source=fifi-in'
                    else:
                        link_url += '?utm_source=fifi-in'

        if link_url:
            citation_marker = f" <a href='{link_url}' target='_blank' title='Source: {link_url}'>[{i}]</a>"
        else:
            citation_marker = f" <a href='#cite-{i}'>[{i}]</a>"

        position = cite.position
        adjusted_position = position + offset

        if adjusted_position <= len(result):
            result = result[:adjusted_position] + citation_marker + result[adjusted_position:]
            offset += len(citation_marker)

    return result

class PineconeAssistantTool:
    """Advanced Pinecone Assistant with clickable inline citations, token limit detection and enhanced error handling."""

    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE:
            error_context = error_handler.handle_api_error("Pinecone", "Import", ImportError("Pinecone client not available"))
            error_handler.display_error_to_user(error_context)
            raise ImportError("Pinecone client not available.")

        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

    @handle_api_errors("Pinecone", "Initialize Assistant")
    def _initialize_assistant(self):
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
                st.info(f"🔧 Creating new Pinecone assistant: '{self.assistant_name}'")
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name,
                    instructions=instructions
                )
            else:
                st.success(f"✅ Connected to Pinecone assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            raise e

    @handle_api_errors("Pinecone", "Query Knowledge Base", show_to_user=False)
    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant:
            return {
                "content": "Pinecone assistant not available.",
                "success": False,
                "source": "error",
                "error_type": "unavailable"
            }

        try:
            pinecone_messages = [
                PineconeMessage(
                    role="user" if isinstance(msg, HumanMessage) else "assistant",
                    content=msg.content
                ) for msg in chat_history
            ]

            response = self.assistant.chat(
                messages=pinecone_messages,
                model="gpt-4o",
                include_highlights=True
            )

            content_with_inline_citations = insert_citations(response)

            has_citations = False
            has_inline_citations = False

            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                has_inline_citations = True

                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
                seen_items = set()

                for i, citation in enumerate(response.citations, 1):
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
                                    markdown_link = f"[{display_text}]({link_url})"
                                    final_item = f"<a id='cite-{i}'></a>{i}. {markdown_link}"
                                    citations_list.append(final_item)
                                    seen_items.add(display_text)
                            else:
                                display_text = getattr(reference.file, 'name', 'Unknown Source')
                                if display_text not in seen_items:
                                    final_item = f"<a id='cite-{i}'></a>{i}. {display_text}"
                                    citations_list.append(final_item)
                                    seen_items.add(display_text)

                if citations_list:
                    content_with_inline_citations += citations_header + "\n".join(citations_list)

            return {
                "content": content_with_inline_citations,
                "success": True,
                "source": "FiFi Knowledge Base",
                "has_citations": has_citations,
                "has_inline_citations": has_inline_citations,
                "response_length": len(content_with_inline_citations)
            }

        except Exception as e:
            raise e

# =============================================================================
# AI SYSTEM - TAVILY FALLBACK
# =============================================================================

class TavilyFallbackAgent:
    """Tavily fallback agent with smart result synthesis, inline citations, UTM tracking."""

    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            error_context = error_handler.handle_api_error("Tavily", "Import", ImportError("Tavily client not available"))
            error_handler.display_error_to_user(error_context)
            raise ImportError("Tavily client not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    def add_utm_to_url(self, url: str) -> str:
        if not url:
            return url
        utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
        if '?' in url:
            new_url = f"{url}&{utm_params}"
        else:
            new_url = f"{url}?{utm_params}"
        return new_url

    def synthesize_search_results(self, results, query: str) -> str:
        if isinstance(results, str):
            return f"Based on my search: {results}"

        if isinstance(results, dict):
            if results.get('answer'):
                return f"Based on my search: {results['answer']}"

            search_results = results.get('results', [])
            if not search_results:
                return "I couldn't find any relevant information for your query."

            relevant_info = []
            sources = []
            source_urls = []

            for i, result in enumerate(search_results[:5], 1):
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or
                             result.get('snippet') or
                             result.get('description') or
                             result.get('summary', ''))
                    url = result.get('url', '')

                    if content:
                        if len(content) > 400:
                            content = content[:400] + "..."
                        
                        if url:
                            url_with_utm = self.add_utm_to_url(url)
                            content_with_citation = f"{content} <a href='{url_with_utm}' target='_blank' title='Source: {title}'>[{i}]</a>"
                            relevant_info.append(content_with_citation)
                            sources.append(f"[{title}]({url_with_utm})")
                            source_urls.append(url_with_utm)
                        else:
                            content_with_citation = f"{content} <a href='#cite-{i}'>[{i}]</a>"
                            relevant_info.append(content_with_citation)
                            sources.append(f"{title}")

            if not relevant_info:
                return "I found search results but couldn't extract readable content. Please try rephrasing your query."

            response_parts = []

            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search, here's what I found:")
                for i, info in enumerate(relevant_info, 1):
                    response_parts.append(f"\n\n**{i}.** {info}")

            if sources:
                response_parts.append(f"\n\n---\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    if source_urls and i <= len(source_urls):
                        response_parts.append(f"\n<a id='cite-{i}'></a>{i}. {source}")
                    else:
                        response_parts.append(f"\n<a id='cite-{i}'></a>{i}. {source}")

            return "".join(response_parts)

        if isinstance(results, list):
            relevant_info = []
            sources = []
            source_urls = []

            for i, result in enumerate(results[:5], 1):
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or
                             result.get('snippet') or
                             result.get('description', ''))
                    url = result.get('url', '')

                    if content:
                        if len(content) > 400:
                            content = content[:400] + "..."
                        
                        if url:
                            url_with_utm = self.add_utm_to_url(url)
                            content_with_citation = f"{content} <a href='{url_with_utm}' target='_blank' title='Source: {title}'>[{i}]</a>"
                            relevant_info.append(content_with_citation)
                            sources.append(f"[{title}]({url_with_utm})")
                            source_urls.append(url_with_utm)
                        else:
                            content_with_citation = f"{content} <a href='#cite-{i}'>[{i}]</a>"
                            relevant_info.append(content_with_citation)
                            sources.append(title)

            if not relevant_info:
                return "I couldn't find relevant information for your query."

            response_parts = []
            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search:")
                for i, info in enumerate(relevant_info, 1):
                    response_parts.append(f"\n**{i}.** {info}")

            if sources:
                response_parts.append(f"\n\n---\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    if source_urls and i <= len(source_urls):
                        response_parts.append(f"\n<a id='cite-{i}'></a>{i}. {source}")
                    else:
                        response_parts.append(f"\n<a id='cite-{i}'></a>{i}. {source}")

            return "".join(response_parts)

        return "I couldn't find any relevant information for your query."

    @handle_api_errors("Tavily", "Web Search", show_to_user=False)
    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            search_results = self.tavily_tool.invoke({"query": message})
            synthesized_content = self.synthesize_search_results(search_results, message)
            
            return {
                "content": synthesized_content,
                "success": True,
                "source": "FiFi Web Search",
                "has_inline_citations": True
            }
        except Exception as e:
            raise e

# =============================================================================
# ENHANCED AI SYSTEM
# =============================================================================

class EnhancedAI:
    """Enhanced AI with Pinecone knowledge base, inline citations, smart Tavily fallback."""

    def __init__(self, config: Config):
        self.config = config
        self.pinecone_tool = None
        self.tavily_agent = None
        self.openai_client = None
        self.langchain_llm = None

        # Initialize OpenAI clients
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                error_context = error_handler.handle_api_error("OpenAI", "Initialize Client", e)
                error_handler.log_error(error_context)

        if LANGCHAIN_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.langchain_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=self.config.OPENAI_API_KEY,
                    temperature=0.7
                )
                error_handler.mark_component_healthy("LangChain")
            except Exception as e:
                logger.error(f"LangChain LLM initialization failed: {e}")
                error_context = error_handler.handle_api_error("LangChain", "Initialize LLM", e)
                error_handler.log_error(error_context)

        # Initialize Pinecone Assistant
        if PINECONE_AVAILABLE and self.config.PINECONE_API_KEY and self.config.PINECONE_ASSISTANT_NAME:
            try:
                self.pinecone_tool = PineconeAssistantTool(
                    api_key=self.config.PINECONE_API_KEY,
                    assistant_name=self.config.PINECONE_ASSISTANT_NAME
                )
                logger.info("Pinecone Assistant initialized successfully")
            except Exception as e:
                logger.error(f"Pinecone Assistant initialization failed: {e}")
                self.pinecone_tool = None

        # Initialize Tavily Fallback Agent
        if TAVILY_AVAILABLE and self.config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(tavily_api_key=self.config.TAVILY_API_KEY)
                logger.info("Tavily Fallback Agent initialized successfully")
            except Exception as e:
                logger.error(f"Tavily Fallback Agent initialization failed: {e}")
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
            return True

        # PRIORITY 3: Detect fake files/images/paths (CRITICAL SAFETY)
        fake_file_patterns = [
            ".jpg", ".jpeg", ".png", ".html", ".gif", ".doc", ".docx",
            ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".avi", ".mp3",
            "/uploads/", "/files/", "/images/", "/documents/", "/media/",
            "file://", "ftp://", "path:", "directory:", "folder:"
        ]

        has_real_citations = pinecone_response.get("has_citations", False)
        has_inline_citations = pinecone_response.get("has_inline_citations", False)

        if any(pattern in content_raw for pattern in fake_file_patterns):
            if not has_real_citations:
                return True

        # PRIORITY 4: Detect potential fake citations (CRITICAL)
        if "[1]" in content_raw or "**Sources:**" in content_raw:
            suspicious_patterns = [
                "http://", ".org", ".net",
                "example.com", "website.com", "source.com", "domain.com"
            ]
            if not has_real_citations and any(pattern in content_raw for pattern in suspicious_patterns):
                return True

        # PRIORITY 5: NO CITATIONS = MANDATORY FALLBACK (unless very short or inline citations present)
        if not has_real_citations and not has_inline_citations:
            if "[1]" not in content_raw and "**Sources:**" in content_raw:
                if len(content_raw.strip()) > 30:
                    return True

        # PRIORITY 6: General knowledge indicators (likely hallucination)
        general_knowledge_red_flags = [
            "generally", "typically", "usually", "commonly", "often", "most",
            "according to", "it is known", "studies show", "research indicates",
            "experts say", "based on", "in general", "as a rule"
        ]
        if any(flag in content for flag in general_knowledge_red_flags):
            return True

        # PRIORITY 7: Question-answering patterns that suggest general knowledge
        qa_patterns = [
            "the answer is", "this is because", "the reason", "due to the fact",
            "this happens when", "the cause of", "this occurs"
        ]
        if any(pattern in content for pattern in qa_patterns):
            if not pinecone_response.get("has_citations", False) and not pinecone_response.get("has_inline_citations", False):
                return True

        # PRIORITY 8: Response length suggests substantial answer without sources
        response_length = pinecone_response.get("response_length", 0)
        if response_length > 100 and not pinecone_response.get("has_citations", False) and not pinecone_response.get("has_inline_citations", False):
            return True

        return False

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get enhanced AI response with comprehensive error handling and recovery."""
        try:
            # Convert chat history to LangChain format
            langchain_history = []
            if chat_history:
                for msg in chat_history[-10:]:  # Last 10 messages to avoid token limits
                    if msg.get("role") == "user":
                        langchain_history.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        # Strip HTML tags from assistant messages for context
                        content = re.sub(r'<[^>]+>', '', msg.get("content", ""))
                        langchain_history.append(AIMessage(content=content))

            # Add current prompt
            langchain_history.append(HumanMessage(content=prompt))

            # STEP 1: Try Pinecone Knowledge Base FIRST
            if self.pinecone_tool:
                pinecone_response = self.pinecone_tool.query(langchain_history)

                if pinecone_response and pinecone_response.get("success"):
                    should_fallback = self.should_use_web_fallback(pinecone_response)

                    if not should_fallback:
                        logger.info("Using Pinecone knowledge base response")
                        return {
                            "content": pinecone_response["content"],
                            "source": pinecone_response.get("source", "FiFi Knowledge Base"),
                            "used_search": False,
                            "used_pinecone": True,
                            "has_citations": pinecone_response.get("has_citations", False),
                            "has_inline_citations": pinecone_response.get("has_inline_citations", False),
                            "safety_override": False,
                            "success": True
                        }
                    else:
                        logger.warning("SAFETY OVERRIDE: Detected potentially fabricated information")

            # STEP 2: Fall back to Tavily Web Search
            if self.tavily_agent:
                logger.info("Using Tavily web search fallback")
                tavily_response = self.tavily_agent.query(prompt, langchain_history[:-1])

                if tavily_response and tavily_response.get("success"):
                    return {
                        "content": tavily_response["content"],
                        "source": tavily_response.get("source", "FiFi Web Search"),
                        "used_search": True,
                        "used_pinecone": False,
                        "has_citations": False,
                        "has_inline_citations": True,
                        "safety_override": True if self.pinecone_tool else False,
                        "success": True
                    }
                else:
                    logger.warning("Tavily search failed, proceeding to final fallback")

            # STEP 3: Final fallback
            if LANGCHAIN_AVAILABLE and not self.pinecone_tool and not self.tavily_agent:
                # No AI tools available
                return {
                    "content": "I understand you're asking about: " + prompt + ". However, the AI knowledge base and web search features are not currently configured. Please ensure the Pinecone and Tavily API keys are set in the configuration.",
                    "source": "System",
                    "used_search": False,
                    "used_pinecone": False,
                    "has_citations": False,
                    "has_inline_citations": False,
                    "safety_override": False,
                    "success": False
                }

            # Final fallback with helpful error message
            return {
                "content": "I apologize, but all AI systems are currently experiencing issues. Please try again in a few minutes, or try rephrasing your question.",
                "source": "System Status",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "has_inline_citations": False,
                "safety_override": False,
                "success": False
            }

        except Exception as e:
            logger.error(f"Enhanced AI response error: {e}")
            error_context = error_handler.handle_api_error("AI System", "Generate Response", e)
            error_handler.log_error(error_context)

            return {
                "content": f"I'm experiencing technical difficulties. {error_context.user_message}",
                "source": "Error Recovery",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "has_inline_citations": False,
                "safety_override": False,
                "success": False
            }

# =============================================================================
# CONTENT MODERATION
# =============================================================================

@handle_api_errors("Content Moderation", "Check Prompt")
def check_content_moderation(prompt: str, client: Optional[openai.OpenAI]) -> Optional[Dict[str, Any]]:
    if not client: 
        return {"flagged": False}
    
    if not hasattr(client, 'moderations'):
        logger.warning("Moderation check skipped: OpenAI client missing moderations attribute.")
        return {"flagged": False}
    
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
    
    return {"flagged": False}

# =============================================================================
# FIXED SESSION MANAGER
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.session_timeout_minutes = 5

    def get_session_timeout_minutes(self) -> int:
        return getattr(self, 'session_timeout_minutes', 5)

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

    # --- CORRECTED METHOD ---
    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        """
        FIXED: Enhanced auto-save with proper error handling, logging, and separated logic.
        """
        logger.info(f"=== AUTO SAVE TO CRM STARTED ===")
        logger.info(f"Trigger: {trigger_reason}")
        logger.info(f"Session ID: {session.session_id[:8] if session.session_id else 'None'}")
        logger.info(f"User Type: {session.user_type}")
        logger.info(f"Has Email: {bool(session.email)}")
        logger.info(f"Message Count: {len(session.messages) if session.messages else 0}")
        logger.info(f"Zoho Enabled: {self.zoho.config.ZOHO_ENABLED}")

        # Check prerequisites
        if not session.user_type == UserType.REGISTERED_USER:
            logger.info("SAVE SKIPPED: Not a registered user")
            return
        if not session.email:
            logger.info("SAVE SKIPPED: No email address")
            return
        if not session.messages:
            logger.info("SAVE SKIPPED: No messages to save")
            return
        if not self.zoho.config.ZOHO_ENABLED:
            logger.info("SAVE SKIPPED: Zoho is not enabled")
            return

        is_interactive = trigger_reason in ["Manual Sign Out", "Manual Save to Zoho CRM", "Manual Test"]

        try:
            if is_interactive:
                with st.spinner(f"💾 Saving chat to CRM ({trigger_reason.lower()})..."):
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                
                if success:
                    st.success("✅ Chat saved to Zoho CRM!")
                    logger.info("SAVE COMPLETED: Interactive save successful")
                else:
                    st.error("❌ Failed to save chat to CRM. Please try again.")
                    logger.error("SAVE FAILED: Interactive save failed")
            else:
                # Non-interactive save (e.g., timeout, browser close)
                logger.info(f"Starting non-interactive save...")
                success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success:
                    logger.info("SAVE COMPLETED: Non-interactive save successful")
                else:
                    logger.error("SAVE FAILED: Non-interactive save failed")

        except Exception as e:
            logger.error(f"SAVE FAILED: Unexpected error - {type(e).__name__}: {str(e)}", exc_info=True)
            if is_interactive:
                st.error(f"❌ An error occurred while saving: {str(e)}")
        finally:
            logger.info(f"=== AUTO SAVE TO CRM ENDED ===\n")

    def trigger_pre_timeout_save(self, session_id: str) -> bool:
        """
        SOLUTION: Trigger a save 30 seconds before timeout expires.
        Called by JavaScript via query parameter.
        """
        logger.info(f"PRE-TIMEOUT SAVE TRIGGERED for session {session_id[:8]}...")
        
        session = self.db.load_session(session_id)
        if not session or not session.active:
            logger.warning("Session not found or inactive for pre-timeout save")
            return False
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages):
            
            # Create a backup copy to ensure data integrity
            session_backup = copy.deepcopy(session)
            
            try:
                # Mark in session that save is in progress
                session.last_activity = datetime.now()  # Update activity to prevent concurrent timeout
                self.db.save_session(session)
                
                # Perform the save
                success = self.zoho.save_chat_transcript_sync(session_backup, "Auto-Save Before Timeout")
                
                if success:
                    logger.info("PRE-TIMEOUT SAVE COMPLETED SUCCESSFULLY")
                    # Mark session as saved
                    if 'pre_timeout_saved' not in st.session_state:
                        st.session_state.pre_timeout_saved = {}
                    st.session_state.pre_timeout_saved[session_id] = True
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
        """
        Enhanced session retrieval with pre-timeout save check.
        """
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                if self._is_session_expired(session):
                    logger.info(f"Session {session_id[:8]}... expired due to inactivity.")
                    
                    # Check if pre-timeout save already happened
                    already_saved = st.session_state.get('pre_timeout_saved', {}).get(session_id, False)
                    
                    if (session.user_type == UserType.REGISTERED_USER and 
                        session.email and 
                        session.messages and
                        not already_saved):  # Only save if not already saved by pre-timeout
                        
                        logger.info("Session expired without pre-timeout save. Attempting emergency save...")
                        session_backup = copy.deepcopy(session)
                        
                        try:
                            self._auto_save_to_crm(session_backup, "Session Timeout (Emergency)")
                        except Exception as e:
                            logger.error(f"Emergency save failed: {e}", exc_info=True)
                    else:
                        if already_saved:
                            logger.info("Session expired. Pre-timeout save was already completed.")
                        else:
                            logger.info("Session expired but was not eligible for saving.")
                    
                    # Clear the pre-timeout save flag
                    if 'pre_timeout_saved' in st.session_state and session_id in st.session_state.pre_timeout_saved:
                        del st.session_state.pre_timeout_saved[session_id]
                    
                    # End the session
                    self._end_session_internal(session)
                    return self._create_guest_session()
                else:
                    # Session is active and not expired
                    self._update_activity(session)
                    return session
        
        # No session or inactive
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

                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.first_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                
                self.db.save_session(current_session)
                
                verification_session = self.db.load_session(current_session.session_id)
                if verification_session and verification_session.user_type == UserType.REGISTERED_USER:
                    st.session_state.current_session_id = current_session.session_id
                    st.success(f"Welcome back, {current_session.first_name}!")
                    return current_session
                else:
                    st.error("Authentication failed - session could not be saved.")
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
        
        if response.get("used_search"):
            response_message["used_search"] = True
        if response.get("used_pinecone"):
            response_message["used_pinecone"] = True
        if response.get("has_citations"):
            response_message["has_citations"] = True
        if response.get("has_inline_citations"):
            response_message["has_inline_citations"] = True
        if response.get("safety_override"):
            response_message["safety_override"] = True
            
        session.messages.append(response_message)
        
        session.messages = session.messages[-100:]
        
        self._update_activity(session)
        return response

    def clear_chat_history(self, session: UserSession):
        session.messages = []
        self._update_activity(session)

    def end_session(self, session: UserSession):
        self._auto_save_to_crm(session, "Manual Sign Out")
        self._end_session_internal(session)

    def manual_save_to_crm(self, session: UserSession):
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._auto_save_to_crm(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_session_manager() -> Optional[SessionManager]:
    """Safely get the session manager from session state."""
    if 'session_manager' not in st.session_state:
        return None
    
    manager = st.session_state.session_manager
    if not hasattr(manager, 'get_session'):
        logger.error("Invalid SessionManager instance in session state")
        return None
    
    return manager

def ensure_initialization():
    """Ensure the application is properly initialized."""
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

# =============================================================================
# UI RENDERING FUNCTIONS
# =============================================================================

def render_auto_logout_component(timeout_seconds: int, session_id: str, session_manager: SessionManager):
    """
    SOLUTION: More robust auto-logout that chains the save and reload events.
    This helps prevent the "message channel closed" error from browser extensions.
    """
    if timeout_seconds <= 0:
        return

    # Calculate when to trigger the save (30 seconds before timeout)
    save_trigger_seconds = max(timeout_seconds - 30, 1)
    
    js_code = f"""
    <script>
    (function() {{
        const sessionId = '{session_id}';
        const baseUrl = window.location.origin + window.location.pathname;
        
        // Clear any existing timers to prevent duplicates
        if (window.streamlitAutoSaveTimer) clearTimeout(window.streamlitAutoSaveTimer);
        if (window.streamlitAutoLogoutTimer) clearTimeout(window.streamlitAutoLogoutTimer);
        
        // Function to trigger the save. It now returns the fetch promise.
        function triggerPreTimeoutSave() {{
            console.log('Triggering pre-timeout save with POST...');
            const saveUrl = `${{baseUrl}}?event=pre_timeout_save&session_id=${{sessionId}}`;
            
            // Show saving indicator
            const savingDiv = document.createElement('div');
            savingDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #ff6b6b; color: white; padding: 10px 20px; border-radius: 5px; z-index: 9999;';
            savingDiv.textContent = 'Auto-saving session...';
            document.body.appendChild(savingDiv);

            // Return the fetch promise so we can chain actions to it
            return fetch(saveUrl, {{
                method: 'POST',
                keepalive: true,
                mode: 'no-cors'
            }});
        }}
        
        // This function will now be responsible for the full sequence
        function executeLogoutSequence() {{
            console.log('Starting logout sequence...');
            triggerPreTimeoutSave()
                .catch(err => console.error('Pre-timeout save fetch failed:', err))
                .finally(() => {{
                    // This block runs whether the save succeeded or failed.
                    console.log('Save request sent. Preparing to reload page in 500ms...');
                    // Give the browser a moment to process the keepalive request
                    // before reloading the page. This helps avoid the extension error.
                    setTimeout(() => {{
                        console.log('Session timeout - reloading page now.');
                        window.parent.location.reload();
                    }}, 500); // 500ms grace period
                }});
        }}
        
        // The main timer now calls the full logout sequence.
        // It's set to fire 30 seconds before the actual timeout.
        window.streamlitAutoLogoutTimer = setTimeout(executeLogoutSequence, {save_trigger_seconds * 1000});
        
        console.log(`Auto-save and logout scheduled in {save_trigger_seconds} seconds.`);
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)
def render_browser_close_component(session_id: str):
    """
    Enhanced browser close detection that avoids redundant saves.
    """
    if not session_id:
        return

    js_code = f"""
    <script>
    (function() {{
        if (window.browserCloseListenerAdded) return;
        window.browserCloseListenerAdded = true;
        
        const sessionId = '{session_id}';
        const baseUrl = window.location.origin + window.location.pathname;
        let saveHasBeenTriggered = false;

        function sendBeaconRequest() {{
            if (!saveHasBeenTriggered) {{
                saveHasBeenTriggered = true;
                const url = `${{baseUrl}}?event=close&session_id=${{sessionId}}`;
                if (navigator.sendBeacon) {{
                    navigator.sendBeacon(url);
                }} else {{
                    const xhr = new XMLHttpRequest();
                    xhr.open('GET', url, false);
                    try {{
                        xhr.send();
                    }} catch(e) {{
                        console.error('Failed to send close event via XHR:', e);
                    }}
                }}
            }}
        }}
        
        document.addEventListener('visibilitychange', () => {{
            if (document.visibilityState === 'hidden') {{
                sendBeaconRequest();
            }}
        }});
        
        window.addEventListener('pagehide', sendBeaconRequest, {{capture: true}});
        window.addEventListener('beforeunload', sendBeaconRequest, {{capture: true}});

    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        fresh_session = session_manager.get_session()

        if fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user":
            st.success("✅ **Authenticated User**") 
            if fresh_session.first_name:
                st.markdown(f"**Welcome:** {fresh_session.first_name}")
            if fresh_session.email:
                st.markdown(f"**Email:** {fresh_session.email}")
                
            col1, col2 = st.columns([3, 1])
            with col1:
                if session_manager.zoho.config.ZOHO_ENABLED:
                    if fresh_session.zoho_contact_id:
                        st.success("🔗 **CRM Linked**")
                    else:
                        st.info("📋 **CRM Ready**")
                else:
                    st.caption("🚫 CRM Disabled")
            
            with col2:
                if session_manager.zoho.config.ZOHO_ENABLED and fresh_session.email:
                    st.caption("💾 Auto-save ON")
            
            if fresh_session.last_activity:
                time_since_activity = datetime.now() - fresh_session.last_activity
                timeout_minutes = session_manager.get_session_timeout_minutes()
                
                total_timeout_seconds = timeout_minutes * 60
                seconds_elapsed = time_since_activity.total_seconds()
                seconds_remaining = total_timeout_seconds - seconds_elapsed
                
                if seconds_remaining > 0:
                    minutes_remaining = seconds_remaining / 60
                    st.caption(f"⏱️ Auto-save & sign out in {minutes_remaining:.1f} minutes")
                    render_auto_logout_component(
                        timeout_seconds=int(seconds_remaining),
                        session_id=fresh_session.session_id,
                        session_manager=session_manager
                    )
                else:
                    st.caption("⏱️ Session will timeout on next interaction")
                    render_auto_logout_component(
                        timeout_seconds=2,
                        session_id=fresh_session.session_id,
                        session_manager=session_manager
                    )
                    
        else:
            st.info("👤 **Guest User**")
            st.markdown("*Sign in for full features*")
        
        st.divider()
        
        st.markdown(f"**Messages:** {len(fresh_session.messages)}")
        st.markdown(f"**Session:** `{fresh_session.session_id[:8]}...`")
        
        st.divider()
        st.subheader("📊 System Status")
        
        if hasattr(st.session_state, 'ai_system'):
            ai = st.session_state.ai_system
            st.write(f"**Pinecone KB:** {'✅' if ai.pinecone_tool else '❌'}")
            st.write(f"**Web Search:** {'✅' if ai.tavily_agent else '❌'}")
            st.write(f"**OpenAI:** {'✅' if ai.openai_client else '❌'}")
        
        with st.expander("🚨 System Health"):
            health_summary = error_handler.get_system_health_summary()
            health_color = {"Healthy": "🟢", "Degraded": "🟡", "Critical": "🔴"}.get(health_summary["overall_health"], "❓")
            st.write(f"**Overall:** {health_color} {health_summary['overall_health']}")
            
            if error_handler.component_status:
                st.write("**Components:**")
                for component, status in error_handler.component_status.items():
                    if status == "healthy":
                        st.write(f"✅ {component}")
                    else:
                        st.write(f"❌ {component}")
        
        # Zoho Diagnostics (optional - remove in production)
        with st.expander("🔧 Zoho Diagnostics", expanded=False):
            if st.button("Run Zoho Diagnostic"):
                # Test 1: Configuration
                st.write("**1. Configuration Check:**")
                config = session_manager.config
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Client ID: {'✅' if config.ZOHO_CLIENT_ID else '❌'}")
                    st.write(f"Client Secret: {'✅' if config.ZOHO_CLIENT_SECRET else '❌'}")
                with col2:
                    st.write(f"Refresh Token: {'✅' if config.ZOHO_REFRESH_TOKEN else '❌'}")
                    st.write(f"Zoho Enabled: {'✅' if config.ZOHO_ENABLED else '❌'}")
                
                # Test 2: Token Generation
                st.write("\n**2. Token Test:**")
                try:
                    zoho = session_manager.zoho
                    with st.spinner("Getting access token..."):
                        token = zoho._get_access_token()
                    if token:
                        st.success(f"✅ Token obtained: {token[:20]}...")
                    else:
                        st.error("❌ Failed to get token")
                except Exception as e:
                    st.error(f"❌ Token error: {str(e)}")
                
                # Test 3: Session State
                st.write("\n**3. Current Session:**")
                st.write(f"Session ID: {fresh_session.session_id[:8]}...")
                st.write(f"Type: {fresh_session.user_type}")
                st.write(f"Email: {fresh_session.email or 'None'}")
                st.write(f"Messages: {len(fresh_session.messages)}")
                
                if st.button("Test Save Now"):
                    if fresh_session.email and fresh_session.messages:
                        with st.spinner("Testing save..."):
                            try:
                                session_manager._auto_save_to_crm(fresh_session, "Manual Test")
                            except Exception as e:
                                st.error(f"Save test failed: {str(e)}")
                    else:
                        st.warning("Need email and messages to test save")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                session_manager.clear_chat_history(fresh_session)
                st.rerun()
        
        with col2:
            if st.button("🚪 Sign Out", use_container_width=True):
                session_manager.end_session(fresh_session)
                st.rerun()

        if (fresh_session.user_type == UserType.REGISTERED_USER or fresh_session.user_type.value == "registered_user") and fresh_session.messages:
            st.divider()
            
            pdf_buffer = pdf_exporter.generate_chat_pdf(fresh_session)
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF", 
                    data=pdf_buffer,
                    file_name=f"fifi_chat_transcript_{fresh_session.session_id[:8]}.pdf",
                    mime="application/pdf", 
                    use_container_width=True
                )
            
            if session_manager.zoho.config.ZOHO_ENABLED and fresh_session.email:
                if st.button("💾 Save to Zoho CRM", use_container_width=True, help="Chat will also auto-save when you sign out or after 5 minutes of inactivity"):
                    session_manager.manual_save_to_crm(fresh_session)
                    
                st.caption("💡 Chat auto-saves to CRM on sign out or timeout")
        
        elif (fresh_session.user_type == UserType.GUEST or fresh_session.user_type.value == "guest") and fresh_session.messages:
            st.divider()
            st.info("💡 **Sign in** to save chat history and export PDF!")
            if st.button("🔑 Go to Sign In", use_container_width=True):
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()

        st.divider()
        st.subheader("💡 Try These Queries")
        example_queries = [
            "Find organic vanilla extract suppliers",
            "Latest trends in plant-based proteins",
            "Current cocoa prices and suppliers",
            "Sustainable packaging suppliers in Europe",
            "Clean label ingredient alternatives"
        ]
        
        for query in example_queries:
            if st.button(f"💬 {query}", key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()

def render_chat_interface(session_manager: SessionManager, session: UserSession):
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with knowledge base and web search")
    
    current_session = session_manager.get_session()
    
    render_browser_close_component(current_session.session_id)
    
    if current_session.user_type == UserType.REGISTERED_USER and current_session.last_activity:
        time_since_activity = datetime.now() - current_session.last_activity
        timeout_minutes = session_manager.get_session_timeout_minutes()
        minutes_remaining = timeout_minutes - (time_since_activity.total_seconds() / 60)
        
        if 0 < minutes_remaining <= 1:
            st.warning(f"⏱️ Session will auto-save and timeout in {minutes_remaining:.1f} minutes due to inactivity")
        elif minutes_remaining <= 0:
            st.error("⏱️ Session expired due to inactivity. Please sign in again.")
    
    for msg in current_session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)
            
            if msg.get("role") == "assistant":
                if "source" in msg:
                    st.caption(f"Source: {msg['source']}")
                
                indicators = []
                if msg.get("used_pinecone"):
                    indicators.append("🧠 Knowledge Base" + (" (with inline citations)" if msg.get("has_inline_citations") else ""))
                
                if msg.get("used_search"):
                    indicators.append("🌐 Web Search")
                
                if indicators:
                    st.caption(f"Enhanced with: {', '.join(indicators)}")
                
                if msg.get("safety_override"):
                    st.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")

    pending_query = st.session_state.get('pending_query')
    if pending_query:
        prompt = pending_query
        del st.session_state.pending_query
    else:
        prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        moderation_result = check_content_moderation(prompt, session_manager.ai.openai_client)
        
        if moderation_result.get("flagged"):
            with st.chat_message("assistant"):
                st.error(f"🚨 {moderation_result['message']}")
            
            current_session.messages.append({
                "role": "assistant",
                "content": moderation_result['message'],
                "source": "Content Safety Policy",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()
        else:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base and web..."):
                    response = session_manager.get_ai_response(current_session, prompt)
                    
                    st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                    
                    if response.get("source"):
                        st.caption(f"Source: {response['source']}")
                    
                    enhancements = []
                    if response.get("used_pinecone"):
                        enhancements.append("🧠 Enhanced with Knowledge Base" + (" (inline citations)" if response.get("has_inline_citations") else ""))
                    
                    if response.get("used_search"):
                        enhancements.append("🌐 Enhanced with verified web search")
                    
                    if enhancements:
                        for enhancement in enhancements:
                            st.success(enhancement)
                    
                    if response.get("safety_override"):
                        st.error("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")
            
            st.rerun()

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
        - Chat transcripts auto-saved to CRM on sign out or after 5 minutes of inactivity
        - Personalized experience with your profile
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")

    # SOLUTION: Handle pre-timeout save requests from JavaScript
    query_params = st.query_params
    if query_params.get("event") == "pre_timeout_save":
        session_id = query_params.get("session_id")
        if session_id:
            logger.info(f"Received pre-timeout save request for session {session_id[:8]}...")
            
            # Clear the query params to prevent loops
            st.query_params.clear()
            
            # Ensure initialization
            if ensure_initialization():
                session_manager = get_session_manager()
                if session_manager:
                    # Check if we haven't already saved this session
                    already_saved = st.session_state.get('pre_timeout_saved', {}).get(session_id, False)
                    if not already_saved:
                        success = session_manager.trigger_pre_timeout_save(session_id)
                        if success:
                            st.success("✅ Chat automatically saved to CRM before timeout")
                        else:
                            st.warning("⚠️ Could not save chat before timeout")
                    
            # Stop here to prevent normal page rendering during save
            st.stop()

    # Clear any problematic session state first
    if st.button("🔄 Fresh Start (Clear All State)", key="emergency_clear"):
        st.session_state.clear()
        st.success("✅ All state cleared. Refreshing...")
        st.rerun()

    # Ensure initialization
    if not ensure_initialization():
        st.stop()

    # Get session manager safely
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get session manager. Reinitializing...")
        st.session_state.clear()
        st.rerun()
    
    pdf_exporter = st.session_state.pdf_exporter
    
    # Main application routing
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page(session_manager)
    else:
        try:
            session = session_manager.get_session()
            if session and session.active:
                render_sidebar(session_manager, session, pdf_exporter)
                render_chat_interface(session_manager, session)
            else:
                if 'page' in st.session_state:
                    del st.session_state.page
                st.rerun()
        except Exception as e:
            logger.error(f"Error in chat interface: {e}")
            st.error("An error occurred. Please refresh the page.")
            if st.button("🔄 Refresh", key="error_refresh"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
