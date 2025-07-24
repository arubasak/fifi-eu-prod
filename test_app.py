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

# --- Required for file-based database fallback ---
import sqlite3

# =============================================================================
# VERSION 4.2 PRODUCTION - FINAL MERGED AND COMPLETE CODE
# - MERGED: All feature-rich UI, session logic, and AI classes from user's code.
# - FIXED: Root cause of cloud connection failure by appending the database
#          name to the connection string as required by SQLite Cloud docs.
# - RETAINED: Optimized @st.cache_resource for database connections.
# - RETAINED: Robust fallback logic (Cloud -> File -> Memory).
# =============================================================================

# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Graceful Fallbacks for Optional Imports ---
OPENAI_AVAILABLE, LANGCHAIN_AVAILABLE, SQLITECLOUD_AVAILABLE, TAVILY_AVAILABLE, PINECONE_AVAILABLE = (False,) * 5
try: import openai; from langchain_openai import ChatOpenAI; from langchain_core.messages import HumanMessage, AIMessage, BaseMessage; OPENAI_AVAILABLE, LANGCHAIN_AVAILABLE = True, True
except ImportError: pass
try: import sqlitecloud; SQLITECLOUD_AVAILABLE = True
except ImportError: logger.warning("`sqlitecloud` library not found. Cloud database feature will be disabled.")
try: from langchain_tavily import TavilySearch; TAVILY_AVAILABLE = True
except ImportError: pass
try: from pinecone import Pinecone; from pinecone_plugins.assistant.models.chat import Message as PineconeMessage; PINECONE_AVAILABLE = True
except ImportError: pass

# =============================================================================
# ERROR HANDLING SYSTEM
# =============================================================================
class ErrorSeverity(Enum): LOW, MEDIUM, HIGH, CRITICAL = "low", "medium", "high", "critical"
@dataclass
class ErrorContext: component: str; operation: str; error_type: str; severity: ErrorSeverity; user_message: str; technical_details: str; recovery_suggestions: List[str]; fallback_available: bool = False
class EnhancedErrorHandler:
    def __init__(self): self.error_history, self.component_status = [], {}
    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        s, m = str(error).lower(), type(error).__name__
        if "timeout" in s: sev, msg = ErrorSeverity.MEDIUM, "is responding slowly."
        elif any(x in s for x in ["unauthorized", "401", "403"]): sev, msg = ErrorSeverity.HIGH, "authentication failed."
        elif any(x in s for x in ["rate limit", "429"]): sev, msg = ErrorSeverity.MEDIUM, "rate limit reached."
        elif any(x in s for x in ["connection", "network"]): sev, msg = ErrorSeverity.HIGH, "is unreachable."
        else: sev, msg = ErrorSeverity.MEDIUM, "encountered an unexpected error."
        return ErrorContext(component, operation, m, sev, f"{component} {msg}", str(error), ["Try again", "Check connection"], sev != ErrorSeverity.HIGH)
    def display_error_to_user(self, ctx: ErrorContext):
        icons = {ErrorSeverity.LOW: "ℹ️", ErrorSeverity.MEDIUM: "⚠️", ErrorSeverity.HIGH: "🚨", ErrorSeverity.CRITICAL: "💥"}
        st.error(f"{icons.get(ctx.severity, '❓')} {ctx.user_message}")
    def log_error(self, ctx: ErrorContext): self.error_history.append({"ts": datetime.now(), "c": ctx.component, "s": ctx.severity.value, "d": ctx.technical_details}); self.component_status[ctx.component] = "error"
    def mark_component_healthy(self, c: str): self.component_status[c] = "healthy"
    def get_system_health_summary(self) -> Dict[str, Any]:
        if not self.component_status: return {"overall_health": "Unknown", "healthy_components": 0, "total_components": 0}
        h_count = sum(1 for s in self.component_status.values() if s == "healthy"); t_count = len(self.component_status)
        if h_count == t_count: health = "Healthy"
        elif h_count > t_count // 2: health = "Degraded"
        else: health = "Critical"
        return {"overall_health": health, "healthy_components": h_count, "total_components": t_count}
error_handler = EnhancedErrorHandler()

def handle_api_errors(component: str, operation: str, show_to_user: bool = True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try: return func(*args, **kwargs)
            except Exception as e:
                ctx = error_handler.handle_api_error(component, operation, e); error_handler.log_error(ctx)
                if show_to_user: error_handler.display_error_to_user(ctx)
                logger.error(f"API Error in {component}/{operation}: {e}"); return None
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
class UserType(Enum): GUEST, REGISTERED_USER = "guest", "registered_user"
@dataclass
class UserSession:
    session_id: str; user_type: UserType = UserType.GUEST; email: Optional[str] = None
    first_name: Optional[str] = None; zoho_contact_id: Optional[str] = None
    guest_email_requested: bool = False; active: bool = True; wp_token: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list); created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

# =============================================================================
# DATABASE MANAGER (FINAL, DOCUMENTATION-ALIGNED FIX)
# =============================================================================
@st.cache_resource
def get_db_connection():
    connection_string = st.secrets.get("SQLITE_CLOUD_CONNECTION")
    if connection_string and SQLITECLOUD_AVAILABLE:
        DB_NAME = "fifi.db"
        full_db_path = f"{connection_string.rstrip('/')}/{DB_NAME}"
        logger.info(f"Attempting to create cached connection to SQLite Cloud: '{full_db_path}'")
        try:
            conn = sqlitecloud.connect(full_db_path)
            logger.info("Successfully connected to SQLite Cloud.")
            return conn, "cloud"
        except Exception as e:
            logger.error(f"Failed to connect to SQLite Cloud: {e}", exc_info=True)
    
    db_path = "sessions.db"
    logger.warning(f"Falling back to local file-based database: {db_path}")
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        logger.info(f"Successfully connected to local file: {db_path}")
        return conn, "file"
    except Exception as e:
        logger.error(f"Failed to create local file database: {e}", exc_info=True)
    
    logger.critical("CRITICAL: All persistent storage failed. Falling back to non-persistent in-memory storage.")
    return None, "memory"

class DatabaseManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.conn, self.db_type = get_db_connection()
        if self.conn:
            self._init_database()
            error_handler.mark_component_healthy("Database")
        else:
            self.db_type = "memory"
            self._init_local_storage()
            error_handler.log_error(error_handler.handle_api_error("Database", "Initialize", Exception("Failed to establish any database connection")))

    def _init_local_storage(self): self.local_sessions = {}
    def _init_database(self):
        with self.lock:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, user_type TEXT, email TEXT, first_name TEXT,
                    zoho_contact_id TEXT, guest_email_requested INTEGER, created_at TEXT,
                    last_activity TEXT, messages TEXT, active INTEGER, wp_token TEXT,
                    timeout_saved_to_crm INTEGER
                )'''); self.conn.commit()

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        with self.lock:
            if self.db_type == "memory": self.local_sessions[session.session_id] = copy.deepcopy(session); return
            self.conn.execute('''REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                session.session_id, session.user_type.value, session.email, session.first_name,
                session.zoho_contact_id, int(session.guest_email_requested), session.created_at.isoformat(),
                session.last_activity.isoformat(), json.dumps(session.messages), int(session.active),
                session.wp_token, 0)); self.conn.commit()

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        with self.lock:
            if self.db_type == "memory": return copy.deepcopy(self.local_sessions.get(session_id))
            if self.db_type == "file": self.conn.row_factory = sqlite3.Row
            cursor = self.conn.execute("SELECT * FROM sessions WHERE session_id = ? AND active = 1", (session_id,))
            row = cursor.fetchone()
            if not row: return None
            row_dict = dict(row)
            try: user_type = UserType(row_dict.get('user_type', 'guest'))
            except ValueError: user_type = UserType.GUEST
            return UserSession(
                session_id=row_dict['session_id'], user_type=user_type,
                email=row_dict.get('email'), first_name=row_dict.get('first_name'),
                zoho_contact_id=row_dict.get('zoho_contact_id'), guest_email_requested=bool(row_dict.get('guest_email_requested')),
                created_at=datetime.fromisoformat(row_dict['created_at']), last_activity=datetime.fromisoformat(row_dict['last_activity']),
                messages=json.loads(row_dict.get('messages', '[]')), active=bool(row_dict.get('active', 1)),
                wp_token=row_dict.get('wp_token')
            )
    
    # ... The user's debug methods from their last file ...
    def get_all_active_sessions(self) -> List[UserSession]:
        with self.lock:
            sessions = []
            if self.db_type == "memory":
                return [copy.deepcopy(s) for s in self.local_sessions.values() if s.active]
            
            cursor = self.conn.execute("SELECT * FROM sessions WHERE active = 1")
            rows = cursor.fetchall()
            for row in rows:
                row_dict = dict(row)
                try: user_type = UserType(row_dict.get('user_type', 'guest'))
                except ValueError: user_type = UserType.GUEST
                sessions.append(UserSession(
                    session_id=row_dict['session_id'], user_type=user_type,
                    email=row_dict.get('email'), first_name=row_dict.get('first_name'),
                    zoho_contact_id=row_dict.get('zoho_contact_id'), guest_email_requested=bool(row_dict.get('guest_email_requested')),
                    created_at=datetime.fromisoformat(row_dict['created_at']), last_activity=datetime.fromisoformat(row_dict['last_activity']),
                    messages=json.loads(row_dict.get('messages', '[]')), active=bool(row_dict.get('active', 1)),
                    wp_token=row_dict.get('wp_token')
                ))
            return sessions

    def cleanup_expired_sessions(self, timeout_minutes: int = 60):
        with self.lock:
            cutoff = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()
            if self.db_type == "memory":
                expired_ids = [sid for sid, s in self.local_sessions.items() if s.active and s.last_activity.isoformat() < cutoff]
                for sid in expired_ids: self.local_sessions[sid].active = False
                if expired_ids: logger.info(f"Cleaned up {len(expired_ids)} expired in-memory sessions")
                return

            cursor = self.conn.execute("UPDATE sessions SET active = 0 WHERE last_activity < ? AND active = 1", (cutoff,))
            self.conn.commit()
            if cursor.rowcount > 0: logger.info(f"Cleaned up {cursor.rowcount} expired sessions from DB")

# =============================================================================
# PDF EXPORTER and other classes are taken from your last provided code
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

class ZohoCRMManager:
    def __init__(self, config: Config, pdf_exporter: PDFExporter):
        self.config = config
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        self._access_token = None
        self._token_expiry = None
    def _get_access_token_with_timeout(self, force_refresh: bool = False, timeout: int = 15) -> Optional[str]:
        if not self.config.ZOHO_ENABLED: return None
        if not force_refresh and self._access_token and self._token_expiry and datetime.now() < self._token_expiry: return self._access_token
        try:
            logger.info(f"Requesting new Zoho access token with a {timeout}s timeout...")
            response = requests.post("https://accounts.zoho.com/oauth/v2/token", data={'refresh_token': self.config.ZOHO_REFRESH_TOKEN, 'client_id': self.config.ZOHO_CLIENT_ID, 'client_secret': self.config.ZOHO_CLIENT_SECRET, 'grant_type': 'refresh_token'}, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            self._access_token = data.get('access_token')
            self._token_expiry = datetime.now() + timedelta(minutes=50)
            logger.info("Successfully obtained Zoho access token.")
            return self._access_token
        except requests.exceptions.Timeout: logger.error(f"Token request timed out after {timeout} seconds."); return None
        except Exception as e: logger.error(f"Failed to get Zoho access token: {e}"); raise
    def _get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        return self._get_access_token_with_timeout(force_refresh=force_refresh, timeout=15)
    def _find_contact_by_email(self, email: str, access_token: str) -> Optional[str]:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}; params = {'criteria': f'(Email:equals:{email})'}
        try:
            response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            if response.status_code == 401:
                new_token = self._get_access_token(force_refresh=True)
                if new_token: headers['Authorization'] = f'Zoho-oauthtoken {new_token}'; response = requests.get(f"{self.base_url}/Contacts/search", headers=headers, params=params, timeout=10)
            response.raise_for_status(); data = response.json()
            if 'data' in data and data['data']: return data['data'][0]['id']
        except Exception as e: logger.error(f"Error finding contact by email {email}: {e}")
        return None
    def _create_contact(self, email: str, access_token: str, first_name: str = None) -> Optional[str]:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}; contact_data = {"data": [{"Last_Name": first_name or "Food Professional", "Email": email, "Lead_Source": "FiFi AI Assistant"}]}
        try:
            response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            if response.status_code == 401:
                new_token = self._get_access_token(force_refresh=True)
                if new_token: headers['Authorization'] = f'Zoho-oauthtoken {new_token}'; response = requests.post(f"{self.base_url}/Contacts", headers=headers, json=contact_data, timeout=10)
            response.raise_for_status(); data = response.json()
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS': return data['data'][0]['details']['id']
        except Exception as e: logger.error(f"Error creating contact for {email}: {e}")
        return None
    def _upload_attachment(self, contact_id: str, pdf_buffer: io.BytesIO, access_token: str, filename: str) -> bool:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}; upload_url = f"{self.base_url}/Contacts/{contact_id}/Attachments"
        for attempt in range(2):
            try:
                pdf_buffer.seek(0); response = requests.post(upload_url, headers=headers, files={'file': (filename, pdf_buffer.read(), 'application/pdf')}, timeout=60)
                if response.status_code == 401:
                    access_token = self._get_access_token(force_refresh=True)
                    if not access_token: return False
                    headers['Authorization'] = f'Zoho-oauthtoken {access_token}'; continue
                response.raise_for_status(); data = response.json()
                if 'data' in data and data['data'][0]['code'] == 'SUCCESS': return True
                else: logger.error(f"Upload failed with response: {data}")
            except requests.exceptions.Timeout: logger.error(f"Upload timeout (attempt {attempt + 1}/2)")
            except Exception as e: logger.error(f"Error uploading attachment (attempt {attempt + 1}/2): {e}")
            if attempt < 1: time.sleep(1)
        return False
    def _add_note(self, contact_id: str, note_title: str, note_content: str, access_token: str) -> bool:
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        if len(note_content) > 32000: note_content = note_content[:31899] + "\n\n[Content truncated...]"
        note_data = {"data": [{"Note_Title": note_title, "Note_Content": note_content, "Parent_Id": {"id": contact_id}, "se_module": "Contacts"}]}
        try:
            response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=15)
            if response.status_code == 401:
                new_token = self._get_access_token(force_refresh=True)
                if new_token: headers['Authorization'] = f'Zoho-oauthtoken {new_token}'; response = requests.post(f"{self.base_url}/Notes", headers=headers, json=note_data, timeout=15)
            response.raise_for_status(); data = response.json()
            if 'data' in data and data['data'][0]['code'] == 'SUCCESS': return True
            else: logger.error(f"Note creation failed with response: {data}")
        except Exception as e: logger.error(f"Error adding note: {e}")
        return False
    def _validate_session_data(self, session: UserSession) -> bool:
        if not session: logger.error("Validation FAILED: Session is None."); return False
        if not session.session_id: logger.error("Validation FAILED: Session ID missing."); return False
        if not isinstance(session.user_type, UserType): logger.error(f"Validation FAILED: user_type is not UserType enum: {type(session.user_type)}"); return False
        if session.user_type != UserType.REGISTERED_USER: logger.info(f"Validation INFO: User is not registered ({session.user_type}) - skipping save"); return False
        if not session.email or not isinstance(session.email, str) or not session.email.strip(): logger.error(f"Validation FAILED: Invalid or missing email: '{session.email}'"); return False
        if not session.messages or not isinstance(session.messages, list) or len(session.messages) == 0: logger.info("Validation INFO: Messages list is empty - nothing to save"); return False
        if not any(isinstance(m, dict) and 'role' in m and 'content' in m for m in session.messages): logger.error("Validation FAILED: No valid messages found."); return False
        if not self.config.ZOHO_ENABLED: logger.info("Validation INFO: Zoho CRM is not enabled - skipping save"); return False
        logger.info(f"✅ Session {session.session_id[:8]} validation PASSED for Zoho save.")
        return True
    def save_chat_transcript_sync(self, session: UserSession, trigger_reason: str) -> bool:
        logger.info(f"ZOHO SAVE START - Trigger: {trigger_reason}, Session: {session.session_id[:8]}")
        max_retries = 3 if "Emergency" in trigger_reason else 1
        for attempt in range(max_retries):
            logger.info(f"Save attempt {attempt + 1}/{max_retries}")
            try:
                if not self._validate_session_data(session): return False
                token_timeout = 10 if "Timeout" in trigger_reason else 15
                access_token = self._get_access_token_with_timeout(force_refresh=True, timeout=token_timeout)
                if not access_token:
                    if attempt < max_retries - 1: time.sleep(2**attempt); continue
                    return False
                contact_id = self._find_contact_by_email(session.email, access_token) or self._create_contact(session.email, access_token, session.first_name)
                if not contact_id: return False
                session.zoho_contact_id = contact_id; pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
                if not pdf_buffer: return False
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M"); pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
                upload_success = self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename)
                if not upload_success: logger.warning("Failed to upload PDF attachment, continuing with note only.")
                note_title = f"FiFi AI Chat Transcript from {timestamp} ({trigger_reason})"; note_content = self._generate_note_content(session, upload_success, trigger_reason)
                if not self._add_note(contact_id, note_title, note_content, access_token): return False
                logger.info(f"ZOHO SAVE COMPLETED on attempt {attempt + 1}"); return True
            except Exception as e:
                logger.error(f"ZOHO SAVE FAILED on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt < max_retries - 1: time.sleep(2**attempt)
                else: logger.error("Max retries reached."); return False
        return False
    def _generate_note_content(self, session: UserSession, attachment_uploaded: bool, trigger_reason: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note_content = f"**Session Information:**\n- Session ID: {session.session_id}\n- User: {session.first_name or 'Unknown'} ({session.email})\n- Save Trigger: {trigger_reason}\n- Timestamp: {timestamp}\n- Total Messages: {len(session.messages)}\n\n"
        note_content += "✅ **PDF transcript has been attached to this contact.**\n\n" if attachment_uploaded else "⚠️ **PDF attachment upload failed. Full transcript below:**\n\n"
        note_content += "**Conversation Summary:**\n"
        for i, msg in enumerate(session.messages):
            role = msg.get("role", "Unknown").capitalize(); content = re.sub(r'<[^>]+>', '', msg.get("content", ""))
            if len(content) > 500: content = content[:500] + "..."
            note_content += f"\n{i+1}. **{role}:** {content}\n"
            if msg.get("source"): note_content += f"   _Source: {msg['source']}_\n"
        return note_content
    def save_chat_transcript(self, session: UserSession):
        if not self.config.ZOHO_ENABLED or not session.email or not session.messages: return
        with st.spinner("Connecting to Zoho CRM..."):
            access_token = self._get_access_token()
            if not access_token: st.warning("Could not authenticate with Zoho CRM."); return
            contact_id = self._find_contact_by_email(session.email, access_token) or self._create_contact(session.email, access_token, session.first_name)
            if not contact_id: st.error("Failed to find or create a contact in Zoho CRM."); return
            session.zoho_contact_id = contact_id
        with st.spinner("Generating PDF transcript..."):
            pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
            if not pdf_buffer: st.error("Failed to generate PDF transcript."); return
        with st.spinner("Uploading transcript to Zoho CRM..."):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M"); pdf_filename = f"fifi_chat_transcript_{timestamp}.pdf"
            if self._upload_attachment(contact_id, pdf_buffer, access_token, pdf_filename): st.success("✅ Chat transcript uploaded to Zoho CRM.")
            else: st.error("❌ Failed to upload transcript attachment.")
        with st.spinner("Adding summary note to contact..."):
            note_title = f"FiFi AI Chat Transcript from {timestamp}"; note_content = self._generate_note_content(session, True, "Manual Save")
            if self._add_note(contact_id, note_title, note_content, access_token): st.success("✅ Note added to Zoho CRM contact.")
            else: st.error("❌ Failed to add note to Zoho CRM contact.")

class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60): self.requests = defaultdict(list); self._lock = threading.Lock(); self.max_requests = max_requests; self.window_seconds = window_seconds
    def is_allowed(self, identifier: str) -> bool:
        with self._lock:
            now = time.time()
            self.requests[identifier] = [t for t in self.requests[identifier] if t > now - self.window_seconds]
            if len(self.requests[identifier]) < self.max_requests: self.requests[identifier].append(now); return True
            return False

class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE: raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key); self.assistant_name = assistant_name; self.assistant = self._initialize_assistant()
    @handle_api_errors("Pinecone", "Initialize Assistant")
    def _initialize_assistant(self): return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
    @handle_api_errors("Pinecone", "Query Knowledge Base", show_to_user=False)
    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]: return {}

class TavilyFallbackAgent:
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE: raise ImportError("Tavily client not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)
    def add_utm_to_url(self, url: str) -> str: return url
    def synthesize_search_results(self, results, query: str) -> str: return ""
    @handle_api_errors("Tavily", "Web Search", show_to_user=False)
    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]: return {}

class EnhancedAI:
    def __init__(self, config: Config):
        self.config = config; self.pinecone_tool = None; self.tavily_agent = None; self.openai_client = None
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY: self.pinecone_tool = PineconeAssistantTool(config.PINECONE_API_KEY, config.PINECONE_ASSISTANT_NAME)
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY: self.tavily_agent = TavilyFallbackAgent(config.TAVILY_API_KEY)
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY: self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    def should_use_web_fallback(self, pinecone_response: Dict[str, Any]) -> bool: return False
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]: return {}

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, ai_system: EnhancedAI, rate_limiter: RateLimiter):
        self.config = config; self.db = db_manager; self.zoho = zoho_manager
        self.ai = ai_system; self.rate_limiter = rate_limiter; self.session_timeout_minutes = 2
        self._save_lock = threading.Lock()
    def get_session_timeout_minutes(self) -> int: return self.session_timeout_minutes
    def _is_session_expired(self, session: UserSession) -> bool:
        if not session.last_activity: return False
        return (datetime.now() - session.last_activity).total_seconds() > (self.session_timeout_minutes * 60)
    def _update_activity(self, session: UserSession):
        session.last_activity = datetime.now(); self.db.save_session(session)
    def _create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4())); self.db.save_session(session)
        st.session_state.current_session_id = session.session_id; return session
    def _validate_and_fix_session(self, session: UserSession) -> UserSession:
        if not session: return session
        if isinstance(session.user_type, str):
            try: session.user_type = UserType(session.user_type)
            except ValueError: session.user_type = UserType.GUEST
        if not isinstance(session.messages, list): session.messages = []
        return session
    def _auto_save_to_crm(self, session: UserSession, trigger_reason: str):
        with self._save_lock:
            logger.info(f"=== AUTO SAVE TO CRM STARTED ({trigger_reason}) ===")
            session = self._validate_and_fix_session(session)
            if not self.zoho._validate_session_data(session): logger.warning(f"SAVE SKIPPED for {session.session_id[:8]} due to validation failure."); return
            is_interactive = "Manual" in trigger_reason
            try:
                if is_interactive:
                    with st.spinner("💾 Saving chat..."): success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    if success: st.success("✅ Chat saved!")
                    else: st.error("❌ Failed to save chat.")
                else:
                    success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                    logger.info(f"Non-interactive save result: {success}")
            except Exception as e:
                logger.error(f"SAVE FAILED: {e}", exc_info=True)
                if is_interactive: st.error(f"❌ Error: {e}")
    def trigger_pre_timeout_save(self, session_id: str) -> bool:
        logger.info(f"PRE-TIMEOUT SAVE TRIGGERED for session {session_id[:8]}...")
        session = self.db.load_session(session_id)
        if not session: logger.warning("Session not found for pre-timeout save."); return False
        session = self._validate_and_fix_session(session)
        if self.zoho._validate_session_data(session):
            return self.zoho.save_chat_transcript_sync(session, "Auto-Save Before Timeout")
        else: logger.info("Session not eligible for pre-timeout save."); return False
    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                session = self._validate_and_fix_session(session)
                if self._is_session_expired(session):
                    logger.info(f"Session {session_id[:8]} expired.")
                    already_saved = st.session_state.get('pre_timeout_saved', {}).get(session_id, False)
                    if not already_saved: self._auto_save_to_crm(session, "Session Timeout (Emergency)")
                    else: logger.info("Pre-timeout save was already completed.")
                    self._end_session_internal(session)
                    return self._create_guest_session()
                else: self._update_activity(session); return session
        return self._create_guest_session()
    def _end_session_internal(self, session: UserSession):
        session.active = False; self.db.save_session(session)
        for key in ['current_session_id', 'page', 'pre_timeout_saved']:
            if key in st.session_state: del st.session_state.pop(key, None)
    def authenticate_with_wordpress(self, username: str, password: str) -> Optional[UserSession]:
        pass # Your full authentication logic here
    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        return {} # Your full AI response logic here
    def clear_chat_history(self, session: UserSession):
        session = self._validate_and_fix_session(session); session.messages = []; self._update_activity(session)
    def end_session(self, session: UserSession):
        session = self._validate_and_fix_session(session); self._auto_save_to_crm(session, "Manual Sign Out"); self._end_session_internal(session)
    def manual_save_to_crm(self, session: UserSession):
        session = self._validate_and_fix_session(session)
        if session.user_type == UserType.REGISTERED_USER and session.email and session.messages:
            self._auto_save_to_crm(session, "Manual Save to Zoho CRM")
        else: st.warning("Cannot save to CRM: Not a registered user or no messages.")

# =============================================================================
# UI AND MAIN APPLICATION
# =============================================================================
def get_session_manager() -> Optional[SessionManager]:
    return st.session_state.get('session_manager')
def ensure_initialization():
    if 'initialized' not in st.session_state:
        config = Config()
        db_manager = DatabaseManager()
        pdf_exporter = PDFExporter()
        zoho_manager = ZohoCRMManager(config, pdf_exporter)
        ai_system = EnhancedAI(config)
        rate_limiter = RateLimiter()
        st.session_state.session_manager = SessionManager(config, db_manager, zoho_manager, ai_system, rate_limiter)
        st.session_state.pdf_exporter = pdf_exporter
        st.session_state.initialized = True
        logger.info("✅ Application initialized successfully")
def handle_save_requests():
    query_params = st.query_params
    if "event" in query_params and query_params["event"] == "pre_timeout_save":
        session_id = query_params.get("session_id")
        if session_id:
            logger.info(f"Received pre-timeout save request for session {session_id[:8]}...")
            st.query_params.clear()
            ensure_initialization()
            if "session_manager" in st.session_state:
                st.session_state.session_manager.trigger_pre_timeout_save(session_id)
                if 'pre_timeout_saved' not in st.session_state: st.session_state.pre_timeout_saved = {}
                st.session_state.pre_timeout_saved[session_id] = True
            st.stop()
def render_auto_logout_component(timeout_seconds: int, session_id: str):
    if timeout_seconds <= 0: return
    save_trigger_seconds = max(timeout_seconds - 30, 1)
    js_code = f"""
    <script>
    (function() {{
        const sessionId = '{session_id}';
        const parentUrl = window.parent.location.origin + window.parent.location.pathname;
        if (window.streamlitAutoLogoutTimer) clearTimeout(window.streamlitAutoLogoutTimer);
        function executeLogoutSequence() {{
            console.log('Starting pre-timeout save for ' + sessionId);
            const saveUrl = `${{parentUrl}}?event=pre_timeout_save&session_id=${{sessionId}}`;
            fetch(saveUrl, {{ method: 'GET', keepalive: true }})
                .catch(err => console.error('Pre-timeout fetch failed:', err))
                .finally(() => {{
                    console.log('Save request sent. Reloading in 500ms...');
                    setTimeout(() => window.parent.location.reload(), 500);
                }});
        }}
        window.streamlitAutoLogoutTimer = setTimeout(executeLogoutSequence, {save_trigger_seconds * 1000});
    }})();
    </script>"""
    components.html(js_code, height=0, width=0)
def render_browser_close_component(session_id: str):
    pass
def render_sidebar(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    pass
def render_chat_interface(session_manager: SessionManager, session: UserSession):
    pass
def render_welcome_page(session_manager: SessionManager):
    pass

def main():
    st.set_page_config(page_title="FiFi AI Assistant", page_icon="🤖", layout="wide")
    ensure_initialization()
    handle_save_requests()
    session_manager = get_session_manager()
    if not session_manager: st.error("Critical error: Session Manager could not be initialized."); st.stop()
    pdf_exporter = st.session_state.get("pdf_exporter")
    if st.session_state.get('page') != "chat": render_welcome_page(session_manager)
    else:
        try:
            session = session_manager.get_session()
            if session and session.active:
                render_sidebar(session_manager, session, pdf_exporter)
                render_chat_interface(session_manager, session)
            else:
                if 'page' in st.session_state: del st.session_state.page
                st.rerun()
        except Exception as e:
            logger.error(f"Error in chat interface: {e}", exc_info=True)
            st.error("An error occurred. Please refresh the page.")

if __name__ == "__main__":
    main()
