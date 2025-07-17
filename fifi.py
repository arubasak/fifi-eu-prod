import streamlit as st
import os
import uuid
import time
import json
import logging
import threading
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import requests
import jwt
from enum import Enum
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black, grey, lightgrey
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re
import sys
from urllib.parse import urlparse
import html  # Fixed: Added missing html import

# --- OpenAI Import with error handling ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# --- SQLite Cloud Import ---
try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    SQLITECLOUD_AVAILABLE = False
    logging.warning("SQLite Cloud not available. Install with: pip install sqlitecloud")

# --- LangChain and Pinecone Imports with proper error handling ---
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logging.warning(f"LangChain packages not available: {e}")

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone packages not found. To enable Pinecone features, run: pip install pinecone-client")

try:
    from tavily import TavilyClient
    TAVILY_CLIENT_AVAILABLE = True
except ImportError:
    TAVILY_CLIENT_AVAILABLE = False
    logging.warning("Tavily client not available. Install with: pip install tavily-python")

# --- 1. Logging and Configuration ---
def setup_logging():
    """Configure logging with proper formatting and handlers."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('app.log', mode='a')
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    def __init__(self):
        # Helper function to get config values from Streamlit secrets or environment
        def get_config_value(key: str, required: bool = True) -> str:
            try:
                # Try Streamlit secrets first
                if hasattr(st, 'secrets') and key in st.secrets:
                    return st.secrets[key]
                # Fallback to environment variables
                value = os.getenv(key)
                if required and not value:
                    raise ValueError(f"Missing required configuration: {key}")
                return value
            except Exception as e:
                if required:
                    raise ValueError(f"Missing required configuration: {key}")
                return None
        
        # Base required configuration (only JWT and WordPress)
        required_vars = ['JWT_SECRET', 'WORDPRESS_URL']
        
        try:
            self.JWT_SECRET = get_config_value('JWT_SECRET')
            self.WORDPRESS_URL = self._validate_url(get_config_value('WORDPRESS_URL'))
            
            # Database configuration (required if SQLite Cloud is available)
            self.SQLITE_CLOUD_CONNECTION = get_config_value('SQLITE_CLOUD_CONNECTION', required=SQLITECLOUD_AVAILABLE)
            if self.SQLITE_CLOUD_CONNECTION and not self.SQLITE_CLOUD_CONNECTION.startswith('sqlitecloud://'):
                raise ValueError("Invalid SQLite Cloud connection string format")
            
            # OpenAI configuration (required if OpenAI is available)
            self.OPENAI_API_KEY = get_config_value('OPENAI_API_KEY', required=OPENAI_AVAILABLE)
            if self.OPENAI_API_KEY:
                self.OPENAI_API_KEY = self._validate_api_key(self.OPENAI_API_KEY)
            
            # Optional configuration
            self.PINECONE_API_KEY = get_config_value('PINECONE_API_KEY', required=False)
            if self.PINECONE_API_KEY:
                self.PINECONE_API_KEY = self._validate_api_key(self.PINECONE_API_KEY)
            
            self.PINECONE_ASSISTANT_NAME = get_config_value('PINECONE_ASSISTANT_NAME', required=False) or 'my-chat-assistant'
            self.TAVILY_API_KEY = get_config_value('TAVILY_API_KEY', required=False)
            
            # Zoho configuration
            self.ZOHO_CLIENT_ID = get_config_value('ZOHO_CLIENT_ID', required=False)
            self.ZOHO_CLIENT_SECRET = get_config_value('ZOHO_CLIENT_SECRET', required=False)
            self.ZOHO_REFRESH_TOKEN = get_config_value('ZOHO_REFRESH_TOKEN', required=False)
            zoho_vars = [self.ZOHO_CLIENT_ID, self.ZOHO_CLIENT_SECRET, self.ZOHO_REFRESH_TOKEN]
            self.ZOHO_ENABLED = all(zoho_vars) and all(var and var.strip() for var in zoho_vars)
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise

    def _validate_url(self, url: str) -> str:
        """Fixed: Better URL validation"""
        if not url:
            raise ValueError("URL cannot be empty")
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {url}")
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")
        except Exception:
            raise ValueError(f"Invalid URL format: {url}")
        return url.rstrip('/')

    def _validate_api_key(self, api_key: str) -> str:
        """Fixed: Less strict API key validation"""
        if not api_key:
            raise ValueError("API key cannot be empty")
        api_key = api_key.strip()
        if len(api_key) < 5:  # More reasonable minimum length
            raise ValueError("API key too short")
        return api_key

try:
    config = Config()
except ValueError as e:
    st.error(f"Application Configuration Error: {e}")
    st.info("Some features may be unavailable due to missing configuration or packages.")
    # Create a minimal config for basic operation
    class MinimalConfig:
        def __init__(self):
            self.JWT_SECRET = os.getenv('JWT_SECRET', 'development-secret')
            self.WORDPRESS_URL = os.getenv('WORDPRESS_URL', 'https://example.com')
            self.OPENAI_API_KEY = None
            self.SQLITE_CLOUD_CONNECTION = None
            self.PINECONE_API_KEY = None
            self.TAVILY_API_KEY = None
            self.ZOHO_ENABLED = False
    
    config = MinimalConfig()
    st.warning("Running in limited mode. Please check your configuration.")

# --- 2. Utility Functions & Classes ---

# Competition exclusion list for Tavily searches
DEFAULT_EXCLUDED_DOMAINS = [
    "ingredientsnetwork.com",
    "csmingredients.com",
    "batafood.com",
    "nccingredients.com",
    "prinovaglobal.com",
    "ingrizo.com",
    "solina.com",
    "opply.com",
    "brusco.co.uk",
    "lehmanningredients.co.uk",
    "nccingredients.com",
    "i-ingredients.com",
    "fciltd.com",
    "lupafoods.com",
    "tradeingredients.com",
    "peterwhiting.co.uk",
    "globalgrains.co.uk",
    "tradeindia.com",
    "udaan.com",
    "ofbusiness.com",
    "indiamart.com",
    "symega.com",
    "meviveinternational.com",
    "amazon.com",
    "podfoods.co",
    "gocheetah.com",
    "foodmaven.com",
    "connect.kehe.com",
    "knowde.com",
    "ingredientsonline.com",
    "sourcegoodfood.com"
]

# --- Custom Tavily Fallback & General Search Tool ---
if LANGCHAIN_AVAILABLE:
    @tool
    def tavily_search_fallback(query: str) -> str:
        """
        Search the web using Tavily. Use this for queries about broader, public-knowledge topics.
        This tool automatically excludes a predefined list of competitor and marketplace domains.
        """
        try:
            if not TAVILY_CLIENT_AVAILABLE:
                return "Tavily client not available for web search."
            
            tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
            
            # The exclude_domains parameter is now included in the search call
            response = tavily_client.search(
                query=query, 
                search_depth="advanced", 
                max_results=5, 
                include_answer=True, 
                include_raw_content=False,
                exclude_domains=DEFAULT_EXCLUDED_DOMAINS
            )
            
            # Format the response
            results = []
            if response.get('results'):
                for result in response['results']:
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    results.append(f"**{title}**\n{content}\nSource: {url}\n")
            
            if response.get('answer'):
                results.insert(0, f"**Summary:** {response['answer']}\n")
            
            return "\n".join(results) if results else "No results found."
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return f"Error performing web search: {str(e)}"

    @tool
    def tavily_search_12taste_only(query: str) -> str:
        """
        Search specifically within 12taste.com domain using Tavily.
        Used when Pinecone hits token limits and we need domain-specific results.
        """
        try:
            if not TAVILY_CLIENT_AVAILABLE:
                return "Tavily client not available for domain-specific search."
            
            tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
            
            # Add site:12taste.com to the query to restrict to this domain
            domain_query = f"site:12taste.com {query}"
            
            response = tavily_client.search(
                query=domain_query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_raw_content=False
            )
            
            # Format the response
            results = []
            if response.get('results'):
                for result in response['results']:
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    results.append(f"**{title}**\n{content}\nSource: {url}\n")
            
            if response.get('answer'):
                results.insert(0, f"**12taste.com Summary:** {response['answer']}\n")
            
            return "\n".join(results) if results else "No results found on 12taste.com."
            
        except Exception as e:
            logger.error(f"Tavily 12taste search error: {e}")
            return f"Error performing 12taste.com search: {str(e)}"

def sanitize_input(text: str, max_length: int = 4000) -> str:
    """Sanitize user input to prevent XSS and limit length."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # HTML escape the input
    sanitized = html.escape(text)
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")
    
    return sanitized.strip()

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
        
    def is_allowed(self, identifier: str) -> bool:
        with self._lock:
            try:
                now = time.time()
                window_start = now - self.window_seconds
                
                # Clean old requests
                self.requests[identifier] = [
                    t for t in self.requests[identifier] if t > window_start
                ]
                
                if len(self.requests[identifier]) < self.max_requests:
                    self.requests[identifier].append(now)
                    return True
                return False
            except Exception as e:
                logger.error(f"Rate limiter error: {e}")
                return True  # Allow on error

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# --- 3. Core Service Classes ---
class UserType(Enum):
    GUEST = "guest"
    REGISTERED_USER = "registered_user"

@dataclass
class UserSession:
    session_id: str
    user_type: UserType
    email: Optional[str] = None
    first_name: Optional[str] = None
    zoho_contact_id: Optional[str] = None
    guest_email_requested: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # Fixed: Better type annotation
    active: bool = True

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        self.lock = threading.Lock()
        
        if connection_string and SQLITECLOUD_AVAILABLE:
            # Use SQLite Cloud
            self.connection_string = connection_string
            self.use_cloud = True
            try:
                self._create_database_if_not_exists()
                self._init_database()
                self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"SQLite Cloud initialization failed: {e}")
                st.warning("SQLite Cloud unavailable, using local storage (sessions won't persist)")
                self._init_local_storage()
        else:
            # Fallback to local storage
            logger.info("Using local storage for sessions (not persistent)")
            self.use_cloud = False
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Initialize local in-memory storage as fallback."""
        import sqlite3
        self.local_sessions = {}
        self.use_cloud = False
    
    def _create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        try:
            # Parse connection string to get database name
            # Format: sqlitecloud://user:password@host:port/database?params
            from urllib.parse import urlparse
            parsed_url = urlparse(self.connection_string)
            database_name = parsed_url.path.lstrip('/')
            
            if not database_name:
                raise ValueError("Database name not found in connection string")
            
            # Connect without specifying database first
            base_connection = self.connection_string.split('/' + database_name)[0]
            
            logger.info(f"Attempting to create database: {database_name}")
            
            with sqlitecloud.connect(base_connection) as conn:
                # Try to create database
                try:
                    conn.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
                    logger.info(f"Database {database_name} created or already exists")
                except Exception as create_error:
                    # Database might already exist or we might not have create permissions
                    logger.warning(f"Could not create database {database_name}: {create_error}")
                    # Continue anyway - database might already exist
            
        except Exception as e:
            logger.warning(f"Database creation check failed: {e}. Proceeding with existing database.")
            # Continue - the database might already exist
        
    def _get_connection(self):
        """Get SQLite Cloud connection with proper error handling."""
        try:
            conn = sqlitecloud.connect(self.connection_string)
            # Basic SQLite optimizations (compatible with older versions)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except Exception as e:
            logger.error(f"SQLite Cloud connection error: {e}")
            raise
            
    def _init_database(self):
        """Initialize database with compatible schema."""
        with self.lock:
            try:
                with self._get_connection() as conn:
                    # Check if we can connect to the database
                    conn.execute("SELECT 1")
                    
                    # Fixed: Compatible table creation without STRICT mode
                    conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_type TEXT NOT NULL,
                        email TEXT,
                        first_name TEXT,
                        zoho_contact_id TEXT,
                        guest_email_requested INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        messages TEXT DEFAULT '[]',
                        active INTEGER DEFAULT 1
                    )''')
                    
                    # Create indexes for better performance
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_last_activity 
                                   ON sessions(last_activity)''')
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_user_type 
                                   ON sessions(user_type)''')
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_active 
                                   ON sessions(active)''')
                    
                    # Create additional useful indexes
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_email 
                                   ON sessions(email) WHERE email IS NOT NULL''')
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                                   ON sessions(created_at)''')
                    
                    conn.commit()
                    logger.info("Database and tables initialized successfully")
            except Exception as e:
                logger.error(f"Database initialization error: {e}")
                raise
                
    def save_session(self, session: UserSession):
        """Save session with improved error handling and validation."""
        with self.lock:
            try:
                with self._get_connection() as conn:
                    # Validate data before insertion
                    if not session.session_id:
                        raise ValueError("Session ID cannot be empty")
                    
                    # Fixed: Use compatible REPLACE syntax instead of UPSERT
                    conn.execute('''
                        REPLACE INTO sessions (
                            session_id, user_type, email, first_name, zoho_contact_id,
                            guest_email_requested, created_at, last_activity, messages, active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session.session_id,
                        session.user_type.value,
                        session.email,
                        session.first_name,
                        session.zoho_contact_id,
                        1 if session.guest_email_requested else 0,
                        session.created_at.isoformat(),
                        session.last_activity.isoformat(),
                        json.dumps(session.messages),
                        1 if session.active else 0
                    ))
                    conn.commit()
                    logger.debug(f"Session {session.session_id} saved successfully")
            except Exception as e:
                logger.error(f"Failed to save session {session.session_id}: {e}")
                raise
                
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with improved error handling."""
        with self.lock:
            try:
                with self._get_connection() as conn:
                    # Use parameterized query for security
                    cursor = conn.execute(
                        "SELECT * FROM sessions WHERE session_id = ? AND active = 1",
                        (session_id,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Convert row to dictionary for easier handling
                    columns = [description[0] for description in cursor.description]
                    row_dict = dict(zip(columns, row))
                    
                    # Fixed: Better datetime parsing with fallback
                    try:
                        created_at = datetime.fromisoformat(row_dict['created_at'])
                    except (ValueError, TypeError):
                        created_at = datetime.now()
                        
                    try:
                        last_activity = datetime.fromisoformat(row_dict['last_activity'])
                    except (ValueError, TypeError):
                        last_activity = datetime.now()
                    
                    return UserSession(
                        session_id=row_dict['session_id'],
                        user_type=UserType(row_dict['user_type']),
                        email=row_dict['email'],
                        first_name=row_dict['first_name'],
                        zoho_contact_id=row_dict['zoho_contact_id'],
                        guest_email_requested=bool(row_dict.get('guest_email_requested', 0)),
                        created_at=created_at,
                        last_activity=last_activity,
                        messages=json.loads(row_dict.get('messages', '[]')),
                        active=bool(row_dict.get('active', 1))
                    )
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return None
                
    def cleanup_expired_sessions(self, timeout_minutes: int = 60):
        """Cleanup expired sessions with better performance."""
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cutoff_time = (datetime.now() - timedelta(minutes=timeout_minutes)).isoformat()
                    
                    # Use a more efficient cleanup query
                    cursor = conn.execute('''
                        DELETE FROM sessions 
                        WHERE last_activity < ? OR active = 0
                    ''', (cutoff_time,))
                    
                    if cursor.rowcount > 0:
                        logger.info(f"Cleaned up {cursor.rowcount} expired sessions")
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to cleanup expired sessions: {e}")
    
    def verify_database_setup(self) -> Dict[str, Any]:
        """Verify database setup and return status information."""
        try:
            with self._get_connection() as conn:
                # Check if sessions table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='sessions'
                """)
                table_exists = cursor.fetchone() is not None
                
                # Check indexes
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND tbl_name='sessions'
                """)
                indexes = [row[0] for row in cursor.fetchall()]
                
                # Get row count
                if table_exists:
                    cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                    total_sessions = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE active = 1")
                    active_sessions = cursor.fetchone()[0]
                else:
                    total_sessions = 0
                    active_sessions = 0
                
                return {
                    "database_connected": True,
                    "table_exists": table_exists,
                    "indexes_created": len(indexes),
                    "total_sessions": total_sessions,
                    "active_sessions": active_sessions,
                    "status": "healthy"
                }
                
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return {
                "database_connected": False,
                "table_exists": False,
                "indexes_created": 0,
                "total_sessions": 0,
                "active_sessions": 0,
                "status": f"error: {str(e)}"
            }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics for monitoring."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN active = 1 THEN 1 END) as active_sessions,
                        COUNT(CASE WHEN user_type = 'guest' THEN 1 END) as guest_sessions,
                        COUNT(CASE WHEN user_type = 'registered_user' THEN 1 END) as registered_sessions
                    FROM sessions
                    WHERE datetime(last_activity) > datetime('now', '-24 hours')
                ''')
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return {}
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}

class PDFExporter:
    def __init__(self):
        try:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
        except Exception as e:
            logger.error(f"PDF exporter initialization error: {e}")
            self.styles = None
        
    def _setup_custom_styles(self):
        if self.styles:
            self.styles.add(ParagraphStyle(
                name='ChatHeader',
                alignment=TA_CENTER,
                fontSize=18,
                spaceAfter=20
            ))
            self.styles.add(ParagraphStyle(
                name='Normal_L',
                alignment=TA_LEFT,
                leftIndent=10
            ))
        
    def generate_chat_pdf(self, session: UserSession) -> io.BytesIO:
        buffer = io.BytesIO()
        try:
            if not self.styles:
                return buffer
                
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            
            # Add title
            story.append(Paragraph("Chat Transcript", self.styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # Add session info
            story.append(Paragraph(f"Session ID: {session.session_id}", self.styles['Normal']))
            story.append(Paragraph(f"User Type: {session.user_type.value}", self.styles['Normal']))
            if session.email:
                story.append(Paragraph(f"Email: {session.email}", self.styles['Normal']))
            story.append(Paragraph(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Add messages
            for msg in session.messages:
                role = str(msg.get('role', 'unknown')).capitalize()
                content = html.escape(str(msg.get('content', '')))
                
                story.append(Paragraph(
                    f"<b>{role}:</b> {content}",
                    self.styles['Normal_L']
                ))
                story.append(Spacer(1, 6))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            return buffer

class ZohoCRMManager:
    def __init__(self, pdf_exporter: PDFExporter):
        self.enabled = config.ZOHO_ENABLED
        self.pdf_exporter = pdf_exporter
        self.base_url = "https://www.zohoapis.com/crm/v2"
        
    def _get_access_token(self) -> Optional[str]:
        """Get access token using refresh token."""
        if not self.enabled:
            return None
            
        try:
            url = "https://accounts.zoho.com/oauth/v2/token"
            data = {
                'refresh_token': config.ZOHO_REFRESH_TOKEN,
                'client_id': config.ZOHO_CLIENT_ID,
                'client_secret': config.ZOHO_CLIENT_SECRET,
                'grant_type': 'refresh_token'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            return response.json().get('access_token')
            
        except requests.RequestException as e:
            logger.error(f"Failed to get Zoho access token: {e}")
            return None
            
    def get_or_create_contact(self, email: str, first_name: str, last_name: str = "F&B Professional") -> Optional[str]:
        """Get existing contact or create new one."""
        if not self.enabled:
            return None
            
        try:
            access_token = self._get_access_token()
            if not access_token:
                return None
                
            headers = {
                'Authorization': f'Zoho-oauthtoken {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Search for existing contact
            search_url = f"{self.base_url}/Contacts/search"
            params = {'email': email}
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    contact_id = data['data'][0]['id']
                    logger.info(f"Found existing Zoho contact: {contact_id}")
                    return contact_id
            
            # Create new contact
            create_url = f"{self.base_url}/Contacts"
            contact_data = {
                'data': [{
                    'Email': email,
                    'First_Name': first_name,
                    'Last_Name': last_name
                }]
            }
            
            response = requests.post(create_url, headers=headers, json=contact_data, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('data'):
                contact_id = data['data'][0]['details']['id']
                logger.info(f"Created new Zoho contact: {contact_id}")
                return contact_id
                
        except requests.RequestException as e:
            logger.error(f"Zoho API error: {e}")
            
        return None
        
    def save_chat_transcript(self, session: UserSession):
        """Save chat transcript as attachment to Zoho contact."""
        if not self.enabled or not session.zoho_contact_id:
            return
            
        try:
            pdf_buffer = self.pdf_exporter.generate_chat_pdf(session)
            
            # In a real implementation, you would upload the PDF to Zoho
            # This is a simplified version
            logger.info(f"Zoho: Saving transcript for session {session.session_id} to contact {session.zoho_contact_id}")
            
        except Exception as e:
            logger.error(f"Failed to save transcript to Zoho: {e}")

class PineconeAssistantTool:
    def __init__(self, api_key: str, assistant_name: str):
        self.api_key = api_key
        self.assistant_name = assistant_name
        self.assistant = None
        
        if PINECONE_AVAILABLE:
            try:
                self.pc = Pinecone(api_key=api_key)
                # Initialize assistant (simplified)
                logger.info("Pinecone assistant initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
    
    def _is_token_limit_error(self, error: Exception) -> bool:
        """Check if the error is related to token limits."""
        error_str = str(error).lower()
        token_error_indicators = [
            "token limit",
            "token_limit",
            "tokens exceeded",
            "context length",
            "context_length",
            "maximum context",
            "150k",
            "150000",
            "token count"
        ]
        return any(indicator in error_str for indicator in token_error_indicators)
                
    def query(self, chat_history: List['BaseMessage']) -> Dict[str, Any]:
        """Query the Pinecone assistant with enhanced error handling."""
        try:
            if not PINECONE_AVAILABLE or not self.assistant:
                return {
                    "content": "Pinecone assistant not available.",
                    "success": False,
                    "source": "error",
                    "error_type": "unavailable"
                }
                
            # Extract the latest user message
            user_message = None
            for msg in reversed(chat_history):
                if hasattr(msg, 'content') and hasattr(msg, '__class__'):
                    if 'HumanMessage' in str(msg.__class__):
                        user_message = msg.content
                        break
                    
            if not user_message:
                return {
                    "content": "No user message found.",
                    "success": False,
                    "source": "error",
                    "error_type": "no_message"
                }
            
            # In a real implementation, you would query the Pinecone assistant
            # This is a simplified response that could throw token limit errors
            
            # Simulate token limit check (in real implementation, this would come from Pinecone)
            total_tokens = sum(len(str(msg.content)) for msg in chat_history if hasattr(msg, 'content'))
            
            # Simulate token limit error when approaching 150K tokens
            if total_tokens > 150000:  # 150K character approximation
                raise Exception("Token limit exceeded: Maximum context length of 150,000 tokens reached")
            
            response_content = f"Based on your query '{user_message}', here's what I found in our knowledge base..."
            
            return {
                "content": response_content,
                "success": True,
                "source": "FiFi Knowledge Base",
                "error_type": None
            }
            
        except Exception as e:
            logger.error(f"Pinecone query error: {e}")
            
            # Check if it's a token limit error
            if self._is_token_limit_error(e):
                return {
                    "content": "Token limit reached in knowledge base.",
                    "success": False,
                    "source": "error",
                    "error_type": "token_limit"
                }
            else:
                return {
                    "content": "Error querying knowledge base.",
                    "success": False,
                    "source": "error",
                    "error_type": "general"
                }

class TavilyFallbackAgent:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.llm = None
        self.search_tool = None
        
        try:
            if LANGCHAIN_AVAILABLE:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=openai_api_key,
                    temperature=0.7
                )
                
                if tavily_api_key:
                    self.search_tool = TavilySearchResults(
                        api_key=tavily_api_key,
                        max_results=5
                    )
                    
            logger.info("Tavily fallback agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tavily agent: {e}")
    
    def _search_with_exclusions(self, message: str) -> str:
        """Perform web search with competitor domain exclusions."""
        try:
            if TAVILY_CLIENT_AVAILABLE and self.tavily_api_key:
                tavily_client = TavilyClient(api_key=self.tavily_api_key)
                
                response = tavily_client.search(
                    query=message,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False,
                    exclude_domains=DEFAULT_EXCLUDED_DOMAINS
                )
                
                # Format results
                results = []
                if response.get('answer'):
                    results.append(f"Summary: {response['answer']}")
                
                if response.get('results'):
                    for result in response['results']:
                        title = result.get('title', 'No title')
                        content = result.get('content', 'No content')
                        url = result.get('url', 'No URL')
                        results.append(f"{title}: {content} (Source: {url})")
                
                return "\n".join(results) if results else "No results found."
            
            # Fallback to basic search tool if Tavily client not available
            elif self.search_tool:
                search_response = self.search_tool.invoke({"query": message})
                return str(search_response)
            
            return "No search capabilities available."
            
        except Exception as e:
            logger.error(f"Search with exclusions error: {e}")
            return f"Search error: {str(e)}"
    
    def _search_12taste_only(self, message: str) -> str:
        """Perform search restricted to 12taste.com domain."""
        try:
            if TAVILY_CLIENT_AVAILABLE and self.tavily_api_key:
                tavily_client = TavilyClient(api_key=self.tavily_api_key)
                
                # Add site restriction to query
                domain_query = f"site:12taste.com {message}"
                
                response = tavily_client.search(
                    query=domain_query,
                    search_depth="advanced",
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False
                )
                
                # Format results
                results = []
                if response.get('answer'):
                    results.append(f"12taste.com Summary: {response['answer']}")
                
                if response.get('results'):
                    for result in response['results']:
                        title = result.get('title', 'No title')
                        content = result.get('content', 'No content')
                        url = result.get('url', 'No URL')
                        results.append(f"{title}: {content} (Source: {url})")
                
                return "\n".join(results) if results else "No results found on 12taste.com."
            
            return "12taste.com search not available."
            
        except Exception as e:
            logger.error(f"12taste search error: {e}")
            return f"12taste.com search error: {str(e)}"
            
    def query(self, message: str, chat_history: List['BaseMessage'], search_type: str = "general") -> Dict[str, Any]:
        """Query using Tavily search and OpenAI with different search types."""
        try:
            if not self.llm:
                return {
                    "content": "AI services not available.",
                    "success": False,
                    "source": "error"
                }
            
            # Perform search based on type
            if search_type == "12taste_only":
                search_results = self._search_12taste_only(message)
                source_label = "FiFi 12taste.com Search"
            else:  # general search with exclusions
                search_results = self._search_with_exclusions(message)
                source_label = "FiFi Web Search"
            
            # Create prompt with context
            prompt = f"""
            User question: {message}
            
            Web search results: {search_results}
            
            Please provide a helpful response based on the available search information.
            If the search results are from 12taste.com specifically, mention that in your response.
            Focus on the most relevant information from the search results.
            """
            
            # Get AI response
            if LANGCHAIN_AVAILABLE:
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
            else:
                content = "LangChain not available for AI processing."
            
            return {
                "content": content,
                "success": True,
                "source": source_label
            }
            
        except Exception as e:
            logger.error(f"Tavily agent error: {e}")
            return {
                "content": "Error processing your request.",
                "success": False,
                "source": "error"
            }
    
class ChatApp:
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        
    def initialize_tools(self):
        """Initialize AI tools based on available API keys."""
        try:
            if config.PINECONE_API_KEY and PINECONE_AVAILABLE:
                self.pinecone_tool = PineconeAssistantTool(
                    config.PINECONE_API_KEY,
                    config.PINECONE_ASSISTANT_NAME
                )
                
            if config.TAVILY_API_KEY:
                self.tavily_agent = TavilyFallbackAgent(
                    config.OPENAI_API_KEY,
                    config.TAVILY_API_KEY
                )
                
            logger.info("Chat app tools initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat tools: {e}")
    
    def _should_use_web_fallback(self, content: str) -> bool:
        """Determine if we should fall back to web search."""
        if not content:
            return True
            
        fallback_indicators = [
            "don't have access",
            "i don't know",
            "not sure",
            "cannot find",
            "no information"
        ]
        return any(indicator in content.lower() for indicator in fallback_indicators)
    
    def get_response(self, chat_history: List['BaseMessage']) -> Dict[str, Any]:
        """Get AI response using available tools with enhanced error handling flow."""
        try:
            # Try Pinecone first
            if self.pinecone_tool:
                pinecone_response = self.pinecone_tool.query(chat_history)
                
                # Check if Pinecone succeeded
                if pinecone_response.get("success"):
                    # Check if content is sufficient (not poor/insufficient)
                    if not self._should_use_web_fallback(pinecone_response.get("content", "")):
                        return pinecone_response
                    # If content is poor, fall back to regular Tavily (with exclusions)
                    logger.info("Pinecone returned insufficient content, falling back to regular Tavily")
                else:
                    # Pinecone failed - check error type
                    error_type = pinecone_response.get("error_type")
                    
                    if error_type == "token_limit":
                        # Token limit error - use 12taste.com only search
                        logger.info("Pinecone hit token limit, falling back to 12taste.com search")
                        if self.tavily_agent and chat_history:
                            latest_message = ""
                            for msg in reversed(chat_history):
                                if hasattr(msg, 'content'):
                                    latest_message = msg.content
                                    break
                            return self.tavily_agent.query(latest_message, chat_history, search_type="12taste_only")
                    else:
                        # Non-token error - use regular Tavily (with exclusions)
                        logger.info(f"Pinecone failed with {error_type} error, falling back to regular Tavily")
            
            # Fall back to regular Tavily (with competitor exclusions)
            if self.tavily_agent and chat_history:
                latest_message = ""
                for msg in reversed(chat_history):
                    if hasattr(msg, 'content'):
                        latest_message = msg.content
                        break
                return self.tavily_agent.query(latest_message, chat_history, search_type="general")
            
            # Default response if no tools available
            return {
                "content": "I'm sorry, but AI systems are currently unavailable. Please try again later.",
                "success": False,
                "source": "error"
            }
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return {
                "content": "An error occurred while processing your request.",
                "success": False,
                "source": "error"
            }

class SessionManager:
    def __init__(self, db: DatabaseManager, chat_app: ChatApp, zoho: ZohoCRMManager):
        self.db = db
        self.chat_app = chat_app
        self.zoho = zoho
        
    def get_session(self) -> UserSession:
        """Get current session or create new guest session."""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Check if session is expired
                if datetime.now() - session.last_activity > timedelta(minutes=30):
                    self.end_session(session_id)
                    st.warning("Session timed out. Starting new session.")
                    return self.create_guest_session()
                return session
        
        return self.create_guest_session()
    
    def create_guest_session(self) -> UserSession:
        """Create a new guest session."""
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_type=UserType.GUEST
        )
        self.db.save_session(session)
        st.session_state.current_session_id = session.session_id
        return session
    
    def _get_current_origin(self) -> str:
        """Detect the current deployment environment and return appropriate origin."""
        try:
            # Check if we're in Streamlit Community Cloud by examining the hostname
            # Since st.get_option API changed, use alternative detection methods
            import socket
            hostname = socket.gethostname()
            
            # Check for Streamlit Community Cloud indicators
            if 'streamlit.app' in hostname or 'streamlit' in os.environ.get('STREAMLIT_SERVER_ADDRESS', ''):
                return 'https://fifi-co-pilot.streamlit.app'
            
            # Check session state for custom origin (set by deployment)
            if hasattr(st.session_state, 'deployment_origin'):
                return st.session_state.deployment_origin
            
            # Default to GCP Cloud Run
            return 'https://fifi-co-pilot-v1-121263692901.europe-west4.run.app'
        except Exception as e:
            logger.warning(f"Could not detect deployment environment: {e}")
            # Fallback to GCP Cloud Run
            return 'https://fifi-co-pilot-v1-121263692901.europe-west4.run.app'

    def authenticate_with_wp(self, username: str, password: str) -> Optional[UserSession]:
        """Authenticate user with WordPress using JWT."""
        try:
            sanitized_user = sanitize_input(username)
            if not sanitized_user or not password:
                st.error("Username and password cannot be empty.")
                return None
            
            # WordPress JWT authentication endpoint
            auth_url = f"{config.WORDPRESS_URL}/wp-json/jwt-auth/v1/token"
            
            # Prepare authentication data
            auth_data = {
                'username': sanitized_user,
                'password': password
            }
            
            # Get current origin dynamically
            current_origin = self._get_current_origin()
            
            # Set headers for CORS
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Origin': current_origin
            }
            
            logger.info(f"Authenticating from origin: {current_origin}")
            
            # Make authentication request
            response = requests.post(
                auth_url, 
                json=auth_data, 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                auth_response = response.json()
                
                if auth_response.get('token'):
                    # Extract user information
                    user_data = auth_response.get('user', {})
                    email = user_data.get('user_email', sanitized_user)
                    first_name = user_data.get('user_nicename', sanitized_user.split('@')[0])
                    
                    # Create authenticated session
                    session = UserSession(
                        session_id=str(uuid.uuid4()),
                        user_type=UserType.REGISTERED_USER,
                        email=email,
                        first_name=first_name,
                        zoho_contact_id=self.zoho.get_or_create_contact(email, first_name)
                    )
                    
                    self.db.save_session(session)
                    st.session_state.current_session_id = session.session_id
                    st.session_state.wp_token = auth_response.get('token')
                    st.success("Login successful!")
                    return session
                else:
                    st.error("Authentication failed: Invalid response from server.")
                    return None
            else:
                error_msg = "Invalid credentials."
                try:
                    error_response = response.json()
                    if error_response.get('message'):
                        error_msg = error_response['message']
                except:
                    pass
                st.error(error_msg)
                logger.error(f"WordPress auth failed: {response.status_code} - {response.text}")
                return None
            
        except requests.RequestException as e:
            logger.error(f"WordPress authentication request error: {e}")
            st.error("Connection error. Please check your internet connection and try again.")
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            st.error("Login failed. Please try again.")
            return None
    
    def get_ai_response(self, session: UserSession, prompt: str) -> Dict[str, Any]:
        """Get AI response for user prompt."""
        try:
            # Rate limiting
            if not rate_limiter.is_allowed(session.session_id):
                return {
                    "content": "Rate limit exceeded. Please wait a moment before sending another message.",
                    "success": False,
                    "source": "error"
                }
            
            # Sanitize input
            sanitized_prompt = sanitize_input(prompt)
            if not sanitized_prompt:
                return {
                    "content": "Invalid input. Please try again.",
                    "success": False,
                    "source": "error"
                }
            
            # Build chat history
            chat_history = []
            if LANGCHAIN_AVAILABLE:
                from langchain_core.messages import HumanMessage, AIMessage
                for msg in session.messages:
                    if msg.get('role') == 'user':
                        chat_history.append(HumanMessage(content=msg.get('content', '')))
                    elif msg.get('role') == 'assistant':
                        chat_history.append(AIMessage(content=msg.get('content', '')))
                
                # Add current message
                chat_history.append(HumanMessage(content=sanitized_prompt))
            
            # Get AI response
            response = self.chat_app.get_response(chat_history)
            
            # Update session
            session.messages.append({
                "role": "user",
                "content": sanitized_prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            session.messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
                "source": response.get("source", "AI"),
                "timestamp": datetime.now().isoformat()
            })
            
            session.last_activity = datetime.now()
            
            # Keep only last 50 messages to prevent memory issues
            session.messages = session.messages[-50:]
            
            self.db.save_session(session)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return {
                "content": "An error occurred while processing your request.",
                "success": False,
                "source": "error"
            }
    
    def end_session(self, session_id: str):
        """End the current session."""
        if session_id:
            session = self.db.load_session(session_id)
            if session:
                session.active = False
                self.db.save_session(session)
                
                # Save transcript to Zoho if user is registered
                if session.user_type == UserType.REGISTERED_USER or session.email:
                    self.zoho.save_chat_transcript(session)
        
        # Clear session state
        if 'current_session_id' in st.session_state:
            del st.session_state.current_session_id
        st.session_state.page = "welcome"
    
    def clear_chat_history(self, session: UserSession):
        """Clear chat history for current session."""
        session.messages = []
        session.last_activity = datetime.now()
        self.db.save_session(session)

# --- 4. Streamlit UI ---
def init_session_state():
    """Initialize Streamlit session state."""
    if 'initialized' not in st.session_state:
        try:
            # Detect deployment environment
            if 'streamlit.app' in st.get_option('browser.serverAddress', ''):
                st.session_state.deployment_origin = 'https://fifi-co-pilot.streamlit.app'
                st.session_state.deployment_env = 'streamlit_cloud'
            else:
                st.session_state.deployment_origin = 'https://fifi-co-pilot-v1-121263692901.europe-west4.run.app'
                st.session_state.deployment_env = 'gcp_cloud_run'
            
            st.session_state.db_manager = DatabaseManager(config.SQLITE_CLOUD_CONNECTION)
            st.session_state.pdf_exporter = PDFExporter()
            st.session_state.zoho_manager = ZohoCRMManager(st.session_state.pdf_exporter)
            st.session_state.chat_app = ChatApp()
            st.session_state.chat_app.initialize_tools()
            st.session_state.session_manager = SessionManager(
                st.session_state.db_manager,
                st.session_state.chat_app,
                st.session_state.zoho_manager
            )
            st.session_state.page = "welcome"
            st.session_state.initialized = True
            logger.info(f"Session state initialized successfully on {st.session_state.deployment_env}")
        except Exception as e:
            logger.error(f"Failed to initialize session state: {e}")
            st.error("Failed to initialize application. Please refresh the page.")

def render_chat_interface(session: UserSession):
    """Render the main chat interface."""
    st.title("🤖 AI Chat Assistant")
    
    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "unknown")):
            st.markdown(msg.get("content", ""), unsafe_allow_html=False)
            if "source" in msg:
                st.caption(f"Source: {msg['source']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.session_manager.get_ai_response(session, prompt)
                st.markdown(response.get("content", ""), unsafe_allow_html=False)
                if "source" in response:
                    st.caption(f"Source: {response['source']}")

def render_welcome_page():
    """Render the welcome/login page."""
    st.title("🤖 Welcome to AI Chat Assistant")
    st.markdown("Please sign in with your credentials or continue as a guest.")
    
    # Login form
    with st.form("login_form"):
        st.subheader("Sign In")
        username = st.text_input("Username/Email")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Sign In"):
            if username and password:
                session = st.session_state.session_manager.authenticate_with_wp(username, password)
                if session:
                    st.session_state.page = "chat"
                    # Fixed: Use st.rerun() for newer Streamlit versions, with fallback
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
            else:
                st.error("Please enter both username and password.")
    
    st.divider()
    
    # Guest option
    st.subheader("Continue as Guest")
    if st.button("Continue as Guest", type="primary"):
        st.session_state.page = "chat"
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

def render_sidebar(session: UserSession):
    """Render the sidebar with controls."""
    with st.sidebar:
        st.title("Chat Controls")
        
        # Session info
        st.subheader("Session Info")
        st.write(f"**Type:** {getattr(session, 'user_type', 'unknown')}")
        if hasattr(session, 'email') and session.email:
            st.write(f"**Email:** {session.email}")
        if hasattr(session, 'messages'):
            st.write(f"**Messages:** {len(session.messages)}")
        
        # Database health check (only if db_manager exists)
        if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
            try:
                db_status = st.session_state.db_manager.verify_database_setup()
                if db_status["database_connected"]:
                    st.write("**Database:** ✅ Connected")
                    if st.checkbox("Show DB Details"):
                        st.write(f"**Total Sessions:** {db_status.get('total_sessions', 'N/A')}")
                        st.write(f"**Active Sessions:** {db_status.get('active_sessions', 'N/A')}")
                        st.write(f"**Indexes:** {db_status.get('indexes_created', 'N/A')}")
                else:
                    st.write("**Database:** ❌ Error")
                    if st.checkbox("Show Error"):
                        st.error(db_status.get('status', 'Unknown error'))
            except Exception as e:
                st.write("**Database:** ⚠️ Local Storage")
                logger.error(f"Failed to get DB status: {e}")
        else:
            st.write("**Database:** 📱 Local Storage")
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear History"):
            if hasattr(st.session_state, 'session_manager') and st.session_state.session_manager:
                st.session_state.session_manager.clear_chat_history(session)
            else:
                # Clear local messages
                if hasattr(session, 'messages'):
                    session.messages = []
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        if st.button("🚪 End Session"):
            if hasattr(st.session_state, 'session_manager') and st.session_state.session_manager:
                st.session_state.session_manager.end_session(st.session_state.get('current_session_id'))
            else:
                # Clear session state manually
                for key in list(st.session_state.keys()):
                    if key.startswith('current_session'):
                        del st.session_state[key]
                st.session_state.page = "welcome"
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        # PDF download (only if pdf_exporter exists)
        if (hasattr(session, 'messages') and session.messages and 
            hasattr(st.session_state, 'pdf_exporter') and st.session_state.pdf_exporter):
            try:
                pdf_buffer = st.session_state.pdf_exporter.generate_chat_pdf(session)
                if pdf_buffer.getvalue():  # Check if PDF has content
                    st.download_button(
                        label="📄 Download Chat PDF",
                        data=pdf_buffer,
                        file_name=f"chat_{getattr(session, 'session_id', 'temp')[:8]}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                logger.error(f"PDF generation error: {e}")
                st.error("Failed to generate PDF")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="AI Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    try:
        # Initialize session state
        init_session_state()
        
        # Check if initialization was successful
        if not st.session_state.get('initialized', False):
            st.error("Application failed to initialize properly.")
            st.info("Please refresh the page or contact support if the problem persists.")
            if st.button("Retry Initialization"):
                # Clear session state and try again
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            return
        
        # Check if session manager is available
        if not hasattr(st.session_state, 'session_manager') or st.session_state.session_manager is None:
            st.error("Session manager not available.")
            st.info("Running in limited mode without session persistence.")
            
            # Create a minimal session for basic functionality
            from dataclasses import dataclass
            from datetime import datetime
            
            @dataclass
            class MinimalSession:
                session_id: str = "temp-session"
                user_type: str = "guest"
                email: str = None
                messages: list = None
                
                def __post_init__(self):
                    if self.messages is None:
                        self.messages = []
            
            session = MinimalSession()
            
            # Render basic welcome page
            st.title("🤖 AI Chat Assistant")
            st.warning("Running in limited mode. Some features may not be available.")
            st.info("Please check your configuration and refresh the page.")
            return
        
        # Get current session
        session = st.session_state.session_manager.get_session()
        
        # Render appropriate page
        if st.session_state.get('page') == "chat":
            if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
                render_sidebar(session)
            render_chat_interface(session)
        else:
            render_welcome_page()
            
    except Exception as e:
        logger.critical(f"Critical application error: {e}", exc_info=True)
        st.error("A critical error occurred.")
        
        # Show error details in an expander
        with st.expander("Error Details"):
            st.exception(e)
        
        # Provide recovery options
        st.subheader("Recovery Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("📋 Copy Error"):
                st.code(str(e))

if __name__ == "__main__":
    main()
