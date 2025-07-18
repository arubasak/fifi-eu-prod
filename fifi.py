import streamlit as st
import os
import uuid
import json
import logging
import re
import time
import functools
import threading
import html
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import requests
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_tavily import TavilySearch
from urllib.parse import urlparse

# =============================================================================
# VERSION 2.4 COMPLETE - ALL ENHANCEMENTS INTEGRATED
# CHANGELOG:
# v2.1: Enhanced Configuration Management with validation & fallbacks
# v2.2: FIXED Rate limiter bug - proper consumption tracking with rollback
# v2.3: FIXED SQLite Cloud display - user-friendly status messages  
# v2.4: NEW Inline Citation Links - citations appear after product/supplier mentions
# =============================================================================

# Setup enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False
TAVILY_AVAILABLE = False
PINECONE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Install with: pip install langchain-openai")

try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    logger.warning("SQLite Cloud not available. Install with: pip install sqlitecloud")

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    logger.warning("Tavily not available. Install with: pip install langchain-tavily")

try:
    from pinecone import Pinecone
    from pinecone_plugins.assistant.models.chat import Message as PineconeMessage
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install with: pip install pinecone-client")

try:
    from tavily import TavilyClient
    TAVILY_CLIENT_AVAILABLE = True
except ImportError:
    TAVILY_CLIENT_AVAILABLE = False
    logger.warning("Tavily client not available. Install with: pip install tavily-python")

# =============================================================================
# v2.1: Enhanced Configuration Management
# =============================================================================

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
        
        try:
            # Base required configuration
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
            
            # Optional API configurations
            self.PINECONE_API_KEY = get_config_value('PINECONE_API_KEY', required=False)
            if self.PINECONE_API_KEY:
                self.PINECONE_API_KEY = self._validate_api_key(self.PINECONE_API_KEY)
            
            self.PINECONE_ASSISTANT_NAME = get_config_value('PINECONE_ASSISTANT_NAME', required=False) or 'my-chat-assistant'
            self.TAVILY_API_KEY = get_config_value('TAVILY_API_KEY', required=False)
            if self.TAVILY_API_KEY:
                self.TAVILY_API_KEY = self._validate_api_key(self.TAVILY_API_KEY)
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            raise

    def _validate_url(self, url: str) -> str:
        """Validate and clean URL format"""
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
        return url.rstrip('/')  # Remove trailing slashes

    def _validate_api_key(self, api_key: str) -> str:
        """Validate API key format and length"""
        if not api_key:
            raise ValueError("API key cannot be empty")
        api_key = api_key.strip()
        if len(api_key) < 5:
            raise ValueError("API key too short")
        return api_key

# Enhanced config initialization with fallback
try:
    config = Config()
    logger.info("Configuration loaded successfully")
except ValueError as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.info("Some features may be unavailable. Running in limited mode.")
    
    # Create minimal fallback config
    class MinimalConfig:
        def __init__(self):
            self.JWT_SECRET = os.getenv('JWT_SECRET', 'development-secret-key')
            self.WORDPRESS_URL = os.getenv('WORDPRESS_URL', 'https://example.com')
            self.OPENAI_API_KEY = None
            self.SQLITE_CLOUD_CONNECTION = None
            self.PINECONE_API_KEY = None
            self.PINECONE_ASSISTANT_NAME = 'my-chat-assistant'
            self.TAVILY_API_KEY = None
    
    config = MinimalConfig()
    st.warning("⚠️ Running in limited mode. Check your environment variables or Streamlit secrets.")
    logger.warning("Using minimal configuration fallback")

# =============================================================================
# v2.2: Enhanced Rate Limiting System with Rollback
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter with rollback mechanism for failed requests."""
    
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.pending_requests: Dict[str, List[float]] = defaultdict(list)  # Track pending requests
        self._lock = threading.Lock()
        
    def check_available(self, identifier: str) -> bool:
        """Check if request would be allowed WITHOUT consuming a slot."""
        with self._lock:
            try:
                now = time.time()
                window_start = now - self.window_seconds
                
                # Clean old requests
                self.requests[identifier] = [
                    timestamp for timestamp in self.requests[identifier] 
                    if timestamp > window_start
                ]
                
                # Clean old pending requests
                self.pending_requests[identifier] = [
                    timestamp for timestamp in self.pending_requests[identifier] 
                    if timestamp > window_start
                ]
                
                # Check if we're under the limit (including pending)
                total_requests = len(self.requests[identifier]) + len(self.pending_requests[identifier])
                return total_requests < self.max_requests
                
            except Exception as e:
                logger.error(f"Rate limiter check error: {e}")
                return True  # Allow on error
    
    def reserve_slot(self, identifier: str) -> str:
        """Reserve a slot for a request. Returns reservation_id or None if not available."""
        with self._lock:
            try:
                if self.check_available(identifier):
                    now = time.time()
                    reservation_id = f"{identifier}_{now}_{uuid.uuid4().hex[:8]}"
                    self.pending_requests[identifier].append(now)
                    return reservation_id
                return None
                
            except Exception as e:
                logger.error(f"Rate limiter reservation error: {e}")
                return f"error_{identifier}_{time.time()}"  # Allow on error
    
    def confirm_request(self, identifier: str, reservation_id: str) -> bool:
        """Confirm a successful request - moves from pending to confirmed."""
        with self._lock:
            try:
                now = time.time()
                # Remove one pending request (most recent)
                if self.pending_requests[identifier]:
                    self.pending_requests[identifier].pop()
                    
                # Add to confirmed requests
                self.requests[identifier].append(now)
                return True
                
            except Exception as e:
                logger.error(f"Rate limiter confirm error: {e}")
                return False
    
    def rollback_request(self, identifier: str, reservation_id: str) -> bool:
        """Rollback a failed request - removes from pending without confirming."""
        with self._lock:
            try:
                # Remove one pending request (most recent)
                if self.pending_requests[identifier]:
                    self.pending_requests[identifier].pop()
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Rate limiter rollback error: {e}")
                return False
        
    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for identifier."""
        with self._lock:
            try:
                now = time.time()
                window_start = now - self.window_seconds
                
                # Clean old requests
                self.requests[identifier] = [
                    timestamp for timestamp in self.requests[identifier] 
                    if timestamp > window_start
                ]
                
                # Clean old pending requests
                self.pending_requests[identifier] = [
                    timestamp for timestamp in self.pending_requests[identifier] 
                    if timestamp > window_start
                ]
                
                # Calculate remaining (subtract both confirmed and pending)
                total_used = len(self.requests[identifier]) + len(self.pending_requests[identifier])
                return max(0, self.max_requests - total_used)
                
            except Exception as e:
                logger.error(f"Error getting remaining requests: {e}")
                return self.max_requests
    
    def get_reset_time(self, identifier: str) -> float:
        """Get timestamp when the rate limit will reset for identifier."""
        with self._lock:
            try:
                all_requests = self.requests[identifier] + self.pending_requests[identifier]
                if not all_requests:
                    return time.time()
                
                # Find the oldest request in the current window
                oldest_request = min(all_requests)
                return oldest_request + self.window_seconds
                
            except Exception as e:
                logger.error(f"Error getting reset time: {e}")
                return time.time()

# Initialize global rate limiter
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

def check_and_reserve_rate_limit(session_id: str) -> tuple[bool, str, Optional[str]]:
    """
    Check rate limit and reserve a slot if available.
    
    Returns:
        tuple: (is_allowed, message_for_user, reservation_id)
    """
    if rate_limiter.check_available(session_id):
        reservation_id = rate_limiter.reserve_slot(session_id)
        if reservation_id:
            remaining = rate_limiter.get_remaining_requests(session_id)
            return True, f"✅ Request reserved ({remaining} remaining)", reservation_id
        else:
            return False, "⏱️ Rate limit exceeded. Please wait a moment.", None
    else:
        reset_time = rate_limiter.get_reset_time(session_id)
        wait_seconds = int(reset_time - time.time())
        return False, f"⏱️ Rate limit exceeded. Try again in {wait_seconds} seconds.", None

# =============================================================================
# Input Sanitization & Validation
# =============================================================================

def sanitize_input(text: str, max_length: int = 4000) -> str:
    """Sanitize user input to prevent XSS attacks and limit length."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # HTML escape the input to prevent XSS
    sanitized = html.escape(text)
    
    # Remove potentially dangerous characters (additional safety)
    sanitized = re.sub(r'[<>"\'\n\r\t]', '', sanitized)
    
    # Remove any script-like patterns
    sanitized = re.sub(r'(?i)(javascript|vbscript|onload|onerror|onclick)', '', sanitized)
    
    # Limit length to prevent DoS
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
    
    # Strip whitespace
    return sanitized.strip()

def sanitize_chat_message(message: str) -> tuple[str, bool]:
    """Sanitize chat message and check if it's valid."""
    try:
        # Basic validation
        if not message or not isinstance(message, str):
            return "", False
        
        # Sanitize
        sanitized = sanitize_input(message, max_length=2000)  # Shorter limit for chat
        
        # Check if message is not empty after sanitization
        if not sanitized or len(sanitized.strip()) < 2:
            return "", False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'(?i)(exec|eval|system|import\s+os)',  # Code injection attempts
            r'(?i)(drop\s+table|delete\s+from)',    # SQL injection attempts
            r'<script|javascript:',                  # XSS attempts
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized):
                logger.warning(f"Suspicious pattern detected in message: {pattern}")
                return "", False
        
        return sanitized, True
        
    except Exception as e:
        logger.error(f"Error sanitizing chat message: {e}")
        return "", False

# =============================================================================
# Domain Exclusions & Enhanced Token Detection
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

def is_token_limit_error(error: Exception) -> bool:
    """Enhanced detection for token limit errors."""
    error_str = str(error).lower()
    token_error_indicators = [
        "token limit", "token_limit", "tokens exceeded", "context length",
        "context_length", "maximum context", "context too long", "150k",
        "150000", "token count", "max_tokens", "input too long",
        "sequence length", "context window"
    ]
    return any(indicator in error_str for indicator in token_error_indicators)

def add_utm_to_links(content: str, utm_source: str = "fifi-chat") -> str:
    """Add UTM tracking parameters to all links in content."""
    def replacer(match):
        url = match.group(1)
        utm_params = f"utm_source={utm_source}&utm_medium=ai-chat"
        if '?' in url:
            new_url = f"{url}&{utm_params}"
        else:
            new_url = f"{url}?{utm_params}"
        return f"({new_url})"
    
    return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

# =============================================================================
# v2.4: NEW Inline Citation Processing
# =============================================================================

def process_inline_citations(content: str, citations_list: List[str]) -> str:
    """
    Process content to add inline citations after product/supplier mentions.
    
    Args:
        content: Original content from Pinecone
        citations_list: List of citation strings like "[1] Source A"
        
    Returns:
        Content with inline citations added
    """
    if not citations_list or not content:
        return content
    
    # Create mapping of citation numbers
    citation_map = {}
    for i, citation in enumerate(citations_list, 1):
        citation_map[i] = citation
    
    # Keywords that likely indicate products, suppliers, or companies
    citation_triggers = [
        # Product-related terms
        r'\b(vanilla extract|cocoa|chocolate|protein|ingredient|supplement|additive|flavor|spice|extract|powder|oil|syrup)\b',
        # Supplier/Company patterns
        r'\b([A-Z][a-z]+ (?:Inc|LLC|Corp|Corporation|Company|Co|Ltd|Limited|Group|International|Foods|Ingredients)\.?)\b',
        # Brand/Company names (2+ capitalized words)
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
        # Supplier-related terms
        r'\b(supplier|manufacturer|distributor|producer|vendor|provider|source|company|brand|farm|factory|mill)\b',
        # Geographic suppliers
        r'\b(European|American|Asian|organic|local|regional|sustainable|premium|specialty)\s+(supplier|manufacturer|producer|source)\b',
        # Price/market terms
        r'\b(price|cost|market|trading|wholesale|retail|bulk|commodity)\b'
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(f'({pattern})' for pattern in citation_triggers)
    
    # Split content into sentences for better citation placement
    sentences = re.split(r'(?<=[.!?])\s+', content)
    processed_sentences = []
    
    citation_counter = 1
    used_citations = set()
    
    for sentence in sentences:
        # Check if sentence contains citation-worthy content
        if re.search(combined_pattern, sentence, re.IGNORECASE):
            # Add citation at the end of sentence if not already present
            if not re.search(r'\[\d+\]', sentence):
                # Distribute citations across different sentences
                citation_num = ((citation_counter - 1) % len(citations_list)) + 1
                sentence = sentence.rstrip('.!?') + f' [{citation_num}]' + sentence[-1] if sentence and sentence[-1] in '.!?' else sentence + f' [{citation_num}]'
                used_citations.add(citation_num)
                citation_counter += 1
        
        processed_sentences.append(sentence)
    
    # If no citations were added, add them to the first substantial sentence
    if not used_citations and len(processed_sentences) > 0:
        for i, sentence in enumerate(processed_sentences):
            if len(sentence.strip()) > 50:  # Substantial sentence
                if not re.search(r'\[\d+\]', sentence):
                    processed_sentences[i] = sentence.rstrip('.!?') + ' [1]' + (sentence[-1] if sentence and sentence[-1] in '.!?' else '')
                    used_citations.add(1)
                break
    
    return ' '.join(processed_sentences)

def extract_company_mentions(content: str) -> List[str]:
    """
    Extract company and product mentions from content for better citation placement.
    
    Args:
        content: Text content to analyze
        
    Returns:
        List of company/product mentions found
    """
    mentions = []
    
    # Company name patterns
    company_patterns = [
        r'\b([A-Z][a-z]+\s+(?:Inc|LLC|Corp|Corporation|Company|Co|Ltd|Limited|Group|International|Foods|Ingredients)\.?)\b',
        r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # Multi-word capitalized names
    ]
    
    # Product/ingredient patterns
    product_patterns = [
        r'\b(organic\s+\w+|premium\s+\w+|sustainable\s+\w+|natural\s+\w+)\b',
        r'\b(vanilla\s+extract|cocoa\s+powder|plant-based\s+protein|clean\s+label)\b',
    ]
    
    all_patterns = company_patterns + product_patterns
    
    for pattern in all_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                mention = next(m for m in match if m)  # Get first non-empty group
            else:
                mention = match
            
            if mention and len(mention.strip()) > 3:
                mentions.append(mention.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_mentions = []
    for mention in mentions:
        if mention.lower() not in seen:
            seen.add(mention.lower())
            unique_mentions.append(mention)
    
    return unique_mentions[:5]  # Limit to top 5 mentions

# =============================================================================
# Enhanced Error Handling System
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
    """Enhanced error handling with user-friendly messages and recovery suggestions."""
    
    def __init__(self):
        self.error_history = []
        self.component_status = {}
        
    def handle_api_error(self, component: str, operation: str, error: Exception) -> ErrorContext:
        """Handle API-related errors with smart classification."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Enhanced token limit detection
        if is_token_limit_error(error):
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="TokenLimitError",
                severity=ErrorSeverity.MEDIUM,
                user_message=f"{component} token limit reached. Switching to alternative search.",
                technical_details=str(error),
                recovery_suggestions=[
                    "The system will automatically use domain-specific search",
                    "Try a shorter or more specific query",
                    "This is normal for very long conversations"
                ],
                fallback_available=True
            )
        
        # Other error handling (keeping original logic)
        elif "timeout" in error_str or "timed out" in error_str:
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="TimeoutError",
                severity=ErrorSeverity.MEDIUM,
                user_message=f"{component} is responding slowly. Please try again in a moment.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try your request again",
                    "Check your internet connection",
                    "Try a simpler query"
                ],
                fallback_available=True
            )
        
        elif "unauthorized" in error_str or "forbidden" in error_str or "401" in error_str or "403" in error_str:
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="AuthenticationError",
                severity=ErrorSeverity.HIGH,
                user_message=f"{component} authentication failed. Please check your API configuration.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Verify your API key is correct",
                    "Check if your API key has expired",
                    "Ensure you have proper API permissions"
                ],
                fallback_available=False
            )
        
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="RateLimitError",
                severity=ErrorSeverity.MEDIUM,
                user_message=f"{component} rate limit reached. Please wait a moment before trying again.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Wait 1-2 minutes before trying again",
                    "Try a shorter query",
                    "Consider upgrading your API plan"
                ],
                fallback_available=True
            )
        
        else:
            # Generic error
            return ErrorContext(
                component=component,
                operation=operation,
                error_type=error_type,
                severity=ErrorSeverity.MEDIUM,
                user_message=f"{component} encountered an unexpected error. We're switching to backup systems.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try your request again",
                    "Try a different approach to your question",
                    "Contact support if the issue persists"
                ],
                fallback_available=True
            )
    
    def display_error_to_user(self, error_context: ErrorContext):
        """Display user-friendly error message in Streamlit."""
        severity_icons = {
            ErrorSeverity.LOW: "ℹ️",
            ErrorSeverity.MEDIUM: "⚠️",
            ErrorSeverity.HIGH: "🚨", 
            ErrorSeverity.CRITICAL: "💥"
        }
        
        # Main error message
        icon = severity_icons.get(error_context.severity, "❓")
        if error_context.severity == ErrorSeverity.CRITICAL:
            st.error(f"{icon} **{error_context.user_message}**")
        elif error_context.severity == ErrorSeverity.HIGH:
            st.error(f"{icon} {error_context.user_message}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            st.warning(f"{icon} {error_context.user_message}")
        else:
            st.info(f"{icon} {error_context.user_message}")
        
        # Recovery suggestions
        if error_context.recovery_suggestions:
            with st.expander("💡 What you can do:"):
                for suggestion in error_context.recovery_suggestions:
                    st.write(f"• {suggestion}")
        
        # Fallback availability
        if error_context.fallback_available:
            st.success("✅ Backup systems are available and will be used automatically.")
    
    def log_error(self, error_context: ErrorContext):
        """Log error for monitoring."""
        self.error_history.append({
            "timestamp": datetime.now(),
            "component": error_context.component,
            "operation": error_context.operation,
            "error_type": error_context.error_type,
            "severity": error_context.severity.value,
            "technical_details": error_context.technical_details
        })
        
        # Update component status
        self.component_status[error_context.component] = {
            "status": "error",
            "last_error": datetime.now(),
            "error_type": error_context.error_type,
            "severity": error_context.severity.value
        }
        
        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history.pop(0)
    
    def mark_component_healthy(self, component: str):
        """Mark a component as healthy."""
        self.component_status[component] = {
            "status": "healthy",
            "last_check": datetime.now()
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        if not self.component_status:
            return {"overall_health": "Unknown", "healthy_components": 0, "total_components": 0}
        
        healthy_count = sum(1 for status in self.component_status.values() if status.get("status") == "healthy")
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

# Initialize error handler
error_handler = EnhancedErrorHandler()

# Enhanced error handling decorators
def handle_api_errors(component: str, operation: str, show_to_user: bool = True):
    """Decorator for API error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Mark component as healthy on success
                error_handler.mark_component_healthy(component)
                return result
            except Exception as e:
                error_context = error_handler.handle_api_error(component, operation, e)
                error_handler.log_error(error_context)
                
                if show_to_user:
                    error_handler.display_error_to_user(error_context)
                
                logger.error(f"{component} {operation} failed: {e}")
                return None
        return wrapper
    return decorator

# =============================================================================
# Enhanced Session Management
# =============================================================================

@dataclass
class UserSession:
    session_id: str
    user_type: str = "guest"
    email: Optional[str] = None
    first_name: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class SimpleSessionManager:
    """Simple session manager with in-memory storage."""
    
    def __init__(self):
        self.sessions = {}
    
    def get_session(self) -> UserSession:
        session_id = st.session_state.get('current_session_id')
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            return session
        
        return self.create_guest_session()
    
    def create_guest_session(self) -> UserSession:
        session = UserSession(session_id=str(uuid.uuid4()))
        self.sessions[session.session_id] = session
        st.session_state.current_session_id = session.session_id
        return session
    
    def clear_chat_history(self, session: UserSession):
        session.messages = []
        session.last_activity = datetime.now()

# =============================================================================
# v2.4: Enhanced PineconeAssistantTool with Inline Citations
# =============================================================================

class PineconeAssistantTool:
    """Advanced Pinecone Assistant with inline citations and enhanced token limit detection."""
    
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE: 
            error_context = error_handler.handle_import_error("pinecone", "Pinecone Knowledge Base")
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
            # This will be caught by the decorator
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
            
            response = self.assistant.chat(messages=pinecone_messages, model="gpt-4o")
            content = response.message.content
            has_citations = False
            citations_list = []
            
            # Process citations
            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                seen_items = set()
                
                for citation in response.citations:
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
                                    link = f"[{len(seen_items) + 1}] [{display_text}]({link_url})"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                            else:
                                display_text = getattr(reference.file, 'name', 'Unknown Source')
                                if display_text not in seen_items:
                                    link = f"[{len(seen_items) + 1}] {display_text}"
                                    citations_list.append(link)
                                    seen_items.add(display_text)
                
                # v2.4: NEW Process inline citations
                if citations_list:
                    # Add inline citations to content
                    content_with_inline = process_inline_citations(content, citations_list)
                    
                    # Add traditional bottom citations section
                    citations_header = "\n\n---\n**Sources:**\n"
                    content = content_with_inline + citations_header + "\n".join(citations_list)
                    
                    # Extract company mentions for better citation context
                    company_mentions = extract_company_mentions(content)
                    logger.info(f"Found company/product mentions: {company_mentions}")
            
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi Knowledge Base",
                "has_citations": has_citations,
                "inline_citations": has_citations and len(citations_list) > 0,
                "response_length": len(content)
            }
            
        except Exception as e:
            # Enhanced token limit detection
            if is_token_limit_error(e):
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

# =============================================================================
# Enhanced TavilyFallbackAgent with Domain Exclusions
# =============================================================================

class TavilyFallbackAgent:
    """Enhanced Tavily fallback agent with domain exclusions and smart result synthesis."""
    
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE and not TAVILY_CLIENT_AVAILABLE:
            error_context = error_handler.handle_import_error("langchain-tavily", "Web Search")
            error_handler.display_error_to_user(error_context)
            raise ImportError("Tavily client not available.")
        
        self.tavily_api_key = tavily_api_key
        self.excluded_domains = DEFAULT_EXCLUDED_DOMAINS.copy()
        
        # Initialize appropriate Tavily client
        if TAVILY_CLIENT_AVAILABLE:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        elif TAVILY_AVAILABLE:
            self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)
        else:
            raise ImportError("No Tavily client available")

    def search_with_exclusions(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform search with competitor domain exclusions."""
        try:
            if TAVILY_CLIENT_AVAILABLE and hasattr(self, 'tavily_client'):
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=False,
                    exclude_domains=self.excluded_domains
                )
                
                # Format response
                formatted_results = []
                if response.get('results'):
                    for result in response['results']:
                        # Double-check domain exclusion (extra safety)
                        url = result.get('url', '')
                        if not any(domain in url for domain in self.excluded_domains):
                            formatted_results.append({
                                'title': result.get('title', 'No title'),
                                'content': result.get('content', 'No content'),
                                'url': url
                            })
                
                return {
                    "results": formatted_results,
                    "answer": response.get('answer'),
                    "success": True,
                    "excluded_count": len(response.get('results', [])) - len(formatted_results)
                }
            
            elif hasattr(self, 'tavily_tool'):
                # Fallback to basic TavilySearch
                search_results = self.tavily_tool.invoke({"query": query})
                return {
                    "results": [{"content": str(search_results)}],
                    "answer": str(search_results),
                    "success": True,
                    "excluded_count": 0
                }
            
            return {"results": [], "success": False, "error": "No search client available"}
            
        except Exception as e:
            logger.error(f"Search with exclusions error: {e}")
            return {"results": [], "success": False, "error": str(e)}

    def search_domain_specific(self, query: str, domain: str, max_results: int = 5) -> Dict[str, Any]:
        """Search within a specific domain only."""
        try:
            if TAVILY_CLIENT_AVAILABLE and hasattr(self, 'tavily_client'):
                # Add site restriction to query
                domain_query = f"site:{domain} {query}"
                
                response = self.tavily_client.search(
                    query=domain_query,
                    search_depth="advanced", 
                    max_results=max_results,
                    include_answer=True,
                    include_raw_content=False
                )
                
                # Format response
                formatted_results = []
                if response.get('results'):
                    for result in response['results']:
                        formatted_results.append({
                            'title': result.get('title', 'No title'),
                            'content': result.get('content', 'No content'),
                            'url': result.get('url', 'No URL')
                        })
                
                return {
                    "results": formatted_results,
                    "answer": response.get('answer'),
                    "success": True,
                    "domain": domain
                }
            
            return {"results": [], "success": False, "error": "Domain-specific search not available"}
            
        except Exception as e:
            logger.error(f"Domain-specific search error: {e}")
            return {"results": [], "success": False, "error": str(e)}

    def synthesize_search_results(self, search_result: Dict[str, Any], source_label: str) -> str:
        """Format search results into readable response."""
        if not search_result.get("success"):
            return f"I apologize, but I encountered an error: {search_result.get('error', 'Unknown error')}"
        
        response_parts = []
        
        # Add summary if available
        if search_result.get("answer"):
            response_parts.append(f"**Summary:** {search_result['answer']}")
        
        # Add results
        results = search_result.get("results", [])
        if results:
            response_parts.append(f"\n**Detailed Information:**")
            for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', '')
                    
                    # Truncate content if too long
                    if len(content) > 200:
                        content = content[:200] + "..."
                    
                    response_parts.append(f"\n{i}. **{title}**")
                    response_parts.append(f"   {content}")
                    if url:
                        response_parts.append(f"   [Read more]({url})")
                else:
                    # Handle string results
                    content = str(result)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    response_parts.append(f"\n{i}. {content}")
        
        # Add source attribution
        if search_result.get("domain"):
            response_parts.append(f"\n*Source: {source_label} ({search_result['domain']})*")
        else:
            response_parts.append(f"\n*Source: {source_label}*")
            
        # Add exclusion info if applicable
        if search_result.get("excluded_count", 0) > 0:
            response_parts.append(f"\n*Note: {search_result['excluded_count']} competitor results excluded*")
        
        formatted_response = "".join(response_parts)
        return add_utm_to_links(formatted_response)

    @handle_api_errors("Tavily", "Web Search", show_to_user=False)
    def query(self, message: str, chat_history: List[BaseMessage], search_type: str = "general") -> Dict[str, Any]:
        try:
            # Perform search based on type
            if search_type == "12taste_only":
                search_result = self.search_domain_specific(message, "12taste.com")
                source_label = "FiFi 12taste.com Search"
            else:  # general search with exclusions
                search_result = self.search_with_exclusions(message)
                source_label = "FiFi Web Search"
            
            # Synthesize results
            final_content = self.synthesize_search_results(search_result, source_label)
            
            return {
                "content": final_content,
                "success": True,
                "source": source_label
            }
        except Exception as e:
            # This will be caught by the decorator
            raise e

# =============================================================================
# v2.4: Enhanced AI with Inline Citations and Smart Fallback
# =============================================================================

class EnhancedAI:
    """Enhanced AI with inline citations, proper rate limiting, and smart fallback."""
    
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        self.openai_client = None
        self.langchain_llm = None
        
        # Initialize OpenAI clients for potential LLM fallback
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                error_handler.mark_component_healthy("OpenAI")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                error_context = error_handler.handle_api_error("OpenAI", "Initialize Client", e)
                error_handler.log_error(error_context)
        
        if LANGCHAIN_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.langchain_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=config.OPENAI_API_KEY,
                    temperature=0.7
                )
                error_handler.mark_component_healthy("LangChain")
            except Exception as e:
                logger.error(f"LangChain LLM initialization failed: {e}")
                error_context = error_handler.handle_api_error("LangChain", "Initialize LLM", e)
                error_handler.log_error(error_context)
        
        # Initialize Pinecone Assistant
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY and config.PINECONE_ASSISTANT_NAME:
            try:
                self.pinecone_tool = PineconeAssistantTool(
                    api_key=config.PINECONE_API_KEY,
                    assistant_name=config.PINECONE_ASSISTANT_NAME
                )
                logger.info("Pinecone Assistant initialized successfully")
            except Exception as e:
                logger.error(f"Pinecone Assistant initialization failed: {e}")
                self.pinecone_tool = None
        
        # Initialize Tavily Fallback Agent
        if (TAVILY_AVAILABLE or TAVILY_CLIENT_AVAILABLE) and config.TAVILY_API_KEY:
            try:
                self.tavily_agent = TavilyFallbackAgent(tavily_api_key=config.TAVILY_API_KEY)
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
        
        # PRIORITY 5: NO CITATIONS = MANDATORY FALLBACK (unless very short)
        if not has_real_citations:
            if "[1]" not in content_raw and "**Sources:**" not in content_raw:
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
        
        # PRIORITY 7: Response length suggests substantial answer without sources
        response_length = pinecone_response.get("response_length", 0)
        if response_length > 100 and not pinecone_response.get("has_citations", False):
            return True
        
        return False

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get enhanced AI response with inline citations, rate limiting, and comprehensive error handling."""
        
        session_id = st.session_state.get('current_session_id', 'anonymous')
        reservation_id = None
        
        try:
            # STEP 1: Check and reserve rate limit slot
            allowed, rate_message, reservation_id = check_and_reserve_rate_limit(session_id)
            
            if not allowed:
                return {
                    "content": rate_message,
                    "source": "Rate Limiter",
                    "used_search": False,
                    "used_pinecone": False,
                    "has_citations": False,
                    "inline_citations": False,
                    "safety_override": False,
                    "success": False
                }
            
            # STEP 2: Sanitize input
            sanitized_prompt, is_valid = sanitize_chat_message(prompt)
            if not is_valid:
                # Rollback rate limit reservation
                rate_limiter.rollback_request(session_id, reservation_id)
                return {
                    "content": "⚠️ Invalid input detected. Please rephrase your question.",
                    "source": "Input Validation",
                    "used_search": False,
                    "used_pinecone": False,
                    "has_citations": False,
                    "inline_citations": False,
                    "safety_override": False,
                    "success": False
                }
            
            # STEP 3: Convert chat history to LangChain format
            langchain_history = []
            if chat_history:
                for msg in chat_history[-10:]:  # Last 10 messages to avoid token limits
                    if msg.get("role") == "user":
                        langchain_history.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        langchain_history.append(AIMessage(content=msg.get("content", "")))
            
            # Add current prompt (sanitized)
            langchain_history.append(HumanMessage(content=sanitized_prompt))
            
            # STEP 4: Try Pinecone Knowledge Base FIRST
            if self.pinecone_tool:
                pinecone_response = self.pinecone_tool.query(langchain_history)
                
                if pinecone_response and pinecone_response.get("success"):
                    should_fallback = self.should_use_web_fallback(pinecone_response)
                    
                    if not should_fallback:
                        # SUCCESS: Confirm rate limit usage
                        rate_limiter.confirm_request(session_id, reservation_id)
                        logger.info("Using Pinecone knowledge base response with inline citations")
                        return {
                            "content": pinecone_response["content"],
                            "source": pinecone_response.get("source", "FiFi Knowledge Base"),
                            "used_search": False,
                            "used_pinecone": True,
                            "has_citations": pinecone_response.get("has_citations", False),
                            "inline_citations": pinecone_response.get("inline_citations", False),
                            "safety_override": False,
                            "success": True
                        }
                    else:
                        logger.warning("SAFETY OVERRIDE: Detected potentially fabricated information")
                        # Continue to Tavily fallback with safety override flag
                else:
                    # Check for token limit error
                    error_type = pinecone_response.get("error_type")
                    if error_type == "token_limit":
                        logger.info("Token limit detected, falling back to domain-specific search")
                        # Use domain-specific search for token limits
                        if self.tavily_agent:
                            tavily_response = self.tavily_agent.query(sanitized_prompt, langchain_history[:-1], search_type="12taste_only")
                            
                            if tavily_response and tavily_response.get("success"):
                                # SUCCESS: Confirm rate limit usage
                                rate_limiter.confirm_request(session_id, reservation_id)
                                return {
                                    "content": tavily_response["content"],
                                    "source": tavily_response.get("source", "FiFi 12taste.com Search"),
                                    "used_search": True,
                                    "used_pinecone": False,
                                    "has_citations": False,
                                    "inline_citations": False,
                                    "safety_override": False,
                                    "token_limit_fallback": True,
                                    "success": True
                                }
            
            # STEP 5: Fall back to Tavily Web Search (with exclusions)
            if self.tavily_agent:
                logger.info("Using Tavily web search fallback")
                tavily_response = self.tavily_agent.query(sanitized_prompt, langchain_history[:-1])
                
                if tavily_response and tavily_response.get("success"):
                    # SUCCESS: Confirm rate limit usage
                    rate_limiter.confirm_request(session_id, reservation_id)
                    return {
                        "content": tavily_response["content"],
                        "source": tavily_response.get("source", "FiFi Web Search"),
                        "used_search": True,
                        "used_pinecone": False,
                        "has_citations": False,
                        "inline_citations": False,
                        "safety_override": True if self.pinecone_tool else False,
                        "success": True
                    }
                else:
                    # Tavily failed, log the issue
                    logger.warning("Tavily search failed, proceeding to final fallback")
            
            # STEP 6: Final fallback with helpful error message
            # Rollback rate limit since we couldn't provide a useful response
            rate_limiter.rollback_request(session_id, reservation_id)
            return {
                "content": "I apologize, but all AI systems are currently experiencing issues. Please try again in a few minutes, or try rephrasing your question.",
                "source": "System Status",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "inline_citations": False,
                "safety_override": False,
                "success": False
            }
            
        except Exception as e:
            # Rollback rate limit on any error
            if reservation_id:
                rate_limiter.rollback_request(session_id, reservation_id)
                
            logger.error(f"Enhanced AI response error: {e}")
            error_context = error_handler.handle_api_error("AI System", "Generate Response", e)
            error_handler.log_error(error_context)
            
            return {
                "content": f"I'm experiencing technical difficulties. {error_context.user_message}",
                "source": "Error Recovery",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "inline_citations": False,
                "safety_override": False,
                "success": False
            }

def init_session_state():
    """Initialize session state safely with enhanced error handling."""
    if 'initialized' not in st.session_state:
        try:
            st.session_state.session_manager = SimpleSessionManager()
            st.session_state.ai = EnhancedAI()
            st.session_state.error_handler = error_handler
            st.session_state.page = "chat"
            st.session_state.initialized = True
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            error_context = error_handler.handle_api_error("Session", "Initialize", e)
            error_handler.display_error_to_user(error_context)
            st.session_state.initialized = False

def render_chat_interface():
    """Render the main chat interface with enhanced inline citation features."""
    st.title("🤖 FiFi AI Assistant v2.4")
    st.caption("✨ Enhanced with inline citation links after product/supplier mentions")
    
    session = st.session_state.session_manager.get_session()
    
    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            # Use safe content display
            content = msg.get("content", "")
            st.markdown(content, unsafe_allow_html=False)  # Streamlit's built-in safety
            
            # Show source information for assistant messages
            if msg.get("role") == "assistant":
                source_indicators = []
                
                if "source" in msg:
                    st.caption(f"Source: {msg['source']}")
                
                # Show knowledge base usage
                if msg.get("used_pinecone"):
                    if msg.get("has_citations"):
                        if msg.get("inline_citations"):
                            source_indicators.append("🧠 Knowledge Base (with inline citations)")
                        else:
                            source_indicators.append("🧠 Knowledge Base (with citations)")
                    else:
                        source_indicators.append("🧠 Knowledge Base")
                
                # Show web search usage  
                if msg.get("used_search"):
                    if msg.get("token_limit_fallback"):
                        source_indicators.append("🌐 Domain-Specific Search (Token Limit)")
                    else:
                        source_indicators.append("🌐 Web Search (Filtered)")
                
                if source_indicators:
                    st.caption(f"Enhanced with: {', '.join(source_indicators)}")
                
                # v2.4: Show inline citation info
                if msg.get("inline_citations"):
                    st.success("✨ NEW: Citations now appear inline after product/supplier mentions!")
                
                # Show safety override warning
                if msg.get("safety_override"):
                    st.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")
                
                # Show token limit fallback info
                if msg.get("token_limit_fallback"):
                    st.info("🔄 TOKEN LIMIT: Switched to domain-specific search for better results.")
    
    # Chat input with enhanced validation
    if prompt := st.chat_input("Ask me about ingredients, suppliers, market trends, or sourcing..."):
        # Sanitize input before processing
        sanitized_prompt, is_valid = sanitize_chat_message(prompt)
        
        if not is_valid:
            st.error("⚠️ Invalid message detected. Please rephrase your question.")
            return
        
        # Display user message (using sanitized version)
        with st.chat_message("user"):
            st.markdown(sanitized_prompt)
        
        # Add user message to history (sanitized)
        session.messages.append({
            "role": "user",
            "content": sanitized_prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying FiFi (Enhanced with Inline Citations v2.4)..."):
                response = st.session_state.ai.get_response(sanitized_prompt, session.messages)
                
                # Handle enhanced response format
                if isinstance(response, dict):
                    content = response.get("content", "No response generated.")
                    source = response.get("source", "Unknown")
                    used_search = response.get("used_search", False)
                    used_pinecone = response.get("used_pinecone", False)
                    has_citations = response.get("has_citations", False)
                    inline_citations = response.get("inline_citations", False)
                    safety_override = response.get("safety_override", False)
                    token_limit_fallback = response.get("token_limit_fallback", False)
                else:
                    # Fallback for simple string responses
                    content = str(response)
                    source = "FiFi AI"
                    used_search = False
                    used_pinecone = False
                    has_citations = False
                    inline_citations = False
                    safety_override = False
                    token_limit_fallback = False
                
                st.markdown(content, unsafe_allow_html=False)  # Safe display
                
                # Show enhancement indicators
                enhancements = []
                if used_pinecone:
                    if has_citations:
                        if inline_citations:
                            enhancements.append("🧠 Enhanced with Knowledge Base (with inline citations)")
                        else:
                            enhancements.append("🧠 Enhanced with Knowledge Base (with citations)")
                    else:
                        enhancements.append("🧠 Enhanced with Knowledge Base")
                
                if used_search:
                    if token_limit_fallback:
                        enhancements.append("🌐 Enhanced with domain-specific search")
                    else:
                        enhancements.append("🌐 Enhanced with filtered web search")
                
                if enhancements:
                    for enhancement in enhancements:
                        st.success(enhancement)
                
                # v2.4: Show inline citation info
                if inline_citations:
                    st.success("✨ NEW: Citations now appear inline after product/supplier mentions!")
                
                # Show safety override warning
                if safety_override:
                    st.error("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")
                
                # Show token limit fallback info
                if token_limit_fallback:
                    st.info("🔄 TOKEN LIMIT: Switched to domain-specific search for better results.")
        
        # Add AI response to history
        session.messages.append({
            "role": "assistant",
            "content": content,
            "source": source,
            "used_search": used_search,
            "used_pinecone": used_pinecone,
            "has_citations": has_citations,
            "inline_citations": inline_citations,
            "safety_override": safety_override,
            "token_limit_fallback": token_limit_fallback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session
        session.last_activity = datetime.now()
        
        st.rerun()

def render_sidebar():
    """v2.4: Render the enhanced sidebar with all improvements and inline citation features."""
    with st.sidebar:
        st.title("Chat Controls")
        
        session = st.session_state.session_manager.get_session()
        
        # Session info
        st.subheader("Session Info")
        st.write(f"**ID:** {session.session_id[:8]}...")
        st.write(f"**Type:** {session.user_type}")
        if session.email:
            st.write(f"**Email:** {session.email}")
        st.write(f"**Messages:** {len(session.messages)}")
        
        # v2.3: Enhanced system status with better SQLite Cloud display
        st.subheader("Enhanced System Status")
        st.write(f"**OpenAI:** {'✅' if OPENAI_AVAILABLE and config.OPENAI_API_KEY else '❌'}")
        st.write(f"**LangChain:** {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
        st.write(f"**Tavily Search:** {'✅' if TAVILY_AVAILABLE and config.TAVILY_API_KEY else '❌'}")
        st.write(f"**Tavily Client:** {'✅' if TAVILY_CLIENT_AVAILABLE and config.TAVILY_API_KEY else '❌'}")
        st.write(f"**Pinecone:** {'✅' if PINECONE_AVAILABLE and config.PINECONE_API_KEY else '❌'}")
        
        # v2.3: Better SQLite Cloud status display
        if SQLITECLOUD_AVAILABLE:
            if config.SQLITE_CLOUD_CONNECTION:
                st.write("**SQLite Cloud:** ✅ Available & Configured")
            else:
                st.write("**SQLite Cloud:** ⚠️ Available but Not Configured")
        else:
            st.write("**SQLite Cloud:** ℹ️ Not Installed")
        
        # v2.2: Rate limiter status with accurate display
        if hasattr(st.session_state, 'session_manager'):
            session_id = st.session_state.get('current_session_id', 'anonymous')
            remaining = rate_limiter.get_remaining_requests(session_id)
            st.write(f"**Rate Limit:** {remaining}/20 requests remaining")
            
            # Add additional rate limit details in expander
            with st.expander("📊 Rate Limit Details"):
                confirmed_requests = len(rate_limiter.requests.get(session_id, []))
                pending_requests = len(rate_limiter.pending_requests.get(session_id, []))
                st.write(f"Confirmed: {confirmed_requests}")
                st.write(f"Pending: {pending_requests}")
                st.write(f"Available: {remaining}")
                
                if confirmed_requests + pending_requests > 0:
                    reset_time = rate_limiter.get_reset_time(session_id)
                    reset_in = max(0, int(reset_time - time.time()))
                    st.write(f"Resets in: {reset_in}s")
        
        # Component Status
        if hasattr(st.session_state, 'ai'):
            ai = st.session_state.ai
            pinecone_status = "✅ Connected" if ai.pinecone_tool else "❌ Failed"
            tavily_status = "✅ Connected" if ai.tavily_agent else "❌ Failed"
            st.write(f"**Pinecone Assistant:** {pinecone_status}")
            st.write(f"**Tavily Fallback Agent:** {tavily_status}")
        
        # Enhanced Error Dashboard
        with st.expander("🚨 System Health & Error Monitoring"):
            if hasattr(st.session_state, 'error_handler'):
                health_summary = error_handler.get_system_health_summary()
                
                # Overall health indicator
                health_color = {"Healthy": "🟢", "Degraded": "🟡", "Critical": "🔴"}.get(health_summary["overall_health"], "❓")
                st.write(f"**System Health:** {health_color} {health_summary['overall_health']}")
                
                # Component health details
                if error_handler.component_status:
                    st.write("**Component Status:**")
                    for component, status_info in error_handler.component_status.items():
                        if status_info.get("status") == "error":
                            severity = status_info.get("severity", "medium")
                            icon = "🚨" if severity in ["high", "critical"] else "⚠️"
                            st.write(f"{icon} **{component}**: {status_info.get('error_type', 'Error')}")
                            if "last_error" in status_info:
                                st.caption(f"Last error: {status_info['last_error'].strftime('%H:%M:%S')}")
                        else:
                            st.write(f"✅ **{component}**: Healthy")
                
                # Recent errors
                if error_handler.error_history:
                    st.write("**Recent Errors:**")
                    for error in error_handler.error_history[-3:]:  # Last 3 errors
                        severity_icon = {"low": "ℹ️", "medium": "⚠️", "high": "🚨", "critical": "💥"}
                        icon = severity_icon.get(error["severity"], "❓")
                        time_str = error["timestamp"].strftime("%H:%M:%S")
                        st.text(f"{icon} {time_str} [{error['component']}] {error['error_type']}")
                
                # System metrics
                if health_summary["total_components"] > 0:
                    health_percentage = (health_summary["healthy_components"] / health_summary["total_components"]) * 100
                    st.metric("System Health", f"{health_percentage:.0f}%")
                    st.metric("Error Count", health_summary["error_count"])
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear History"):
            st.session_state.session_manager.clear_chat_history(session)
            st.rerun()
        
        if st.button("🔄 New Session"):
            if 'current_session_id' in st.session_state:
                del st.session_state.current_session_id
            st.rerun()
        
        # v2.4: Enhanced feature status
        st.subheader("✨ v2.4 Features")
        st.write("✨ NEW: Inline Citation Links")
        st.write("✅ FIXED: SQLite Cloud Display (User-Friendly)")
        st.write("✅ FIXED: Rate Limiting (Accurate Display)")
        st.write("✅ Enhanced F&B AI Chat")
        st.write("✅ Input Sanitization & XSS Protection")
        st.write("✅ Domain Exclusions (Competitor Filtering)")
        st.write("✅ Enhanced Token Limit Detection")
        st.write("✅ Session Management")
        st.write("✅ Anti-Hallucination Safety")
        st.write("✅ Smart Fallback Logic")
        st.write("✅ Enhanced Error Handling")
        
        if LANGCHAIN_AVAILABLE:
            st.write("✅ LangChain Support")
        else:
            st.write("❌ LangChain (install required)")
        
        # Enhanced Safety Features Info
        with st.expander("🛡️ Enhanced Safety Features"):
            st.write("**✨ v2.4 NEW: Inline Citations:**")
            st.write("- Citations appear after product mentions")
            st.write("- Citations appear after supplier mentions")
            st.write("- Company names get inline citations")
            st.write("- Traditional bottom citations still included")
            st.write("- Better citation context mapping")
            st.write("")
            st.write("**v2.3 Display Improvements:**")
            st.write("- User-friendly SQLite Cloud status")
            st.write("- Better package availability messaging")
            st.write("- Clearer configuration guidance")
            st.write("")
            st.write("**v2.2 Rate Limiting Fix:**")
            st.write("- Proper request reservation system")
            st.write("- Rollback on failed requests")
            st.write("- Accurate remaining count display")
            st.write("- Thread-safe sliding window")
            st.write("")
            st.write("**Security Enhancements:**")
            st.write("- XSS Protection & Input Sanitization")
            st.write("- Suspicious pattern detection")
            st.write("- Thread-safe operations")
            st.write("")
            st.write("**Business Logic:**")
            st.write("- Competitor domain exclusions")
            st.write("- UTM tracking on all links")
            st.write("- Domain-specific fallback for token limits")
        
        # Example queries with focus on inline citations
        st.subheader("💡 Try These Queries (with Inline Citations)")
        example_queries = [
            "Find organic vanilla extract suppliers with certifications",
            "Latest trends in plant-based proteins from major manufacturers", 
            "Current cocoa prices and top suppliers in Europe",
            "Sustainable packaging suppliers with eco-friendly solutions",
            "Clean label ingredient alternatives from certified companies"
        ]
        
        for query in example_queries:
            if st.button(f"💬 {query}", key=f"example_{hash(query)}", use_container_width=True):
                # Sanitize example query before adding
                sanitized_query, is_valid = sanitize_chat_message(query)
                if is_valid:
                    session.messages.append({
                        "role": "user", 
                        "content": sanitized_query,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()
        
        # v2.3: Enhanced configuration status with better messaging
        st.subheader("🔧 API Configuration")
        st.write(f"**OpenAI:** {'✅ Configured' if config.OPENAI_API_KEY else '❌ Missing'}")
        st.write(f"**Tavily:** {'✅ Configured' if config.TAVILY_API_KEY else '❌ Missing'}")
        st.write(f"**Pinecone:** {'✅ Configured' if config.PINECONE_API_KEY else '❌ Missing'}")
        st.write(f"**WordPress:** {'✅ Configured' if config.WORDPRESS_URL else '❌ Missing'}")
        
        # v2.3: Better SQLite Cloud configuration display
        if SQLITECLOUD_AVAILABLE:
            if config.SQLITE_CLOUD_CONNECTION:
                st.write("**SQLite Cloud:** ✅ Configured")
            else:
                st.write("**SQLite Cloud:** ⚠️ Package Available, API Key Missing")
        else:
            st.write("**SQLite Cloud:** ℹ️ Package Not Installed")
            if st.button("📦 Install SQLite Cloud", help="Run: pip install sqlitecloud"):
                st.code("pip install sqlitecloud", language="bash")

def main():
    """Main application function with all enhancements integrated."""
    st.set_page_config(
        page_title="FiFi AI Assistant v2.4 - Complete Enhanced Edition",
        page_icon="🤖",
        layout="wide"
    )
    
    try:
        # Initialize session state
        init_session_state()
        
        if not st.session_state.get('initialized', False):
            st.error("⚠️ Application initialization failed")
            st.info("Please refresh the page or check your configuration.")
            return
        
        # Render interface
        render_sidebar()
        render_chat_interface()
        
        # Show comprehensive welcome message
        if not st.session_state.session_manager.get_session().messages:
            st.info("""
            👋 **Welcome to FiFi AI Chat Assistant v2.4 - Complete Enhanced Edition!**
            
            **✨ v2.4 NEW FEATURE - Inline Citation Links:**
            - ✅ Citations now appear inline after product mentions
            - ✅ Citations appear after supplier/company names  
            - ✅ Better citation context mapping for F&B content
            - ✅ Traditional bottom citations section still included
            - ✅ Smart company/product mention detection
            
            **🔧 Previous Enhancements:**
            - ✅ v2.3: User-friendly SQLite Cloud display
            - ✅ v2.2: Fixed rate limiting with proper consumption tracking
            - ✅ v2.1: Enhanced configuration with validation & fallbacks
            
            **🔒 Enhanced Security Features:**
            - ✅ XSS Protection & Input Sanitization
            - ✅ Rate Limiting (20 requests per minute) - Working Correctly
            - ✅ Suspicious Pattern Detection
            - ✅ Thread-Safe Operations
            
            **🎯 Smart Business Logic:**
            - ✅ Competitor Domain Exclusions (30+ domains filtered)
            - ✅ UTM Tracking on All External Links
            - ✅ Domain-Specific Fallback for Token Limits
            - ✅ Enhanced Token Limit Detection
            
            **🛡️ Anti-Hallucination System:**
            - ✅ Blocks fake citations and non-existent file references
            - ✅ Prevents hallucinated image paths (.jpg, .png, etc.)
            - ✅ Validates all sources before presenting information
            - ✅ Falls back to verified web search when information is questionable
            
            **🔄 Enhanced Fallback Logic:**
            - 🔍 **First**: Searches your internal knowledge base via Pinecone
            - 🛡️ **Safety Override**: Detects and blocks fabricated information
            - 🌐 **Smart Fallback**: Regular web search with competitor exclusions
            - 🎯 **Token Limit**: Domain-specific search when context is full
            
            **✨ NEW: Ask about suppliers, products, or companies to see inline citations in action!**
            
            **📈 Version History:**
            - v2.4: Inline Citation Links
            - v2.3: SQLite Cloud Display Fixes
            - v2.2: Rate Limiter Bug Fixes  
            - v2.1: Enhanced Configuration Management
            - v2.0: Base Enhanced System
            """)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        error_context = error_handler.handle_api_error("Application", "Main Function", e)
        error_handler.display_error_to_user(error_context)
        
        if st.button("🔄 Restart App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
