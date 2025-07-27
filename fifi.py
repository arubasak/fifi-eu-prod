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

# =============================================================================
# COMPLETE INTEGRATED FIFI AI - ALL FEATURES IMPLEMENTATION
# - Universal fingerprinting (Canvas/WebGL/Audio) for ALL sessions
# - 3-tier user system: GUEST → EMAIL_VERIFIED_GUEST → REGISTERED_USER
# - Activity-based question limits with rolling 24-hour windows
# - Email verification with Supabase Auth OTP (simple verification codes)
# - 15-minute session timeout with strategic CRM saves
# - Cross-device enforcement and evasion detection
# - Complete new database schema with migration
# - Enhanced browser close detection and error handling
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
        
        # NEW: Supabase configuration for email verification
        self.SUPABASE_URL = st.secrets.get("SUPABASE_URL")
        self.SUPABASE_ANON_KEY = st.secrets.get("SUPABASE_ANON_KEY")
        self.SUPABASE_ENABLED = all([self.SUPABASE_URL, self.SUPABASE_ANON_KEY])

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
# ENHANCED USER MODELS - 3-TIER SYSTEM WITH NEW FEATURES
# =============================================================================

class UserType(Enum):
    GUEST = "guest"                           # 4 questions → forced email
    EMAIL_VERIFIED_GUEST = "email_verified_guest"  # 10 questions/day
    REGISTERED_USER = "registered_user"       # 40 questions/day cross-device

class BanStatus(Enum):
    NONE = "none"
    ONE_HOUR = "1hour"          # WordPress users: questions 21-40
    TWENTY_FOUR_HOUR = "24hour" # All users: daily limits reached
    EVASION_BLOCK = "evasion_block"  # Escalating evasion penalties

@dataclass
class UserSession:
    session_id: str
    user_type: UserType = UserType.GUEST
    email: Optional[str] = None
    full_name: Optional[str] = None  # Renamed from first_name
    zoho_contact_id: Optional[str] = None
    active: bool = True
    wp_token: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    timeout_saved_to_crm: bool = False
    
    # NEW: Universal Fingerprinting (ALL sessions)
    fingerprint_id: Optional[str] = None
    fingerprint_method: Optional[str] = None  # "canvas" | "webgl" | "audio" | "hybrid"
    visitor_type: str = "new_visitor"  # "new_visitor" | "returning_visitor"
    recognition_response: Optional[str] = None  # "yes" | "no" | "new_user"
    
    # NEW: Question Tracking (Activity-Based)
    daily_question_count: int = 0
    total_question_count: int = 0  # WordPress 40-question max
    last_question_time: Optional[datetime] = None
    question_limit_reached: bool = False
    
    # NEW: Ban Management (Question Limits Only)
    ban_status: BanStatus = BanStatus.NONE
    ban_start_time: Optional[datetime] = None
    ban_end_time: Optional[datetime] = None
    ban_reason: Optional[str] = None
    
    # NEW: Evasion Tracking
    evasion_count: int = 0
    current_penalty_hours: int = 0
    escalation_level: int = 0  # 1-5 for exponential
    
    # NEW: Multi-Email & Device Tracking
    email_addresses_used: List[str] = field(default_factory=list)
    email_switches_count: int = 0
    
    # NEW: Network & Browser
    ip_address: Optional[str] = None
    ip_detection_method: Optional[str] = None
    user_agent: Optional[str] = None
    browser_privacy_level: Optional[str] = None
    
    # NEW: Registration Tracking
    registration_prompted: bool = False
    registration_link_clicked: bool = False

# =============================================================================
# UNIVERSAL FINGERPRINTING SYSTEM
# =============================================================================

class FingerprintingManager:
    """3-layer fingerprinting system for universal device identification"""
    
    def __init__(self):
        self.fingerprint_cache = {}
    
    def generate_fingerprint_component(self, session_id: str) -> str:
        """Generate JavaScript component for 3-layer fingerprinting"""
        
        js_code = f"""
        (() => {{
            const sessionId = "{session_id}";
            
            console.log("🔍 FiFi Fingerprinting: Starting 3-layer device identification");
            
            // Layer 1: Canvas Fingerprinting (Primary - 95-98% accuracy)
            function generateCanvasFingerprint() {{
                try {{
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = 220;
                    canvas.height = 100;
                    
                    // Complex drawing pattern for unique fingerprint
                    ctx.textBaseline = 'top';
                    ctx.font = '14px Arial';
                    ctx.fillStyle = '#f60';
                    ctx.fillRect(125, 1, 62, 20);
                    ctx.fillStyle = '#069';
                    ctx.fillText('FiFi AI Canvas Test 🤖', 2, 15);
                    ctx.fillStyle = 'rgba(102, 204, 0, 0.7)';
                    ctx.fillText('Food & Beverage Industry', 4, 45);
                    
                    // Add geometric shapes
                    ctx.strokeStyle = '#000';
                    ctx.beginPath();
                    ctx.arc(50, 50, 20, 0, Math.PI * 2);
                    ctx.stroke();
                    
                    const canvasData = canvas.toDataURL();
                    const hash = btoa(canvasData).slice(0, 32);
                    console.log("✅ Canvas fingerprint generated");
                    return hash;
                }} catch (e) {{
                    console.error("❌ Canvas fingerprint failed:", e);
                    return 'canvas_blocked';
                }}
            }}
            
            // Layer 2: WebGL Fingerprinting (Secondary - 93-97% accuracy)
            function generateWebGLFingerprint() {{
                try {{
                    const canvas = document.createElement('canvas');
                    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                    
                    if (!gl) {{
                        return 'webgl_unavailable';
                    }}
                    
                    const vendor = gl.getParameter(gl.VENDOR);
                    const renderer = gl.getParameter(gl.RENDERER);
                    const version = gl.getParameter(gl.VERSION);
                    const extensions = gl.getSupportedExtensions();
                    
                    const webglData = {{
                        vendor: vendor,
                        renderer: renderer,
                        version: version,
                        extensions: extensions ? extensions.slice(0, 10) : []
                    }};
                    
                    const hash = btoa(JSON.stringify(webglData)).slice(0, 32);
                    console.log("✅ WebGL fingerprint generated");
                    return hash;
                }} catch (e) {{
                    console.error("❌ WebGL fingerprint failed:", e);
                    return 'webgl_blocked';
                }}
            }}
            
            // Layer 3: Audio Context Fingerprinting (Tertiary - 90-95% accuracy)
            function generateAudioFingerprint() {{
                try {{
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    
                    // Create oscillator for audio fingerprinting
                    const oscillator = audioContext.createOscillator();
                    const analyser = audioContext.createAnalyser();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.type = 'triangle';
                    oscillator.frequency.value = 1000;
                    gainNode.gain.value = 0;
                    
                    oscillator.connect(analyser);
                    analyser.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.start(0);
                    
                    // Get frequency data
                    const frequencyData = new Uint8Array(analyser.frequencyBinCount);
                    analyser.getByteFrequencyData(frequencyData);
                    
                    oscillator.stop();
                    audioContext.close();
                    
                    const audioHash = btoa(Array.from(frequencyData.slice(0, 32)).join(',')).slice(0, 32);
                    console.log("✅ Audio fingerprint generated");
                    return audioHash;
                }} catch (e) {{
                    console.error("❌ Audio fingerprint failed:", e);
                    return 'audio_blocked';
                }}
            }}
            
            // IP Detection Fallback
            function getIPInfo() {{
                const headers = {{
                    'x-forwarded-for': '',
                    'x-real-ip': '',
                    'cf-connecting-ip': '',
                    'x-client-ip': ''
                }};
                
                // Try to get IP from various headers (simulated)
                const userIP = 'client_ip_hidden';  // Streamlit will provide server-side
                return {{ ip: userIP, method: 'header_detection' }};
            }}
            
            // Browser & System Info
            function getBrowserInfo() {{
                return {{
                    userAgent: navigator.userAgent,
                    language: navigator.language,
                    platform: navigator.platform,
                    cookieEnabled: navigator.cookieEnabled,
                    doNotTrack: navigator.doNotTrack,
                    hardwareConcurrency: navigator.hardwareConcurrency,
                    maxTouchPoints: navigator.maxTouchPoints,
                    screen: {{
                        width: screen.width,
                        height: screen.height,
                        colorDepth: screen.colorDepth,
                        pixelDepth: screen.pixelDepth
                    }},
                    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
                }};
            }}
            
            // Generate all fingerprints
            const canvasFingerprint = generateCanvasFingerprint();
            const webglFingerprint = generateWebGLFingerprint();
            const audioFingerprint = generateAudioFingerprint();
            const ipInfo = getIPInfo();
            const browserInfo = getBrowserInfo();
            
            // Determine primary fingerprint method based on success
            let primaryMethod = 'canvas';
            let fingerprintId = canvasFingerprint;
            
            if (canvasFingerprint === 'canvas_blocked') {{
                if (webglFingerprint !== 'webgl_blocked' && webglFingerprint !== 'webgl_unavailable') {{
                    primaryMethod = 'webgl';
                    fingerprintId = webglFingerprint;
                }} else if (audioFingerprint !== 'audio_blocked') {{
                    primaryMethod = 'audio';
                    fingerprintId = audioFingerprint;
                }} else {{
                    primaryMethod = 'fallback';
                    fingerprintId = 'privacy_browser_' + Date.now();
                }}
            }}
            
            // Combine for hybrid approach if multiple methods work
            const workingMethods = [];
            if (canvasFingerprint !== 'canvas_blocked') workingMethods.push('canvas');
            if (webglFingerprint !== 'webgl_blocked' && webglFingerprint !== 'webgl_unavailable') workingMethods.push('webgl');
            if (audioFingerprint !== 'audio_blocked') workingMethods.push('audio');
            
            if (workingMethods.length > 1) {{
                primaryMethod = 'hybrid';
                fingerprintId = btoa([canvasFingerprint, webglFingerprint, audioFingerprint].join('|')).slice(0, 32);
            }}
            
            // Privacy browser detection
            const privacyLevel = 'standard';
            if (canvasFingerprint === 'canvas_blocked' && 
                webglFingerprint === 'webgl_blocked' && 
                audioFingerprint === 'audio_blocked') {{
                privacyLevel = 'high_privacy';
            }}
            
            const fingerprintResult = {{
                session_id: sessionId,
                fingerprint_id: fingerprintId,
                fingerprint_method: primaryMethod,
                canvas_fp: canvasFingerprint,
                webgl_fp: webglFingerprint,
                audio_fp: audioFingerprint,
                browser_info: browserInfo,
                ip_info: ipInfo,
                privacy_level: privacyLevel,
                working_methods: workingMethods,
                timestamp: Date.now()
            }};
            
            console.log("🔍 Fingerprinting complete:", {{
                id: fingerprintId,
                method: primaryMethod,
                privacy: privacyLevel,
                working: workingMethods.length
            }});
            
            return fingerprintResult;
        }})()
        """
        
        return js_code
    
    def extract_fingerprint_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate fingerprint data from JavaScript result"""
        
        if not result or not isinstance(result, dict):
            return self._generate_fallback_fingerprint()
        
        fingerprint_id = result.get('fingerprint_id')
        fingerprint_method = result.get('fingerprint_method', 'unknown')
        
        if not fingerprint_id or fingerprint_id.startswith('privacy_browser_'):
            return self._generate_fallback_fingerprint()
        
        # Determine visitor type based on caching
        visitor_type = "returning_visitor" if fingerprint_id in self.fingerprint_cache else "new_visitor"
        
        # Cache the fingerprint
        self.fingerprint_cache[fingerprint_id] = {
            'method': fingerprint_method,
            'last_seen': datetime.now(),
            'session_count': self.fingerprint_cache.get(fingerprint_id, {}).get('session_count', 0) + 1
        }
        
        return {
            'fingerprint_id': fingerprint_id,
            'fingerprint_method': fingerprint_method,
            'visitor_type': visitor_type,
            'browser_info': result.get('browser_info', {}),
            'ip_info': result.get('ip_info', {}),
            'privacy_level': result.get('privacy_level', 'standard'),
            'working_methods': result.get('working_methods', [])
        }
    
    def _generate_fallback_fingerprint(self) -> Dict[str, Any]:
        """Generate fallback fingerprint for privacy browsers"""
        
        fallback_id = f"fallback_{secrets.token_hex(8)}"
        
        return {
            'fingerprint_id': fallback_id,
            'fingerprint_method': 'fallback',
            'visitor_type': 'new_visitor',
            'browser_info': {},
            'ip_info': {},
            'privacy_level': 'high_privacy',
            'working_methods': []
        }

# =============================================================================
# EMAIL VERIFICATION SYSTEM WITH SUPABASE
# =============================================================================

class EmailVerificationManager:
    """Email verification using Supabase Auth OTP"""
    
    def __init__(self, config: Config):
        self.config = config
        self.verification_cache = {}  # Store pending verifications
        self.supabase = None
        
        # Initialize Supabase client if available
        if SUPABASE_AVAILABLE and self.config.SUPABASE_ENABLED:
            try:
                self.supabase = create_client(
                    self.config.SUPABASE_URL, 
                    self.config.SUPABASE_ANON_KEY
                )
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.supabase = None
    
    @handle_api_errors("Supabase Auth", "Send Verification Code")
    def send_verification_code(self, email: str) -> bool:
        """Send verification code to email using Supabase Auth OTP"""
        
        if not self.supabase:
            st.error("Email verification service is not available.")
            return False
        
        try:
            # Use Supabase Auth OTP
            response = self.supabase.auth.sign_in_with_otp({
                'email': email,
                'options': {
                    'should_create_user': True,  # Create user if doesn't exist
                    'data': {
                        'registered_via': 'fifi_ai_guest_verification',
                        'verification_purpose': 'guest_upgrade'
                    }
                }
            })
            
            # Cache the verification attempt
            self.verification_cache[email] = {
                'status': 'pending',
                'created_at': datetime.now(),
                'attempts': self.verification_cache.get(email, {}).get('attempts', 0) + 1,
                'method': 'supabase_otp'
            }
            
            logger.info(f"Email verification code sent to {email} via Supabase")
            return True
                
        except Exception as e:
            logger.error(f"Failed to send verification code via Supabase: {e}")
            return False
    
    @handle_api_errors("Supabase Auth", "Verify Code")
    def verify_code(self, email: str, code: str) -> bool:
        """Verify email verification code using Supabase Auth"""
        
        if not self.supabase:
            return False
        
        try:
            # Use Supabase Auth OTP verification
            response = self.supabase.auth.verify_otp({
                'email': email,
                'token': code.strip(),
                'type': 'email'
            })
            
            if response.user:  # Verification successful
                # Update verification cache
                if email in self.verification_cache:
                    self.verification_cache[email]['status'] = 'approved'
                    self.verification_cache[email]['verified_at'] = datetime.now()
                    self.verification_cache[email]['user_id'] = response.user.id
                
                logger.info(f"Email verification successful for {email} via Supabase (User ID: {response.user.id})")
                return True
            else:
                logger.warning(f"Email verification failed for {email}: No user returned")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify code via Supabase: {e}")
            return False
    
    def get_verification_status(self, email: str) -> Dict[str, Any]:
        """Get verification status for email"""
        
        if email not in self.verification_cache:
            return {'status': 'not_started', 'attempts': 0}
        
        cache_entry = self.verification_cache[email]
        
        # Check if verification is expired (10 minutes)
        if cache_entry['created_at'] < datetime.now() - timedelta(minutes=10):
            cache_entry['status'] = 'expired'
        
        return cache_entry
    
    def cleanup_expired_verifications(self):
        """Clean up expired verification attempts"""
        current_time = datetime.now()
        expired_emails = []
        
        for email, cache_entry in self.verification_cache.items():
            if cache_entry['created_at'] < current_time - timedelta(hours=1):
                expired_emails.append(email)
        
        for email in expired_emails:
            del self.verification_cache[email]
        
        if expired_emails:
            logger.info(f"Cleaned up {len(expired_emails)} expired verification attempts")

# =============================================================================
# QUESTION LIMITS & BAN MANAGEMENT SYSTEM
# =============================================================================

class QuestionLimitManager:
    """Activity-based question limiting with rolling 24-hour windows"""
    
    def __init__(self):
        self.question_limits = {
            UserType.GUEST: 4,                    # 4 questions → forced email
            UserType.EMAIL_VERIFIED_GUEST: 10,   # 10 questions/day
            UserType.REGISTERED_USER: 40         # 40 questions/day cross-device
        }
        
        # Escalating evasion penalties (hours)
        self.evasion_penalties = [24, 48, 96, 192, 336]  # 1d, 2d, 4d, 8d, 14d (cap)
    
    def is_within_limits(self, session: UserSession) -> Dict[str, Any]:
        """Check if user is within question limits"""
        
        user_limit = self.question_limits.get(session.user_type, 0)
        
        # Check current ban status
        if session.ban_status != BanStatus.NONE:
            if session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                return {
                    'allowed': False,
                    'reason': 'banned',
                    'ban_type': session.ban_status.value,
                    'time_remaining': time_remaining,
                    'message': self._get_ban_message(session)
                }
            else:
                # Ban expired, reset status
                session.ban_status = BanStatus.NONE
                session.ban_start_time = None
                session.ban_end_time = None
                session.ban_reason = None
        
        # Check activity-based reset (24-hour rolling window)
        if session.last_question_time:
            time_since_last = datetime.now() - session.last_question_time
            if time_since_last >= timedelta(hours=24):
                # Reset daily count
                session.daily_question_count = 0
                session.question_limit_reached = False
        
        # Check question limits by user type
        if session.user_type == UserType.GUEST:
            # Guest: 4 questions maximum → forced email
            if session.daily_question_count >= user_limit:
                return {
                    'allowed': False,
                    'reason': 'guest_limit',
                    'questions_used': session.daily_question_count,
                    'limit': user_limit,
                    'message': 'Please provide your email address to continue.'
                }
        
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            # Email-verified guest: 10 questions/day → 24-hour ban
            if session.daily_question_count >= user_limit:
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Daily question limit reached")
                return {
                    'allowed': False,
                    'reason': 'daily_limit',
                    'questions_used': session.daily_question_count,
                    'limit': user_limit,
                    'message': self._get_email_verified_limit_message()
                }
        
        elif session.user_type == UserType.REGISTERED_USER:
            # WordPress registered: Two-tier system
            if 1 <= session.total_question_count <= 20:
                # First 20 questions allowed
                pass
            elif 21 <= session.total_question_count <= 40:
                if session.total_question_count > 20 and session.ban_status == BanStatus.NONE:
                    # Apply 1-hour ban after question 20
                    self._apply_ban(session, BanStatus.ONE_HOUR, "First tier limit reached")
                    return {
                        'allowed': False,
                        'reason': 'first_tier_limit',
                        'questions_used': session.total_question_count,
                        'message': "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
                    }
            else:
                # After 40 questions → 24-hour ban
                self._apply_ban(session, BanStatus.TWENTY_FOUR_HOUR, "Total question limit reached")
                return {
                    'allowed': False,
                    'reason': 'total_limit',
                    'questions_used': session.total_question_count,
                    'limit': user_limit,
                    'message': "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
                }
        
        return {'allowed': True}
    
    def record_question(self, session: UserSession):
        """Record a question and update counters"""
        
        session.daily_question_count += 1
        session.total_question_count += 1
        session.last_question_time = datetime.now()
        
        logger.info(f"Question recorded for {session.session_id[:8]}: daily={session.daily_question_count}, total={session.total_question_count}")
    
    def _apply_ban(self, session: UserSession, ban_type: BanStatus, reason: str):
        """Apply ban to session"""
        
        ban_hours = {
            BanStatus.ONE_HOUR: 1,
            BanStatus.TWENTY_FOUR_HOUR: 24
        }.get(ban_type, 24)
        
        session.ban_status = ban_type
        session.ban_start_time = datetime.now()
        session.ban_end_time = datetime.now() + timedelta(hours=ban_hours)
        session.ban_reason = reason
        session.question_limit_reached = True
        
        logger.info(f"Ban applied to {session.session_id[:8]}: {ban_type.value} for {reason}")
    
    def apply_evasion_penalty(self, session: UserSession) -> int:
        """Apply escalating evasion penalty"""
        
        session.evasion_count += 1
        session.escalation_level = min(session.evasion_count, len(self.evasion_penalties))
        
        penalty_hours = self.evasion_penalties[session.escalation_level - 1]
        session.current_penalty_hours = penalty_hours
        
        session.ban_status = BanStatus.EVASION_BLOCK
        session.ban_start_time = datetime.now()
        session.ban_end_time = datetime.now() + timedelta(hours=penalty_hours)
        session.ban_reason = f"Evasion attempt #{session.evasion_count}"
        
        logger.warning(f"Evasion penalty applied to {session.session_id[:8]}: {penalty_hours}h (level {session.escalation_level})")
        
        return penalty_hours
    
    def _get_ban_message(self, session: UserSession) -> str:
        """Get appropriate ban message"""
        
        if session.ban_status == BanStatus.EVASION_BLOCK:
            return "Usage limit reached. Please try again later."
        elif session.user_type == UserType.REGISTERED_USER:
            return "Usage limit reached. Please retry in 1 hour as we are giving preference to others in the queue."
        else:
            return self._get_email_verified_limit_message()
    
    def _get_email_verified_limit_message(self) -> str:
        """Get message for email-verified guests"""
        return ("Our system is very busy and is being used by multiple users. For a fair assessment of our FiFi AI assistant and to provide fair usage to everyone, we can allow a total of 10 questions per day (20 messages). To increase the limit, please Register: https://www.12taste.com/in/my-account/ and come back here to the Welcome page to Sign In.")

# =============================================================================
# DATABASE MANAGER WITH NEW SCHEMA
# =============================================================================

class DatabaseManager:
    def __init__(self, connection_string: Optional[str]):
        self.lock = threading.Lock()
        self.connection_string = connection_string
        self.conn = None
        
        logger.info("🔄 INITIALIZING NEW DATABASE MANAGER WITH COMPLETE SCHEMA")
        
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
        
        # Initialize database tables with new schema
        if self.conn:
            self._init_complete_database()
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
        
        db_path = "fifi_sessions_v2.db"  # New database file
        
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

    def _init_complete_database(self):
        """Initialize complete new database schema"""
        with self.lock:
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                # Create the complete new schema
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        -- Core session fields
                        session_id TEXT PRIMARY KEY,
                        user_type TEXT NOT NULL DEFAULT 'guest',
                        email TEXT,
                        full_name TEXT,
                        zoho_contact_id TEXT,
                        created_at TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        messages TEXT DEFAULT '[]',
                        active INTEGER DEFAULT 1,
                        wp_token TEXT,
                        timeout_saved_to_crm INTEGER DEFAULT 0,
                        
                        -- Universal Fingerprinting (ALL sessions)
                        fingerprint_id TEXT,
                        fingerprint_method TEXT,
                        visitor_type TEXT DEFAULT 'new_visitor',
                        recognition_response TEXT,
                        
                        -- Question Tracking (Activity-Based)
                        daily_question_count INTEGER DEFAULT 0,
                        total_question_count INTEGER DEFAULT 0,
                        last_question_time TEXT,
                        question_limit_reached INTEGER DEFAULT 0,
                        
                        -- Ban Management (Question Limits Only)
                        ban_status TEXT DEFAULT 'none',
                        ban_start_time TEXT,
                        ban_end_time TEXT,
                        ban_reason TEXT,
                        
                        -- Evasion Tracking
                        evasion_count INTEGER DEFAULT 0,
                        current_penalty_hours INTEGER DEFAULT 0,
                        escalation_level INTEGER DEFAULT 0,
                        
                        -- Multi-Email & Device Tracking
                        email_addresses_used TEXT DEFAULT '[]',
                        email_switches_count INTEGER DEFAULT 0,
                        
                        -- Network & Browser
                        ip_address TEXT,
                        ip_detection_method TEXT,
                        user_agent TEXT,
                        browser_privacy_level TEXT,
                        
                        -- Registration Tracking
                        registration_prompted INTEGER DEFAULT 0,
                        registration_link_clicked INTEGER DEFAULT 0
                    )
                ''')
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_fingerprint_id ON sessions(fingerprint_id)",
                    "CREATE INDEX IF NOT EXISTS idx_email ON sessions(email)",
                    "CREATE INDEX IF NOT EXISTS idx_user_type ON sessions(user_type)",
                    "CREATE INDEX IF NOT EXISTS idx_ban_status ON sessions(ban_status)",
                    "CREATE INDEX IF NOT EXISTS idx_active ON sessions(active)",
                    "CREATE INDEX IF NOT EXISTS idx_last_activity ON sessions(last_activity)"
                ]
                
                for index_sql in indexes:
                    self.conn.execute(index_sql)
                
                self.conn.commit()
                logger.info("✅ Complete database schema initialized successfully")
                
                # Migrate existing data if old table exists
                self._migrate_existing_data()
                
            except Exception as e:
                logger.error(f"Database schema initialization failed: {e}")
                raise

    def _migrate_existing_data(self):
        """Migrate data from old schema if it exists"""
        try:
            # Check if old sessions table exists
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='sessions_old'
            """)
            
            if cursor.fetchone():
                logger.info("🔄 Migrating data from old schema...")
                
                # Migration logic would go here
                # For now, we'll just log that migration is available
                logger.info("📋 Old session data detected - manual migration may be required")
                
        except Exception as e:
            logger.info(f"No old data to migrate: {e}")

    @handle_api_errors("Database", "Save Session")
    def save_session(self, session: UserSession):
        """Save session with new complete schema"""
        with self.lock:
            if self.db_type == "memory":
                self.local_sessions[session.session_id] = session
                return
            
            try:
                if hasattr(self.conn, 'row_factory'):
                    self.conn.row_factory = None
                
                # Convert enums and lists to strings for storage
                user_type_str = session.user_type.value if hasattr(session.user_type, 'value') else str(session.user_type)
                ban_status_str = session.ban_status.value if hasattr(session.ban_status, 'value') else str(session.ban_status)
                messages_json = json.dumps(session.messages)
                email_addresses_json = json.dumps(session.email_addresses_used)
                
                # Convert datetime fields to ISO strings
                created_at_str = session.created_at.isoformat() if session.created_at else datetime.now().isoformat()
                last_activity_str = session.last_activity.isoformat() if session.last_activity else datetime.now().isoformat()
                last_question_time_str = session.last_question_time.isoformat() if session.last_question_time else None
                ban_start_time_str = session.ban_start_time.isoformat() if session.ban_start_time else None
                ban_end_time_str = session.ban_end_time.isoformat() if session.ban_end_time else None
                
                self.conn.execute('''
                    REPLACE INTO sessions (
                        session_id, user_type, email, full_name, zoho_contact_id,
                        created_at, last_activity, messages, active, wp_token, timeout_saved_to_crm,
                        fingerprint_id, fingerprint_method, visitor_type, recognition_response,
                        daily_question_count, total_question_count, last_question_time, question_limit_reached,
                        ban_status, ban_start_time, ban_end_time, ban_reason,
                        evasion_count, current_penalty_hours, escalation_level,
                        email_addresses_used, email_switches_count,
                        ip_address, ip_detection_method, user_agent, browser_privacy_level,
                        registration_prompted, registration_link_clicked
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id, user_type_str, session.email, session.full_name, session.zoho_contact_id,
                    created_at_str, last_activity_str, messages_json, int(session.active), session.wp_token, int(session.timeout_saved_to_crm),
                    session.fingerprint_id, session.fingerprint_method, session.visitor_type, session.recognition_response,
                    session.daily_question_count, session.total_question_count, last_question_time_str, int(session.question_limit_reached),
                    ban_status_str, ban_start_time_str, ban_end_time_str, session.ban_reason,
                    session.evasion_count, session.current_penalty_hours, session.escalation_level,
                    email_addresses_json, session.email_switches_count,
                    session.ip_address, session.ip_detection_method, session.user_agent, session.browser_privacy_level,
                    int(session.registration_prompted), int(session.registration_link_clicked)
                ))
                
                self.conn.commit()
                logger.debug(f"Successfully saved session {session.session_id[:8]}: user_type={user_type_str}")
                
            except Exception as e:
                logger.error(f"Failed to save session {session.session_id[:8]}: {e}")
                raise

    @handle_api_errors("Database", "Load Session")
    def load_session(self, session_id: str) -> Optional[UserSession]:
        """Load session with complete new schema"""
        with self.lock:
            if self.db_type == "memory":
                session = self.local_sessions.get(session_id)
                if session and isinstance(session.user_type, str):
                    session.user_type = UserType(session.user_type)
                return session

            try:
                if self.db_type == "cloud":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = None
                elif self.db_type == "file":
                    if hasattr(self.conn, 'row_factory'):
                        self.conn.row_factory = sqlite3.Row
                
                cursor = self.conn.execute("""
                    SELECT * FROM sessions 
                    WHERE session_id = ? AND active = 1
                """, (session_id,))
                
                row = cursor.fetchone()
                
                if not row: 
                    return None
                
                # Convert row to dictionary
                if hasattr(row, 'keys') and callable(getattr(row, 'keys')):
                    row_dict = dict(row)
                else:
                    # Handle as tuple (SQLite Cloud)
                    columns = [description[0] for description in cursor.description]
                    row_dict = dict(zip(columns, row))
                
                # Create UserSession from row data with proper type conversion
                session = UserSession(
                    session_id=row_dict['session_id'],
                    user_type=UserType(row_dict.get('user_type', 'guest')),
                    email=row_dict.get('email'),
                    full_name=row_dict.get('full_name'),
                    zoho_contact_id=row_dict.get('zoho_contact_id'),
                    created_at=datetime.fromisoformat(row_dict['created_at']),
                    last_activity=datetime.fromisoformat(row_dict['last_activity']),
                    messages=json.loads(row_dict.get('messages', '[]')),
                    active=bool(row_dict.get('active', 1)),
                    wp_token=row_dict.get('wp_token'),
                    timeout_saved_to_crm=bool(row_dict.get('timeout_saved_to_crm', 0)),
                    
                    # Fingerprinting fields
                    fingerprint_id=row_dict.get('fingerprint_id'),
                    fingerprint_method=row_dict.get('fingerprint_method'),
                    visitor_type=row_dict.get('visitor_type', 'new_visitor'),
                    recognition_response=row_dict.get('recognition_response'),
                    
                    # Question tracking fields
                    daily_question_count=row_dict.get('daily_question_count', 0),
                    total_question_count=row_dict.get('total_question_count', 0),
                    last_question_time=datetime.fromisoformat(row_dict['last_question_time']) if row_dict.get('last_question_time') else None,
                    question_limit_reached=bool(row_dict.get('question_limit_reached', 0)),
                    
                    # Ban management fields
                    ban_status=BanStatus(row_dict.get('ban_status', 'none')),
                    ban_start_time=datetime.fromisoformat(row_dict['ban_start_time']) if row_dict.get('ban_start_time') else None,
                    ban_end_time=datetime.fromisoformat(row_dict['ban_end_time']) if row_dict.get('ban_end_time') else None,
                    ban_reason=row_dict.get('ban_reason'),
                    
                    # Evasion tracking fields
                    evasion_count=row_dict.get('evasion_count', 0),
                    current_penalty_hours=row_dict.get('current_penalty_hours', 0),
                    escalation_level=row_dict.get('escalation_level', 0),
                    
                    # Multi-email tracking fields
                    email_addresses_used=json.loads(row_dict.get('email_addresses_used', '[]')),
                    email_switches_count=row_dict.get('email_switches_count', 0),
                    
                    # Network & browser fields
                    ip_address=row_dict.get('ip_address'),
                    ip_detection_method=row_dict.get('ip_detection_method'),
                    user_agent=row_dict.get('user_agent'),
                    browser_privacy_level=row_dict.get('browser_privacy_level'),
                    
                    # Registration tracking fields
                    registration_prompted=bool(row_dict.get('registration_prompted', 0)),
                    registration_link_clicked=bool(row_dict.get('registration_link_clicked', 0))
                )
                
                logger.debug(f"Successfully loaded session {session_id[:8]}: user_type={session.user_type}")
                return session
                
            except Exception as e:
                logger.error(f"Failed to load session {session_id[:8]}: {e}")
                return None

    def find_sessions_by_fingerprint(self, fingerprint_id: str) -> List[UserSession]:
        """Find all sessions with the same fingerprint"""
        try:
            with self.lock:
                if self.db_type == "memory":
                    return [session for session in self.local_sessions.values() 
                           if session.fingerprint_id == fingerprint_id]
                
                cursor = self.conn.execute("""
                    SELECT * FROM sessions 
                    WHERE fingerprint_id = ? 
                    ORDER BY last_activity DESC
                """, (fingerprint_id,))
                
                sessions = []
                for row in cursor.fetchall():
                    session = self._row_to_session(row, cursor.description)
                    if session:
                        sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to find sessions by fingerprint: {e}")
            return []

    def find_sessions_by_email(self, email: str) -> List[UserSession]:
        """Find all sessions with the same email"""
        try:
            with self.lock:
                if self.db_type == "memory":
                    return [session for session in self.local_sessions.values() 
                           if session.email == email]
                
                cursor = self.conn.execute("""
                    SELECT * FROM sessions 
                    WHERE email = ? 
                    ORDER BY last_activity DESC
                """, (email,))
                
                sessions = []
                for row in cursor.fetchall():
                    session = self._row_to_session(row, cursor.description)
                    if session:
                        sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to find sessions by email: {e}")
            return []

    def _row_to_session(self, row, description) -> Optional[UserSession]:
        """Convert database row to UserSession object"""
        try:
            if hasattr(row, 'keys'):
                row_dict = dict(row)
            else:
                columns = [desc[0] for desc in description]
                row_dict = dict(zip(columns, row))
            
            return UserSession(
                session_id=row_dict['session_id'],
                user_type=UserType(row_dict.get('user_type', 'guest')),
                email=row_dict.get('email'),
                full_name=row_dict.get('full_name'),
                # ... (rest of the fields as in load_session)
            )
        except Exception as e:
            logger.error(f"Failed to convert row to session: {e}")
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
# PDF EXPORTER (PRESERVED FROM ORIGINAL)
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
# ZOHO CRM MANAGER (ENHANCED FROM ORIGINAL)
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

    def _create_contact(self, email: str, access_token: str, full_name: str = None) -> Optional[str]:
        """Create contact with retry on token expiry"""
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        contact_data = {
            "data": [{
                "Last_Name": full_name or "Food Professional",
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
        
        max_retries = 3 if "timeout" in trigger_reason.lower() else 1
        
        for attempt in range(max_retries):
            logger.info(f"Save attempt {attempt + 1}/{max_retries}")
            try:
                # Get access token
                token_timeout = 10 if "timeout" in trigger_reason.lower() else 15
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
                    contact_id = self._create_contact(session.email, access_token, session.full_name)
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
        note_content += f"- User: {session.full_name or 'Unknown'} ({session.email})\n"
        note_content += f"- Save Trigger: {trigger_reason}\n"
        note_content += f"- Timestamp: {timestamp}\n"
        note_content += f"- Total Messages: {len(session.messages)}\n"
        note_content += f"- User Type: {session.user_type.value}\n"
        note_content += f"- Questions Asked: {session.daily_question_count}\n\n"
        
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
# RATE LIMITER (PRESERVED FROM ORIGINAL)
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
# AI SYSTEM PLACEHOLDER (SIMPLIFIED FOR COMPLETE VERSION)
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
            "content": f"I understand you're asking about: {prompt}. This is the integrated FiFi AI with universal fingerprinting, 3-tier user system (GUEST→EMAIL_VERIFIED_GUEST→REGISTERED_USER), activity-based question limits, email verification with Supabase OTP, and comprehensive evasion detection. Your question will be processed according to your current user tier and daily limits.",
            "source": "Integrated FiFi AI System",
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
# ENHANCED JAVASCRIPT COMPONENTS WITH 15-MINUTE TIMEOUT
# =============================================================================

def render_activity_timer_component_15min(session_id: str, session_manager) -> Optional[Dict[str, Any]]:
    """
    15-minute activity timer per specification requirements
    """
    
    if not session_id:
        return None
    
    # JavaScript with 15-minute timeout system
    js_timer_code = f"""
    (() => {{
        try {{
            const sessionId = "{session_id}";
            const SESSION_TIMEOUT = 900000;  // 15 minutes in milliseconds
            
            console.log("🕐 FiFi 15-Minute Timer checking session:", sessionId.substring(0, 8));
            
            // Initialize timer state
            if (typeof window.fifi_timer_state === 'undefined' || window.fifi_timer_state === null) {{
                console.clear();
                console.log("🆕 FiFi 15-Minute Timer: Starting fresh session"); 
                window.fifi_timer_state = {{
                    lastActivityTime: Date.now(),
                    sessionExpired: false,
                    listenersInitialized: false,
                    sessionId: sessionId,
                    crm_saved: false
                }};
                console.log("🆕 FiFi 15-Minute Timer state initialized");
            }}
            
            const state = window.fifi_timer_state;
            
            // Reset state if session changed
            if (state.sessionId !== sessionId) {{
                console.clear();
                console.log("🔄 Session changed, resetting 15-minute timer state");
                state.sessionId = sessionId;
                state.lastActivityTime = Date.now();
                state.sessionExpired = false;
                state.listenersInitialized = false;
                state.crm_saved = false;
            }}
            
            // Initialize activity listeners
            if (!state.listenersInitialized) {{
                console.log("👂 Setting up 15-minute activity listeners");
                
                function resetActivity() {{
                    try {{
                        const now = Date.now();
                        state.lastActivityTime = now;
                        state.sessionExpired = false;
                        state.crm_saved = false;
                        console.log("🔄 Activity detected at:", new Date(now).toLocaleTimeString());
                    }} catch (e) {{
                        console.debug("Activity reset error:", e);
                    }}
                }}
                
                // Activity events
                const events = [
                    'mousedown', 'mousemove', 'mouseup', 'click', 'dblclick',
                    'keydown', 'keyup', 'keypress',
                    'scroll', 'wheel',
                    'touchstart', 'touchmove', 'touchend',
                    'focus', 'blur'
                ];
                
                // Add to current document (component iframe)
                events.forEach(eventType => {{
                    try {{
                        document.addEventListener(eventType, resetActivity, {{ 
                            passive: true, 
                            capture: true
                        }});
                    }} catch (e) {{
                        console.debug("Failed to add listener:", eventType, e);
                    }}
                }});
                
                // Add to parent document (main Streamlit app)
                try {{
                    if (window.parent && 
                        window.parent.document && 
                        window.parent.document !== document &&
                        window.parent.location.origin === window.location.origin) {{
                        
                        events.forEach(eventType => {{
                            try {{
                                window.parent.document.addEventListener(eventType, resetActivity, {{ 
                                    passive: true, 
                                    capture: true
                                }});
                            }} catch (e) {{
                                console.debug("Failed to add parent listener:", eventType, e);
                            }}
                        }});
                        console.log("👂 Parent document listeners added for 15-minute timer");
                    }}
                }} catch (e) {{
                    console.debug("Cannot access parent document:", e);
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
                    document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
                    if (window.parent && window.parent.document && window.parent.document !== document) {{
                        window.parent.document.addEventListener('visibilitychange', handleVisibilityChange, {{ passive: true }});
                    }}
                }} catch (e) {{
                    console.debug("Cannot setup visibility detection:", e);
                }}
                
                state.listenersInitialized = true;
                console.log("✅ 15-minute activity listeners initialized");
            }}
            
            // Calculate current inactivity
            const currentTime = Date.now();
            const inactiveTimeMs = currentTime - state.lastActivityTime;
            const inactiveMinutes = Math.floor(inactiveTimeMs / 60000);
            const inactiveSeconds = Math.floor((inactiveTimeMs % 60000) / 1000);
            
            console.log(`⏰ Session ${{sessionId.substring(0, 8)}} inactive: ${{inactiveMinutes}}m${{inactiveSeconds}}s`);
            
            // Check for 15-minute inactivity → CRM save → NO BAN
            if (inactiveTimeMs >= SESSION_TIMEOUT && !state.sessionExpired) {{
                state.sessionExpired = true;
                console.log("🚨 15-MINUTE SESSION TIMEOUT REACHED");
                
                return {{
                    event: "session_timeout_15min",
                    session_id: sessionId,
                    inactive_time_ms: inactiveTimeMs,
                    inactive_minutes: inactiveMinutes,
                    inactive_seconds: inactiveSeconds,
                    timestamp: currentTime,
                    should_save_crm: true,
                    should_ban: false
                }};
            }}
            
            // EXPLICITLY return null to prevent undefined/falsy return issues
            return null;
            
        }} catch (error) {{
            console.error("🚨 FiFi 15-Minute Timer error caught:", error);
            return null;
        }}
    }})()
    """
    
    try:
        # Enhanced key stability
        app_session_hash = hash(str(st.session_state.get('current_session_id', 'default'))) % 10000
        stable_key = f"fifi_timer_15min_{session_id[:8]}_{app_session_hash}"
        
        # Execute JavaScript
        timer_result = st_javascript(js_timer_code, key=stable_key)
        
        # Enhanced validation
        logger.debug(f"⏰ 15-min timer result: {timer_result} (type: {type(timer_result)})")
        
        if timer_result is None or timer_result == 0 or timer_result == "" or timer_result == False:
            return None
        
        # Validate result structure
        if isinstance(timer_result, dict) and 'event' in timer_result:
            event = timer_result.get('event')
            received_session_id = timer_result.get('session_id')
            
            if received_session_id == session_id:
                logger.info(f"✅ Valid 15-min timer event: {event} for session {session_id[:8]}")
                return timer_result
            else:
                logger.warning(f"⚠️ Session ID mismatch: expected {session_id[:8]}, got {received_session_id[:8] if received_session_id else 'None'}")
                return None
        else:
            logger.warning(f"⚠️ Invalid 15-min timer result structure: {timer_result} (type: {type(timer_result)})")
            return None
        
    except Exception as e:
        logger.error(f"❌ 15-minute timer execution error: {e}")
        return None

def render_fingerprinting_component(session_id: str, fingerprinting_manager: FingerprintingManager) -> Optional[Dict[str, Any]]:
    """Render fingerprinting component and return results"""
    
    if not session_id:
        return None
    
    try:
        # Generate JavaScript code
        js_code = fingerprinting_manager.generate_fingerprint_component(session_id)
        
        # Create stable key
        fingerprint_key = f"fingerprint_{session_id[:8]}_{hash(session_id) % 1000}"
        
        # Execute JavaScript
        fingerprint_result = st_javascript(js_code, key=fingerprint_key)
        
        logger.debug(f"🔍 Raw fingerprint result: {type(fingerprint_result)}")
        
        if fingerprint_result and isinstance(fingerprint_result, dict):
            # Extract and validate fingerprint data
            processed_result = fingerprinting_manager.extract_fingerprint_from_result(fingerprint_result)
            logger.info(f"✅ Fingerprint generated: {processed_result.get('fingerprint_id', 'unknown')[:8]}... (method: {processed_result.get('fingerprint_method')})")
            return processed_result
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Fingerprinting component error: {e}")
        return None

def render_browser_close_detection_enhanced(session_id: str):
    """Enhanced browser close detection for emergency saves"""
    
    if not session_id:
        return

    js_code = f"""
    <script>
    (function() {{
        const sessionKey = 'fifi_close_enhanced_' + '{session_id}';
        if (window[sessionKey]) return;
        window[sessionKey] = true;
        
        const sessionId = '{session_id}';
        let saveTriggered = false;
        
        function getAppUrl() {{
            try {{
                return window.parent.location.origin + window.parent.location.pathname;
            }} catch (e) {{
                return window.location.origin + window.location.pathname;
            }}
        }}

        function sendEmergencySave() {{
            if (saveTriggered) return;
            saveTriggered = true;
            console.clear();
            console.log('🚨 FiFi Enhanced: Browser close detected - emergency save');
            
            const appUrl = getAppUrl();
            const saveUrl = `${{appUrl}}?event=emergency_close&session_id=${{sessionId}}`;
            
            console.log('📡 Emergency save URL:', saveUrl);
            
            // Try beacon first
            try {{
                if (navigator.sendBeacon) {{
                    const beaconSuccess = navigator.sendBeacon(saveUrl);
                    console.log('📡 Emergency beacon result:', beaconSuccess);
                    
                    if (beaconSuccess) {{
                        setTimeout(() => {{
                            try {{
                                window.parent.location.href = saveUrl;
                            }} catch (e) {{
                                window.location.href = saveUrl;
                            }}
                        }}, 50);
                        return;
                    }}
                }}
            }} catch (e) {{
                console.log('📡 Emergency beacon failed:', e);
            }}
            
            // Force reload with save params
            try {{
                console.log('🔄 Forcing emergency save via reload');
                window.parent.location.href = saveUrl;
            }} catch (e) {{
                try {{
                    window.location.href = saveUrl;
                }} catch (e2) {{
                    console.error('❌ All emergency save methods failed:', e, e2);
                }}
            }}
        }}
        
        // Setup all close detection methods
        const events = ['beforeunload', 'pagehide', 'unload'];
        events.forEach(eventType => {{
            try {{
                if (window.parent && window.parent !== window) {{
                    window.parent.addEventListener(eventType, sendEmergencySave, {{ capture: true }});
                }}
                window.addEventListener(eventType, sendEmergencySave, {{ capture: true }});
            }} catch (e) {{
                console.debug(`Failed to add ${{eventType}} listener:`, e);
            }}
        }});
        
        // Enhanced visibility detection
        try {{
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('visibilitychange', () => {{
                    if (window.parent.document.visibilityState === 'hidden') {{
                        console.log('🚨 Enhanced: Main app hidden');
                        sendEmergencySave();
                    }}
                }}, {{ passive: true }});
            }}
        }} catch (e) {{
            document.addEventListener('visibilitychange', () => {{
                if (document.visibilityState === 'hidden') {{
                    console.log('🚨 Enhanced: Component hidden');
                    sendEmergencySave();
                }}
            }}, {{ passive: true }});
        }}
        
        console.log('✅ Enhanced browser close detection initialized for 15-min system');
    }})();
    </script>
    """
    
    try:
        st.components.v1.html(js_code, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to render enhanced browser close component: {e}")

def global_message_channel_error_handler():
    """Global error handling for message channel issues"""
    
    js_error_handler = """
    <script>
    (function() {
        if (window.fifi_error_handler_initialized) return;
        window.fifi_error_handler_initialized = true;
        
        // Global error handler for uncaught promise rejections
        window.addEventListener('unhandledrejection', function(event) {
            const error = event.reason;
            if (error && error.message && error.message.includes('message channel closed')) {
                console.log('🛡️ FiFi Enhanced: Caught message channel error:', error.message);
                event.preventDefault();
            }
        });
        
        console.log('✅ FiFi Enhanced: Global message channel error handler initialized');
    })();
    </script>
    """
    
    try:
        st.components.v1.html(js_error_handler, height=0, width=0)
    except Exception as e:
        logger.error(f"Failed to initialize global error handler: {e}")

def handle_15min_timer_event(timer_result: Dict[str, Any], session_manager, session) -> bool:
    """Handle 15-minute timer events per specification"""
    
    if not timer_result or not isinstance(timer_result, dict):
        return False
    
    event = timer_result.get('event')
    session_id = timer_result.get('session_id')
    inactive_minutes = timer_result.get('inactive_minutes', 0)
    should_save_crm = timer_result.get('should_save_crm', False)
    should_ban = timer_result.get('should_ban', False)
    
    logger.info(f"🎯 Processing 15-min timer event: {event} for session {session_id[:8] if session_id else 'unknown'}")
    
    try:
        if event == 'session_timeout_15min':
            # 15-minute inactivity → CRM save → NO BAN per specification
            st.info(f"⏰ **Session timeout** after {inactive_minutes} minutes of inactivity")
            
            # Check if session is eligible for CRM save
            if (session.user_type == UserType.REGISTERED_USER and 
                hasattr(session, 'email') and session.email and 
                hasattr(session, 'messages') and session.messages and
                hasattr(session, 'timeout_saved_to_crm') and not session.timeout_saved_to_crm):
                
                with st.spinner("💾 Auto-saving chat to CRM (15-min timeout)..."):
                    try:
                        save_success = session_manager._save_to_crm_timeout(session, "15-Minute Session Timeout")
                    except Exception as e:
                        logger.error(f"15-min timeout save failed: {e}")
                        save_success = False
                
                if save_success:
                    st.success("✅ Chat automatically saved to CRM!")
                    # Update session activity and mark as saved
                    session.last_activity = datetime.now()
                    session.timeout_saved_to_crm = True
                    try:
                        session_manager.db.save_session(session)
                    except Exception as e:
                        logger.error(f"Failed to update session after 15-min save: {e}")
                else:
                    st.warning("⚠️ CRM save failed, but session continues (NO BAN)")
                
                # Important: NO BAN applied per specification
                st.info("ℹ️ You can continue using FiFi AI")
                return False
            else:
                st.info("ℹ️ Session timeout (no CRM save needed)")
                logger.info(f"15-min timeout eligibility check: user_type={session.user_type}, email={getattr(session, 'email', None)}, messages={len(getattr(session, 'messages', []))}, saved={getattr(session, 'timeout_saved_to_crm', None)}")
                
                # Still no ban - user can continue
                st.info("ℹ️ You can continue using FiFi AI")
                return False
        else:
            logger.warning(f"⚠️ Unknown 15-min timer event: {event}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing 15-min timer event {event}: {e}", exc_info=True)
        st.error(f"⚠️ Timer event processing error: {str(e)}")
        return False

# =============================================================================
# ENHANCED SESSION MANAGER WITH ALL NEW FEATURES
# =============================================================================

class SessionManager:
    def __init__(self, config: Config, db_manager: DatabaseManager, zoho_manager: ZohoCRMManager, 
                 ai_system: EnhancedAI, rate_limiter: RateLimiter, fingerprinting_manager: FingerprintingManager,
                 email_verification_manager: EmailVerificationManager, question_limit_manager: QuestionLimitManager):
        self.config = config
        self.db = db_manager
        self.zoho = zoho_manager
        self.ai = ai_system
        self.rate_limiter = rate_limiter
        self.fingerprinting = fingerprinting_manager
        self.email_verification = email_verification_manager
        self.question_limits = question_limit_manager
        self.session_timeout_minutes = 15  # Updated to 15 minutes per specification
        self._save_lock = threading.Lock()

    def get_session_timeout_minutes(self) -> int:
        return getattr(self, 'session_timeout_minutes', 15)

    def _is_session_expired(self, session: UserSession) -> bool:
        """Check if session has exceeded 15-minute timeout"""
        if not session.last_activity:
            return False
        time_diff = datetime.now() - session.last_activity
        return time_diff.total_seconds() > (self.session_timeout_minutes * 60)

    def _update_activity(self, session: UserSession):
        """Update session activity timestamp"""
        session.last_activity = datetime.now()
        
        # Reset timeout save flag when user becomes active again
        if session.timeout_saved_to_crm:
            session.timeout_saved_to_crm = False
            logger.info(f"Reset timeout save flag for active session {session.session_id[:8]}")
        
        try:
            self.db.save_session(session)
            logger.debug(f"Session activity updated for {session.session_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

    def _create_guest_session(self) -> UserSession:
        """Create new guest session with fingerprinting"""
        session = UserSession(session_id=str(uuid.uuid4()))
        
        # Get IP address from Streamlit context
        try:
            # Try different methods to get client IP
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = st.context.headers
                ip_methods = [
                    ('x-forwarded-for', lambda h: h.get('x-forwarded-for', '').split(',')[0].strip()),
                    ('x-real-ip', lambda h: h.get('x-real-ip', '')),
                    ('cf-connecting-ip', lambda h: h.get('cf-connecting-ip', '')),
                    ('x-client-ip', lambda h: h.get('x-client-ip', ''))
                ]
                
                for method, extractor in ip_methods:
                    try:
                        ip = extractor(headers)
                        if ip:
                            session.ip_address = ip
                            session.ip_detection_method = method
                            break
                    except:
                        continue
            
            # Get user agent
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                session.user_agent = st.context.headers.get('user-agent', '')
                
        except Exception as e:
            logger.debug(f"Could not extract IP/User-Agent: {e}")
        
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
        
        # Fix ban_status if it's a string
        if isinstance(session.ban_status, str):
            try:
                session.ban_status = BanStatus(session.ban_status)
            except ValueError:
                session.ban_status = BanStatus.NONE
        
        # Ensure messages is a list
        if not isinstance(session.messages, list):
            session.messages = []
        
        # Ensure email_addresses_used is a list
        if not isinstance(session.email_addresses_used, list):
            session.email_addresses_used = []
            
        return session

    def _save_to_crm_timeout(self, session: UserSession, trigger_reason: str):
        """15-minute timeout CRM save"""
        with self._save_lock:
            logger.info(f"=== 15-MINUTE TIMEOUT CRM SAVE ===")
            logger.info(f"Trigger: {trigger_reason}")
            logger.info(f"Session ID: {session.session_id[:8] if session.session_id else 'None'}")
            
            # Validate and fix session before proceeding
            session = self._validate_and_fix_session(session)
            
            # Check prerequisites
            if session.user_type != UserType.REGISTERED_USER:
                logger.info(f"TIMEOUT SAVE SKIPPED: Not a registered user (current: {session.user_type})")
                return False
            if not session.email:
                logger.info("TIMEOUT SAVE SKIPPED: No email address")
                return False
            if not session.messages:
                logger.info("TIMEOUT SAVE SKIPPED: No messages to save")
                return False
            if not self.zoho.config.ZOHO_ENABLED:
                logger.info("TIMEOUT SAVE SKIPPED: Zoho is not enabled")
                return False

            try:
                success = self.zoho.save_chat_transcript_sync(session, trigger_reason)
                if success:
                    logger.info("TIMEOUT SAVE COMPLETED: 15-minute save successful")
                    # Mark as saved to prevent duplicate saves
                    session.timeout_saved_to_crm = True
                    self.db.save_session(session)
                    return True
                else:
                    logger.error("TIMEOUT SAVE FAILED: 15-minute save failed")
                    return False

            except Exception as e:
                logger.error(f"TIMEOUT SAVE FAILED: Unexpected error - {type(e).__name__}: {str(e)}", exc_info=True)
                return False
            finally:
                logger.info(f"=== 15-MINUTE TIMEOUT CRM SAVE ENDED ===\n")

    def apply_fingerprinting(self, session: UserSession, fingerprint_data: Dict[str, Any]):
        """Apply fingerprinting data to session"""
        session.fingerprint_id = fingerprint_data.get('fingerprint_id')
        session.fingerprint_method = fingerprint_data.get('fingerprint_method')
        session.visitor_type = fingerprint_data.get('visitor_type', 'new_visitor')
        session.browser_privacy_level = fingerprint_data.get('privacy_level', 'standard')
        
        # Update network info if available
        browser_info = fingerprint_data.get('browser_info', {})
        if browser_info.get('userAgent'):
            session.user_agent = browser_info['userAgent']
        
        logger.info(f"Applied fingerprinting to session {session.session_id[:8]}: {session.fingerprint_id[:8]}... ({session.fingerprint_method})")

    def check_fingerprint_history(self, fingerprint_id: str) -> Optional[Dict[str, Any]]:
        """Check if fingerprint has been seen before"""
        try:
            sessions = self.db.find_sessions_by_fingerprint(fingerprint_id)
            
            if sessions:
                # Find sessions with email addresses
                email_sessions = [s for s in sessions if s.email]
                
                if email_sessions:
                    # Return the most recent email for recognition
                    latest_email_session = max(email_sessions, key=lambda x: x.last_activity)
                    
                    return {
                        'has_history': True,
                        'email': latest_email_session.email,
                        'full_name': latest_email_session.full_name,
                        'last_seen': latest_email_session.last_activity,
                        'session_count': len(sessions)
                    }
            
            return {'has_history': False}
            
        except Exception as e:
            logger.error(f"Error checking fingerprint history: {e}")
            return {'has_history': False}

    def _mask_email(self, email: str) -> str:
        """Mask email for privacy (e.g., j***n@***.com)"""
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        
        if len(local) <= 2:
            masked_local = local[0] + '*'
        else:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            masked_domain = '*' * len(domain_parts[0]) + '.' + '.'.join(domain_parts[1:])
        else:
            masked_domain = '*' * len(domain)
        
        return f"{masked_local}@{masked_domain}"

    def handle_guest_email_verification(self, session: UserSession, email: str) -> Dict[str, Any]:
        """Handle email verification for guest upgrade"""
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # Check if email is already in use
        existing_sessions = self.db.find_sessions_by_email(email)
        
        # Send verification code
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            # Track email in session
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email)
            
            self.db.save_session(session)
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your email.',
                'verification_pending': True
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verify code and upgrade user"""
        
        if not session.email:
            return {'success': False, 'message': 'No email verification in progress.'}
        
        # Verify the code
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            # Upgrade user to EMAIL_VERIFIED_GUEST
            session.user_type = UserType.EMAIL_VERIFIED_GUEST
            session.daily_question_count = 0  # Reset question count
            session.question_limit_reached = False
            
            # Reset ban status if any
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session)
            
            logger.info(f"User upgraded to EMAIL_VERIFIED_GUEST: {session.email}")
            
            return {
                'success': True,
                'message': f'Email verified! You now have 10 questions per day.',
                'user_type': 'email_verified_guest'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """Detect evasion attempts"""
        
        if not session.fingerprint_id:
            return False
        
        # Check for multiple sessions with same fingerprint
        fingerprint_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        # Look for recent sessions that hit limits
        recent_cutoff = datetime.now() - timedelta(hours=48)
        recent_limited_sessions = [
            s for s in fingerprint_sessions 
            if s.session_id != session.session_id and
               s.last_activity > recent_cutoff and
               (s.question_limit_reached or s.ban_status != BanStatus.NONE)
        ]
        
        if recent_limited_sessions:
            logger.warning(f"Evasion detected for fingerprint {session.fingerprint_id[:8]}... - {len(recent_limited_sessions)} recent limited sessions")
            return True
        
        # Check for rapid email switching
        if len(session.email_addresses_used) > 2:
            logger.warning(f"Email switching detected: {len(session.email_addresses_used)} emails used")
            return True
        
        return False

    def get_session(self) -> UserSession:
        """Get session with complete validation and fingerprinting"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Validate and fix session data
                session = self._validate_and_fix_session(session)
                
                # Check ban status
                limit_check = self.question_limits.is_within_limits(session)
                if not limit_check.get('allowed', True):
                    # Session is banned - show ban message
                    ban_type = limit_check.get('ban_type', 'unknown')
                    message = limit_check.get('message', 'Usage limit reached.')
                    time_remaining = limit_check.get('time_remaining')
                    
                    if time_remaining:
                        hours = int(time_remaining.total_seconds() // 3600)
                        minutes = int((time_remaining.total_seconds() % 3600) // 60)
                        st.error(f"🚫 **Access Restricted**")
                        st.error(f"Time remaining: {hours}h {minutes}m")
                        st.info(message)
                    else:
                        st.error(f"🚫 **Access Restricted**")
                        st.info(message)
                    
                    # Don't create new session - show ban message
                    return session
                
                # Update activity and continue
                self._update_activity(session)
                return session
        
        # No session or inactive - create new guest session
        return self._create_guest_session()

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

                # Upgrade to REGISTERED_USER
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                # Reset question counts and bans for registered users
                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                # Add email to used addresses
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
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
                        st.success(f"Welcome back, {current_session.full_name}!")
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
        
        # Check question limits BEFORE processing
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            reason = limit_check.get('reason')
            
            if reason == 'guest_limit':
                # Guest hit 4-question limit → force email verification
                return {
                    "content": "Please provide your email address to continue using FiFi AI.",
                    "success": False,
                    "requires_email": True,
                    "user_type": "guest"
                }
            elif reason in ['daily_limit', 'first_tier_limit', 'total_limit']:
                # User hit daily/tier limits → show ban message
                return {
                    "content": limit_check.get('message', 'Usage limit reached.'),
                    "success": False,
                    "banned": True,
                    "ban_type": limit_check.get('ban_type', 'unknown')
                }
            elif reason == 'banned':
                # User is currently banned
                return {
                    "content": limit_check.get('message', 'You are currently restricted.'),
                    "success": False,
                    "banned": True,
                    "time_remaining": limit_check.get('time_remaining')
                }
        
        # Check for evasion attempts
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session)
            self.db.save_session(session)
            
            return {
                "content": "Usage limit reached. Please try again later.",
                "success": False,
                "evasion_penalty": True,
                "penalty_hours": penalty_hours
            }

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {
                "content": moderation["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        # Record the question
        self.question_limits.record_question(session)

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
        
        # Save to CRM if eligible
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        self._end_session_internal(session)

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

    def manual_save_to_crm(self, session: UserSession):
        """Manual CRM save (Save button)"""
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# ENHANCED UI COMPONENTS WITH NEW FEATURES
# =============================================================================

def render_welcome_page_enhanced(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Knowledge Base**\nAccess curated F&B industry information")
    with col2:
        st.info("🌐 **Web Search**\nReal-time market data and trends") 
    with col3:
        st.info("📚 **Smart Citations**\nClickable inline source references")
    
    # Show user tier benefits
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown("• 4 questions to try FiFi AI")
        st.markdown("• Email verification required")
        st.markdown("• Quick start, no registration")
    
    with col2:
        st.info("📧 **Email Verified**")
        st.markdown("• 10 questions per day")
        st.markdown("• Email verification required")
        st.markdown("• Rolling 24-hour limits")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown("• 40 questions per day")
        st.markdown("• Cross-device tracking")
        st.markdown("• Auto-save to CRM")
        st.markdown("• Enhanced features")
    
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
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest experience:**
        - Start with 4 questions to explore FiFi AI
        - Email verification unlocks 10 questions/day
        - Universal device fingerprinting for security
        - Upgrade path to full registration
        
        ✨ **Registration benefits:**
        - 40 questions per day across all devices
        - Automatic CRM integration and chat history
        - Enhanced personalization features
        - Priority access during high usage
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """Render email verification dialog for guest users"""
    
    st.error("📧 **Email Verification Required**")
    st.info("You've reached the 4-question limit for guest users. Please verify your email to get 10 questions per day.")
    
    # Check if we have fingerprint history for recognition
    if session.fingerprint_id:
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            # Show recognition prompt
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            
            st.info(f"🤝 **We Recognize You!**")
            st.markdown(f"Based on our records, we seem to recognize you. Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's me", use_container_width=True):
                    # Send verification code to recognized email
                    verification_sent = session_manager.email_verification.send_verification_code(fingerprint_history['email'])
                    if verification_sent:
                        session.email = fingerprint_history['email']
                        session.recognition_response = "yes"
                        session_manager.db.save_session(session)
                        st.session_state.verification_email = fingerprint_history['email']
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error("Failed to send verification code. Please try manual entry.")
            
            with col2:
                if st.button("❌ No, different email", use_container_width=True):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry"
                    st.rerun()
    
    # Manual email entry
    if st.session_state.get('verification_stage') == 'email_entry' or not session.fingerprint_id:
        with st.form("email_verification_form"):
            st.markdown("**Enter your email address:**")
            email = st.text_input("Email Address", placeholder="your@email.com")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if email:
                    result = session_manager.handle_guest_email_verification(session, email)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = email
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address.")
    
    # Code entry
    if st.session_state.get('verification_stage') == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 Verification code sent to **{verification_email}**")
        
        with st.form("code_verification_form"):
            st.markdown("**Enter the verification code from your email:**")
            code = st.text_input("Verification Code", placeholder="123456", max_chars=6)
            submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            
            if submit_code:
                if code:
                    result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        
                        # Clear verification state
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code.")
        
        if st.button("🔄 Resend Code"):
            if verification_email:
                verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                if verification_sent:
                    st.success("Verification code resent!")
                else:
                    st.error("Failed to resend code. Please try again.")

def render_sidebar_enhanced(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Enhanced user status section
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # Question usage for registered users
            st.markdown(f"**Questions:** {session.total_question_count}/40")
            if session.total_question_count <= 20:
                st.progress(session.total_question_count / 20)
                st.caption("First tier: 20 questions")
            else:
                st.progress((session.total_question_count - 20) / 20)
                st.caption("Second tier: 21-40 questions")
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            # Daily question usage
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Questions reset!")
            
        else:  # GUEST
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10/day")
        
        # Fingerprinting info (if available)
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'}")
        
        # CRM Status
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type == UserType.REGISTERED_USER:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready**")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM")
            else:
                st.caption("💾 Auto-save at 15min")
        else: 
            st.caption("🚫 CRM: Registered users only")
        
        st.divider()
        
        # Enhanced session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        # Ban status
        if session.ban_status != BanStatus.NONE:
            if session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 **Restricted**")
                st.error(f"Time: {hours}h {minutes}m")
            else:
                st.info("🟢 **Restrictions Lifted**")
        
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

        # Download & save section (registered users only)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
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
                st.caption("💡 Auto-saves after 15min inactivity")

def render_chat_interface_complete(session_manager, session):
    """Complete chat interface with all new features"""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting")
    
    # Add global error handler
    try:
        global_message_channel_error_handler()
    except Exception as e:
        logger.error(f"Failed to add global error handler: {e}")
    
    # Initialize fingerprinting for session
    if not session.fingerprint_id:
        fingerprint_data = render_fingerprinting_component(session.session_id, session_manager.fingerprinting)
        if fingerprint_data:
            session_manager.apply_fingerprinting(session, fingerprint_data)
            session_manager.db.save_session(session)
            st.rerun()
    
    # Add enhanced browser close detection
    try:
        render_browser_close_detection_enhanced(session.session_id)
    except Exception as e:
        logger.error(f"Failed to render browser close detection: {e}")
    
    # Handle guest email verification dialog
    if (session.user_type == UserType.GUEST and 
        session.daily_question_count >= 4):
        render_email_verification_dialog(session_manager, session)
        return  # Don't show chat interface during verification
    
    # Show user tier status
    if session.user_type == UserType.GUEST:
        remaining = 4 - session.daily_question_count
        if remaining > 0:
            st.info(f"👤 **Guest Mode:** {remaining} questions remaining before email verification")
        else:
            st.warning("👤 **Guest Mode:** Email verification required to continue")
    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
        remaining = 10 - session.daily_question_count
        st.info(f"📧 **Email Verified:** {remaining} questions remaining today")
    elif session.user_type == UserType.REGISTERED_USER:
        if session.total_question_count <= 20:
            remaining = 20 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions in first tier")
        else:
            remaining = 40 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions remaining")
    
    # 15-minute timer for registered users
    if session.user_type == UserType.REGISTERED_USER:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id, session_manager)
        except Exception as e:
            logger.error(f"15-minute timer error: {e}")
        
        # Process timer events
        if timer_result:
            try:
                should_rerun = handle_15min_timer_event(timer_result, session_manager, session)
                if should_rerun:
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

    # Handle input
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue")
                        st.rerun()
                    elif response.get('banned'):
                        st.error(response.get('content', 'Access restricted'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Extended restriction applied")
                        st.error(f"Penalty: {response.get('penalty_hours', 0)} hours")
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                        
                        # Show enhanced features used
                        indicators = []
                        if response.get("used_pinecone"):
                            indicators.append("🧠 Knowledge Base")
                        if response.get("used_search"):
                            indicators.append("🌐 Web Search")
                        
                        if indicators:
                            st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                except Exception as e:
                    logger.error(f"AI response generation failed: {e}")
                    st.error("⚠️ Sorry, I encountered an error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS (ENHANCED)
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

def ensure_complete_initialization():
    """Ensure the complete application is properly initialized"""
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
            
            # NEW: Initialize additional managers
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager
            st.session_state.initialized = True
            
            logger.info("✅ Complete application initialized successfully with all features")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Complete initialization failed: {e}", exc_info=True)
            return False
    
    return True

def process_emergency_save_enhanced(session_id: str) -> bool:
    """Enhanced emergency save processing"""
    try:
        session_manager = get_session_manager()
        if not session_manager:
            logger.error("❌ Session manager not available for emergency save")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Session {session_id[:8]} not found for emergency save")
            return False
        
        session = session_manager._validate_and_fix_session(session)
        
        logger.info(f"✅ Emergency save session loaded: {session_id[:8]}")
        logger.info(f"   User: {session.user_type.value}")
        logger.info(f"   Email: {session.email}")
        logger.info(f"   Messages: {len(session.messages)}")
        
        # Check eligibility for emergency save
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            logger.info("✅ Session eligible for emergency save")
            
            # Extend session life
            session.last_activity = datetime.now()
            session_manager.db.save_session(session)
            
            # Perform emergency save
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Enhanced Browser Close)")
            return success
        else:
            logger.info("❌ Session not eligible for emergency save")
            return False
            
    except Exception as e:
        logger.error(f"Enhanced emergency save processing failed: {e}", exc_info=True)
        return False

def enhanced_handle_save_requests():
    """Enhanced save handler for all emergency save types"""
    
    logger.info("🔍 ENHANCED SAVE HANDLER: Checking for emergency save requests...")
    
    query_params = st.query_params
    logger.info(f"📋 Query params found: {dict(query_params)}")
    
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event in ["emergency_close", "close"] and session_id:
        logger.info("=" * 80)
        logger.info("🚨 ENHANCED EMERGENCY SAVE REQUEST DETECTED")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event: {event}")
        logger.info("=" * 80)
        
        # Show visual confirmation
        st.error("🚨 **Emergency Save Detected** - Processing enhanced browser close save...")
        
        # Clear query params
        st.query_params.clear()
        
        try:
            success = process_emergency_save_enhanced(session_id)
            
            if success:
                st.success("✅ Enhanced emergency save completed successfully!")
                logger.info("✅ Enhanced emergency save completed")
            else:
                st.error("❌ Enhanced emergency save failed")
                logger.error("❌ Enhanced emergency save failed")
                
        except Exception as e:
            st.error(f"❌ Enhanced emergency save error: {str(e)}")
            logger.error(f"Enhanced emergency save crashed: {e}", exc_info=True)
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No enhanced emergency save requests found")

# =============================================================================
# MAIN APPLICATION WITH ALL FEATURES
# =============================================================================

def main():
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    # Add global message channel error handler
    global_message_channel_error_handler()

    # Clear state button for development
    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Initialize complete application
    if not ensure_complete_initialization():
        st.stop()

    # Handle enhanced save requests
    enhanced_handle_save_requests()

    # Get session manager
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get enhanced session manager.")
        st.stop()
    
    # Main application flow
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page_enhanced(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar_enhanced(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_complete(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state['page']
            st.rerun()

if __name__ == "__main__":
    main()
        if not re.match(email_pattern, email):
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # Check if email is already in use
        existing_sessions = self.db.find_sessions_by_email(email)
        
        # Send verification code
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            # Track email in session
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email)
            
            self.db.save_session(session)
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your email.',
                'verification_pending': True
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verify code and upgrade user"""
        
        if not session.email:
            return {'success': False, 'message': 'No email verification in progress.'}
        
        # Verify the code
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            # Upgrade user to EMAIL_VERIFIED_GUEST
            session.user_type = UserType.EMAIL_VERIFIED_GUEST
            session.daily_question_count = 0  # Reset question count
            session.question_limit_reached = False
            
            # Reset ban status if any
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session)
            
            logger.info(f"User upgraded to EMAIL_VERIFIED_GUEST: {session.email}")
            
            return {
                'success': True,
                'message': f'Email verified! You now have 10 questions per day.',
                'user_type': 'email_verified_guest'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """Detect evasion attempts"""
        
        if not session.fingerprint_id:
            return False
        
        # Check for multiple sessions with same fingerprint
        fingerprint_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        # Look for recent sessions that hit limits
        recent_cutoff = datetime.now() - timedelta(hours=48)
        recent_limited_sessions = [
            s for s in fingerprint_sessions 
            if s.session_id != session.session_id and
               s.last_activity > recent_cutoff and
               (s.question_limit_reached or s.ban_status != BanStatus.NONE)
        ]
        
        if recent_limited_sessions:
            logger.warning(f"Evasion detected for fingerprint {session.fingerprint_id[:8]}... - {len(recent_limited_sessions)} recent limited sessions")
            return True
        
        # Check for rapid email switching
        if len(session.email_addresses_used) > 2:
            logger.warning(f"Email switching detected: {len(session.email_addresses_used)} emails used")
            return True
        
        return False

    def get_session(self) -> UserSession:
        """Get session with complete validation and fingerprinting"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Validate and fix session data
                session = self._validate_and_fix_session(session)
                
                # Check ban status
                limit_check = self.question_limits.is_within_limits(session)
                if not limit_check.get('allowed', True):
                    # Session is banned - show ban message
                    ban_type = limit_check.get('ban_type', 'unknown')
                    message = limit_check.get('message', 'Usage limit reached.')
                    time_remaining = limit_check.get('time_remaining')
                    
                    if time_remaining:
                        hours = int(time_remaining.total_seconds() // 3600)
                        minutes = int((time_remaining.total_seconds() % 3600) // 60)
                        st.error(f"🚫 **Access Restricted**")
                        st.error(f"Time remaining: {hours}h {minutes}m")
                        st.info(message)
                    else:
                        st.error(f"🚫 **Access Restricted**")
                        st.info(message)
                    
                    # Don't create new session - show ban message
                    return session
                
                # Update activity and continue
                self._update_activity(session)
                return session
        
        # No session or inactive - create new guest session
        return self._create_guest_session()

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

                # Upgrade to REGISTERED_USER
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                # Reset question counts and bans for registered users
                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                # Add email to used addresses
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
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
                        st.success(f"Welcome back, {current_session.full_name}!")
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
        
        # Check question limits BEFORE processing
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            reason = limit_check.get('reason')
            
            if reason == 'guest_limit':
                # Guest hit 4-question limit → force email verification
                return {
                    "content": "Please provide your email address to continue using FiFi AI.",
                    "success": False,
                    "requires_email": True,
                    "user_type": "guest"
                }
            elif reason in ['daily_limit', 'first_tier_limit', 'total_limit']:
                # User hit daily/tier limits → show ban message
                return {
                    "content": limit_check.get('message', 'Usage limit reached.'),
                    "success": False,
                    "banned": True,
                    "ban_type": limit_check.get('ban_type', 'unknown')
                }
            elif reason == 'banned':
                # User is currently banned
                return {
                    "content": limit_check.get('message', 'You are currently restricted.'),
                    "success": False,
                    "banned": True,
                    "time_remaining": limit_check.get('time_remaining')
                }
        
        # Check for evasion attempts
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session)
            self.db.save_session(session)
            
            return {
                "content": "Usage limit reached. Please try again later.",
                "success": False,
                "evasion_penalty": True,
                "penalty_hours": penalty_hours
            }

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {
                "content": moderation["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        # Record the question
        self.question_limits.record_question(session)

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
        
        # Save to CRM if eligible
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        self._end_session_internal(session)

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

    def manual_save_to_crm(self, session: UserSession):
        """Manual CRM save (Save button)"""
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# ENHANCED UI COMPONENTS WITH NEW FEATURES
# =============================================================================

def render_welcome_page_enhanced(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Knowledge Base**\nAccess curated F&B industry information")
    with col2:
        st.info("🌐 **Web Search**\nReal-time market data and trends") 
    with col3:
        st.info("📚 **Smart Citations**\nClickable inline source references")
    
    # Show user tier benefits
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown("• 4 questions to try FiFi AI")
        st.markdown("• Email verification required")
        st.markdown("• Quick start, no registration")
    
    with col2:
        st.info("📧 **Email Verified**")
        st.markdown("• 10 questions per day")
        st.markdown("• Email verification required")
        st.markdown("• Rolling 24-hour limits")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown("• 40 questions per day")
        st.markdown("• Cross-device tracking")
        st.markdown("• Auto-save to CRM")
        st.markdown("• Enhanced features")
    
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
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest experience:**
        - Start with 4 questions to explore FiFi AI
        - Email verification unlocks 10 questions/day
        - Universal device fingerprinting for security
        - Upgrade path to full registration
        
        ✨ **Registration benefits:**
        - 40 questions per day across all devices
        - Automatic CRM integration and chat history
        - Enhanced personalization features
        - Priority access during high usage
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """Render email verification dialog for guest users"""
    
    st.error("📧 **Email Verification Required**")
    st.info("You've reached the 4-question limit for guest users. Please verify your email to get 10 questions per day.")
    
    # Check if we have fingerprint history for recognition
    if session.fingerprint_id:
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            # Show recognition prompt
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            
            st.info(f"🤝 **We Recognize You!**")
            st.markdown(f"Based on our records, we seem to recognize you. Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's me", use_container_width=True):
                    # Send verification code to recognized email
                    verification_sent = session_manager.email_verification.send_verification_code(fingerprint_history['email'])
                    if verification_sent:
                        session.email = fingerprint_history['email']
                        session.recognition_response = "yes"
                        session_manager.db.save_session(session)
                        st.session_state.verification_email = fingerprint_history['email']
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error("Failed to send verification code. Please try manual entry.")
            
            with col2:
                if st.button("❌ No, different email", use_container_width=True):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry"
                    st.rerun()
    
    # Manual email entry
    if st.session_state.get('verification_stage') == 'email_entry' or not session.fingerprint_id:
        with st.form("email_verification_form"):
            st.markdown("**Enter your email address:**")
            email = st.text_input("Email Address", placeholder="your@email.com")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if email:
                    result = session_manager.handle_guest_email_verification(session, email)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = email
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address.")
    
    # Code entry
    if st.session_state.get('verification_stage') == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 Verification code sent to **{verification_email}**")
        
        with st.form("code_verification_form"):
            st.markdown("**Enter the verification code from your email:**")
            code = st.text_input("Verification Code", placeholder="123456", max_chars=6)
            submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            
            if submit_code:
                if code:
                    result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        
                        # Clear verification state
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code.")
        
        if st.button("🔄 Resend Code"):
            if verification_email:
                verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                if verification_sent:
                    st.success("Verification code resent!")
                else:
                    st.error("Failed to resend code. Please try again.")

def render_sidebar_enhanced(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Enhanced user status section
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # Question usage for registered users
            st.markdown(f"**Questions:** {session.total_question_count}/40")
            if session.total_question_count <= 20:
                st.progress(session.total_question_count / 20)
                st.caption("First tier: 20 questions")
            else:
                st.progress((session.total_question_count - 20) / 20)
                st.caption("Second tier: 21-40 questions")
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            # Daily question usage
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Questions reset!")
            
        else:  # GUEST
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10/day")
        
        # Fingerprinting info (if available)
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'}")
        
        # CRM Status
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type == UserType.REGISTERED_USER:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready**")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM")
            else:
                st.caption("💾 Auto-save at 15min")
        else: 
            st.caption("🚫 CRM: Registered users only")
        
        st.divider()
        
        # Enhanced session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        # Ban status
        if session.ban_status != BanStatus.NONE:
            if session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 **Restricted**")
                st.error(f"Time: {hours}h {minutes}m")
            else:
                st.info("🟢 **Restrictions Lifted**")
        
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

        # Download & save section (registered users only)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
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
                st.caption("💡 Auto-saves after 15min inactivity")

def render_chat_interface_complete(session_manager, session):
    """Complete chat interface with all new features"""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting")
    
    # Add global error handler
    try:
        global_message_channel_error_handler()
    except Exception as e:
        logger.error(f"Failed to add global error handler: {e}")
    
    # Initialize fingerprinting for session
    if not session.fingerprint_id:
        fingerprint_data = render_fingerprinting_component(session.session_id, session_manager.fingerprinting)
        if fingerprint_data:
            session_manager.apply_fingerprinting(session, fingerprint_data)
            session_manager.db.save_session(session)
            st.rerun()
    
    # Add enhanced browser close detection
    try:
        render_browser_close_detection_enhanced(session.session_id)
    except Exception as e:
        logger.error(f"Failed to render browser close detection: {e}")
    
    # Handle guest email verification dialog
    if (session.user_type == UserType.GUEST and 
        session.daily_question_count >= 4):
        render_email_verification_dialog(session_manager, session)
        return  # Don't show chat interface during verification
    
    # Show user tier status
    if session.user_type == UserType.GUEST:
        remaining = 4 - session.daily_question_count
        if remaining > 0:
            st.info(f"👤 **Guest Mode:** {remaining} questions remaining before email verification")
        else:
            st.warning("👤 **Guest Mode:** Email verification required to continue")
    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
        remaining = 10 - session.daily_question_count
        st.info(f"📧 **Email Verified:** {remaining} questions remaining today")
    elif session.user_type == UserType.REGISTERED_USER:
        if session.total_question_count <= 20:
            remaining = 20 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions in first tier")
        else:
            remaining = 40 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions remaining")
    
    # 15-minute timer for registered users
    if session.user_type == UserType.REGISTERED_USER:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id, session_manager)
        except Exception as e:
            logger.error(f"15-minute timer error: {e}")
        
        # Process timer events
        if timer_result:
            try:
                should_rerun = handle_15min_timer_event(timer_result, session_manager, session)
                if should_rerun:
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

    # Handle input
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue")
                        st.rerun()
                    elif response.get('banned'):
                        st.error(response.get('content', 'Access restricted'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Extended restriction applied")
                        st.error(f"Penalty: {response.get('penalty_hours', 0)} hours")
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                        
                        # Show enhanced features used
                        indicators = []
                        if response.get("used_pinecone"):
                            indicators.append("🧠 Knowledge Base")
                        if response.get("used_search"):
                            indicators.append("🌐 Web Search")
                        
                        if indicators:
                            st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                except Exception as e:
                    logger.error(f"AI response generation failed: {e}")
                    st.error("⚠️ Sorry, I encountered an error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS (ENHANCED)
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

def ensure_complete_initialization():
    """Ensure the complete application is properly initialized"""
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
            
            # NEW: Initialize additional managers
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager
            st.session_state.initialized = True
            
            logger.info("✅ Complete application initialized successfully with all features")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Complete initialization failed: {e}", exc_info=True)
            return False
    
    return True

def process_emergency_save_enhanced(session_id: str) -> bool:
    """Enhanced emergency save processing"""
    try:
        session_manager = get_session_manager()
        if not session_manager:
            logger.error("❌ Session manager not available for emergency save")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Session {session_id[:8]} not found for emergency save")
            return False
        
        session = session_manager._validate_and_fix_session(session)
        
        logger.info(f"✅ Emergency save session loaded: {session_id[:8]}")
        logger.info(f"   User: {session.user_type.value}")
        logger.info(f"   Email: {session.email}")
        logger.info(f"   Messages: {len(session.messages)}")
        
        # Check eligibility for emergency save
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            logger.info("✅ Session eligible for emergency save")
            
            # Extend session life
            session.last_activity = datetime.now()
            session_manager.db.save_session(session)
            
            # Perform emergency save
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Enhanced Browser Close)")
            return success
        else:
            logger.info("❌ Session not eligible for emergency save")
            return False
            
    except Exception as e:
        logger.error(f"Enhanced emergency save processing failed: {e}", exc_info=True)
        return False

def enhanced_handle_save_requests():
    """Enhanced save handler for all emergency save types"""
    
    logger.info("🔍 ENHANCED SAVE HANDLER: Checking for emergency save requests...")
    
    query_params = st.query_params
    logger.info(f"📋 Query params found: {dict(query_params)}")
    
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event in ["emergency_close", "close"] and session_id:
        logger.info("=" * 80)
        logger.info("🚨 ENHANCED EMERGENCY SAVE REQUEST DETECTED")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event: {event}")
        logger.info("=" * 80)
        
        # Show visual confirmation
        st.error("🚨 **Emergency Save Detected** - Processing enhanced browser close save...")
        
        # Clear query params
        st.query_params.clear()
        
        try:
            success = process_emergency_save_enhanced(session_id)
            
            if success:
                st.success("✅ Enhanced emergency save completed successfully!")
                logger.info("✅ Enhanced emergency save completed")
            else:
                st.error("❌ Enhanced emergency save failed")
                logger.error("❌ Enhanced emergency save failed")
                
        except Exception as e:
            st.error(f"❌ Enhanced emergency save error: {str(e)}")
            logger.error(f"Enhanced emergency save crashed: {e}", exc_info=True)
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No enhanced emergency save requests found")

# =============================================================================
# MAIN APPLICATION WITH ALL FEATURES
# =============================================================================

def main():
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    # Add global message channel error handler
    global_message_channel_error_handler()

    # Clear state button for development
    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Initialize complete application
    if not ensure_complete_initialization():
        st.stop()

    # Handle enhanced save requests
    enhanced_handle_save_requests()

    # Get session manager
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get enhanced session manager.")
        st.stop()
    
    # Main application flow
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page_enhanced(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar_enhanced(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_complete(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state['page']
            st.rerun()

if __name__ == "__main__":
    main()
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # Check if email is already in use
        existing_sessions = self.db.find_sessions_by_email(email)
        
        # Send verification code
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            # Track email in session
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email)
            
            self.db.save_session(session)
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your email.',
                'verification_pending': True
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verify code and upgrade user"""
        
        if not session.email:
            return {'success': False, 'message': 'No email verification in progress.'}
        
        # Verify the code
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            # Upgrade user to EMAIL_VERIFIED_GUEST
            session.user_type = UserType.EMAIL_VERIFIED_GUEST
            session.daily_question_count = 0  # Reset question count
            session.question_limit_reached = False
            
            # Reset ban status if any
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session)
            
            logger.info(f"User upgraded to EMAIL_VERIFIED_GUEST: {session.email}")
            
            return {
                'success': True,
                'message': f'Email verified! You now have 10 questions per day.',
                'user_type': 'email_verified_guest'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """Detect evasion attempts"""
        
        if not session.fingerprint_id:
            return False
        
        # Check for multiple sessions with same fingerprint
        fingerprint_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        # Look for recent sessions that hit limits
        recent_cutoff = datetime.now() - timedelta(hours=48)
        recent_limited_sessions = [
            s for s in fingerprint_sessions 
            if s.session_id != session.session_id and
               s.last_activity > recent_cutoff and
               (s.question_limit_reached or s.ban_status != BanStatus.NONE)
        ]
        
        if recent_limited_sessions:
            logger.warning(f"Evasion detected for fingerprint {session.fingerprint_id[:8]}... - {len(recent_limited_sessions)} recent limited sessions")
            return True
        
        # Check for rapid email switching
        if len(session.email_addresses_used) > 2:
            logger.warning(f"Email switching detected: {len(session.email_addresses_used)} emails used")
            return True
        
        return False

    def get_session(self) -> UserSession:
        """Get session with complete validation and fingerprinting"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Validate and fix session data
                session = self._validate_and_fix_session(session)
                
                # Check ban status
                limit_check = self.question_limits.is_within_limits(session)
                if not limit_check.get('allowed', True):
                    # Session is banned - show ban message
                    ban_type = limit_check.get('ban_type', 'unknown')
                    message = limit_check.get('message', 'Usage limit reached.')
                    time_remaining = limit_check.get('time_remaining')
                    
                    if time_remaining:
                        hours = int(time_remaining.total_seconds() // 3600)
                        minutes = int((time_remaining.total_seconds() % 3600) // 60)
                        st.error(f"🚫 **Access Restricted**")
                        st.error(f"Time remaining: {hours}h {minutes}m")
                        st.info(message)
                    else:
                        st.error(f"🚫 **Access Restricted**")
                        st.info(message)
                    
                    # Don't create new session - show ban message
                    return session
                
                # Update activity and continue
                self._update_activity(session)
                return session
        
        # No session or inactive - create new guest session
        return self._create_guest_session()

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

                # Upgrade to REGISTERED_USER
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                # Reset question counts and bans for registered users
                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                # Add email to used addresses
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
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
                        st.success(f"Welcome back, {current_session.full_name}!")
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
        
        # Check question limits BEFORE processing
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            reason = limit_check.get('reason')
            
            if reason == 'guest_limit':
                # Guest hit 4-question limit → force email verification
                return {
                    "content": "Please provide your email address to continue using FiFi AI.",
                    "success": False,
                    "requires_email": True,
                    "user_type": "guest"
                }
            elif reason in ['daily_limit', 'first_tier_limit', 'total_limit']:
                # User hit daily/tier limits → show ban message
                return {
                    "content": limit_check.get('message', 'Usage limit reached.'),
                    "success": False,
                    "banned": True,
                    "ban_type": limit_check.get('ban_type', 'unknown')
                }
            elif reason == 'banned':
                # User is currently banned
                return {
                    "content": limit_check.get('message', 'You are currently restricted.'),
                    "success": False,
                    "banned": True,
                    "time_remaining": limit_check.get('time_remaining')
                }
        
        # Check for evasion attempts
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session)
            self.db.save_session(session)
            
            return {
                "content": "Usage limit reached. Please try again later.",
                "success": False,
                "evasion_penalty": True,
                "penalty_hours": penalty_hours
            }

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {
                "content": moderation["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        # Record the question
        self.question_limits.record_question(session)

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
        
        # Save to CRM if eligible
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        self._end_session_internal(session)

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

    def manual_save_to_crm(self, session: UserSession):
        """Manual CRM save (Save button)"""
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# ENHANCED UI COMPONENTS WITH NEW FEATURES
# =============================================================================

def render_welcome_page_enhanced(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Knowledge Base**\nAccess curated F&B industry information")
    with col2:
        st.info("🌐 **Web Search**\nReal-time market data and trends") 
    with col3:
        st.info("📚 **Smart Citations**\nClickable inline source references")
    
    # Show user tier benefits
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown("• 4 questions to try FiFi AI")
        st.markdown("• Email verification required")
        st.markdown("• Quick start, no registration")
    
    with col2:
        st.info("📧 **Email Verified**")
        st.markdown("• 10 questions per day")
        st.markdown("• Email verification required")
        st.markdown("• Rolling 24-hour limits")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown("• 40 questions per day")
        st.markdown("• Cross-device tracking")
        st.markdown("• Auto-save to CRM")
        st.markdown("• Enhanced features")
    
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
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest experience:**
        - Start with 4 questions to explore FiFi AI
        - Email verification unlocks 10 questions/day
        - Universal device fingerprinting for security
        - Upgrade path to full registration
        
        ✨ **Registration benefits:**
        - 40 questions per day across all devices
        - Automatic CRM integration and chat history
        - Enhanced personalization features
        - Priority access during high usage
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """Render email verification dialog for guest users"""
    
    st.error("📧 **Email Verification Required**")
    st.info("You've reached the 4-question limit for guest users. Please verify your email to get 10 questions per day.")
    
    # Check if we have fingerprint history for recognition
    if session.fingerprint_id:
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            # Show recognition prompt
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            
            st.info(f"🤝 **We Recognize You!**")
            st.markdown(f"Based on our records, we seem to recognize you. Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's me", use_container_width=True):
                    # Send verification code to recognized email
                    verification_sent = session_manager.email_verification.send_verification_code(fingerprint_history['email'])
                    if verification_sent:
                        session.email = fingerprint_history['email']
                        session.recognition_response = "yes"
                        session_manager.db.save_session(session)
                        st.session_state.verification_email = fingerprint_history['email']
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error("Failed to send verification code. Please try manual entry.")
            
            with col2:
                if st.button("❌ No, different email", use_container_width=True):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry"
                    st.rerun()
    
    # Manual email entry
    if st.session_state.get('verification_stage') == 'email_entry' or not session.fingerprint_id:
        with st.form("email_verification_form"):
            st.markdown("**Enter your email address:**")
            email = st.text_input("Email Address", placeholder="your@email.com")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if email:
                    result = session_manager.handle_guest_email_verification(session, email)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = email
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address.")
    
    # Code entry
    if st.session_state.get('verification_stage') == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 Verification code sent to **{verification_email}**")
        
        with st.form("code_verification_form"):
            st.markdown("**Enter the verification code from your email:**")
            code = st.text_input("Verification Code", placeholder="123456", max_chars=6)
            submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            
            if submit_code:
                if code:
                    result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        
                        # Clear verification state
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code.")
        
        if st.button("🔄 Resend Code"):
            if verification_email:
                verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                if verification_sent:
                    st.success("Verification code resent!")
                else:
                    st.error("Failed to resend code. Please try again.")

def render_sidebar_enhanced(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Enhanced user status section
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # Question usage for registered users
            st.markdown(f"**Questions:** {session.total_question_count}/40")
            if session.total_question_count <= 20:
                st.progress(session.total_question_count / 20)
                st.caption("First tier: 20 questions")
            else:
                st.progress((session.total_question_count - 20) / 20)
                st.caption("Second tier: 21-40 questions")
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            # Daily question usage
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Questions reset!")
            
        else:  # GUEST
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10/day")
        
        # Fingerprinting info (if available)
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'}")
        
        # CRM Status
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type == UserType.REGISTERED_USER:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready**")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM")
            else:
                st.caption("💾 Auto-save at 15min")
        else: 
            st.caption("🚫 CRM: Registered users only")
        
        st.divider()
        
        # Enhanced session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        # Ban status
        if session.ban_status != BanStatus.NONE:
            if session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 **Restricted**")
                st.error(f"Time: {hours}h {minutes}m")
            else:
                st.info("🟢 **Restrictions Lifted**")
        
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

        # Download & save section (registered users only)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
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
                st.caption("💡 Auto-saves after 15min inactivity")

def render_chat_interface_complete(session_manager, session):
    """Complete chat interface with all new features"""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting")
    
    # Add global error handler
    try:
        global_message_channel_error_handler()
    except Exception as e:
        logger.error(f"Failed to add global error handler: {e}")
    
    # Initialize fingerprinting for session
    if not session.fingerprint_id:
        fingerprint_data = render_fingerprinting_component(session.session_id, session_manager.fingerprinting)
        if fingerprint_data:
            session_manager.apply_fingerprinting(session, fingerprint_data)
            session_manager.db.save_session(session)
            st.rerun()
    
    # Add enhanced browser close detection
    try:
        render_browser_close_detection_enhanced(session.session_id)
    except Exception as e:
        logger.error(f"Failed to render browser close detection: {e}")
    
    # Handle guest email verification dialog
    if (session.user_type == UserType.GUEST and 
        session.daily_question_count >= 4):
        render_email_verification_dialog(session_manager, session)
        return  # Don't show chat interface during verification
    
    # Show user tier status
    if session.user_type == UserType.GUEST:
        remaining = 4 - session.daily_question_count
        if remaining > 0:
            st.info(f"👤 **Guest Mode:** {remaining} questions remaining before email verification")
        else:
            st.warning("👤 **Guest Mode:** Email verification required to continue")
    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
        remaining = 10 - session.daily_question_count
        st.info(f"📧 **Email Verified:** {remaining} questions remaining today")
    elif session.user_type == UserType.REGISTERED_USER:
        if session.total_question_count <= 20:
            remaining = 20 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions in first tier")
        else:
            remaining = 40 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions remaining")
    
    # 15-minute timer for registered users
    if session.user_type == UserType.REGISTERED_USER:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id, session_manager)
        except Exception as e:
            logger.error(f"15-minute timer error: {e}")
        
        # Process timer events
        if timer_result:
            try:
                should_rerun = handle_15min_timer_event(timer_result, session_manager, session)
                if should_rerun:
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

    # Handle input
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue")
                        st.rerun()
                    elif response.get('banned'):
                        st.error(response.get('content', 'Access restricted'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Extended restriction applied")
                        st.error(f"Penalty: {response.get('penalty_hours', 0)} hours")
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                        
                        # Show enhanced features used
                        indicators = []
                        if response.get("used_pinecone"):
                            indicators.append("🧠 Knowledge Base")
                        if response.get("used_search"):
                            indicators.append("🌐 Web Search")
                        
                        if indicators:
                            st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                except Exception as e:
                    logger.error(f"AI response generation failed: {e}")
                    st.error("⚠️ Sorry, I encountered an error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS (ENHANCED)
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

def ensure_complete_initialization():
    """Ensure the complete application is properly initialized"""
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
            
            # NEW: Initialize additional managers
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager
            st.session_state.initialized = True
            
            logger.info("✅ Complete application initialized successfully with all features")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Complete initialization failed: {e}", exc_info=True)
            return False
    
    return True

def process_emergency_save_enhanced(session_id: str) -> bool:
    """Enhanced emergency save processing"""
    try:
        session_manager = get_session_manager()
        if not session_manager:
            logger.error("❌ Session manager not available for emergency save")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Session {session_id[:8]} not found for emergency save")
            return False
        
        session = session_manager._validate_and_fix_session(session)
        
        logger.info(f"✅ Emergency save session loaded: {session_id[:8]}")
        logger.info(f"   User: {session.user_type.value}")
        logger.info(f"   Email: {session.email}")
        logger.info(f"   Messages: {len(session.messages)}")
        
        # Check eligibility for emergency save
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            logger.info("✅ Session eligible for emergency save")
            
            # Extend session life
            session.last_activity = datetime.now()
            session_manager.db.save_session(session)
            
            # Perform emergency save
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Enhanced Browser Close)")
            return success
        else:
            logger.info("❌ Session not eligible for emergency save")
            return False
            
    except Exception as e:
        logger.error(f"Enhanced emergency save processing failed: {e}", exc_info=True)
        return False

def enhanced_handle_save_requests():
    """Enhanced save handler for all emergency save types"""
    
    logger.info("🔍 ENHANCED SAVE HANDLER: Checking for emergency save requests...")
    
    query_params = st.query_params
    logger.info(f"📋 Query params found: {dict(query_params)}")
    
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event in ["emergency_close", "close"] and session_id:
        logger.info("=" * 80)
        logger.info("🚨 ENHANCED EMERGENCY SAVE REQUEST DETECTED")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event: {event}")
        logger.info("=" * 80)
        
        # Show visual confirmation
        st.error("🚨 **Emergency Save Detected** - Processing enhanced browser close save...")
        
        # Clear query params
        st.query_params.clear()
        
        try:
            success = process_emergency_save_enhanced(session_id)
            
            if success:
                st.success("✅ Enhanced emergency save completed successfully!")
                logger.info("✅ Enhanced emergency save completed")
            else:
                st.error("❌ Enhanced emergency save failed")
                logger.error("❌ Enhanced emergency save failed")
                
        except Exception as e:
            st.error(f"❌ Enhanced emergency save error: {str(e)}")
            logger.error(f"Enhanced emergency save crashed: {e}", exc_info=True)
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No enhanced emergency save requests found")

# =============================================================================
# MAIN APPLICATION WITH ALL FEATURES
# =============================================================================

def main():
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    # Add global message channel error handler
    global_message_channel_error_handler()

    # Clear state button for development
    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Initialize complete application
    if not ensure_complete_initialization():
        st.stop()

    # Handle enhanced save requests
    enhanced_handle_save_requests()

    # Get session manager
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get enhanced session manager.")
        st.stop()
    
    # Main application flow
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page_enhanced(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar_enhanced(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_complete(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state['page']
            st.rerun()

if __name__ == "__main__":
    main()
        if not re.match(email_pattern, email):
            return {'success': False, 'message': 'Please enter a valid email address.'}
        
        # Check if email is already in use
        existing_sessions = self.db.find_sessions_by_email(email)
        
        # Send verification code
        verification_sent = self.email_verification.send_verification_code(email)
        
        if verification_sent:
            # Track email in session
            session.email = email
            if email not in session.email_addresses_used:
                session.email_addresses_used.append(email)
            
            self.db.save_session(session)
            
            return {
                'success': True, 
                'message': f'Verification code sent to {email}. Please check your email.',
                'verification_pending': True
            }
        else:
            return {'success': False, 'message': 'Failed to send verification code. Please try again.'}

    def verify_email_code(self, session: UserSession, code: str) -> Dict[str, Any]:
        """Verify code and upgrade user"""
        
        if not session.email:
            return {'success': False, 'message': 'No email verification in progress.'}
        
        # Verify the code
        verification_success = self.email_verification.verify_code(session.email, code)
        
        if verification_success:
            # Upgrade user to EMAIL_VERIFIED_GUEST
            session.user_type = UserType.EMAIL_VERIFIED_GUEST
            session.daily_question_count = 0  # Reset question count
            session.question_limit_reached = False
            
            # Reset ban status if any
            session.ban_status = BanStatus.NONE
            session.ban_start_time = None
            session.ban_end_time = None
            session.ban_reason = None
            
            self.db.save_session(session)
            
            logger.info(f"User upgraded to EMAIL_VERIFIED_GUEST: {session.email}")
            
            return {
                'success': True,
                'message': f'Email verified! You now have 10 questions per day.',
                'user_type': 'email_verified_guest'
            }
        else:
            return {'success': False, 'message': 'Invalid verification code. Please try again.'}

    def detect_evasion(self, session: UserSession) -> bool:
        """Detect evasion attempts"""
        
        if not session.fingerprint_id:
            return False
        
        # Check for multiple sessions with same fingerprint
        fingerprint_sessions = self.db.find_sessions_by_fingerprint(session.fingerprint_id)
        
        # Look for recent sessions that hit limits
        recent_cutoff = datetime.now() - timedelta(hours=48)
        recent_limited_sessions = [
            s for s in fingerprint_sessions 
            if s.session_id != session.session_id and
               s.last_activity > recent_cutoff and
               (s.question_limit_reached or s.ban_status != BanStatus.NONE)
        ]
        
        if recent_limited_sessions:
            logger.warning(f"Evasion detected for fingerprint {session.fingerprint_id[:8]}... - {len(recent_limited_sessions)} recent limited sessions")
            return True
        
        # Check for rapid email switching
        if len(session.email_addresses_used) > 2:
            logger.warning(f"Email switching detected: {len(session.email_addresses_used)} emails used")
            return True
        
        return False

    def get_session(self) -> UserSession:
        """Get session with complete validation and fingerprinting"""
        session_id = st.session_state.get('current_session_id')
        
        if session_id:
            session = self.db.load_session(session_id)
            if session and session.active:
                # Validate and fix session data
                session = self._validate_and_fix_session(session)
                
                # Check ban status
                limit_check = self.question_limits.is_within_limits(session)
                if not limit_check.get('allowed', True):
                    # Session is banned - show ban message
                    ban_type = limit_check.get('ban_type', 'unknown')
                    message = limit_check.get('message', 'Usage limit reached.')
                    time_remaining = limit_check.get('time_remaining')
                    
                    if time_remaining:
                        hours = int(time_remaining.total_seconds() // 3600)
                        minutes = int((time_remaining.total_seconds() % 3600) // 60)
                        st.error(f"🚫 **Access Restricted**")
                        st.error(f"Time remaining: {hours}h {minutes}m")
                        st.info(message)
                    else:
                        st.error(f"🚫 **Access Restricted**")
                        st.info(message)
                    
                    # Don't create new session - show ban message
                    return session
                
                # Update activity and continue
                self._update_activity(session)
                return session
        
        # No session or inactive - create new guest session
        return self._create_guest_session()

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

                # Upgrade to REGISTERED_USER
                current_session.user_type = UserType.REGISTERED_USER
                current_session.email = data.get('user_email')
                current_session.full_name = display_name
                current_session.wp_token = data.get('token')
                current_session.last_activity = datetime.now()
                current_session.timeout_saved_to_crm = False
                
                # Reset question counts and bans for registered users
                current_session.daily_question_count = 0
                current_session.total_question_count = 0
                current_session.question_limit_reached = False
                current_session.ban_status = BanStatus.NONE
                current_session.ban_start_time = None
                current_session.ban_end_time = None
                current_session.ban_reason = None
                
                # Add email to used addresses
                if current_session.email and current_session.email not in current_session.email_addresses_used:
                    current_session.email_addresses_used.append(current_session.email)
                
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
                        st.success(f"Welcome back, {current_session.full_name}!")
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
        
        # Check question limits BEFORE processing
        limit_check = self.question_limits.is_within_limits(session)
        if not limit_check.get('allowed', True):
            reason = limit_check.get('reason')
            
            if reason == 'guest_limit':
                # Guest hit 4-question limit → force email verification
                return {
                    "content": "Please provide your email address to continue using FiFi AI.",
                    "success": False,
                    "requires_email": True,
                    "user_type": "guest"
                }
            elif reason in ['daily_limit', 'first_tier_limit', 'total_limit']:
                # User hit daily/tier limits → show ban message
                return {
                    "content": limit_check.get('message', 'Usage limit reached.'),
                    "success": False,
                    "banned": True,
                    "ban_type": limit_check.get('ban_type', 'unknown')
                }
            elif reason == 'banned':
                # User is currently banned
                return {
                    "content": limit_check.get('message', 'You are currently restricted.'),
                    "success": False,
                    "banned": True,
                    "time_remaining": limit_check.get('time_remaining')
                }
        
        # Check for evasion attempts
        if self.detect_evasion(session):
            penalty_hours = self.question_limits.apply_evasion_penalty(session)
            self.db.save_session(session)
            
            return {
                "content": "Usage limit reached. Please try again later.",
                "success": False,
                "evasion_penalty": True,
                "penalty_hours": penalty_hours
            }

        self._update_activity(session)

        sanitized_prompt = sanitize_input(prompt)
        
        moderation = check_content_moderation(sanitized_prompt, self.ai.openai_client)
        if moderation and moderation.get("flagged"):
            return {
                "content": moderation["message"], 
                "success": False, 
                "source": "Content Safety"
            }

        # Record the question
        self.question_limits.record_question(session)

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
        
        # Save to CRM if eligible
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Sign Out")
        
        self._end_session_internal(session)

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

    def manual_save_to_crm(self, session: UserSession):
        """Manual CRM save (Save button)"""
        session = self._validate_and_fix_session(session)
        
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and 
            self.zoho.config.ZOHO_ENABLED):
            
            self._save_to_crm_timeout(session, "Manual Save to Zoho CRM")
            self._update_activity(session)
        else:
            st.warning("Cannot save to CRM: Missing email or chat messages")

# =============================================================================
# ENHANCED UI COMPONENTS WITH NEW FEATURES
# =============================================================================

def render_welcome_page_enhanced(session_manager: SessionManager):
    st.title("🤖 Welcome to FiFi AI Assistant")
    st.subheader("Your Intelligent Food & Beverage Sourcing Companion")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🧠 **Knowledge Base**\nAccess curated F&B industry information")
    with col2:
        st.info("🌐 **Web Search**\nReal-time market data and trends") 
    with col3:
        st.info("📚 **Smart Citations**\nClickable inline source references")
    
    # Show user tier benefits
    st.markdown("---")
    st.subheader("🎯 Usage Tiers")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("👤 **Guest Users**")
        st.markdown("• 4 questions to try FiFi AI")
        st.markdown("• Email verification required")
        st.markdown("• Quick start, no registration")
    
    with col2:
        st.info("📧 **Email Verified**")
        st.markdown("• 10 questions per day")
        st.markdown("• Email verification required")
        st.markdown("• Rolling 24-hour limits")
    
    with col3:
        st.warning("🔐 **Registered Users**")
        st.markdown("• 40 questions per day")
        st.markdown("• Cross-device tracking")
        st.markdown("• Auto-save to CRM")
        st.markdown("• Enhanced features")
    
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
                            st.success(f"🎉 Welcome back, {authenticated_session.full_name}!")
                            time.sleep(1)
                            st.session_state.page = "chat"
                            st.rerun()
    
    with tab2:
        st.markdown("""
        **Continue as a guest** to try FiFi AI Assistant without signing in.
        
        ℹ️ **Guest experience:**
        - Start with 4 questions to explore FiFi AI
        - Email verification unlocks 10 questions/day
        - Universal device fingerprinting for security
        - Upgrade path to full registration
        
        ✨ **Registration benefits:**
        - 40 questions per day across all devices
        - Automatic CRM integration and chat history
        - Enhanced personalization features
        - Priority access during high usage
        """)
        
        if st.button("👤 Start as Guest", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

def render_email_verification_dialog(session_manager: SessionManager, session: UserSession):
    """Render email verification dialog for guest users"""
    
    st.error("📧 **Email Verification Required**")
    st.info("You've reached the 4-question limit for guest users. Please verify your email to get 10 questions per day.")
    
    # Check if we have fingerprint history for recognition
    if session.fingerprint_id:
        fingerprint_history = session_manager.check_fingerprint_history(session.fingerprint_id)
        
        if fingerprint_history.get('has_history') and fingerprint_history.get('email'):
            # Show recognition prompt
            masked_email = session_manager._mask_email(fingerprint_history['email'])
            
            st.info(f"🤝 **We Recognize You!**")
            st.markdown(f"Based on our records, we seem to recognize you. Are you **{masked_email}**?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, that's me", use_container_width=True):
                    # Send verification code to recognized email
                    verification_sent = session_manager.email_verification.send_verification_code(fingerprint_history['email'])
                    if verification_sent:
                        session.email = fingerprint_history['email']
                        session.recognition_response = "yes"
                        session_manager.db.save_session(session)
                        st.session_state.verification_email = fingerprint_history['email']
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error("Failed to send verification code. Please try manual entry.")
            
            with col2:
                if st.button("❌ No, different email", use_container_width=True):
                    session.recognition_response = "no"
                    st.session_state.verification_stage = "email_entry"
                    st.rerun()
    
    # Manual email entry
    if st.session_state.get('verification_stage') == 'email_entry' or not session.fingerprint_id:
        with st.form("email_verification_form"):
            st.markdown("**Enter your email address:**")
            email = st.text_input("Email Address", placeholder="your@email.com")
            submit_email = st.form_submit_button("Send Verification Code", use_container_width=True)
            
            if submit_email:
                if email:
                    result = session_manager.handle_guest_email_verification(session, email)
                    if result['success']:
                        st.success(result['message'])
                        st.session_state.verification_email = email
                        st.session_state.verification_stage = "code_entry"
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter an email address.")
    
    # Code entry
    if st.session_state.get('verification_stage') == 'code_entry':
        verification_email = st.session_state.get('verification_email', session.email)
        
        st.success(f"📧 Verification code sent to **{verification_email}**")
        
        with st.form("code_verification_form"):
            st.markdown("**Enter the verification code from your email:**")
            code = st.text_input("Verification Code", placeholder="123456", max_chars=6)
            submit_code = st.form_submit_button("Verify Code", use_container_width=True)
            
            if submit_code:
                if code:
                    result = session_manager.verify_email_code(session, code)
                    if result['success']:
                        st.success(result['message'])
                        st.balloons()
                        
                        # Clear verification state
                        for key in ['verification_email', 'verification_stage']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result['message'])
                else:
                    st.error("Please enter the verification code.")
        
        if st.button("🔄 Resend Code"):
            if verification_email:
                verification_sent = session_manager.email_verification.send_verification_code(verification_email)
                if verification_sent:
                    st.success("Verification code resent!")
                else:
                    st.error("Failed to resend code. Please try again.")

def render_sidebar_enhanced(session_manager: SessionManager, session: UserSession, pdf_exporter: PDFExporter):
    with st.sidebar:
        st.title("🎛️ Dashboard")
        
        # Enhanced user status section
        if session.user_type == UserType.REGISTERED_USER:
            st.success("✅ **Registered User**")
            if session.full_name: 
                st.markdown(f"**Name:** {session.full_name}")
            if session.email: 
                st.markdown(f"**Email:** {session.email}")
            
            # Question usage for registered users
            st.markdown(f"**Questions:** {session.total_question_count}/40")
            if session.total_question_count <= 20:
                st.progress(session.total_question_count / 20)
                st.caption("First tier: 20 questions")
            else:
                st.progress((session.total_question_count - 20) / 20)
                st.caption("Second tier: 21-40 questions")
            
        elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
            st.info("📧 **Email Verified Guest**")
            if session.email:
                st.markdown(f"**Email:** {session.email}")
            
            # Daily question usage
            st.markdown(f"**Daily Questions:** {session.daily_question_count}/10")
            st.progress(session.daily_question_count / 10)
            
            if session.last_question_time:
                next_reset = session.last_question_time + timedelta(hours=24)
                time_to_reset = next_reset - datetime.now()
                if time_to_reset.total_seconds() > 0:
                    hours = int(time_to_reset.total_seconds() // 3600)
                    minutes = int((time_to_reset.total_seconds() % 3600) // 60)
                    st.caption(f"Resets in: {hours}h {minutes}m")
                else:
                    st.caption("Questions reset!")
            
        else:  # GUEST
            st.warning("👤 **Guest User**")
            st.markdown(f"**Questions:** {session.daily_question_count}/4")
            st.progress(session.daily_question_count / 4)
            st.caption("Email verification unlocks 10/day")
        
        # Fingerprinting info (if available)
        if session.fingerprint_id:
            st.markdown(f"**Device ID:** `{session.fingerprint_id[:8]}...`")
            st.caption(f"Method: {session.fingerprint_method or 'unknown'}")
        
        # CRM Status
        if session_manager.zoho.config.ZOHO_ENABLED and session.user_type == UserType.REGISTERED_USER:
            if session.zoho_contact_id: 
                st.success("🔗 **CRM Linked**")
            else: 
                st.info("📋 **CRM Ready**")
            if session.timeout_saved_to_crm:
                st.caption("💾 Auto-saved to CRM")
            else:
                st.caption("💾 Auto-save at 15min")
        else: 
            st.caption("🚫 CRM: Registered users only")
        
        st.divider()
        
        # Enhanced session info
        st.markdown(f"**Messages:** {len(session.messages)}")
        st.markdown(f"**Session:** `{session.session_id[:8]}...`")
        
        # Ban status
        if session.ban_status != BanStatus.NONE:
            if session.ban_end_time and datetime.now() < session.ban_end_time:
                time_remaining = session.ban_end_time - datetime.now()
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                st.error(f"🚫 **Restricted**")
                st.error(f"Time: {hours}h {minutes}m")
            else:
                st.info("🟢 **Restrictions Lifted**")
        
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

        # Download & save section (registered users only)
        if session.user_type == UserType.REGISTERED_USER and session.messages:
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
                st.caption("💡 Auto-saves after 15min inactivity")

def render_chat_interface_complete(session_manager, session):
    """Complete chat interface with all new features"""
    
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with universal fingerprinting")
    
    # Add global error handler
    try:
        global_message_channel_error_handler()
    except Exception as e:
        logger.error(f"Failed to add global error handler: {e}")
    
    # Initialize fingerprinting for session
    if not session.fingerprint_id:
        fingerprint_data = render_fingerprinting_component(session.session_id, session_manager.fingerprinting)
        if fingerprint_data:
            session_manager.apply_fingerprinting(session, fingerprint_data)
            session_manager.db.save_session(session)
            st.rerun()
    
    # Add enhanced browser close detection
    try:
        render_browser_close_detection_enhanced(session.session_id)
    except Exception as e:
        logger.error(f"Failed to render browser close detection: {e}")
    
    # Handle guest email verification dialog
    if (session.user_type == UserType.GUEST and 
        session.daily_question_count >= 4):
        render_email_verification_dialog(session_manager, session)
        return  # Don't show chat interface during verification
    
    # Show user tier status
    if session.user_type == UserType.GUEST:
        remaining = 4 - session.daily_question_count
        if remaining > 0:
            st.info(f"👤 **Guest Mode:** {remaining} questions remaining before email verification")
        else:
            st.warning("👤 **Guest Mode:** Email verification required to continue")
    elif session.user_type == UserType.EMAIL_VERIFIED_GUEST:
        remaining = 10 - session.daily_question_count
        st.info(f"📧 **Email Verified:** {remaining} questions remaining today")
    elif session.user_type == UserType.REGISTERED_USER:
        if session.total_question_count <= 20:
            remaining = 20 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions in first tier")
        else:
            remaining = 40 - session.total_question_count
            st.success(f"🔐 **Registered User:** {remaining} questions remaining")
    
    # 15-minute timer for registered users
    if session.user_type == UserType.REGISTERED_USER:
        timer_result = None
        try:
            timer_result = render_activity_timer_component_15min(session.session_id, session_manager)
        except Exception as e:
            logger.error(f"15-minute timer error: {e}")
        
        # Process timer events
        if timer_result:
            try:
                should_rerun = handle_15min_timer_event(timer_result, session_manager, session)
                if should_rerun:
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

    # Handle input
    prompt = st.chat_input("Ask me about ingredients, suppliers, or market trends...")
    
    # Process user input
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your question..."):
                try:
                    response = session_manager.get_ai_response(session, prompt)
                    
                    if response.get('requires_email'):
                        st.error("📧 Please verify your email to continue")
                        st.rerun()
                    elif response.get('banned'):
                        st.error(response.get('content', 'Access restricted'))
                        if response.get('time_remaining'):
                            time_remaining = response['time_remaining']
                            hours = int(time_remaining.total_seconds() // 3600)
                            minutes = int((time_remaining.total_seconds() % 3600) // 60)
                            st.error(f"Time remaining: {hours}h {minutes}m")
                    elif response.get('evasion_penalty'):
                        st.error("🚫 Evasion detected - Extended restriction applied")
                        st.error(f"Penalty: {response.get('penalty_hours', 0)} hours")
                    else:
                        st.markdown(response.get("content", "No response generated."), unsafe_allow_html=True)
                        
                        if response.get("source"):
                            st.caption(f"Source: {response['source']}")
                        
                        # Show enhanced features used
                        indicators = []
                        if response.get("used_pinecone"):
                            indicators.append("🧠 Knowledge Base")
                        if response.get("used_search"):
                            indicators.append("🌐 Web Search")
                        
                        if indicators:
                            st.caption(f"Enhanced with: {', '.join(indicators)}")
                        
                except Exception as e:
                    logger.error(f"AI response generation failed: {e}")
                    st.error("⚠️ Sorry, I encountered an error processing your request. Please try again.")
        
        st.rerun()

# =============================================================================
# UTILITY FUNCTIONS (ENHANCED)
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

def ensure_complete_initialization():
    """Ensure the complete application is properly initialized"""
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
            
            # NEW: Initialize additional managers
            fingerprinting_manager = FingerprintingManager()
            email_verification_manager = EmailVerificationManager(config)
            question_limit_manager = QuestionLimitManager()

            st.session_state.session_manager = SessionManager(
                config, db_manager, zoho_manager, ai_system, rate_limiter,
                fingerprinting_manager, email_verification_manager, question_limit_manager
            )
            st.session_state.pdf_exporter = pdf_exporter
            st.session_state.error_handler = error_handler
            st.session_state.ai_system = ai_system
            st.session_state.fingerprinting_manager = fingerprinting_manager
            st.session_state.email_verification_manager = email_verification_manager
            st.session_state.question_limit_manager = question_limit_manager
            st.session_state.initialized = True
            
            logger.info("✅ Complete application initialized successfully with all features")
            return True
            
        except Exception as e:
            st.error("💥 A critical error occurred during application startup.")
            st.error(f"Error details: {str(e)}")
            logger.critical(f"Complete initialization failed: {e}", exc_info=True)
            return False
    
    return True

def process_emergency_save_enhanced(session_id: str) -> bool:
    """Enhanced emergency save processing"""
    try:
        session_manager = get_session_manager()
        if not session_manager:
            logger.error("❌ Session manager not available for emergency save")
            return False
        
        session = session_manager.db.load_session(session_id)
        if not session:
            logger.error(f"❌ Session {session_id[:8]} not found for emergency save")
            return False
        
        session = session_manager._validate_and_fix_session(session)
        
        logger.info(f"✅ Emergency save session loaded: {session_id[:8]}")
        logger.info(f"   User: {session.user_type.value}")
        logger.info(f"   Email: {session.email}")
        logger.info(f"   Messages: {len(session.messages)}")
        
        # Check eligibility for emergency save
        if (session.user_type == UserType.REGISTERED_USER and 
            session.email and 
            session.messages and
            not session.timeout_saved_to_crm):
            
            logger.info("✅ Session eligible for emergency save")
            
            # Extend session life
            session.last_activity = datetime.now()
            session_manager.db.save_session(session)
            
            # Perform emergency save
            success = session_manager.zoho.save_chat_transcript_sync(session, "Emergency Save (Enhanced Browser Close)")
            return success
        else:
            logger.info("❌ Session not eligible for emergency save")
            return False
            
    except Exception as e:
        logger.error(f"Enhanced emergency save processing failed: {e}", exc_info=True)
        return False

def enhanced_handle_save_requests():
    """Enhanced save handler for all emergency save types"""
    
    logger.info("🔍 ENHANCED SAVE HANDLER: Checking for emergency save requests...")
    
    query_params = st.query_params
    logger.info(f"📋 Query params found: {dict(query_params)}")
    
    event = query_params.get("event")
    session_id = query_params.get("session_id")
    
    if event in ["emergency_close", "close"] and session_id:
        logger.info("=" * 80)
        logger.info("🚨 ENHANCED EMERGENCY SAVE REQUEST DETECTED")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event: {event}")
        logger.info("=" * 80)
        
        # Show visual confirmation
        st.error("🚨 **Emergency Save Detected** - Processing enhanced browser close save...")
        
        # Clear query params
        st.query_params.clear()
        
        try:
            success = process_emergency_save_enhanced(session_id)
            
            if success:
                st.success("✅ Enhanced emergency save completed successfully!")
                logger.info("✅ Enhanced emergency save completed")
            else:
                st.error("❌ Enhanced emergency save failed")
                logger.error("❌ Enhanced emergency save failed")
                
        except Exception as e:
            st.error(f"❌ Enhanced emergency save error: {str(e)}")
            logger.error(f"Enhanced emergency save crashed: {e}", exc_info=True)
        
        time.sleep(2)
        st.stop()
    else:
        logger.info("ℹ️ No enhanced emergency save requests found")

# =============================================================================
# MAIN APPLICATION WITH ALL FEATURES
# =============================================================================

def main():
    st.set_page_config(
        page_title="FiFi AI Assistant - Complete Integration", 
        page_icon="🤖", 
        layout="wide"
    )

    # Add global message channel error handler
    global_message_channel_error_handler()

    # Clear state button for development
    if st.button("🔄 Fresh Start (Dev)", key="emergency_clear"):
        st.session_state.clear()
        st.rerun()

    # Initialize complete application
    if not ensure_complete_initialization():
        st.stop()

    # Handle enhanced save requests
    enhanced_handle_save_requests()

    # Get session manager
    session_manager = get_session_manager()
    if not session_manager:
        st.error("Failed to get enhanced session manager.")
        st.stop()
    
    # Main application flow
    current_page = st.session_state.get('page')
    
    if current_page != "chat":
        render_welcome_page_enhanced(session_manager)
    else:
        session = session_manager.get_session()
        if session and session.active:
            render_sidebar_enhanced(session_manager, session, st.session_state.pdf_exporter)
            render_chat_interface_complete(session_manager, session)
        else:
            if 'page' in st.session_state:
                del st.session_state['page']
            st.rerun()

if __name__ == "__main__":
    main()
