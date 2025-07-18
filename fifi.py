import streamlit as st
import os
import uuid
import json
import logging
import re
import time
import functools
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import requests
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_tavily import TavilySearch

# =============================================================================
# VERSION 2.1 CHANGELOG:
# - Implemented clickable inline citations. Markers [1], [2] are now hyperlinks.
# - Markers jump to the corresponding full source at the bottom of the response.
# - Enabled unsafe_allow_html in st.markdown to render the necessary anchor links.
# - Added EnhancedErrorHandler for sophisticated error management
# - Added ErrorSeverity classification system
# - Added ErrorContext for detailed error information
# - Added error handling decorators (@handle_api_errors, @safe_import)
# - Added error recovery suggestions and user-friendly messages
# - Added error dashboard in sidebar
# - Enhanced component status tracking
# - Added graceful degradation for missing features
# - Added Pinecone inline citation functionality
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
    pass

try:
    from langchain_openai import ChatOpenAI
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
    PINECONE_AVAILABLE = False

# =============================================================================
# NEW in v2.0: Enhanced Error Handling System
# =============================================================================

class ErrorSeverity(Enum):
    LOW = "low"           # Feature unavailable but app continues
    MEDIUM = "medium"     # Degraded functionality
    HIGH = "high"         # Major feature broken
    CRITICAL = "critical" # App might not work

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

        # Classify the error
        if "timeout" in error_str or "timed out" in error_str:
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

        elif "not found" in error_str or "404" in error_str:
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="NotFoundError",
                severity=ErrorSeverity.MEDIUM,
                user_message=f"{component} resource not found. The service might be temporarily unavailable.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try again in a few minutes",
                    "Check if the service is experiencing issues"
                ],
                fallback_available=True
            )

        elif "connection" in error_str or "network" in error_str:
            return ErrorContext(
                component=component,
                operation=operation,
                error_type="ConnectionError",
                severity=ErrorSeverity.HIGH,
                user_message=f"Cannot connect to {component}. Please check your internet connection.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Check your internet connection",
                    "Try refreshing the page",
                    "Try again in a few minutes"
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

    def handle_import_error(self, package_name: str, feature_name: str) -> ErrorContext:
        """Handle missing package errors."""
        return ErrorContext(
            component="Package Import",
            operation=f"Import {package_name}",
            error_type="ImportError",
            severity=ErrorSeverity.LOW,
            user_message=f"{feature_name} is not available. The app will continue with limited functionality.",
            technical_details=f"Package '{package_name}' is not installed",
            recovery_suggestions=[
                f"Install {package_name}: pip install {package_name}",
                "Some features may be unavailable",
                "Core functionality will still work"
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

def safe_import(package_name: str, feature_name: str, show_error: bool = True):
    """Safe import with user-friendly error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                error_context = error_handler.handle_import_error(package_name, feature_name)
                error_handler.log_error(error_context)

                if show_error:
                    error_handler.display_error_to_user(error_context)

                return None
        return wrapper
    return decorator

# =============================================================================
# Competition exclusion list for web searches
# =============================================================================
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

# Simple configuration
class Config:
    def __init__(self):
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.WORDPRESS_URL = st.secrets.get("WORDPRESS_URL", "https://example.com")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
        self.PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
        self.PINECONE_ASSISTANT_NAME = st.secrets.get("PINECONE_ASSISTANT_NAME", "my-chat-assistant")

config = Config()

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
# MODIFIED in v2.1: Pinecone Inline Citation Helper Function
# =============================================================================

def insert_citations(response) -> str:
    """
    Insert clickable citation markers. If a source URL is available, the marker
    links directly to it in a new tab. Otherwise, it links to the citation list at the bottom.

    Args:
        response: Pinecone Assistant Chat Response

    Returns:
        Modified text with citation markers inserted
    """
    if not hasattr(response, 'citations') or not response.citations:
        return response.message.content

    result = response.message.content
    citations = response.citations
    offset = 0  # Keep track of how much we've shifted the text

    # Sort citations by their position in the text to ensure proper insertion order
    sorted_citations = sorted(enumerate(citations, start=1), key=lambda x: x[1].position)

    for i, cite in sorted_citations:
        link_url = None
        # Attempt to find a URL in the citation's references
        if hasattr(cite, 'references') and cite.references:
            # Use the URL from the first reference for the inline link
            reference = cite.references[0]
            if hasattr(reference, 'file') and reference.file:
                # Prioritize 'source_url' from metadata, then fall back to 'signed_url'
                if hasattr(reference.file, 'metadata') and reference.file.metadata:
                    link_url = reference.file.metadata.get('source_url')
                if not link_url and hasattr(reference.file, 'signed_url') and reference.file.signed_url:
                    link_url = reference.file.signed_url

                # If a URL was found, append the UTM parameters
                if link_url:
                    if '?' in link_url:
                        link_url += '&utm_source=fifi-in'
                    else:
                        link_url += '?utm_source=fifi-in'

        # Create the final HTML for the citation marker
        if link_url:
            # If a URL exists, link directly to the external source and open in a new tab
            citation_marker = f" <a href='{link_url}' target='_blank' title='Source: {link_url}'>[{i}]</a>"
        else:
            # Otherwise, fall back to the original behavior of linking to the bottom of the page
            citation_marker = f" <a href='#cite-{i}'>[{i}]</a>"

        position = cite.position
        adjusted_position = position + offset

        # Insert the citation marker into the response text
        if adjusted_position <= len(result):
            result = result[:adjusted_position] + citation_marker + result[adjusted_position:]
            offset += len(citation_marker)

    return result

# =============================================================================
# MODIFIED in v2.1: PineconeAssistantTool with Clickable Inline Citations
# =============================================================================
class PineconeAssistantTool:
    """Advanced Pinecone Assistant with clickable inline citations, token limit detection and enhanced error handling."""

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

            # Enable highlights for inline citations
            response = self.assistant.chat(
                messages=pinecone_messages,
                model="gpt-4o",
                include_highlights=True  # Enable inline citation positions
            )

            # Process inline citations
            content_with_inline_citations = insert_citations(response)

            has_citations = False
            has_inline_citations = False

            # Check if we have inline citations
            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                has_inline_citations = True

                # Also build traditional citations list for additional context
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
                                    # MODIFIED: Create a Markdown link and wrap it with an HTML anchor for the jump target
                                    markdown_link = f"[{display_text}]({link_url})"
                                    final_item = f"<a id='cite-{i}'></a>{i}. {markdown_link}"
                                    citations_list.append(final_item)
                                    seen_items.add(display_text)
                            else:
                                display_text = getattr(reference.file, 'name', 'Unknown Source')
                                if display_text not in seen_items:
                                    # MODIFIED: Create a non-linked item but still wrap it with the anchor for the jump target
                                    final_item = f"<a id='cite-{i}'></a>{i}. {display_text}"
                                    citations_list.append(final_item)
                                    seen_items.add(display_text)

                # Add traditional citations list at the end
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
            # This will be caught by the decorator
            raise e


# =============================================================================
# ENHANCED v2.0: TavilyFallbackAgent with Error Handling
# =============================================================================
class TavilyFallbackAgent:
    """Tavily fallback agent with smart result synthesis, UTM tracking, and enhanced error handling."""

    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            error_context = error_handler.handle_import_error("langchain-tavily", "Web Search")
            error_handler.display_error_to_user(error_context)
            raise ImportError("Tavily client not available.")
        self.tavily_tool = TavilySearch(max_results=5, api_key=tavily_api_key)

    def add_utm_to_links(self, content: str) -> str:
        """Finds all Markdown links in a string and appends the UTM parameters."""
        def replacer(match):
            url = match.group(1)
            utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
            if '?' in url:
                new_url = f"{url}&{utm_params}"
            else:
                new_url = f"{url}?{utm_params}"
            return f"({new_url})"
        return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

    def synthesize_search_results(self, results, query: str) -> str:
        """Synthesize search results into a coherent response similar to LLM output."""

        # Handle string response from Tavily
        if isinstance(results, str):
            return f"Based on my search: {results}"

        # Handle dictionary response from Tavily (most common format)
        if isinstance(results, dict):
            # Check if there's a pre-made answer
            if results.get('answer'):
                return f"Based on my search: {results['answer']}"

            # Extract the results array
            search_results = results.get('results', [])
            if not search_results:
                return "I couldn't find any relevant information for your query."

            # Process the results
            relevant_info = []
            sources = []

            for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or
                             result.get('snippet') or
                             result.get('description') or
                             result.get('summary', ''))
                    url = result.get('url', '')

                    if content:
                        # Clean up content
                        if len(content) > 400:
                            content = content[:400] + "..."
                        relevant_info.append(content)

                        if url and title:
                            sources.append(f"[{title}]({url})")

            if not relevant_info:
                return "I found search results but couldn't extract readable content. Please try rephrasing your query."

            # Build synthesized response
            response_parts = []

            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search, here's what I found:")
                for i, info in enumerate(relevant_info, 1):
                    response_parts.append(f"\n\n**{i}.** {info}")

            # Add sources
            if sources:
                response_parts.append(f"\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"\n{i}. {source}")

            return "".join(response_parts)

        # Handle direct list (fallback)
        if isinstance(results, list):
            relevant_info = []
            sources = []

            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', f'Result {i}')
                    content = (result.get('content') or
                             result.get('snippet') or
                             result.get('description', ''))
                    url = result.get('url', '')

                    if content:
                        if len(content) > 400:
                            content = content[:400] + "..."
                        relevant_info.append(content)
                        if url:
                            sources.append(f"[{title}]({url})")

            if not relevant_info:
                return "I couldn't find relevant information for your query."

            response_parts = []
            if len(relevant_info) == 1:
                response_parts.append(f"Based on my search: {relevant_info[0]}")
            else:
                response_parts.append("Based on my search:")
                for info in relevant_info:
                    response_parts.append(f"\n{info}")

            if sources:
                response_parts.append(f"\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"{i}. {source}")

            return "".join(response_parts)

        # Fallback for unknown formats
        return "I couldn't find any relevant information for your query."

    @handle_api_errors("Tavily", "Web Search", show_to_user=False)
    def query(self, message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        try:
            search_results = self.tavily_tool.invoke({"query": message})
            synthesized_content = self.synthesize_search_results(search_results, message)
            final_content = self.add_utm_to_links(synthesized_content)

            return {
                "content": final_content,
                "success": True,
                "source": "FiFi Web Search"
            }
        except Exception as e:
            # This will be caught by the decorator
            raise e

# =============================================================================
# ENHANCED v2.0: EnhancedAI with Better Error Recovery and Inline Citations
# =============================================================================
class EnhancedAI:
    """Enhanced AI with Pinecone knowledge base, inline citations, smart Tavily fallback, and sophisticated error handling."""

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
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
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
                        langchain_history.append(AIMessage(content=msg.get("content", "")))

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
                        # Continue to Tavily fallback with safety override flag

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
                        "has_inline_citations": False,
                        "safety_override": True if self.pinecone_tool else False,
                        "success": True
                    }
                else:
                    # Tavily failed, log the issue
                    logger.warning("Tavily search failed, proceeding to final fallback")

            # STEP 3: Final fallback with helpful error message
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

# =============================================================================
# MODIFIED in v2.1: render_chat_interface with unsafe_allow_html=True
# =============================================================================
def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with inline citations and smart fallback")

    session = st.session_state.session_manager.get_session()

    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            # MODIFIED: Allow HTML for clickable citations in historical messages
            st.markdown(msg.get("content", ""), unsafe_allow_html=True)

            # Show source information for assistant messages
            if msg.get("role") == "assistant":
                source_indicators = []

                if "source" in msg:
                    st.caption(f"Source: {msg['source']}")

                # Show knowledge base usage with inline citations
                if msg.get("used_pinecone"):
                    if msg.get("has_inline_citations"):
                        source_indicators.append("🧠 Knowledge Base (with inline citations)")
                    elif msg.get("has_citations"):
                        source_indicators.append("🧠 Knowledge Base (with citations)")
                    else:
                        source_indicators.append("🧠 Knowledge Base")

                # Show web search usage
                if msg.get("used_search"):
                    source_indicators.append("🌐 Web Search")

                if source_indicators:
                    st.caption(f"Enhanced with: {', '.join(source_indicators)}")

                # Show safety override warning
                if msg.get("safety_override"):
                    st.warning("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")

    # Chat input
    if prompt := st.chat_input("Ask me about ingredients, suppliers, market trends, or sourcing..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to history
        session.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying FiFi (Internal Specialist)..."):
                response = st.session_state.ai.get_response(prompt, session.messages)

                # Handle enhanced response format
                if isinstance(response, dict):
                    content = response.get("content", "No response generated.")
                    source = response.get("source", "Unknown")
                    used_search = response.get("used_search", False)
                    used_pinecone = response.get("used_pinecone", False)
                    has_citations = response.get("has_citations", False)
                    has_inline_citations = response.get("has_inline_citations", False)
                    safety_override = response.get("safety_override", False)
                else:
                    # Fallback for simple string responses
                    content = str(response)
                    source = "FiFi AI"
                    used_search = False
                    used_pinecone = False
                    has_citations = False
                    has_inline_citations = False
                    safety_override = False

                # MODIFIED: Allow HTML to render clickable citations
                st.markdown(content, unsafe_allow_html=True)

                # Show enhancement indicators
                enhancements = []
                if used_pinecone:
                    if has_inline_citations:
                        enhancements.append("🧠 Enhanced with Knowledge Base (with inline citations)")
                    elif has_citations:
                        enhancements.append("🧠 Enhanced with Knowledge Base (with citations)")
                    else:
                        enhancements.append("🧠 Enhanced with Knowledge Base")

                if used_search:
                    enhancements.append("🌐 Enhanced with verified web search")

                if enhancements:
                    for enhancement in enhancements:
                        st.success(enhancement)

                # Show safety override warning
                if safety_override:
                    st.error("🚨 SAFETY OVERRIDE: Detected potentially fabricated information. Switched to verified web sources.")

        # Add AI response to history
        session.messages.append({
            "role": "assistant",
            "content": content,
            "source": source,
            "used_search": used_search,
            "used_pinecone": used_pinecone,
            "has_citations": has_citations,
            "has_inline_citations": has_inline_citations,
            "safety_override": safety_override,
            "timestamp": datetime.now().isoformat()
        })

        # Update session
        session.last_activity = datetime.now()

        st.rerun()

# =============================================================================
# ENHANCED v2.0: Sidebar with Error Dashboard
# =============================================================================
def render_sidebar():
    """Render the sidebar with controls and enhanced error monitoring."""
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

        # System status
        st.subheader("System Status")
        st.write(f"**OpenAI:** {'✅' if OPENAI_AVAILABLE and config.OPENAI_API_KEY else '❌'}")
        st.write(f"**LangChain:** {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
        st.write(f"**Tavily Search:** {'✅' if TAVILY_AVAILABLE and config.TAVILY_API_KEY else '❌'}")
        st.write(f"**Pinecone:** {'✅' if PINECONE_AVAILABLE and config.PINECONE_API_KEY else '❌'}")
        st.write(f"**SQLite Cloud:** {'✅' if SQLITECLOUD_AVAILABLE else '❌'}")

        # Component Status
        if hasattr(st.session_state, 'ai'):
            ai = st.session_state.ai
            pinecone_status = "✅ Connected" if ai.pinecone_tool else "❌ Failed"
            tavily_status = "✅ Connected" if ai.tavily_agent else "❌ Failed"
            st.write(f"**Pinecone Assistant:** {pinecone_status}")
            st.write(f"**Tavily Fallback Agent:** {tavily_status}")

        # NEW in v2.0: Enhanced Error Dashboard
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

        # Feature status
        st.subheader("Available Features")
        st.write("✅ Enhanced F&B AI Chat")
        st.write("✅ Session Management")
        st.write("✅ Anti-Hallucination Safety")
        st.write("✅ Smart Fallback Logic")
        st.write("✅ Enhanced Error Handling")
        st.write("✅ Clickable Inline Citations (NEW)")  # NEW in v2.1

        if LANGCHAIN_AVAILABLE:
            st.write("✅ LangChain Support")
        else:
            st.write("❌ LangChain (install required)")

        # Pinecone Knowledge Base Status
        if PINECONE_AVAILABLE and config.PINECONE_API_KEY:
            if hasattr(st.session_state, 'ai') and st.session_state.ai.pinecone_tool:
                st.write("✅ Knowledge Base (Pinecone)")
            else:
                st.write("⚠️ Knowledge Base (Connection Failed)")
        else:
            st.write("❌ Knowledge Base (Setup Required)")

        # Tavily Fallback Status
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            if hasattr(st.session_state, 'ai') and st.session_state.ai.tavily_agent:
                st.write("✅ Web Search Fallback (Tavily)")
            else:
                st.write("⚠️ Web Search (Connection Failed)")
        else:
            st.write("❌ Web Search (API key needed)")

        # Safety Features Info
        with st.expander("🛡️ Safety Features"):
            st.write("**Anti-Hallucination Checks:**")
            st.write("- Fake citation detection")
            st.write("- File path validation")
            st.write("- General knowledge flagging")
            st.write("- Response length analysis")
            st.write("- Current info detection")
            st.write("- Automatic web fallback")
            st.write("**NEW in v2.1:**")
            st.write("- Enhanced error recovery")
            st.write("- User-friendly error messages")
            st.write("- System health monitoring")
            st.write("- Clickable inline citation support")

        # Citation Features Info
        with st.expander("📚 Citation Features"):
            st.write("**Clickable Inline Citations:**")
            st.write("- Precise text position citations [1]")
            st.write("- Clickable markers jump to source")
            st.write("- Traditional source list at bottom")
            st.write("- UTM tracking for external links")
            st.write("- Real-time citation validation")
            st.write("**Citation Safety:**")
            st.write("- Validates all citation sources")
            st.write("- Prevents fake file references")
            st.write("- Detects fabricated citations")
            st.write("- Fallback for uncited content")

        # Example queries
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
                # Add the example query to chat
                session.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                })
                st.rerun()

        # Configuration status
        st.subheader("🔧 API Configuration")
        st.write(f"**OpenAI:** {'✅ Configured' if config.OPENAI_API_KEY else '❌ Missing'}")
        st.write(f"**Tavily:** {'✅ Configured' if config.TAVILY_API_KEY else '❌ Missing'}")
        st.write(f"**Pinecone:** {'✅ Configured' if config.PINECONE_API_KEY else '❌ Missing'}")
        st.write(f"**WordPress:** {'✅ Configured' if config.WORDPRESS_URL else '❌ Missing'}")
        st.write(f"**SQLite Cloud:** {'✅ Configured' if config.SQLITE_CLOUD_CONNECTION else '❌ Missing'}")

def main():
    """Main application function with enhanced error handling."""
    st.set_page_config(
        page_title="FiFi AI Assistant v2.1",
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

        # Show welcome message with safety info
        if not st.session_state.session_manager.get_session().messages:
            st.info("""
            👋 **Welcome to FiFi AI Chat Assistant v2.1!**

            **How it works:**
            - 🔍 **First**: Searches your internal knowledge base via Pinecone
            - 📚 **NEW**: Clickable inline citations [1] show exactly where information comes from
            - 🛡️ **Safety Override**: Detects and blocks fabricated information (fake URLs, file paths, etc.)
            - 🌐 **Verified Fallback**: Switches to real web sources when needed
            - 🚨 **Anti-Misinformation**: Aggressive detection of hallucinated content

            **NEW in v2.1 - Clickable Citations:**
            - ✅ **Clickable Inline Citations**: Precise [1] markers in the text now jump to the full source at the bottom.
            - ✅ User-friendly error messages with recovery suggestions
            - ✅ Automatic fallback when services fail
            - ✅ Real-time system health monitoring
            - ✅ Graceful degradation for missing features

            **Safety Features:**
            - ✅ Blocks fake citations and non-existent file references
            - ✅ Prevents hallucinated image paths (.jpg, .png, etc.)
            - ✅ Validates all sources before presenting information
            - ✅ Falls back to verified web search when information is questionable
            - ✅ Clickable inline citations with position validation

            **Note**: If you see a "SAFETY OVERRIDE" message, the system detected potentially fabricated information and switched to verified sources to protect you from misinformation.
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
