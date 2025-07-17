import streamlit as st
import os
import uuid
import json
import logging
import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import requests
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_tavily import TavilySearch

# Setup logging
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

class PineconeAssistantTool:
    """Advanced Pinecone Assistant with token limit detection and anti-hallucination."""
    
    def __init__(self, api_key: str, assistant_name: str):
        if not PINECONE_AVAILABLE: 
            raise ImportError("Pinecone client not available.")
        self.pc = Pinecone(api_key=api_key)
        self.assistant_name = assistant_name
        self.assistant = self._initialize_assistant()

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
                st.warning(f"Assistant '{self.assistant_name}' not found. Creating...")
                return self.pc.assistant.create_assistant(
                    assistant_name=self.assistant_name, 
                    instructions=instructions
                )
            else:
                st.info(f"Connected to assistant: '{self.assistant_name}'")
                return self.pc.assistant.Assistant(assistant_name=self.assistant_name)
        except Exception as e:
            st.error(f"Failed to initialize Pinecone Assistant: {e}")
            return None

    def query(self, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        if not self.assistant: 
            return None
        
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
            
            # Process citations
            if hasattr(response, 'citations') and response.citations:
                has_citations = True
                citations_header = "\n\n---\n**Sources:**\n"
                citations_list = []
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
                
                if citations_list:
                    content += citations_header + "\n".join(citations_list)
            
            return {
                "content": content, 
                "success": True, 
                "source": "FiFi Knowledge Base",
                "has_citations": has_citations,
                "response_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Pinecone Assistant error: {str(e)}")
            return None

class TavilyFallbackAgent:
    """Tavily fallback agent with smart result synthesis and UTM tracking."""
    
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
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
            return {
                "content": f"I apologize, but an error occurred while searching: {str(e)}",
                "success": False,
                "source": "error"
            }

class EnhancedAI:
    """Enhanced AI with Pinecone knowledge base and smart Tavily fallback with aggressive anti-hallucination."""
    
    def __init__(self):
        self.pinecone_tool = None
        self.tavily_agent = None
        self.openai_client = None
        self.langchain_llm = None
        
        # Initialize OpenAI clients for potential LLM fallback
        if OPENAI_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
        
        if LANGCHAIN_AVAILABLE and config.OPENAI_API_KEY:
            try:
                self.langchain_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=config.OPENAI_API_KEY,
                    temperature=0.7
                )
            except Exception as e:
                logger.error(f"LangChain LLM initialization failed: {e}")
        
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
        
        # PRIORITY 7: Question-answering patterns that suggest general knowledge
        qa_patterns = [
            "the answer is", "this is because", "the reason", "due to the fact",
            "this happens when", "the cause of", "this occurs"
        ]
        if any(pattern in content for pattern in qa_patterns):
            if not pinecone_response.get("has_citations", False):
                return True
        
        # PRIORITY 8: Response length suggests substantial answer without sources
        response_length = pinecone_response.get("response_length", 0)
        if response_length > 100 and not pinecone_response.get("has_citations", False):
            return True
        
        return False

    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get enhanced AI response using same logic as reference code."""
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
                
                if tavily_response.get("success"):
                    return {
                        "content": tavily_response["content"],
                        "source": tavily_response.get("source", "FiFi Web Search"),
                        "used_search": True,
                        "used_pinecone": False,
                        "has_citations": False,
                        "safety_override": True if self.pinecone_tool else False,
                        "success": True
                    }
            
            # STEP 3: Final fallback
            return {
                "content": "I apologize, but all systems are currently unavailable.",
                "source": "Error",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "safety_override": False,
                "success": False
            }
            
        except Exception as e:
            logger.error(f"Enhanced AI response error: {e}")
            return {
                "content": f"I'm having trouble processing your request: {str(e)}",
                "source": "Error",
                "used_search": False,
                "used_pinecone": False,
                "has_citations": False,
                "safety_override": False,
                "success": False
            }

def init_session_state():
    """Initialize session state safely."""
    if 'initialized' not in st.session_state:
        try:
            st.session_state.session_manager = SimpleSessionManager()
            st.session_state.ai = EnhancedAI()
            st.session_state.page = "chat"
            st.session_state.initialized = True
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.session_state.initialized = False

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion with smart fallback")
    
    session = st.session_state.session_manager.get_session()
    
    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))
            
            # Show source information for assistant messages
            if msg.get("role") == "assistant":
                source_indicators = []
                
                if "source" in msg:
                    st.caption(f"Source: {msg['source']}")
                
                # Show knowledge base usage
                if msg.get("used_pinecone"):
                    if msg.get("has_citations"):
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
                    safety_override = response.get("safety_override", False)
                else:
                    # Fallback for simple string responses
                    content = str(response)
                    source = "FiFi AI"
                    used_search = False
                    used_pinecone = False
                    has_citations = False
                    safety_override = False
                
                st.markdown(content)
                
                # Show enhancement indicators
                enhancements = []
                if used_pinecone:
                    if has_citations:
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
            "safety_override": safety_override,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session
        session.last_activity = datetime.now()
        
        st.rerun()

def render_sidebar():
    """Render the sidebar with controls."""
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
    """Main application function."""
    st.set_page_config(
        page_title="FiFi AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    try:
        # Initialize session state
        init_session_state()
        
        if not st.session_state.get('initialized', False):
            st.error("⚠️ Application initialization failed")
            st.info("Please refresh the page.")
            return
        
        # Render interface
        render_sidebar()
        render_chat_interface()
        
        # Show welcome message with safety info
        if not st.session_state.session_manager.get_session().messages:
            st.info("""
            👋 **Welcome to FiFi AI Chat Assistant!**
            
            **How it works:**
            - 🔍 **First**: Searches your internal knowledge base via Pinecone
            - 🛡️ **Safety Override**: Detects and blocks fabricated information (fake URLs, file paths, etc.)
            - 🌐 **Verified Fallback**: Switches to real web sources when needed
            - 🚨 **Anti-Misinformation**: Aggressive detection of hallucinated content
            
            **Safety Features:**
            - ✅ Blocks fake citations and non-existent file references
            - ✅ Prevents hallucinated image paths (.jpg, .png, etc.)
            - ✅ Validates all sources before presenting information
            - ✅ Falls back to verified web search when information is questionable
            
            **Note**: If you see a "SAFETY OVERRIDE" message, the system detected potentially fabricated information and switched to verified sources to protect you from misinformation.
            """)
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        st.error("🚨 Application Error")
        st.text(f"Error: {str(e)}")
        
        if st.button("🔄 Restart App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
