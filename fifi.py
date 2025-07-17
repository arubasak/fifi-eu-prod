import streamlit as st
import os
import uuid
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with graceful fallbacks
OPENAI_AVAILABLE = False
LANGCHAIN_AVAILABLE = False
SQLITECLOUD_AVAILABLE = False
TAVILY_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlitecloud
    SQLITECLOUD_AVAILABLE = True
except ImportError:
    pass

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    pass

# Competition exclusion list for web searches
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

# Simple configuration
class Config:
    def __init__(self):
        self.JWT_SECRET = st.secrets.get("JWT_SECRET", "default-secret")
        self.WORDPRESS_URL = st.secrets.get("WORDPRESS_URL", "https://example.com")
        self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        self.SQLITE_CLOUD_CONNECTION = st.secrets.get("SQLITE_CLOUD_CONNECTION")
        self.TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

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

class EnhancedAI:
    """Enhanced AI interface with web search and F&B domain expertise."""
    
    def __init__(self):
        self.openai_client = None
        self.langchain_llm = None
        self.tavily_client = None
        
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
        
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            try:
                self.tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
            except Exception as e:
                logger.error(f"Tavily client initialization failed: {e}")
    
    def _search_web(self, query: str, search_type: str = "general") -> str:
        """Perform web search with domain filtering."""
        try:
            if not self.tavily_client:
                return "Web search not available."
            
            if search_type == "12taste_only":
                # Search only 12taste.com domain
                domain_query = f"site:12taste.com {query}"
                response = self.tavily_client.search(
                    query=domain_query,
                    search_depth="advanced",
                    max_results=3,
                    include_answer=True,
                    include_raw_content=False
                )
                source_label = "12taste.com"
            else:
                # General search with competitor exclusions
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced", 
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False,
                    exclude_domains=DEFAULT_EXCLUDED_DOMAINS
                )
                source_label = "Web Search"
            
            # Format results
            results = []
            if response.get('answer'):
                results.append(f"**Summary:** {response['answer']}")
            
            if response.get('results'):
                results.append("\n**Sources:**")
                for i, result in enumerate(response['results'][:3], 1):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    results.append(f"{i}. **{title}**: {content[:200]}... ([Source]({url}))")
            
            return f"\n🔍 **{source_label} Results:**\n" + "\n".join(results)
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search encountered an error: {str(e)}"
    
    def _should_use_web_search(self, prompt: str) -> bool:
        """Determine if the query would benefit from web search."""
        search_indicators = [
            "find", "search", "suppliers", "companies", "latest", "current", 
            "price", "cost", "market", "trends", "news", "available",
            "where to buy", "who sells", "contact", "locate"
        ]
        return any(indicator in prompt.lower() for indicator in search_indicators)
    
    def get_response(self, prompt: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get enhanced AI response with web search capability."""
        try:
            # Enhanced F&B domain prompt
            enhanced_prompt = f"""You are FiFi, an AI assistant specializing in food & beverage sourcing and ingredients. 
You help F&B professionals find suppliers, understand market trends, source ingredients, and navigate the food industry.

User Question: {prompt}

Please provide helpful, specific advice relevant to the food & beverage industry. If you need current market information or supplier details, I can search the web for you."""

            # Check if we should use web search
            use_search = self._should_use_web_search(prompt)
            search_results = ""
            
            if use_search and self.tavily_client:
                search_results = self._search_web(prompt)
                enhanced_prompt += f"\n\nWeb Search Results:\n{search_results}\n\nPlease incorporate this search information into your response when relevant."
            
            # Get AI response
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                source = "FiFi AI + Web Search" if search_results else "FiFi AI"
                
            elif self.langchain_llm:
                from langchain_core.messages import HumanMessage
                response = self.langchain_llm.invoke([HumanMessage(content=enhanced_prompt)])
                content = response.content
                source = "FiFi AI + Web Search" if search_results else "FiFi AI"
                
            else:
                content = "AI services are not available. Please configure your OpenAI API key."
                source = "Error"
            
            return {
                "content": content,
                "source": source,
                "used_search": bool(search_results),
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Enhanced AI response error: {e}")
            return {
                "content": "I'm having trouble processing your request right now. Please try again.",
                "source": "Error",
                "used_search": False,
                "success": False
            }

def init_session_state():
    """Initialize session state safely."""
    if 'initialized' not in st.session_state:
        try:
            # Simple initialization without complex detection
            st.session_state.session_manager = SimpleSessionManager()
            st.session_state.ai = EnhancedAI()  # Updated to use EnhancedAI
            st.session_state.page = "chat"  # Start directly in chat
            st.session_state.initialized = True
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Session state initialization failed: {e}")
            st.session_state.initialized = False

def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion")
    
    session = st.session_state.session_manager.get_session()
    
    # Display chat history
    for msg in session.messages:
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))
            if "source" in msg:
                st.caption(f"Source: {msg['source']}")
            if msg.get("used_search"):
                st.caption("🔍 Enhanced with web search")
    
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
            with st.spinner("Searching and analyzing..."):
                response = st.session_state.ai.get_response(prompt, session.messages)
                
                # Handle enhanced response format
                if isinstance(response, dict):
                    content = response.get("content", "No response generated.")
                    source = response.get("source", "Unknown")
                    used_search = response.get("used_search", False)
                else:
                    # Fallback for simple string responses
                    content = str(response)
                    source = "FiFi AI"
                    used_search = False
                
                st.markdown(content)
                
                # Show search indicator
                if used_search:
                    st.success("🔍 Enhanced with real-time web search")
        
        # Add AI response to history
        session.messages.append({
            "role": "assistant",
            "content": content,
            "source": source,
            "used_search": used_search,
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
        st.write(f"**Web Search:** {'✅' if TAVILY_AVAILABLE and config.TAVILY_API_KEY else '❌'}")
        st.write(f"**SQLite Cloud:** {'✅' if SQLITECLOUD_AVAILABLE else '❌'}")
        
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
        st.write("✅ OpenAI Integration")
        
        if LANGCHAIN_AVAILABLE:
            st.write("✅ LangChain Support")
        else:
            st.write("❌ LangChain (install required)")
            
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            st.write("✅ Web Search (Competitor-Filtered)")
            st.write("✅ 12taste.com Search")
        else:
            st.write("❌ Web Search (API key needed)")
        
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
        
        # Search info
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            with st.expander("🔍 Search Features"):
                st.write("**Competitor Exclusions:**")
                st.write(f"- {len(DEFAULT_EXCLUDED_DOMAINS)} domains filtered")
                st.write("- Focus on relevant suppliers")
                st.write("- Excludes marketplaces & competitors")
                
                st.write("**Smart Search Triggers:**")
                st.write("- Supplier/company queries")
                st.write("- Price & market information")
                st.write("- Current trends & news")
                st.write("- 'Find', 'locate', 'where to buy'")

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
