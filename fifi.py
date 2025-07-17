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

    def _is_token_limit_error(self, error: Exception) -> bool:
        """Check if the error is related to token limits (150K tokens)."""
        error_str = str(error).lower()
        token_error_indicators = [
            "token limit", "token_limit", "tokens exceeded", "context length",
            "context_length", "maximum context", "150k", "150000", "token count"
        ]
        return any(indicator in error_str for indicator in token_error_indicators)

    def query(self, chat_history: List[HumanMessage]) -> Dict[str, Any]:
        if not self.assistant: 
            return {"content": "Pinecone assistant not available.", "success": False, "source": "error", "error_type": "unavailable"}
        
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
                "response_length": len(content),
                "error_type": None
            }
            
        except Exception as e:
            logger.error(f"Pinecone Assistant error: {str(e)}")
            
            # Check if it's a token limit error
            if self._is_token_limit_error(e):
                return {"content": "Token limit reached in knowledge base.", "success": False, "source": "error", "error_type": "token_limit"}
            else:
                return {"content": "Error querying knowledge base.", "success": False, "source": "error", "error_type": "general"}

class DirectTavilySearch:
    """Direct Tavily search without OpenAI orchestration - handles both general and 12taste-only searches."""
    
    def __init__(self, tavily_api_key: str):
        if not TAVILY_AVAILABLE:
            raise ImportError("Tavily client not available.")
        self.client = TavilyClient(api_key=tavily_api_key)

    def _add_utm_to_links(self, content: str) -> str:
        """Add UTM parameters to links."""
        def replacer(match):
            url = match.group(1)
            utm_params = "utm_source=12taste.com&utm_medium=fifi-chat"
            if '?' in url:
                new_url = f"{url}&{utm_params}"
            else:
                new_url = f"{url}?{utm_params}"
            return f"({new_url})"
        return re.sub(r'(?<=\])\(([^)]+)\)', replacer, content)

    def _synthesize_search_results(self, results, query: str, search_type: str = "general") -> str:
        """Synthesize search results into a coherent response similar to LLM output."""
        
        # Handle string response from Tavily
        if isinstance(results, str):
            return f"Based on my search: {results}"
        
        # Handle dictionary response from Tavily (most common format)
        if isinstance(results, dict):
            # Check if there's a pre-made answer
            if results.get('answer'):
                prefix = "Based on 12taste.com" if search_type == "12taste_only" else "Based on my web search"
                return f"{prefix}: {results['answer']}"
            
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
                            # Add UTM tracking
                            if '?' in url:
                                url += '&utm_source=12taste.com&utm_medium=fifi-chat'
                            else:
                                url += '?utm_source=12taste.com&utm_medium=fifi-chat'
                            sources.append(f"[{title}]({url})")
            
            if not relevant_info:
                return "I found search results but couldn't extract readable content. Please try rephrasing your query."
            
            # Build synthesized response
            response_parts = []
            prefix = "Based on 12taste.com" if search_type == "12taste_only" else "Based on my web search"
            
            if len(relevant_info) == 1:
                response_parts.append(f"{prefix}: {relevant_info[0]}")
            else:
                response_parts.append(f"{prefix}, here's what I found:")
                for i, info in enumerate(relevant_info, 1):
                    response_parts.append(f"\n\n**{i}.** {info}")
            
            # Add sources
            if sources:
                response_parts.append(f"\n\n**Sources:**")
                for i, source in enumerate(sources, 1):
                    response_parts.append(f"\n{i}. {source}")
            
            return "".join(response_parts)
        
        # Fallback for unknown formats
        return "I couldn't find any relevant information for your query."

    def search_general(self, query: str) -> Dict[str, Any]:
        """Search web with competitor exclusions."""
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_raw_content=False,
                exclude_domains=DEFAULT_EXCLUDED_DOMAINS
            )
            
            content = self._synthesize_search_results(response, query, "general")
            final_content = self._add_utm_to_links(content)
            
            return {"content": final_content, "success": True, "source": "FiFi Web Search"}
            
        except Exception as e:
            return {"content": f"Search error: {str(e)}", "success": False, "source": "error"}

    def search_12taste_only(self, query: str) -> Dict[str, Any]:
        """Search only 12taste.com domain."""
        try:
            domain_query = f"site:12taste.com {query}"
            response = self.client.search(
                query=domain_query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=False
            )
            
            content = self._synthesize_search_results(response, query, "12taste_only")
            final_content = self._add_utm_to_links(content)
            
            return {"content": final_content, "success": True, "source": "12taste.com Search"}
            
        except Exception as e:
            return {"content": f"12taste.com search error: {str(e)}", "success": False, "source": "error"}

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
            # Check if we should use web search
            use_search = self._should_use_web_search(prompt)
            search_results = ""
            search_attempted = False
            search_error = None
            
            # Debug info
            logger.info(f"Query: {prompt[:50]}...")
            logger.info(f"Should use search: {use_search}")
            logger.info(f"Tavily available: {TAVILY_AVAILABLE}")
            logger.info(f"Tavily API key configured: {bool(config.TAVILY_API_KEY)}")
            logger.info(f"Tavily client initialized: {bool(self.tavily_client)}")
            
            if use_search and self.tavily_client:
                search_attempted = True
                try:
                    search_results = self._search_web(prompt)
                    logger.info(f"Search results length: {len(search_results)}")
                except Exception as e:
                    search_error = str(e)
                    logger.error(f"Search failed: {e}")
                    search_results = f"Search error: {e}"
            
            # Enhanced F&B domain prompt
            enhanced_prompt = f"""You are FiFi, an AI assistant specializing in food & beverage sourcing and ingredients. 
You help F&B professionals find suppliers, understand market trends, source ingredients, and navigate the food industry.

User Question: {prompt}"""

            if search_results:
                enhanced_prompt += f"\n\nWeb Search Results:\n{search_results}\n\nPlease incorporate this search information into your response when relevant."
            else:
                enhanced_prompt += "\n\nPlease provide helpful, specific advice relevant to the food & beverage industry based on your knowledge."
            
            # Get AI response
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                
            elif self.langchain_llm:
                from langchain_core.messages import HumanMessage
                response = self.langchain_llm.invoke([HumanMessage(content=enhanced_prompt)])
                content = response.content
                
            else:
                content = "AI services are not available. Please configure your OpenAI API key."
            
            # Determine source and add debug info
            if search_results and not search_error:
                source = "FiFi AI + Web Search"
                # Add debug info to response
                content += f"\n\n---\n🔍 **Search Debug Info:**\n- Search triggered: {use_search}\n- Search attempted: {search_attempted}\n- Results length: {len(search_results)}"
            elif search_attempted and search_error:
                source = "FiFi AI (Search Failed)"
                content += f"\n\n---\n❌ **Search Failed:**\n- Error: {search_error}\n- Falling back to AI knowledge"
            elif use_search and not self.tavily_client:
                source = "FiFi AI (Search Unavailable)"
                content += f"\n\n---\n⚠️ **Search Status:**\n- Search triggered but Tavily client not available\n- Tavily available: {TAVILY_AVAILABLE}\n- API key configured: {bool(config.TAVILY_API_KEY)}"
            else:
                source = "FiFi AI"
                content += f"\n\n---\n📝 **Response Info:**\n- Search triggered: {use_search}\n- Using knowledge-based response"
                
            return {
                "content": content,
                "source": source,
                "used_search": bool(search_results and not search_error),
                "search_attempted": search_attempted,
                "search_error": search_error,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Enhanced AI response error: {e}")
            return {
                "content": f"I'm having trouble processing your request: {str(e)}",
                "source": "Error",
                "used_search": False,
                "search_attempted": False,
                "search_error": str(e),
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
        
        # Search info and testing
        if TAVILY_AVAILABLE and config.TAVILY_API_KEY:
            with st.expander("🔍 Search Features & Testing"):
                st.write("**Competitor Exclusions:**")
                st.write(f"- {len(DEFAULT_EXCLUDED_DOMAINS)} domains filtered")
                st.write("- Focus on relevant suppliers")
                st.write("- Excludes marketplaces & competitors")
                
                st.write("**Smart Search Triggers:**")
                st.write("- Supplier/company queries")
                st.write("- Price & market information")
                st.write("- Current trends & news")
                st.write("- 'Find', 'locate', 'where to buy'")
                
                st.divider()
                
                # Manual search test
                st.write("**🧪 Test Search:**")
                test_query = st.text_input("Test search query:", placeholder="organic vanilla suppliers")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 Test General Search"):
                        if test_query:
                            with st.spinner("Testing search..."):
                                try:
                                    ai = st.session_state.ai
                                    search_result = ai._search_web(test_query, "general")
                                    st.success("✅ Search worked!")
                                    st.text_area("Results:", search_result, height=200)
                                except Exception as e:
                                    st.error(f"❌ Search failed: {e}")
                
                with col2:
                    if st.button("🎯 Test 12taste Search"):
                        if test_query:
                            with st.spinner("Testing 12taste search..."):
                                try:
                                    ai = st.session_state.ai
                                    search_result = ai._search_web(test_query, "12taste_only")
                                    st.success("✅ 12taste search worked!")
                                    st.text_area("Results:", search_result, height=200)
                                except Exception as e:
                                    st.error(f"❌ 12taste search failed: {e}")
        else:
            with st.expander("❌ Search Not Available"):
                st.write("**Status Check:**")
                st.write(f"- Tavily package: {'✅' if TAVILY_AVAILABLE else '❌'}")
                st.write(f"- API key configured: {'✅' if config.TAVILY_API_KEY else '❌'}")
                
                if not TAVILY_AVAILABLE:
                    st.warning("Install tavily-python package")
                
                if not config.TAVILY_API_KEY:
                    st.warning("Add TAVILY_API_KEY to secrets")
                    st.code("TAVILY_API_KEY = 'your-api-key-here'")
        
        # Configuration status
        st.subheader("🔧 API Configuration")
        st.write(f"**OpenAI:** {'✅ Configured' if config.OPENAI_API_KEY else '❌ Missing'}")
        st.write(f"**Tavily:** {'✅ Configured' if config.TAVILY_API_KEY else '❌ Missing'}")
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
