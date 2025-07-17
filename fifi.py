def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 FiFi AI Assistant")
    st.caption("Your intelligent food & beverage sourcing companion")
    
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
                    source_indicators.append("🔍 Direct Search")
                
                if source_indicators:
                    st.caption(f"Enhanced with: {', '.join(source_indicators)}")
    
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
            with st.spinner("Searching knowledge base and web..."):
                response = st.session_state.ai.get_response(prompt, session.messages)
                
                # Handle enhanced response format
                if isinstance(response, dict):
                    content = response.get("content", "No response generated.")
                    source = response.get("source", "Unknown")
                    used_search = response.get("used_search", False)
                    used_pinecone = response.get("used_pinecone", False)
                    pinecone_result = response.get("pinecone_result", {})
                    has_citations = response.get("has_citations", False)
                else:
                    # Fallback for simple string responses
                    content = str(response)
                    source = "FiFi AI"
                    used_search = False
                    used_pinecone = False
                    pinecone_result = {}
                    has_citations = False
                
                st.markdown(content)
                
                # Show enhancement indicators
                enhancements = []
                if used_pinecone:
                    if has_citations:
                        enhancements.append("🧠 Enhanced with Knowledge Base (with citations)")
                    else:
                        enhancements.append("🧠 Enhanced with Knowledge Base")
                
                if used_search:
                    enhancements.append("🔍 Enhanced with direct Tavily search")
                
                # Show specific error messages for debugging
                if pinecone_result.get("error_type") == "token_limit":
                    st.warning("⚠️ Knowledge base token limit reached - using direct search fallback")
                elif pinecone_result.get("error_type") == "general":
                    st.warning("⚠️ Knowledge base temporarily unavailable - using direct search fallback")
                
                if enhancements:
                    for enhancement in enhancements:
                        st.success(enhancement)
        
        # Add AI response to history
        session.messages.append({
            "role": "assistant",
            "content": content,
            "source": source,
            "used_search": used_search,
            "used_pinecone": used_pinecone,
            "has_citations": has_citations,
            "pinecone_result": pinecone_result,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update session
        session.last_activity = datetime.now()
        
        st.rerun()
