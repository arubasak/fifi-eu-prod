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
        
        # --- START OF NEW (v2) STRICT MODERATION LOGIC ---

        # Check the user's input for policy violations
        moderation_result = check_content_moderation(prompt, st.session_state.ai.openai_client)

        # First, handle cases where the prompt should be blocked
        if moderation_result["flagged"] or moderation_result["check_failed"]:
            # If content is flagged or the check failed, display the blocking message and stop
            with st.chat_message("assistant"):
                st.error(f"🚨 {moderation_result['message']}") # Use st.error for high visibility
            
            # Add the moderation response to chat history
            session.messages.append({
                "role": "assistant",
                "content": moderation_result['message'],
                "source": "Content Safety Policy",
                "timestamp": datetime.now().isoformat()
            })
            st.rerun()

        else:
            # If content is safe, proceed to get the AI response
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

                    # Allow HTML to render clickable citations
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
