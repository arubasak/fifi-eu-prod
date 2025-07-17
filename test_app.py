import streamlit as st
import os
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="Test App", page_icon="🧪")
    
    st.title("🧪 Minimal Test App")
    
    try:
        # Test basic functionality
        st.write(f"✅ Streamlit version: {st.__version__}")
        st.write(f"✅ Python version: {os.sys.version}")
        
        # Test session state
        if 'counter' not in st.session_state:
            st.session_state.counter = 0
        
        if st.button("Test Button"):
            st.session_state.counter += 1
        
        st.write(f"✅ Button clicks: {st.session_state.counter}")
        
        # Test environment detection
        env_vars = [key for key in os.environ.keys() if 'STREAMLIT' in key.upper()]
        st.write(f"✅ Streamlit env vars: {env_vars}")
        
        # Test basic chat
        if prompt := st.chat_input("Test chat input"):
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                st.write(f"Echo: {prompt}")
        
        st.success("🎉 All basic tests passed!")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
