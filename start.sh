#!/bin/sh

# This script starts two processes: Nginx and Streamlit.

# Start the Nginx server in the background.
nginx

# Start the Streamlit application in the foreground.
# --server.port=8501 tells Streamlit to listen on the internal port for Nginx.
# --server.enableCORS=false is important because Nginx is now handling all
# external traffic, so Streamlit doesn't need its own CORS protection.
streamlit run fifi.py --server.port=8501 --server.enableCORS=false
