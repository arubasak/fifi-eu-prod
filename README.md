# FiFi AI Chat Assistant

An intelligent chat assistant built with Streamlit, featuring WordPress authentication, AI-powered responses, and CRM integration.

## Features

- 🤖 AI-powered chat with Pinecone knowledge base and Tavily web search
- 🔐 WordPress JWT authentication 
- 📊 SQLite Cloud database for session management
- 📄 PDF export of chat transcripts
- 🔄 Zoho CRM integration
- 🌐 Multi-platform deployment support

## Deployment

### Streamlit Community Cloud

1. **Fork this repository**
2. **Connect to Streamlit Community Cloud**
3. **Configure secrets** in the Streamlit dashboard:
   - Go to your app settings
   - Add all variables from `.streamlit/secrets.toml.example`

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fifi-chat-assistant.git
   cd fifi-chat-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Required Environment Variables

### Essential (Required)
- `JWT_SECRET`: Secret key for WordPress JWT authentication
- `OPENAI_API_KEY`: OpenAI API key for AI responses
- `WORDPRESS_URL`: Your WordPress site URL
- `SQLITE_CLOUD_CONNECTION`: SQLite Cloud connection string

### Optional (Enhanced Features)
- `PINECONE_API_KEY`: For knowledge base queries
- `TAVILY_API_KEY`: For web search capabilities
- `ZOHO_CLIENT_ID`, `ZOHO_CLIENT_SECRET`, `ZOHO_REFRESH_TOKEN`: For CRM integration

## WordPress Setup

1. Install JWT Authentication plugin
2. Add CORS configuration to `wp-config.php`
3. Configure allowed origins for your deployment

See the documentation for detailed WordPress setup instructions.

## License

MIT License
