# FiFi AI Chat Assistant

An enterprise-grade AI-powered chat assistant for B2B food & beverage ingredient sourcing, built with Streamlit. Features WordPress authentication, multi-tier user management, AI-powered responses with knowledge base retrieval, and comprehensive CRM integration.

## Features

### AI & Knowledge Retrieval
- **OpenAI GPT-4o-mini** for intelligent responses and content moderation
- **Pinecone Vector Database** for knowledge base querying
- **Tavily Web Search** fallback when knowledge base unavailable
- **Industry Context Validation** - specialized for food & beverage sourcing
- **Meta-conversation Detection** - distinguishes user questions about the chat from industry queries

### User Management System

| User Type | Daily Questions | Reset Period |
|-----------|----------------|--------------|
| Guest | 4 | Per session |
| Email Verified Guest | 10 | 24 hours |
| Registered User (Tier 1) | 10 | 1-hour break |
| Registered User (Tier 2) | +10 (20 total) | 24 hours |

### Session & Security
- **Device Fingerprinting** - recognizes returning users across sessions
- **Ban Evasion Detection** - escalating penalties (72-hour block)
- **Rate Limiting** - 2 requests per 60 seconds
- **Content Moderation** - OpenAI moderation API integration
- **Session Recovery** - by email or fingerprint

### Integrations
- **WordPress** - JWT-based authentication with fallback mode
- **WooCommerce** - Order status lookup and customer queries
- **Zoho CRM** - Contact management and chat transcript archival
- **Supabase** - Email OTP verification
- **SQLite Cloud** - Distributed session storage (with local fallback)

### Additional Features
- PDF export of chat transcripts
- Emergency save on page unload
- 15-minute session timeout with auto-save
- Chat history persistence and recovery
- Multi-device session synchronization

## Deployment

### Google Cloud Run (Production)

The app is configured for Cloud Run deployment with Docker and Nginx reverse proxy.

```bash
gcloud run deploy fifi-eu-prod \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated
```

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fifi-eu-prod.git
   cd fifi-eu-prod
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure secrets**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Edit with your actual values
   ```

4. **Run the app**
   ```bash
   streamlit run fifi.py
   ```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for AI responses and moderation |
| `JWT_SECRET` | Secret key for WordPress JWT authentication |
| `WORDPRESS_URL` | WordPress site URL (include protocol) |

### Database

| Variable | Description |
|----------|-------------|
| `SQLITE_CLOUD_CONNECTION` | SQLite Cloud connection string (optional, falls back to local SQLite) |

### Search & Knowledge Base

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Pinecone vector database API key |
| `PINECONE_ASSISTANT_NAME` | Pinecone assistant name (default: "my-chat-assistant") |
| `TAVILY_API_KEY` | Tavily web search API key |

### Email Verification (Supabase)

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anonymous key |

### WooCommerce Integration

| Variable | Description |
|----------|-------------|
| `WOOCOMMERCE_URL` | WooCommerce REST API base URL |
| `WOOCOMMERCE_CONSUMER_KEY` | WooCommerce API consumer key |
| `WOOCOMMERCE_CONSUMER_SECRET` | WooCommerce API consumer secret |

### Zoho CRM Integration

| Variable | Description |
|----------|-------------|
| `ZOHO_CLIENT_ID` | Zoho OAuth client ID |
| `ZOHO_CLIENT_SECRET` | Zoho OAuth client secret |
| `ZOHO_REFRESH_TOKEN` | Zoho OAuth refresh token |

## Configuration Constants

Key configuration values in `production_config.py`:

```python
# Question Limits
GUEST_QUESTION_LIMIT = 4
EMAIL_VERIFIED_QUESTION_LIMIT = 10
REGISTERED_USER_QUESTION_LIMIT = 20
REGISTERED_USER_TIER_1_LIMIT = 10

# Ban Durations
TIER_1_BAN_HOURS = 1
TIER_2_BAN_HOURS = 24
EMAIL_VERIFIED_BAN_HOURS = 24
EVASION_BAN_HOURS = 72

# Session Settings
SESSION_TIMEOUT_MINUTES = 15
DAILY_RESET_WINDOW_HOURS = 24
FINGERPRINT_TIMEOUT_SECONDS = 20

# Rate Limiting
RATE_LIMIT_REQUESTS = 2
RATE_LIMIT_WINDOW_SECONDS = 60

# Content Limits
MAX_MESSAGE_LENGTH = 1000
MAX_PDF_MESSAGES = 50
CRM_SAVE_MIN_QUESTIONS = 2
```

## Architecture

```
fifi.py (Main Application - 8000+ lines)
├── Config                    # Environment & secrets management
├── UserSession               # User state and session data
├── DatabaseManager           # SQLite Cloud / local SQLite
│   ├── FingerprintingManager # Device fingerprinting
│   ├── EmailVerificationManager # Supabase OTP
│   └── QuestionLimitManager  # Quota enforcement
├── SessionManager            # Main orchestrator
│   ├── WordPress Auth        # JWT authentication
│   ├── WooCommerceManager    # Order queries
│   └── ZohoCRMManager        # CRM integration
├── EnhancedAI                # AI response orchestration
│   ├── Pinecone Assistant    # Knowledge base
│   └── Tavily Fallback       # Web search
├── PDFExporter               # Chat transcript PDF
└── UI Functions              # Streamlit interface
```

## WordPress Setup

1. Install **JWT Authentication for WP REST API** plugin
2. Add to `wp-config.php`:
   ```php
   define('JWT_AUTH_SECRET_KEY', 'your-secret-key');
   define('JWT_AUTH_CORS_ENABLE', true);
   ```
3. Configure allowed origins for your deployment domain

## Docker Configuration

The app uses a multi-stage Docker build with Nginx:

- **Base**: Python 3.12-slim
- **Internal Port**: 8501 (Streamlit)
- **External Port**: 8080 (Nginx)
- **Health Check**: `/healthz` endpoint

## License

MIT License
