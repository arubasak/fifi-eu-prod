# production_config.py
"""
Production configuration - DO NOT MODIFY WITHOUT APPROVAL
Last updated: September 12, 2025
Version: 1.0.0
"""

from datetime import timedelta

# Timing Configuration (in appropriate units)
DAILY_RESET_WINDOW_HOURS = 24
SESSION_TIMEOUT_MINUTES = 5
FINGERPRINT_TIMEOUT_SECONDS = 20

# Ban Durations (in hours)
TIER_1_BAN_HOURS = 1
TIER_2_BAN_HOURS = 24
EMAIL_VERIFIED_BAN_HOURS = 24

# Question Limits
GUEST_QUESTION_LIMIT = 4
EMAIL_VERIFIED_QUESTION_LIMIT = 10
REGISTERED_USER_QUESTION_LIMIT = 20
REGISTERED_USER_TIER_1_LIMIT = 10  # When tier 1 break triggers

# Rate Limiting
RATE_LIMIT_REQUESTS = 2
RATE_LIMIT_WINDOW_SECONDS = 60

# System Limits
MAX_MESSAGE_LENGTH = 4000
MAX_PDF_MESSAGES = 500 # Maximum messages to include in PDF to prevent memory issues
MAX_FINGERPRINT_CACHE_SIZE = 10000
MAX_RATE_LIMIT_TRACKING = 10000
MAX_ERROR_HISTORY = 100

# CRM Configuration
CRM_SAVE_MIN_QUESTIONS = 1  # Minimum questions before CRM save (no longer checks duration)

# Evasion Penalties (hours)
EVASION_BAN_HOURS = [1, 2, 4, 8, 24]

# FastAPI Integration
FASTAPI_EMERGENCY_SAVE_URL = 'https://fifi-beacon-fastapi-121263692901.europe-west4.run.app/emergency-save'
FASTAPI_EMERGENCY_SAVE_TIMEOUT = 5 # seconds
