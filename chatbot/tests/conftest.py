import os
import sys

os.environ.setdefault("DB_NAME", "testdb")
os.environ.setdefault("DB_USER", "testuser")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test-client-secret")
os.environ.setdefault("GOOGLE_REDIRECT_URI", "http://testhost/oauth2callback")
os.environ.setdefault("SESSION_SECRET", "test-session-secret")
os.environ.setdefault("ALLOWED_EMAILS", "allowed@example.com")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
