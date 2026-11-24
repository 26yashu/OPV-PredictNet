import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.security import (
    create_access_token,
    verify_password,
)
from backend.app.db.base import Base
import backend.app.models
from backend.app.models.user import User
from backend.app.schemas.user import UserRegisterRequest, UserLoginRequest
from backend.app.api.v1.auth import register, login, get_current_user_profile
from backend.app.api.deps import get_current_user
from fastapi.security import HTTPAuthorizationCredentials


class TestAuthenticationAndDatabase(unittest.TestCase):
    """Automated test suite for Phase 2: Database and Authentication."""

    def setUp(self):
        # Create a temporary SQLite database for testing isolation
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_successful_registration(self):
        """Verify user registration returns safe profile and sets correct database fields."""
        reg_in = UserRegisterRequest(
            email="solar.researcher@institute.org",
            password="SecurePassword2026!",
            full_name="Dr. Alex Rivera",
            institution="Solar Energy Institute"
        )
        user = register(reg_in, self.db)

        self.assertIsInstance(user.id, str)
        self.assertTrue(user.id.startswith("usr_"))
        self.assertEqual(user.email, "solar.researcher@institute.org")
        self.assertEqual(user.full_name, "Dr. Alex Rivera")
        self.assertEqual(user.institution, "Solar Energy Institute")
        self.assertTrue(user.is_active)
        self.assertIsNotNone(user.created_at)

    def test_duplicate_email_rejection(self):
        """Verify registering an existing email raises HTTP 400 Bad Request."""
        reg_in = UserRegisterRequest(
            email="duplicate@lab.org",
            password="SecurePassword2026!",
            full_name="User One"
        )
        register(reg_in, self.db)

        # Attempt to register with the same email (different case)
        duplicate_in = UserRegisterRequest(
            email="DUPLICATE@lab.org",
            password="AnotherPassword123!",
            full_name="User Two"
        )
        with self.assertRaises(HTTPException) as ctx:
            register(duplicate_in, self.db)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("already registered", ctx.exception.detail)

    def test_password_hashing_not_plaintext(self):
        """Verify that stored password is an Argon2id hash and never plaintext."""
        raw_password = "MyComplexPassword99!"
        reg_in = UserRegisterRequest(
            email="hash.test@lab.org",
            password=raw_password,
            full_name="Hash Check"
        )
        user = register(reg_in, self.db)

        # Query directly from database
        db_user = self.db.query(User).filter(User.id == user.id).first()
        self.assertIsNotNone(db_user)
        self.assertNotEqual(db_user.hashed_password, raw_password)
        self.assertTrue(db_user.hashed_password.startswith("$argon2id$"))

        # Verify hash matches through argon2 verification
        self.assertTrue(verify_password(raw_password, db_user.hashed_password))
        self.assertFalse(verify_password("WrongPassword123!", db_user.hashed_password))

    def test_successful_login(self):
        """Verify valid login returns a signed JWT bearer token with expiration."""
        reg_in = UserRegisterRequest(
            email="login.valid@lab.org",
            password="ValidPassword123!",
            full_name="Login Success"
        )
        register(reg_in, self.db)

        login_in = UserLoginRequest(
            email="login.valid@lab.org",
            password="ValidPassword123!"
        )
        token_resp = login(login_in, self.db)

        self.assertIsNotNone(token_resp.access_token)
        self.assertEqual(token_resp.token_type, "bearer")
        self.assertGreater(token_resp.expires_in, 0)

    def test_invalid_login(self):
        """Verify invalid credentials raise HTTP 401 Unauthorized with generic message."""
        reg_in = UserRegisterRequest(
            email="login.fail@lab.org",
            password="ValidPassword123!",
            full_name="Login Fail"
        )
        register(reg_in, self.db)

        # Case 1: Wrong password
        with self.assertRaises(HTTPException) as ctx1:
            login(UserLoginRequest(email="login.fail@lab.org", password="WrongPassword999!"), self.db)
        self.assertEqual(ctx1.exception.status_code, 401)
        self.assertEqual(ctx1.exception.detail, "Invalid email or password")

        # Case 2: Non-existent email
        with self.assertRaises(HTTPException) as ctx2:
            login(UserLoginRequest(email="nonexistent@lab.org", password="ValidPassword123!"), self.db)
        self.assertEqual(ctx2.exception.status_code, 401)
        self.assertEqual(ctx2.exception.detail, "Invalid email or password")

        # Case 3: Inactive user
        user = self.db.query(User).filter(User.email == "login.fail@lab.org").first()
        user.is_active = False
        self.db.commit()

        with self.assertRaises(HTTPException) as ctx3:
            login(UserLoginRequest(email="login.fail@lab.org", password="ValidPassword123!"), self.db)
        self.assertEqual(ctx3.exception.status_code, 401)

    def test_missing_jwt(self):
        """Verify missing authorization header or empty credentials raises HTTP 401."""
        with self.assertRaises(HTTPException) as ctx1:
            get_current_user(None, self.db)
        self.assertEqual(ctx1.exception.status_code, 401)

        empty_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        with self.assertRaises(HTTPException) as ctx2:
            get_current_user(empty_creds, self.db)
        self.assertEqual(ctx2.exception.status_code, 401)

    def test_invalid_jwt(self):
        """Verify malformed or expired JWT tokens raise HTTP 401."""
        # Malformed token
        bad_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token.payload")
        with self.assertRaises(HTTPException) as ctx1:
            get_current_user(bad_creds, self.db)
        self.assertEqual(ctx1.exception.status_code, 401)

        # Expired token (delta in the past)
        expired_token = create_access_token(
            subject="usr_123",
            expires_delta=timedelta(seconds=-10)
        )
        expired_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=expired_token)
        with self.assertRaises(HTTPException) as ctx2:
            get_current_user(expired_creds, self.db)
        self.assertEqual(ctx2.exception.status_code, 401)

    def test_authenticated_profile_retrieval(self):
        """Verify authenticated user can retrieve their safe profile via GET /me."""
        reg_in = UserRegisterRequest(
            email="profile.user@lab.org",
            password="ValidPassword123!",
            full_name="Profile User",
            institution="Quantum Electronics"
        )
        user = register(reg_in, self.db)

        token_resp = login(
            UserLoginRequest(email="profile.user@lab.org", password="ValidPassword123!"),
            self.db
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_resp.access_token)
        current_user = get_current_user(creds, self.db)

        profile = get_current_user_profile(current_user)
        self.assertEqual(profile.id, user.id)
        self.assertEqual(profile.email, "profile.user@lab.org")
        self.assertEqual(profile.full_name, "Profile User")
        self.assertEqual(profile.institution, "Quantum Electronics")
        self.assertTrue(profile.is_active)

    def test_user_isolation(self):
        """Verify tokens strictly isolate distinct user sessions."""
        user_a = register(
            UserRegisterRequest(email="usera@lab.org", password="PasswordA123!", full_name="User A"),
            self.db
        )
        user_b = register(
            UserRegisterRequest(email="userb@lab.org", password="PasswordB123!", full_name="User B"),
            self.db
        )

        token_a = login(UserLoginRequest(email="usera@lab.org", password="PasswordA123!"), self.db).access_token
        token_b = login(UserLoginRequest(email="userb@lab.org", password="PasswordB123!"), self.db).access_token

        creds_a = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_a)
        creds_b = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_b)

        resolved_a = get_current_user(creds_a, self.db)
        resolved_b = get_current_user(creds_b, self.db)

        self.assertEqual(resolved_a.id, user_a.id)
        self.assertEqual(resolved_b.id, user_b.id)
        self.assertNotEqual(resolved_a.id, resolved_b.id)


if __name__ == "__main__":
    unittest.main()
