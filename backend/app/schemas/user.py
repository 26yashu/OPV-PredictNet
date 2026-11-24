from datetime import datetime
import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="Valid user email address")
    password: str = Field(..., min_length=8, description="Password with minimum 8 characters, uppercase, lowercase, and digit")
    full_name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    institution: Optional[str] = Field(None, max_length=150, description="Academic or industry institution")

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Email must be a string")
        email_clean = v.strip().lower()
        if not EMAIL_REGEX.match(email_clean):
            raise ValueError("Invalid email address format (e.g., user@domain.com)")
        return email_clean

    @field_validator("full_name")
    @classmethod
    def sanitize_full_name(cls, v: str) -> str:
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                raise ValueError("Full name cannot be empty or whitespace only")
            return v_clean
        return v

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one numeric digit")
        return v


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="Account password")

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Email must be a string")
        email_clean = v.strip().lower()
        if not EMAIL_REGEX.match(email_clean):
            raise ValueError("Invalid email address format (e.g., user@domain.com)")
        return email_clean


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    institution: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
