"""
OWASP A04:2025 — Cryptographic Failures
Also covers: M10 (Insufficient Cryptography), M1 (Improper Credential Usage), API2 (Broken Authentication)

Verifies that tokens are correctly validated cryptographically
and that the API does not accept forged or weakly signed credentials.
"""

import base64
import json
import httpx as httpx_lib
import pytest


def _make_fake_token(algorithm: str = "HS256", expired: bool = False) -> str:
    """Builds a structurally valid but cryptographically invalid JWT for testing."""
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": algorithm, "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "sub": "fake-user-id",
                    "email": "fake@test.com",
                    "exp": 1000000000 if expired else 9999999999,
                }
            ).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}.fakesignature"


class TestA04CryptographicFailures:
    def test_hs256_token_is_rejected(self, client):
        """
        A04-1: Token signed with HS256 instead of RS256.
        Cognito uses RS256 — a different algorithm must be rejected.
        Expected result: 401 Unauthorized
        Covers: M10 (Insufficient Cryptography), API2 (Broken Authentication)
        """
        fake_token = _make_fake_token(algorithm="HS256")
        response = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        assert response.status_code == 401, (
            f"HS256 token should be rejected — API accepts RS256 only. "
            f"Got: {response.status_code} — potential A04 vulnerability!"
        )

    def test_fake_signature_is_rejected(self, client):
        """
        A04-2: Token with valid format but a fake signature.
        Expected result: 401 Unauthorized
        Covers: M10 (Insufficient Cryptography)
        """
        fake_token = _make_fake_token(algorithm="RS256")
        response = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        assert response.status_code == 401, (
            f"Token with fake signature should be rejected. "
            f"Got: {response.status_code} — potential A04 vulnerability!"
        )

    def test_expired_token_is_rejected(self, client):
        """
        A04-3: Token with expiry date in the past (exp: 2001-09-09).
        Expected result: 401 Unauthorized
        Covers: M1 (Improper Credential Usage), API2 (Broken Authentication)
        """
        fake_token = _make_fake_token(expired=True)
        response = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        assert response.status_code == 401, (
            f"Expired token should be rejected. "
            f"Got: {response.status_code} — potential A04 vulnerability!"
        )

    def test_empty_bearer_token_is_rejected(self, client):
        """
        A04-4: Empty string as token.
        httpx rejects 'Bearer ' as invalid HTTP (LocalProtocolError) —
        which is itself correct behavior.
        Also tests 'Bearer' without a token value.
        """
        with pytest.raises(httpx_lib.LocalProtocolError):
            client.get(
                "/places/favourites",
                headers={"Authorization": "Bearer "},
            )

        response = client.get(
            "/places/favourites",
            headers={"Authorization": "Bearer"},
        )
        assert response.status_code in (401, 403, 422), (
            f"'Bearer' without token should be rejected. Got: {response.status_code}"
        )

    def test_random_string_token_is_rejected(self, client):
        """
        A04-5: Completely random string instead of JWT.
        Expected result: 401 Unauthorized
        Covers: M1 (Improper Credential Usage)
        """
        response = client.get(
            "/places/favourites",
            headers={"Authorization": "Bearer abcdef123456"},
        )
        assert response.status_code == 401, (
            f"Random string token should be rejected. "
            f"Got: {response.status_code} — potential A04 vulnerability!"
        )

    def test_missing_authorization_header_returns_401(self, client):
        """
        A04-6: Request with no Authorization header at all.
        Expected result: 401 Unauthorized
        """
        response = client.get("/places/favourites")
        assert response.status_code == 401, (
            f"Missing Authorization header should return 401. "
            f"Got: {response.status_code}"
        )

    def test_valid_token_is_accepted(self, client, token_a):
        """
        A04-7: Valid RS256 token from Cognito.
        Expected result: 200 OK
        Sanity check — cryptographic validation must not block legitimate users.
        """
        response = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 200, (
            f"Valid token should be accepted. Got: {response.status_code}"
        )

    def test_error_response_does_not_expose_crypto_details(self, client):
        """
        A04-8: Error response must not reveal cryptographic implementation details.
        Expected result: generic error message without stack trace,
        algorithm names, or key structure details.
        Covers: M9 (Insecure Data Storage)
        """
        fake_token = _make_fake_token()
        response = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        body = response.text.lower()

        assert "traceback" not in body, (
            "Response contains traceback — do not expose error details!"
        )
        assert "jose" not in body, "Response exposes name of cryptographic library!"
        assert "jwks" not in body, "Response exposes JWT validation details!"
        assert "secret" not in body, "Response may contain sensitive keywords!"
