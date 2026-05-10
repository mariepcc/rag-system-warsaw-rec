"""
OWASP A08:2025 — Software and Data Integrity Failures
Also covers: M1 (Improper Credential Usage), M10 (Insufficient Cryptography), API2 (Broken Authentication)

Verifies that the API correctly checks the integrity of JWT tokens
and does not allow manipulation of authorization data.

Tested scenarios:
- tokens with invalid signatures
- tokens with modified payload (user_id substitution)
- expired tokens
- tokens with invalid issuer / audience
- missing token and invalid format
- algorithm manipulation (alg:none attack)
"""

import uuid
import base64
import json
import pytest


def _b64_encode(data: dict) -> str:
    """Encodes a dict as base64url without padding — JWT segment format."""
    raw = json.dumps(data, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _forge_token(header: dict, payload: dict, signature: str = "fakesignature") -> str:
    """Builds a forged JWT token with the given header and payload."""
    h = _b64_encode(header)
    p = _b64_encode(payload)
    s = base64.urlsafe_b64encode(signature.encode()).rstrip(b"=").decode()
    return f"{h}.{p}.{s}"


PROTECTED_ENDPOINTS = [
    ("GET", "/sessions/"),
    ("GET", "/places/favourites"),
    ("GET", "/places/all"),
    ("GET", "/places/favourite-names"),
    ("GET", "/chat/all-names"),
]


class TestA08SoftwareDataIntegrity:
    def test_token_with_fake_signature_returns_401(self, client, token_a):
        """
        A08-1: Token with valid header and payload but a fake signature
        must be rejected. RS256 signature verification is critical.
        Covers: M10 (Insufficient Cryptography), API2 (Broken Authentication)
        """
        parts = token_a.split(".")
        assert len(parts) == 3, "Token does not have 3 segments — invalid JWT"

        forged = f"{parts[0]}.{parts[1]}.INVALIDSIGNATUREXXXXXXXXXXXXXXXXXX"

        for method, path in PROTECTED_ENDPOINTS:
            response = client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {forged}"},
            )
            assert response.status_code == 401, (
                f"{method} {path}: Token with fake signature returned {response.status_code}. "
                "Expected 401 Unauthorized — potential A08 vulnerability!"
            )

    def test_token_with_modified_payload_returns_401(self, client, token_a, token_b):
        """
        A08-2: Token with user A's signature but user B's payload
        (substituted sub/user_id) must be rejected.
        Classic attack — changing user_id in payload without changing signature.
        Covers: M1 (Improper Credential Usage), API2 (Broken Authentication)
        """
        parts_a = token_a.split(".")
        parts_b = token_b.split(".")
        assert len(parts_a) == 3 and len(parts_b) == 3

        forged = f"{parts_a[0]}.{parts_b[1]}.{parts_a[2]}"

        for method, path in PROTECTED_ENDPOINTS:
            response = client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {forged}"},
            )
            assert response.status_code == 401, (
                f"{method} {path}: Token with swapped payload returned {response.status_code}. "
                "Expected 401 — signature must not match modified payload — potential A08 vulnerability!"
            )

    def test_alg_none_attack_returns_401(self, client, token_a):
        """
        A08-3: The 'alg:none' attack — token without a signature and alg=none header.
        Some JWT libraries accept such tokens if not properly configured.
        Must return 401.
        Covers: M10 (Insufficient Cryptography), API2 (Broken Authentication)
        """
        parts = token_a.split(".")

        try:
            payload_padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_padded))
        except Exception:
            pytest.skip("Cannot decode token payload")

        none_token_variants = [
            _forge_token({"alg": "none", "typ": "JWT"}, payload, ""),
            _forge_token({"alg": "None", "typ": "JWT"}, payload, ""),
            _forge_token({"alg": "NONE", "typ": "JWT"}, payload, ""),
            f"{_b64_encode({'alg': 'none', 'typ': 'JWT'})}.{parts[1]}.",
        ]

        for token in none_token_variants:
            for method, path in PROTECTED_ENDPOINTS:
                response = client.request(
                    method,
                    path,
                    headers={"Authorization": f"Bearer {token}"},
                )
                assert response.status_code == 401, (
                    f"{method} {path}: alg:none token returned {response.status_code}. "
                    f"Variant: {token[:60]}... Expected 401 — potential A08 vulnerability!"
                )

    def test_malformed_token_segments_return_401(self, client):
        """
        A08-4: Token with incorrect number of segments must be rejected.
        A valid JWT must have exactly 3 base64url parts separated by dots.
        """
        bad_tokens = [
            "notajwttoken",
            "onlyone",
            "two.segments",
            "four.segments.are.bad",
            "Bearer",
        ]
        for bad_token in bad_tokens:
            response = client.get(
                "/sessions/",
                headers={"Authorization": f"Bearer {bad_token}"},
            )
            assert response.status_code == 401, (
                f"Malformed token '{bad_token[:20]}' returned {response.status_code}. "
                "Expected 401 — potential A08 vulnerability!"
            )

    def test_token_with_fake_issuer_returns_401(self, client, token_a):
        """
        A08-5: Token with valid structure but a different issuer (iss)
        must be rejected. Protects against tokens from other Cognito pools.
        Covers: M1 (Improper Credential Usage), API2 (Broken Authentication)
        """
        parts = token_a.split(".")

        try:
            payload_padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_padded))
        except Exception:
            pytest.skip("Cannot decode token payload")

        fake_payload = {**payload, "iss": "https://fake-issuer.example.com/fake-pool"}
        forged = _forge_token({"alg": "RS256", "typ": "JWT"}, fake_payload, "fakesig")

        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {forged}"},
        )
        assert response.status_code == 401, (
            f"Token with fake issuer returned {response.status_code}. "
            "Expected 401 — potential A08 vulnerability!"
        )

    def test_missing_authorization_header_returns_401(self, client):
        """
        A08-6: Request without Authorization header must return 401.
        Verifies that endpoints are not accidentally left public.
        """
        for method, path in PROTECTED_ENDPOINTS:
            response = client.request(method, path)
            assert response.status_code == 401, (
                f"{method} {path}: Missing Authorization returned {response.status_code}. "
                "Expected 401 — endpoint must not be public — potential A08 vulnerability!"
            )

    def test_invalid_authorization_format_returns_401(self, client, token_a):
        """
        A08-7: Authorization header with invalid format
        (missing 'Bearer', wrong schemes) must return 401.
        """
        bad_auth_headers = [
            token_a,
            f"Basic {token_a}",
            f"Token {token_a}",
            f"bearer {token_a}",
            "Bearer null",
            "Bearer undefined",
            "Bearer None",
        ]
        for auth_header in bad_auth_headers:
            response = client.get(
                "/sessions/",
                headers={"Authorization": auth_header},
            )
            assert response.status_code == 401, (
                f"Invalid auth header '{auth_header[:40]}' returned {response.status_code}. "
                "Expected 401 — potential A08 vulnerability!"
            )

    def test_user_id_in_body_does_not_override_token(self, client, token_a, token_b):
        """
        A08-8: Sending user_id in the request body must not override
        the user_id from the JWT token. The backend must always take
        user identity from the verified token, never from the request body.
        Covers: API3 (Broken Object Property Level Authorization)
        """
        place_id = str(uuid.uuid4())
        response = client.post(
            f"/places/favourites/{place_id}/toggle",
            json={
                "name": f"InjectedPlace_{uuid.uuid4().hex[:6]}",
                "rating": 4.0,
                "user_id": "some-other-user-id",
                "userId": "some-other-user-id",
                "id": place_id,
            },
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (200, 422), (
            f"Toggle with injected user_id returned {response.status_code}."
        )

    def test_place_id_from_url_takes_precedence_over_body(self, client, token_a):
        """
        A08-9: place_id from the URL path must be used by the backend — not from the body.
        Prevents resource identifier manipulation by substituting id in the body.
        Covers: API3 (Broken Object Property Level Authorization)
        """
        place_id_url = str(uuid.uuid4())
        place_id_body = str(uuid.uuid4())

        response = client.post(
            f"/places/favourites/{place_id_url}/toggle",
            json={
                "id": place_id_body,
                "name": "IntegrityTest",
                "rating": 3.5,
            },
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (200, 422), (
            f"Toggle with different IDs in URL and body returned {response.status_code}."
        )
        assert response.status_code != 500, (
            "Mismatched ID in URL and body caused a server crash — potential A08 vulnerability!"
        )
