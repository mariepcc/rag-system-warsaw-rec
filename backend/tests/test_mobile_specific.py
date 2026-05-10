"""
OWASP Mobile Top 10:2024 — Mobile-Specific Security Tests
Covers categories not already addressed by A01-A10:

- M1: Improper Credential Usage
- M2: Inadequate Supply Chain Security
- M7: Insufficient Binary Protections (API surface only)
- M10: Insufficient Cryptography

Note: M3, M4, M5, M6, M8, M9 are fully covered by test_a01 through test_a10.
"""

import pytest
import ssl
import socket
import base64
import json

import os
from dotenv import load_dotenv

load_dotenv("tests/.env.test")

base_url = os.getenv("BASE_URL", "")


PROTECTED_ENDPOINTS = [
    ("GET", "/sessions/"),
    ("GET", "/places/favourites"),
    ("GET", "/places/all"),
    ("GET", "/places/favourite-names"),
    ("GET", "/chat/all-names"),
    ("GET", "/chat/search-history?q=test"),
]


class TestM01ImproperCredentialUsage:
    def test_token_not_accepted_in_query_string(self, client, token_a):
        """
        M1-1: Credentials must not be accepted via URL query parameters.
        Tokens in URLs are stored in server logs, browser history,
        and proxy caches — a major credential exposure risk.
        """
        response = client.get(f"/places/favourites?token={token_a}")
        assert response.status_code == 401, (
            f"API accepted token in URL query string. "
            f"Got: {response.status_code} — potential M1 vulnerability! "
            "Tokens must only be accepted in the Authorization header."
        )

    def test_token_not_accepted_in_query_string_as_access_token(self, client, token_a):
        """
        M1-2: Token must not be accepted as ?access_token= query parameter.
        Some frameworks support this pattern — it must be disabled.
        """
        response = client.get(f"/sessions/?access_token={token_a}")
        assert response.status_code == 401, (
            f"API accepted token as ?access_token= query param. "
            f"Got: {response.status_code} — potential M1 vulnerability!"
        )

    def test_credentials_not_in_error_response(self, client):
        """
        M1-3: Error responses must not echo back any part of the credentials
        provided in the request. Even partial token exposure is a risk.
        """
        fake_token = "eyJhbGciOiJSUzI1NiJ9.fakepayload.fakesig"
        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        assert response.status_code == 401
        assert "fakepayload" not in response.text, (
            "Error response echoes back part of the submitted token — potential M1 vulnerability!"
        )
        assert fake_token not in response.text, (
            "Error response echoes back the full token — potential M1 vulnerability!"
        )

    def test_basic_auth_credentials_not_accepted(self, client):
        """
        M1-4: Basic authentication credentials must not be accepted.
        The API uses JWT Bearer tokens exclusively.
        Accepting Basic auth could expose plaintext credentials.
        """
        credentials = base64.b64encode(b"user@test.com:password123").decode()
        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Basic {credentials}"},
        )
        assert response.status_code == 401, (
            f"API accepted Basic auth credentials. "
            f"Got: {response.status_code} — potential M1 vulnerability!"
        )

    def test_hardcoded_test_credentials_do_not_work(self, client):
        """
        M1-5: Common hardcoded test credentials must not grant access.
        These are credentials that developers sometimes leave in code
        or that attackers try during reconnaissance.
        """
        test_tokens = [
            "test",
            "admin",
            "password",
            "123456",
            "token",
            "secret",
            "letmein",
        ]
        for token in test_tokens:
            response = client.get(
                "/sessions/",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 401, (
                f"Hardcoded test credential '{token}' was accepted. "
                f"Got: {response.status_code} — potential M1 vulnerability!"
            )

    def test_long_token_does_not_crash_server(self, client):
        """
        M1-6: Extremely long token string must not cause a server crash.
        Protects against buffer overflow or denial-of-service via credential field.
        Must return 401 — not 500.
        """
        long_token = "A" * 10_000
        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {long_token}"},
        )
        assert response.status_code in (400, 401, 413, 422), (
            f"Extremely long token returned {response.status_code}. "
            "Expected 4xx — potential M1/DoS vulnerability!"
        )
        assert response.status_code != 500, (
            "Long token caused server crash — potential M1 vulnerability!"
        )


class TestM02SupplyChainSecurity:
    def test_response_does_not_reveal_dependency_names(self, client, token_a):
        """
        M2-1: API responses must not reveal names of internal dependencies.
        Knowing which packages are used allows targeted CVE-based attacks.
        """
        sensitive_packages = [
            "psycopg2",
            "sqlalchemy",
            "pydantic",
            "uvicorn",
            "fastapi",
            "boto3",
            "openai",
            "jose",
            "jwt",
            "timescale",
            "pgvector",
            "httpx",
            "starlette",
        ]
        endpoints = [
            ("/health", {}),
            ("/sessions/", {"Authorization": f"Bearer {token_a}"}),
        ]
        for path, headers in endpoints:
            response = client.get(path, headers=headers)
            body = response.text.lower()
            for pkg in sensitive_packages:
                assert pkg not in body, (
                    f"Endpoint {path} exposes package name '{pkg}' in response body. "
                    "Remove dependency references from responses — potential M2 vulnerability!"
                )

    def test_no_internal_file_paths_in_responses(self, client, token_a):
        """
        M2-2: Responses must not contain internal file system paths.
        File paths reveal the project structure and can assist supply chain attacks.
        """
        path_indicators = ["/app/", "/usr/local/lib/", "site-packages", '.py"']

        error_responses = [
            client.get("/nonexistent", headers={"Authorization": f"Bearer {token_a}"}),
            client.post(
                "/chat/message",
                content=b"{bad",
                headers={
                    "Authorization": f"Bearer {token_a}",
                    "Content-Type": "application/json",
                },
            ),
        ]
        for response in error_responses:
            body = response.text.lower()
            for indicator in path_indicators:
                assert indicator not in body, (
                    f"Response exposes internal file path: '{indicator}'. "
                    f"Body: {response.text[:200]} — potential M2 vulnerability!"
                )

    def test_openapi_does_not_list_internal_modules(self, client):
        """
        M2-3: The OpenAPI schema must not reference internal module paths
        or package-specific schema names that reveal dependency structure.
        """
        response = client.get("/openapi.json")
        if response.status_code != 200:
            pytest.skip("OpenAPI docs not accessible")

        schema = response.text.lower()
        internal_refs = ["psycopg", "sqlalchemy", "pydantic.v1", "site-packages"]

        for ref in internal_refs:
            assert ref not in schema, (
                f"OpenAPI schema references internal module: '{ref}'. "
                "This reveals dependency structure — potential M2 vulnerability!"
            )


class TestM07InsufficientBinaryProtections:
    def test_stack_trace_not_exposed_in_any_error(self, client, token_a):
        """
        M7-1: No error response should contain a Python stack trace.
        Stack traces reveal file paths, function names, and line numbers —
        all of which assist reverse engineering of the application logic.
        """
        error_triggers = [
            client.get("/nonexistent-xyz"),
            client.get(
                "/sessions/invalid-id/messages",
                headers={"Authorization": f"Bearer {token_a}"},
            ),
            client.post(
                "/chat/message",
                content=b"not json",
                headers={
                    "Authorization": f"Bearer {token_a}",
                    "Content-Type": "application/json",
                },
            ),
        ]
        stack_indicators = ['file "', "line ", "traceback", "most recent call"]

        for response in error_triggers:
            body = response.text.lower()
            for indicator in stack_indicators:
                assert indicator not in body, (
                    f"Response ({response.status_code}) contains stack trace indicator: '{indicator}'. "
                    f"Body: {response.text[:300]} — potential M7 vulnerability!"
                )

    def test_internal_function_names_not_exposed(self, client, token_a):
        """
        M7-2: Error responses must not reveal internal function names,
        class names, or method names from the application code.
        These assist in reverse engineering the application logic.
        """
        internal_names = [
            "handle_message",
            "toggle_favourite",
            "get_sessions",
            "chat_repository",
            "places_repository",
            "vector_store",
            "synthesizer",
            "llm_factory",
            "chat_service",
        ]
        response = client.get(
            "/sessions/nonexistent/messages",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        body = response.text.lower()

        for name in internal_names:
            assert name not in body, (
                f"Response exposes internal function/class name: '{name}'. "
                f"Body: {response.text[:200]} — potential M7 vulnerability!"
            )

    def test_database_schema_not_exposed_in_errors(self, client, token_a):
        """
        M7-3: Error responses must not reveal database table names,
        column names, or schema structure.
        This information assists targeted injection attacks.
        """
        db_internals = [
            "saved_places",
            "chat_sessions",
            "chat_messages",
            "embeddings",
            "user_rating_count",
            "is_favourite",
        ]
        error_response = client.get(
            "/sessions/bad-uuid/messages",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        body = error_response.text.lower()

        for internal in db_internals:
            assert internal not in body, (
                f"Error response exposes database schema detail: '{internal}'. "
                f"Body: {error_response.text[:200]} — potential M7 vulnerability!"
            )


class TestM10InsufficientCryptography:
    def test_api_uses_https(self):
        """
        M10-1: The API base URL must use HTTPS.
        HTTP transmits tokens and data in plaintext — critical vulnerability.
        """
        if "localhost" in base_url or "127.0.0.1" in base_url:
            pytest.skip("Running locally — HTTPS not required for local development")
        assert base_url.startswith("https://"), (
            f"API base URL uses HTTP: '{base_url}' — potential M10 vulnerability!"
        )

    def test_tls_certificate_is_valid(self):
        """
        M10-2: The TLS certificate must be valid and not self-signed
        in production. An invalid certificate allows MITM attacks.
        """
        import os
        from dotenv import load_dotenv

        load_dotenv("tests/.env.test")

        base_url = os.getenv("BASE_URL", "")
        if not base_url.startswith("https://"):
            pytest.skip("BASE_URL is not HTTPS — skipping TLS certificate check")

        hostname = base_url.replace("https://", "").split("/")[0].split(":")[0]

        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    assert cert, (
                        f"Could not retrieve TLS certificate for '{hostname}'. "
                        "Possible M10 vulnerability!"
                    )
        except ssl.SSLCertVerificationError as e:
            pytest.fail(
                f"TLS certificate for '{hostname}' is invalid: {e}. "
                "Use a valid certificate from a trusted CA — potential M10 vulnerability!"
            )
        except (socket.timeout, ConnectionRefusedError, OSError):
            pytest.skip(f"Cannot connect to '{hostname}' — skipping TLS check")

    def test_hsts_header_enforces_https(self, client):
        """
        M10-3: Strict-Transport-Security header must be present.
        HSTS tells clients to always use HTTPS for this domain,
        preventing protocol downgrade attacks.
        """
        response = client.get("/health")
        hsts = response.headers.get("strict-transport-security", "")

        assert hsts, (
            "Missing Strict-Transport-Security (HSTS) header. "
            "Add: Strict-Transport-Security: max-age=31536000; includeSubDomains "
            "— potential M10 vulnerability!"
        )
        assert "max-age" in hsts, (
            f"HSTS header present but missing max-age directive: '{hsts}'. "
            "— potential M10 vulnerability!"
        )

    def test_sensitive_data_not_transmitted_in_plaintext_in_response(
        self, client, token_a
    ):
        """
        M10-4: API responses must not contain sensitive data fields
        that should never leave the server (internal keys, raw passwords,
        connection strings).
        """
        plaintext_risks = [
            "postgresql://",
            "postgres://",
            "password=",
            "sk-",
            "aws_secret",
            "private_key",
        ]
        endpoints = [
            "/sessions/",
            "/places/favourites",
            "/health",
        ]
        for endpoint in endpoints:
            response = client.get(
                endpoint,
                headers={"Authorization": f"Bearer {token_a}"},
            )
            body = response.text.lower()
            for risk in plaintext_risks:
                assert risk not in body, (
                    f"Endpoint {endpoint} transmits sensitive data in plaintext: '{risk}'. "
                    f"Body: {response.text[:200]} — potential M10 vulnerability!"
                )

    def test_jwt_uses_asymmetric_algorithm(self, client, token_a):
        """
        M10-5: The JWT token issued by Cognito must use an asymmetric
        algorithm (RS256) — not a symmetric one (HS256).
        Symmetric JWTs require sharing the secret with every service
        that validates them, which is a cryptographic design flaw.
        """
        parts = token_a.split(".")
        assert len(parts) == 3, "Token is not a valid JWT"

        try:
            padded = parts[0] + "=" * (4 - len(parts[0]) % 4)
            header = json.loads(base64.urlsafe_b64decode(padded))
        except Exception:
            pytest.skip("Cannot decode token header")

        algorithm = header.get("alg", "")
        assert algorithm in ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512"), (
            f"JWT uses symmetric or weak algorithm: '{algorithm}'. "
            "Use RS256 or another asymmetric algorithm — potential M10 vulnerability!"
        )
        assert algorithm != "HS256", (
            "JWT uses HS256 (symmetric HMAC). "
            "Switch to RS256 — potential M10 vulnerability!"
        )
        assert algorithm != "none", (
            "JWT uses 'none' algorithm — critical M10 vulnerability!"
        )
