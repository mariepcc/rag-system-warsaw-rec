"""
OWASP A02:2025 — Security Misconfiguration
Verifies that the API is not exposed due to misconfigured security settings:
CORS policy, HTTP security headers, documentation exposure, and error handling.
"""

SENSITIVE_KEYWORDS = [
    "traceback",
    'file "',
    "line ",
    "sqlalchemy",
    "psycopg",
    "postgres",
    "password",
    "secret",
    "token",
    "stack",
]


class TestA02SecurityMisconfiguration:
    def test_cors_wildcard_with_credentials_is_invalid(self, client, token_a):
        """
        A02-1: allow_origins=["*"] combined with allow_credentials=True
        is an invalid and insecure CORS configuration.
        Browsers will block such responses — and the server should not
        send both headers simultaneously.

        Fix: replace allow_origins=["*"] with an explicit list of allowed origins.
        """
        response = client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        acao = response.headers.get("access-control-allow-origin", "")
        acac = response.headers.get("access-control-allow-credentials", "")

        assert not (acao == "*" and acac == "true"), (
            "CORS misconfiguration: allow_origins=['*'] and allow_credentials=True "
            "cannot be used together. Browsers will block this. "
            "Set explicit allowed origins instead of wildcard."
        )

    def test_cors_untrusted_origin_is_rejected(self, client, token_a):
        """
        A02-2: Requests from untrusted origins should not receive
        Access-Control-Allow-Origin reflecting the untrusted domain.
        If wildcard is used, this test documents the risk.
        """
        response = client.get(
            "/health",
            headers={"Origin": "https://attacker.com"},
        )
        acao = response.headers.get("access-control-allow-origin", "")

        assert acao != "https://attacker.com", (
            "CORS misconfiguration: untrusted origin 'https://attacker.com' "
            "is reflected in Access-Control-Allow-Origin. "
            "Set an explicit allowlist of trusted origins."
        )

    def test_cors_preflight_does_not_allow_arbitrary_methods(self, client):
        """
        A02-3: OPTIONS preflight from untrusted origin should not allow
        arbitrary HTTP methods like DELETE or PATCH to all origins.
        """
        response = client.options(
            "/health",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        acao = response.headers.get("access-control-allow-origin", "")
        acam = response.headers.get("access-control-allow-methods", "")

        assert acao != "*" or "DELETE" not in acam, (
            "CORS preflight allows DELETE from any origin — restrict allowed origins."
        )

    def test_cors_trusted_origin_is_allowed(self, client):
        """
        A02-4: Sanity check — once wildcard is replaced with explicit origins,
        the trusted frontend origin must still be allowed.
        Update ALLOWED_ORIGIN to match your actual frontend domain.
        """
        ALLOWED_ORIGIN = "http://localhost:8081"

        response = client.get(
            "/health",
            headers={"Origin": ALLOWED_ORIGIN},
        )
        acao = response.headers.get("access-control-allow-origin", "")

        assert acao in (ALLOWED_ORIGIN, "*"), (
            f"Trusted origin '{ALLOWED_ORIGIN}' is not allowed by CORS policy. "
            f"Got: '{acao}'"
        )

    def test_x_content_type_options_header_present(self, client):
        """
        A02-5: X-Content-Type-Options: nosniff prevents browsers from
        MIME-sniffing responses away from the declared content-type.
        Without it, a browser might execute a JSON response as a script.

        Fix: add custom middleware or use a library like secure.py.
        """
        response = client.get("/health")
        assert "x-content-type-options" in response.headers, (
            "Missing header: X-Content-Type-Options. "
            "Add it via middleware: response.headers['X-Content-Type-Options'] = 'nosniff'"
        )
        assert response.headers["x-content-type-options"] == "nosniff", (
            f"X-Content-Type-Options should be 'nosniff', "
            f"got: '{response.headers['x-content-type-options']}'"
        )

    def test_x_frame_options_header_present(self, client):
        """
        A02-6: X-Frame-Options: DENY prevents the API from being embedded
        in an iframe — protects against clickjacking attacks.

        Fix: add via middleware.
        """
        response = client.get("/health")
        assert "x-frame-options" in response.headers, (
            "Missing header: X-Frame-Options. "
            "Add: response.headers['X-Frame-Options'] = 'DENY'"
        )
        assert response.headers["x-frame-options"] in ("DENY", "SAMEORIGIN"), (
            f"X-Frame-Options should be DENY or SAMEORIGIN, "
            f"got: '{response.headers['x-frame-options']}'"
        )

    def test_content_type_is_json_for_api_responses(self, client, token_a):
        """
        A02-7: API responses must declare Content-Type: application/json.
        Incorrect or missing content-type can cause browsers to misinterpret
        the response, potentially enabling injection attacks.
        """
        response = client.get(
            "/health",
        )
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type, (
            f"Expected Content-Type: application/json, got: '{content_type}'"
        )

    def test_server_header_does_not_leak_technology(self, client):
        """
        A02-8: The Server header should not reveal implementation details
        like 'uvicorn', 'python', or version numbers.
        This information helps attackers fingerprint the technology stack.

        Fix: remove or override the Server header in middleware.
        """
        response = client.get("/health")
        server = response.headers.get("server", "").lower()

        for tech in ["uvicorn", "python", "fastapi", "starlette"]:
            assert tech not in server, (
                f"Server header leaks technology: '{server}'. "
                f"Override it in middleware: response.headers['server'] = 'Spot-Guide'"
            )

    def test_openapi_docs_accessible_locally(self, client):
        """
        A02-9: /docs should be accessible in local/dev environment.
        This test documents that docs are currently enabled.
        On production, docs should be disabled via:
        app = FastAPI(docs_url=None, redoc_url=None) when ENVIRONMENT=prod.
        """
        response = client.get("/docs")
        assert response.status_code == 200, (
            "/docs should be accessible in development. "
            "Ensure it is disabled on production via ENVIRONMENT variable."
        )

    def test_openapi_json_accessible_locally(self, client):
        """
        A02-10: /openapi.json exposes the full API schema.
        Acceptable locally, must be disabled on production.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data, "openapi.json should contain API paths"

    def test_openapi_json_does_not_expose_sensitive_env_data(self, client):
        """
        A02-11: The OpenAPI schema must not contain secrets, connection strings,
        or internal configuration values in descriptions or examples.
        """
        response = client.get("/openapi.json")
        schema_text = response.text.lower()

        for keyword in ["password", "secret", "sk-", "postgres://"]:
            assert keyword not in schema_text, (
                f"OpenAPI schema contains sensitive keyword: '{keyword}'. "
                "Check endpoint descriptions and example values."
            )

    def test_404_response_does_not_leak_internals(self, client, token_a):
        """
        A02-12: 404 responses must not contain stack traces, file paths,
        or internal implementation details.
        """
        response = client.get(
            "/nonexistent-endpoint-xyz",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 404
        body = response.text.lower()

        for keyword in SENSITIVE_KEYWORDS:
            assert keyword not in body, (
                f"404 response leaks internal detail: '{keyword}' found in body. "
                f"Body: {response.text[:300]}"
            )

    def test_422_response_does_not_leak_internals(self, client, token_a):
        """
        A02-13: Validation error responses (422) must not expose
        internal file paths or stack traces — only field-level errors.
        """
        response = client.post(
            "/chat/message",
            json={"message": None},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 422
        body = response.text.lower()

        for keyword in ["traceback", 'file "', "sqlalchemy", "psycopg"]:
            assert keyword not in body, (
                f"422 response leaks internal detail: '{keyword}'. "
                f"Body: {response.text[:300]}"
            )

    def test_500_response_does_not_expose_traceback(self, client, token_a):
        """
        A02-14: When a 500 error occurs, the response body must contain
        only a generic error message — never a Python traceback, file path,
        or database error details.

        Note: chat.py currently uses traceback.print_exc() which prints to
        stdout/logs — that is acceptable. The risk is if traceback details
        are included in the HTTP response body.
        """
        response = client.post(
            "/places/favourites/trigger-error/toggle",
            json={"name": "x" * 5000},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        if response.status_code == 500:
            body = response.text.lower()
            for keyword in SENSITIVE_KEYWORDS:
                assert keyword not in body, (
                    f"500 response exposes internal detail: '{keyword}'. "
                    f"Body: {response.text[:300]}"
                )

    def test_debug_endpoints_do_not_exist(self, client):
        """
        A02-15: Common debug and admin endpoints must not be exposed.
        These are often left enabled by accident in frameworks.
        """
        sensitive_paths = [
            "/debug",
            "/admin",
            "/metrics",
            "/actuator",
            "/env",
            "/__debug__",
            "/config",
        ]
        for path in sensitive_paths:
            response = client.get(path)
            assert response.status_code in (404, 401, 403, 405), (
                f"Sensitive endpoint '{path}' returned {response.status_code}. "
                "This endpoint should not be accessible."
            )

    def test_health_endpoint_does_not_expose_system_info(self, client):
        """
        A02-16: /health should return minimal status information.
        Must not expose database connection strings, environment variables,
        library versions, or system details.
        """
        response = client.get("/health")
        assert response.status_code == 200
        body = response.text.lower()

        for keyword in ["postgres", "password", "secret", "version", "python"]:
            assert keyword not in body, (
                f"/health exposes sensitive info: '{keyword}' found. "
                f"Body: {response.text}"
            )
