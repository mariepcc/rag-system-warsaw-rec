"""
OWASP A03:2025 — Software Supply Chain Failures
Also covers: M2 (Inadequate Supply Chain Security), API9 (Improper Inventory Management)

Verifies that the API does not expose information that could assist
supply chain attacks, and that the application does not reveal
dependency versions, internal package names, or build artifacts.

Note: Full supply chain testing requires static analysis tools
(pip-audit, bandit, Safety) run separately in CI. These tests cover
the observable API surface — what the running application exposes.
"""

import pytest


SENSITIVE_TECH_KEYWORDS = [
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "psycopg2",
    "sqlalchemy",
    "python",
    "boto3",
    "openai",
    "timescale",
    "pgvector",
]

VERSION_PATTERNS = [
    "/v1/",
    "/v2/",
    "/v3/",
    "version",
    "build",
    "release",
    "commit",
    "sha",
]


class TestA03SoftwareSupplyChain:
    def test_server_header_does_not_expose_framework_version(self, client):
        """
        A03-1: The Server response header must not reveal framework name or version.
        Exposing 'uvicorn/0.x.x' or 'Python/3.11' helps attackers target
        known vulnerabilities in specific versions.
        Covers: M2 (Inadequate Supply Chain Security)
        """
        response = client.get("/health")
        server = response.headers.get("server", "").lower()

        for tech in SENSITIVE_TECH_KEYWORDS:
            assert tech not in server, (
                f"Server header exposes technology: '{server}'. "
                f"Found: '{tech}' — potential A03 vulnerability! "
                "Override Server header in middleware."
            )

    def test_x_powered_by_header_not_present(self, client):
        """
        A03-2: X-Powered-By header must not be present in responses.
        This header explicitly announces the technology stack to attackers.
        Covers: M2 (Inadequate Supply Chain Security)
        """
        response = client.get("/health")
        x_powered_by = response.headers.get("x-powered-by", "")

        assert not x_powered_by, (
            f"X-Powered-By header exposes technology: '{x_powered_by}'. "
            "Remove this header via middleware — potential A03 vulnerability!"
        )

    def test_error_responses_do_not_expose_package_names(self, client, token_a):
        """
        A03-3: Error responses must not reveal internal package names
        or library names used in the application.
        An attacker who knows which packages are in use can look up
        known CVEs for those specific versions.
        Covers: M2 (Inadequate Supply Chain Security)
        """
        error_triggers = [
            client.get(
                "/nonexistent-endpoint-xyz",
                headers={"Authorization": f"Bearer {token_a}"},
            ),
            client.get(
                "/sessions/invalid-uuid/messages",
                headers={"Authorization": f"Bearer {token_a}"},
            ),
        ]

        for response in error_triggers:
            body = response.text.lower()
            for tech in SENSITIVE_TECH_KEYWORDS:
                assert tech not in body, (
                    f"Error response ({response.status_code}) exposes package name: '{tech}'. "
                    f"Body: {response.text[:200]} — potential A03 vulnerability!"
                )

    def test_openapi_schema_does_not_expose_internal_versions(self, client):
        """
        A03-4: The OpenAPI schema must not contain version strings
        of internal dependencies in descriptions or examples.
        Covers: API9 (Improper Inventory Management)
        """
        response = client.get("/openapi.json")
        if response.status_code != 200:
            pytest.skip("OpenAPI schema not accessible — may be disabled on production")

        schema_text = response.text.lower()

        for tech in SENSITIVE_TECH_KEYWORDS:
            assert tech not in schema_text, (
                f"OpenAPI schema contains internal package reference: '{tech}'. "
                "Check endpoint descriptions and example values — potential A03 vulnerability!"
            )

    def test_health_endpoint_does_not_expose_dependency_info(self, client):
        """
        A03-5: /health must return only minimal status information.
        It must not expose database driver versions, Python version,
        or any library-specific details.
        Covers: M2 (Inadequate Supply Chain Security), API9 (Improper Inventory Management)
        """
        response = client.get("/health")
        assert response.status_code == 200

        body = response.text.lower()
        for tech in SENSITIVE_TECH_KEYWORDS:
            assert tech not in body, (
                f"/health exposes dependency info: '{tech}'. "
                f"Body: {response.text} — potential A03 vulnerability!"
            )

        data = response.json()
        forbidden_keys = [
            "version",
            "python_version",
            "dependencies",
            "packages",
            "build",
            "commit",
            "sha",
        ]
        for key in forbidden_keys:
            assert key not in data, (
                f"/health response contains internal key: '{key}'. "
                f"Response: {data} — potential A03 vulnerability!"
            )

    def test_api_does_not_expose_legacy_versioned_endpoints(self, client, token_a):
        """
        A03-6: Legacy or undocumented versioned API endpoints must not be accessible.
        Old API versions with known vulnerabilities are a common supply chain risk.
        Covers: API9 (Improper Inventory Management)
        """
        legacy_paths = [
            "/v1/sessions/",
            "/v2/sessions/",
            "/api/v1/sessions/",
            "/api/v2/places/favourites",
            "/api/sessions/",
            "/api/places/",
        ]
        for path in legacy_paths:
            response = client.get(
                path,
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (404, 405), (
                f"Legacy versioned endpoint '{path}' returned {response.status_code}. "
                "Old API versions must not be accessible — potential A03/API9 vulnerability!"
            )

    def test_debug_and_admin_endpoints_not_exposed(self, client):
        """
        A03-7: Debug and admin endpoints that frameworks expose by default
        must not be accessible in production.
        These often expose dependency versions, configuration, and metrics.
        Covers: M2 (Inadequate Supply Chain Security)
        """
        sensitive_paths = [
            "/debug",
            "/admin",
            "/metrics",
            "/actuator",
            "/actuator/info",
            "/actuator/health",
            "/env",
            "/__debug__",
            "/config",
            "/info",
            "/status",
            "/_debug",
        ]
        for path in sensitive_paths:
            response = client.get(path)
            assert response.status_code in (404, 401, 403, 405), (
                f"Sensitive endpoint '{path}' returned {response.status_code}. "
                "This endpoint must not be accessible — potential A03 vulnerability!"
            )

    def test_response_headers_do_not_contain_version_strings(self, client, token_a):
        """
        A03-8: No response header should contain version strings
        that reveal dependency versions.
        Covers: M2 (Inadequate Supply Chain Security)
        """
        endpoints = [
            "/health",
            "/sessions/",
            "/places/favourites",
        ]
        auth_headers = {"Authorization": f"Bearer {token_a}"}

        for endpoint in endpoints:
            response = client.get(endpoint, headers=auth_headers)
            headers_lower = {k.lower(): v.lower() for k, v in response.headers.items()}

            for key, value in headers_lower.items():
                for tech in SENSITIVE_TECH_KEYWORDS:
                    assert tech not in value, (
                        f"Response header '{key}: {value}' exposes technology: '{tech}'. "
                        f"Endpoint: {endpoint} — potential A03 vulnerability!"
                    )

    def test_404_response_does_not_reveal_routing_structure(self, client, token_a):
        """
        A03-9: 404 responses must not reveal the internal routing structure
        or available endpoints. An attacker could use this to map the API surface
        and identify targets for known vulnerabilities.
        Covers: API9 (Improper Inventory Management)
        """
        response = client.get(
            "/completely/nonexistent/path/xyz",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 404
        body = response.text.lower()

        for phrase in [
            "route",
            "router",
            "available routes",
            "endpoints",
            "registered",
            "path exists",
        ]:
            assert phrase not in body, (
                f"404 response reveals routing structure: '{phrase}'. "
                f"Body: {response.text[:200]} — potential A03/API9 vulnerability!"
            )
