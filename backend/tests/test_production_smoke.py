"""
Production smoke test — HTTPS and security headers.
Run against live deployment:
    BASE_URL=https://twoja-domena.com pytest tests/test_production_smoke.py -v
"""

import pytest
import httpx

BASE_URL = "https://spotguide.me"


@pytest.fixture(scope="session")
def prod_client():
    return httpx.Client(base_url=BASE_URL, follow_redirects=False)


class TestProductionSmoke:
    def test_https_redirect_from_http(self):
        """
        HTTP request to port 80 must be permanently redirected to HTTPS.
        ALB listener on port 80 is configured with redirect action.
        """
        http_url = BASE_URL.replace("https://", "http://")
        response = httpx.get(http_url, follow_redirects=False)
        assert response.status_code in (301, 302), (
            f"Expected redirect from HTTP to HTTPS, got {response.status_code}. "
            "Check ALB HTTP listener redirect action."
        )
        location = response.headers.get("location", "")
        assert location.startswith("https://"), (
            f"Redirect location is not HTTPS: '{location}'"
        )

    def test_hsts_header_present_on_https(self, prod_client):
        """
        Strict-Transport-Security header must be present on HTTPS responses.
        Tells browsers to never use HTTP for this domain.
        Only testable on production with real TLS.
        """
        response = prod_client.get("/health")
        hsts = response.headers.get("strict-transport-security", "")
        assert hsts, (
            "Missing Strict-Transport-Security header. "
            "Add via ALB or middleware: "
            "Strict-Transport-Security: max-age=31536000; includeSubDomains"
        )
        assert "max-age" in hsts, f"HSTS header missing max-age directive: '{hsts}'"

    def test_health_endpoint_returns_200_on_production(self, prod_client):
        """
        Sanity check — backend is running and reachable through ALB.
        """
        response = prod_client.get("/health")
        assert response.status_code == 200, (
            f"Production /health returned {response.status_code}. "
            "Check ECS service status and ALB target group health."
        )
        assert response.json().get("status") == "ok"

    def test_tls_certificate_is_valid(self):
        """
        TLS certificate must be valid — not expired, not self-signed.
        httpx verifies by default — if this passes, cert is trusted.
        """
        try:
            response = httpx.get(f"{BASE_URL}/health", verify=True, timeout=10)
            assert response.status_code == 200
        except httpx.ConnectError as e:
            pytest.fail(
                f"TLS certificate verification failed: {e}. "
                "Check ACM certificate status in AWS Console."
            )

    def test_security_headers_present_on_production(self, prod_client):
        """
        All security headers must be present on production — same as local
        but now verified against the live ALB + ECS stack.
        """
        response = prod_client.get("/health")
        assert "x-content-type-options" in response.headers, (
            "Missing X-Content-Type-Options on production."
        )
        assert "x-frame-options" in response.headers, (
            "Missing X-Frame-Options on production."
        )

    def test_docs_are_disabled_on_production(self, prod_client):
        """
        /docs and /redoc must return 404 on production.
        Disable via: app = FastAPI(docs_url=None, redoc_url=None)
        when ENVIRONMENT=prod.
        """
        for path in ["/docs", "/redoc", "/openapi.json"]:
            response = prod_client.get(path)
            assert response.status_code == 404, (
                f"'{path}' is accessible on production (got {response.status_code}). "
                "Set docs_url=None and redoc_url=None in FastAPI() when ENVIRONMENT=prod."
            )

    def test_cors_does_not_allow_wildcard_on_production(self, prod_client):
        """
        Production CORS must use explicit origin allowlist — not wildcard.
        """
        response = prod_client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        acao = response.headers.get("access-control-allow-origin", "")
        assert acao != "*", (
            "Production CORS allows wildcard origin. "
            "Set explicit allowed origins in CORSMiddleware."
        )
        assert acao != "https://evil.com", (
            "Production CORS reflects untrusted origin 'https://evil.com'."
        )
