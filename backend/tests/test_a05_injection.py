"""
OWASP A05 — Injection
Verifies that the API properly validates input data and is not vulnerable
to SQL injection, command injection, or malformed data attacks.
"""

SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE saved_places; --",
    "' OR '1'='1",
    "' UNION SELECT * FROM users --",
    "1; SELECT * FROM sessions",
    "' OR 1=1 --",
    "%27 OR %271%27=%271",
]

XSS_PAYLOADS = [
    "<script>alert('xss')</script>",
    "javascript:alert(1)",
    "<img src=x onerror=alert(1)>",
]

OVERSIZED_PAYLOADS = [
    "A" * 10_000,
    "A" * 100_000,
]


class TestA05Injection:
    def test_sql_injection_in_category_filter(self, client, token_a):
        """
        A05-1:  SQL injection attempts in the category query parameter.
        Expected: 200 with empty list OR 422. Must NOT return 500.
        """
        for payload in SQL_INJECTION_PAYLOADS:
            response = client.get(
                "/places/favourites",
                params={"category": payload},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (200, 422), (
                f"SQL injection payload '{payload}' caused: "
                f"{response.status_code} — possible injection vulnerability!"
            )

    def test_sql_injection_in_search(self, client, token_a):
        """
        A05-2: SQL injection in session search query.
        Expected: 200 or 422. Must NOT return 500 or other users' data.
        """
        for payload in SQL_INJECTION_PAYLOADS:
            response = client.get(
                "/chat/search-history",
                params={"q": payload},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (200, 422), (
                f"SQL injection in search caused: {response.status_code}"
            )

    def test_sql_injection_in_toggle_place_id(self, client, token_a):
        """
        A05-3: SQL injection in place_id path parameter.
        place_id is passed directly to parameterized query — must not cause 500.
        Expected: 422 (FK violation — place doesn't exist) or 200.
        """
        for payload in SQL_INJECTION_PAYLOADS:
            response = client.post(
                f"/places/favourites/{payload}/toggle",
                json={"sessionId": None},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (200, 422), (
                f"SQL injection in place_id caused: {response.status_code} — "
                f"possible injection vulnerability!"
            )

    def test_toggle_with_no_body_uses_null_session(self, client, token_a):
        """
        A05-4: Toggle with empty body — sessionId defaults to None.
        ToggleFavouriteRequest has only optional sessionId, so empty body
        is valid. Must NOT return 500.
        """
        import uuid

        place_id = str(uuid.uuid4())
        response = client.post(
            f"/places/favourites/{place_id}/toggle",
            json={},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (200, 422), (
            f"Toggle with empty body caused: {response.status_code}"
        )

    def test_toggle_with_sql_injection_in_session_id(self, client, token_a):
        """
        A05-5: SQL injection in sessionId body field.
        sessionId is passed to parameterized query — must be stored as plain text
        or rejected, never executed. Must NOT cause 500.
        """
        import uuid

        place_id = str(uuid.uuid4())
        for payload in SQL_INJECTION_PAYLOADS:
            response = client.post(
                f"/places/favourites/{place_id}/toggle",
                json={"sessionId": payload},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (200, 422), (
                f"SQL injection in sessionId caused: {response.status_code}"
            )

    def test_oversized_session_id_is_handled(self, client, token_a):
        """
        A05-6: Oversized sessionId value — must not crash the server.
        Expected: 200 (stored as-is) or 422 (rejected by validator).
        """
        import uuid

        place_id = str(uuid.uuid4())
        response = client.post(
            f"/places/favourites/{place_id}/toggle",
            json={"sessionId": "A" * 10_000},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (200, 422), (
            f"Oversized sessionId caused: {response.status_code}"
        )

    def test_xss_payload_in_session_id_is_handled(self, client, token_a):
        """
        A05-7: XSS payloads in sessionId — API stores as plain text or rejects.
        Must NOT execute or cause 500.
        """
        import uuid

        place_id = str(uuid.uuid4())
        for payload in XSS_PAYLOADS:
            response = client.post(
                f"/places/favourites/{place_id}/toggle",
                json={"sessionId": payload},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (200, 422), (
                f"XSS payload in sessionId caused: {response.status_code}"
            )

    def test_oversized_chat_message_is_handled(self, client, token_a):
        """
        A05-8: Message over 10k chars sent to chat endpoint.
        ChatRequest validator must reject with 422.
        """
        response = client.post(
            "/chat/message",
            json={"message": "A" * 10_001, "session_id": None},
            headers={"Authorization": f"Bearer {token_a}"},
            timeout=15.0,
        )
        assert response.status_code == 422, (
            f"Message over 10k chars should be rejected. Got: {response.status_code}"
        )

    def test_empty_chat_message_is_rejected(self, client, token_a):
        """
        A05-9: Empty string as chat message — must return 422 immediately.
        """
        response = client.post(
            "/chat/message",
            json={"message": "", "session_id": None},
            headers={"Authorization": f"Bearer {token_a}"},
            timeout=15.0,
        )
        assert response.status_code == 422, (
            f"Empty message should be rejected. Got: {response.status_code}"
        )

    def test_null_message_is_rejected(self, client, token_a):
        """
        A05-10: Null value as chat message — must return 422.
        """
        response = client.post(
            "/chat/message",
            json={"message": None, "session_id": None},
            headers={"Authorization": f"Bearer {token_a}"},
            timeout=15.0,
        )
        assert response.status_code == 422, (
            f"Null message should be rejected. Got: {response.status_code}"
        )
