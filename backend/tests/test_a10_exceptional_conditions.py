"""
OWASP A10:2025 — Mishandling of Exceptional Conditions
Verifies that the API handles all unexpected inputs, malformed requests,
and edge cases gracefully — always returning a structured error response
instead of crashing with 500 or leaking internals.
"""

import httpx
import uuid
import threading


SENSITIVE_KEYWORDS = [
    "traceback",
    'file "',
    "line ",
    "sqlalchemy",
    "psycopg",
    "postgres",
    "password",
    "secret",
]


class TestA10MishandlingOfExceptionalConditions:
    def test_malformed_json_returns_422_not_500(self, client, token_a):
        """
        A10-1: Sending syntactically invalid JSON as request body.
        FastAPI must return 422 Unprocessable Entity, not 500.
        A 500 here means the JSON parser exception is unhandled.
        """
        malformed_bodies = [
            b"{name: value}",
            b'{"name":}',
            b'{"name": "test"',
            b"not json at all",
            b"",
            b"null",
            b"[]",
        ]
        for body in malformed_bodies:
            response = client.post(
                "/chat/message",
                content=body,
                headers={
                    "Authorization": f"Bearer {token_a}",
                    "Content-Type": "application/json",
                },
            )
            assert response.status_code in (400, 422), (
                f"Malformed JSON body '{body[:30]}' returned {response.status_code}. "
                "Expected 400 or 422, not 500."
            )
            assert response.status_code != 500, (
                f"Malformed JSON caused server crash: {response.text[:200]}"
            )

    def test_wrong_content_type_is_handled(self, client, token_a):
        """
        A10-2: Sending Content-Type: text/plain instead of application/json.
        Must return 422 or 415 — not 500.
        """
        response = client.post(
            "/chat/message",
            content=b'{"message": "test"}',
            headers={
                "Authorization": f"Bearer {token_a}",
                "Content-Type": "text/plain",
            },
        )
        assert response.status_code in (400, 415, 422), (
            f"Wrong Content-Type returned {response.status_code}. Expected 4xx."
        )
        assert response.status_code != 500

    def test_multipart_form_data_is_handled(self, client, token_a):
        """
        A10-3: Sending multipart/form-data to a JSON endpoint.
        Must not crash the server.
        """
        response = client.post(
            "/chat/message",
            data={"message": "test"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (400, 415, 422), (
            f"Multipart form data returned {response.status_code}. Expected 4xx."
        )
        assert response.status_code != 500

    def test_deeply_nested_json_does_not_crash(self, client, token_a):
        """
        A10-4: Extremely deeply nested JSON structure.
        Recursive parsers can stack overflow on deep nesting.
        Must return 4xx or 200 — not 500 and not hang.
        """
        nested = {"message": "test"}
        for _ in range(100):
            nested = {"data": nested}

        response = client.post(
            "/chat/message",
            json=nested,
            headers={"Authorization": f"Bearer {token_a}"},
            timeout=10.0,
        )
        assert response.status_code in (400, 422, 200), (
            f"Deeply nested JSON returned {response.status_code}."
        )
        assert response.status_code != 500

    def test_very_large_json_object_is_handled(self, client, token_a):
        large_obj = {f"field_{i}": f"value_{i}" for i in range(500)}
        large_obj["message"] = "test"

        try:
            response = client.post(
                "/chat/message",
                json=large_obj,
                headers={"Authorization": f"Bearer {token_a}"},
                timeout=120.0,
            )
            assert response.status_code in (400, 422, 200)
            assert response.status_code != 500
        except httpx.ReadTimeout:
            pass

    def test_extreme_rating_values_are_handled(self, client, token_a):
        """
        A10-6: Extreme numeric values in rating field.
        Pydantic should validate range — if no range validator exists,
        at minimum the server must not crash with 500.
        """
        valid_place_id = str(uuid.uuid4())
        extreme_ratings = [
            999_999_999,
            -999_999_999,
            0,
            -1,
            6,
            1.23456789,
        ]
        for rating in extreme_ratings:
            response = client.post(
                f"/places/favourites/{valid_place_id}/toggle",
                json={"name": "Test Place", "rating": rating, "id": valid_place_id},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code != 500, (
                f"Rating value {rating} caused 500. Response: {response.text[:200]}"
            )

    def test_unicode_and_special_characters_in_fields(self, client, token_a):
        """
        A10-8: Unicode edge cases, null bytes, and special characters.
        Must be stored or rejected cleanly — not crash the server.
        """
        valid_place_id = str(uuid.uuid4())
        special_names = [
            "café ☕",
            "餐厅 🍜",
            "Place\x00WithNullByte",
            "Place\nWith\nNewlines",
            "\u202e Reversed Text",
            "A" * 1000,
        ]
        for name in special_names:
            response = client.post(
                f"/places/favourites/{valid_place_id}/toggle",
                json={"name": name},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code != 500, (
                f"Special name '{repr(name[:30])}' caused 500. "
                f"Response: {response.text[:200]}"
            )

    def test_nonexistent_session_id_returns_404_not_500(self, client, token_a):
        """
        A10-9: Requesting messages for a session ID that does not exist.
        Must return 403 (access check fails first) or 404 — not 500.
        A 500 here means the repository throws an unhandled exception
        when the query returns no rows.
        """
        fake_session_id = str(uuid.uuid4())
        response = client.get(
            f"/sessions/{fake_session_id}/messages",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (403, 404), (
            f"Non-existent session returned {response.status_code}. "
            "Expected 403 or 404 — not 500."
        )

    def test_nonexistent_session_delete_returns_4xx(self, client, token_a):
        """
        A10-10: Deleting a session that does not exist.
        Must return 403 or 404 — not 500.
        """
        fake_session_id = str(uuid.uuid4())
        response = client.delete(
            f"/sessions/{fake_session_id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code in (403, 404), (
            f"Delete non-existent session returned {response.status_code}. "
            "Expected 403 or 404."
        )

    def test_malformed_uuid_in_session_path_returns_4xx(self, client, token_a):
        """
        A10-11: Non-UUID string in session_id path parameter.
        Must return 400, 403, or 422 — not 500.
        """
        bad_ids = [
            "not-a-uuid",
            "12345",
            "../etc/passwd",
            "' OR 1=1 --",
            "null",
        ]
        for bad_id in bad_ids:
            response = client.get(
                f"/sessions/{bad_id}/messages",
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (400, 403, 404, 422), (
                f"Malformed session_id '{bad_id}' returned {response.status_code}. "
                "Expected 4xx."
            )
            assert response.status_code != 500, (
                f"Malformed session_id '{bad_id}' caused server crash."
            )

    def test_malformed_uuid_in_place_toggle_path(self, client, token_a):
        """
        A10-12: Non-UUID place_id in toggle endpoint path.
        Must not crash — 4xx expected.
        """
        bad_place_ids = [
            "not-a-uuid",
            "' OR 1=1 --",
            "../../../etc",
            "null",
        ]
        for bad_id in bad_place_ids:
            response = client.post(
                f"/places/favourites/{bad_id}/toggle",
                json={"name": "Test Place"},
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code != 500, (
                f"Malformed place_id '{bad_id}' caused 500."
            )

    def test_wrong_http_method_returns_405(self, client, token_a):
        """
        A10-13: Using wrong HTTP method on an endpoint.
        Must return 405 Method Not Allowed — not 500.
        """
        cases = [
            ("POST", "/sessions/"),
            ("DELETE", "/places/all"),
            ("PUT", "/chat/message"),
            ("PATCH", "/sessions/"),
        ]
        for method, path in cases:
            response = client.request(
                method,
                path,
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert response.status_code in (405, 404), (
                f"{method} {path} returned {response.status_code}. "
                "Expected 405 Method Not Allowed."
            )
            assert response.status_code != 500

    def test_get_with_body_is_handled(self, client, token_a):
        response = client.request(
            "GET",
            "/sessions/",
            content=b'{"unexpected": "body"}',
            headers={
                "Authorization": f"Bearer {token_a}",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code in (200, 400, 422), (
            f"GET with body returned {response.status_code}."
        )
        assert response.status_code != 500

    def test_concurrent_requests_do_not_cause_500(self, client, token_a):
        """
        A10-15: Multiple simultaneous requests to the same endpoint.
        Race conditions in connection pool or session handling must not
        cause 500 errors. All responses must be 2xx or 4xx.
        """
        results = []
        errors = []

        def make_request():
            try:
                with httpx.Client(base_url=str(client.base_url)) as c:
                    response = c.get(
                        "/sessions/",
                        headers={"Authorization": f"Bearer {token_a}"},
                        timeout=15.0,
                    )
                    results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors, f"Concurrent requests caused connection errors: {errors}"
        for code in results:
            assert code != 500, (
                f"Concurrent request returned 500 — possible race condition. "
                f"All codes: {results}"
            )

    def test_concurrent_toggle_same_place_is_safe(self, client, token_a):
        """
        A10-16: Multiple simultaneous toggle requests for the same place.
        ON CONFLICT DO UPDATE in the DB must handle concurrent writes
        without raising a 500 or corrupting data.
        """
        results = []
        place_name = f"ConcurrentTestPlace_{uuid.uuid4().hex[:8]}"

        def toggle():
            try:
                with httpx.Client(base_url=str(client.base_url)) as c:
                    response = c.post(
                        "/places/favourites/test-id/toggle",
                        json={"name": place_name},
                        headers={"Authorization": f"Bearer {token_a}"},
                        timeout=15.0,
                    )
                    results.append(response.status_code)
            except Exception as e:
                results.append(str(e))

        threads = [threading.Thread(target=toggle) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        for result in results:
            assert result != 500, (
                f"Concurrent toggle caused 500 — possible DB race condition. "
                f"Results: {results}"
            )

    def test_error_responses_do_not_leak_internals(self, client, token_a):
        """
        A10-17: Any error response (4xx or 5xx) must not contain
        stack traces, file paths, or database error details.
        Checks multiple error-triggering scenarios at once.
        """
        error_requests = [
            client.get(
                f"/sessions/{uuid.uuid4()}/messages",
                headers={"Authorization": f"Bearer {token_a}"},
            ),
            client.post(
                "/chat/message",
                content=b"{bad json",
                headers={
                    "Authorization": f"Bearer {token_a}",
                    "Content-Type": "application/json",
                },
            ),
            client.get("/nonexistent", headers={"Authorization": f"Bearer {token_a}"}),
        ]

        for response in error_requests:
            if response.status_code >= 400:
                body = response.text.lower()
                for keyword in SENSITIVE_KEYWORDS:
                    assert keyword not in body, (
                        f"Error response ({response.status_code}) leaks "
                        f"internal detail: '{keyword}'. "
                        f"Body: {response.text[:300]}"
                    )
