"""
OWASP A01:2025 — Broken Access Control
Also covers: API1 (BOLA), API5 (BFLA), M3 (Insecure Auth/Authorization), M6 (Privacy Controls)

Verifies that users can only access their own resources.
"""


class TestA01AccessControl:
    def test_missing_token_returns_401_favourites(self, client):
        """
        A01-1: Access to favourites without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/places/favourites")
        assert response.status_code == 401, (
            f"Endpoint /places/favourites should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_missing_token_returns_401_sessions(self, client):
        """
        A01-2: Access to sessions without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/sessions/")
        assert response.status_code == 401, (
            f"Endpoint /sessions/ should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_missing_token_returns_401_all_places(self, client):
        """
        A01-3: Access to all places without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/places/all")
        assert response.status_code == 401, (
            f"Endpoint /places/all should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_missing_token_returns_401_favourite_names(self, client):
        """
        A01-4: Access to favourite names without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/places/favourite-names")
        assert response.status_code == 401, (
            f"Endpoint /places/favourite-names should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_missing_token_returns_401_chat_all_names(self, client):
        """
        A01-5: Access to chat all-names without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/chat/all-names")
        assert response.status_code == 401, (
            f"Endpoint /chat/all-names should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_missing_token_returns_401_search_history(self, client):
        """
        A01-6: Access to search history without a token.
        Expected result: 401 Unauthorized
        """
        response = client.get("/chat/search-history?q=test")
        assert response.status_code == 401, (
            f"Endpoint /chat/search-history should require authorization. "
            f"Got: {response.status_code}"
        )

    def test_user_b_cannot_read_user_a_session(self, client, token_b, session_id_a):
        """
        A01-7: User B attempts to read messages from user A's session.
        Expected result: 403 Forbidden
        Covers: API1 (BOLA) — accessing another user's object by ID
        """
        response = client.get(
            f"/sessions/{session_id_a}/messages",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 403, (
            f"User B should not have access to user A's session. "
            f"Got: {response.status_code} — potential A01/API1 vulnerability!"
        )

    def test_user_b_cannot_delete_user_a_session(self, client, token_b, session_id_a):
        """
        A01-8: User B attempts to delete user A's session.
        Expected result: 403 Forbidden
        Covers: API5 (BFLA) — executing a function on another user's resource
        """
        response = client.delete(
            f"/sessions/{session_id_a}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 403, (
            f"User B should not be able to delete user A's session. "
            f"Got: {response.status_code} — potential A01/API5 vulnerability!"
        )

    def test_user_a_sees_only_own_favourites(self, client, token_a, token_b):
        """
        A01-9: User A's favourites are not visible to user B.
        Expected result: favourite lists of both users are disjoint
        or user B has an empty list.
        Covers: API1 (BOLA), M6 (Privacy Controls)
        """
        response_a = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        response_b = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_a.status_code == 200
        assert response_b.status_code == 200

        places_a = {p["name"] for p in response_a.json()}
        places_b = {p["name"] for p in response_b.json()}

        if places_a:
            assert places_a != places_b, (
                "User B sees the same favourites as user A — "
                "potential A01/M6 vulnerability!"
            )

    def test_user_a_can_access_own_session(self, client, token_a, session_id_a):
        """
        A01-10: User A has access to their own session.
        Expected result: 200 OK
        Sanity check — access control must not block legitimate users.
        """
        response = client.get(
            f"/sessions/{session_id_a}/messages",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 200, (
            f"User A should have access to their own session. "
            f"Got: {response.status_code}"
        )

    def test_user_a_sees_only_own_sessions(self, client, token_a, token_b):
        """
        A01-11: Sessions list is scoped per user — user A and user B
        must not share any session IDs.
        Covers: API1 (BOLA)
        """
        response_a = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        response_b = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_a.status_code == 200
        assert response_b.status_code == 200

        ids_a = {s["id"] for s in response_a.json()}
        ids_b = {s["id"] for s in response_b.json()}

        overlap = ids_a & ids_b
        assert not overlap, (
            f"Users A and B share session IDs: {overlap} — "
            "potential A01/API1 vulnerability!"
        )

    def test_search_history_scoped_to_user(self, client, token_a, token_b):
        """
        A01-12: Search history must return only the authenticated user's sessions.
        User B must not see user A's sessions through search.
        Covers: API1 (BOLA)
        """
        response_a = client.get(
            "/chat/search-history?q=",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        response_b = client.get(
            "/chat/search-history?q=",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_a.status_code == 200
        assert response_b.status_code == 200

        ids_a = {s.get("id") for s in response_a.json() if isinstance(s, dict)}
        ids_b = {s.get("id") for s in response_b.json() if isinstance(s, dict)}

        overlap = ids_a & ids_b
        assert not overlap, (
            f"Search history of users A and B overlaps: {overlap} — "
            "potential A01/API1 vulnerability!"
        )
