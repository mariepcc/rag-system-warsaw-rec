"""
OWASP A06:2025 — Insecure Design
Also covers: API1 (BOLA), API5 (BFLA), M3 (Insecure Auth/Authorization), M6 (Privacy Controls)

Verifies that the business logic of the API is architecturally secure —
not only at the implementation level, but at the design level.

Tested scenarios:
- data isolation between users (IDOR)
- no access to other users' sessions and favourites
- no enumeration of other users' resources
- correctness of toggle logic (idempotency)
"""

import uuid
import pytest


class TestA06InsecureDesign:
    def test_user_b_cannot_see_user_a_sessions(self, client, token_a, token_b):
        """
        A06-1: Sessions of user A must not be visible to user B.
        GET /sessions/ must return only the sessions of the authenticated user.
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

        sessions_a = {s["id"] for s in response_a.json()}
        sessions_b = {s["id"] for s in response_b.json()}

        overlap = sessions_a & sessions_b
        assert not overlap, (
            f"Users A and B see the same sessions: {overlap}. "
            "Sessions must be isolated per user — potential A06/API1 vulnerability!"
        )

    def test_user_b_cannot_read_messages_from_user_a_session(
        self, client, token_a, token_b
    ):
        """
        A06-2: User B cannot read messages from user A's session.
        Even if they know the session_id — they must receive 403.
        Covers: API1 (BOLA) — classic IDOR attack
        """
        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 200
        sessions = response.json()

        if not sessions:
            pytest.skip(
                "User A has no sessions — run /chat/message first to create one"
            )

        session_id_a = sessions[0]["id"]
        response_b = client.get(
            f"/sessions/{session_id_a}/messages",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_b.status_code == 403, (
            f"User B accessed user A's session (HTTP {response_b.status_code}). "
            "Expected 403 Forbidden — potential A06/API1 vulnerability!"
        )

    def test_user_b_cannot_delete_user_a_session(self, client, token_a, token_b):
        """
        A06-3: User B cannot delete user A's session.
        DELETE /sessions/{id} must verify ownership before deleting.
        Covers: API5 (BFLA) — broken function level authorization
        """
        response = client.get(
            "/sessions/",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 200
        sessions = response.json()

        if not sessions:
            pytest.skip("User A has no sessions")

        session_id_a = sessions[0]["id"]

        response_b = client.delete(
            f"/sessions/{session_id_a}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_b.status_code in (403, 404), (
            f"User B deleted user A's session (HTTP {response_b.status_code}). "
            "Expected 403 or 404 — potential A06/API5 vulnerability!"
        )

        response_check = client.get(
            f"/sessions/{session_id_a}/messages",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response_check.status_code in (200, 403), (
            "User A's session was deleted by user B — potential A06 vulnerability!"
        )

    def test_favourites_are_isolated_per_user(self, client, token_a, token_b):
        """
        A06-4: GET /places/favourites must return only the favourites
        of the authenticated user. User B must not see user A's favourites.
        Covers: API1 (BOLA), M6 (Privacy Controls)
        """
        place_name = f"TestPlace_A06_{uuid.uuid4().hex[:8]}"
        place_id = str(uuid.uuid4())
        client.post(
            f"/places/favourites/{place_id}/toggle",
            json={"name": place_name, "rating": 4.5},
            headers={"Authorization": f"Bearer {token_a}"},
        )

        response_b = client.get(
            "/places/favourites",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_b.status_code == 200

        names_b = [p["name"] for p in response_b.json()]
        assert place_name not in names_b, (
            f"User A's favourite place ('{place_name}') is visible to user B. "
            "Favourites must be isolated per user — potential A06/M6 vulnerability!"
        )

    def test_favourite_names_are_isolated_per_user(self, client, token_a, token_b):
        """
        A06-5: GET /places/favourite-names must return only the names
        of places favourited by the authenticated user — not a global list.
        Covers: API1 (BOLA), M6 (Privacy Controls)
        """
        place_name = f"UniqueFavName_{uuid.uuid4().hex[:8]}"
        place_id = str(uuid.uuid4())
        client.post(
            f"/places/favourites/{place_id}/toggle",
            json={"name": place_name, "rating": 3.0},
            headers={"Authorization": f"Bearer {token_a}"},
        )

        response_b = client.get(
            "/places/favourite-names",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_b.status_code == 200
        names_b = response_b.json().get("names", [])

        assert place_name not in names_b, (
            "User A's favourite place name is visible to user B. "
            "The favourite-names endpoint does not filter by user — potential A06 vulnerability!"
        )

    def test_toggle_is_deterministic(self, client, token_a):
        """
        A06-6: Toggle logic must be deterministic.
        First call adds, second removes, third adds again.
        A flaw here means an attacker can manipulate state unpredictably.
        Covers: API6 (Unrestricted Access to Sensitive Business Flows)
        """
        place_id = str(uuid.uuid4())
        headers = {"Authorization": f"Bearer {token_a}"}
        payload = {"sessionId": None}

        r1 = client.post(
            f"/places/favourites/{place_id}/toggle", json=payload, headers=headers
        )
        assert r1.status_code in (200, 422), f"First toggle returned {r1.status_code}"

        if r1.status_code == 422:
            pytest.skip("place_id not in DB — toggle correctly rejected with 422")

        state1 = r1.json()["is_favourite"]

        r2 = client.post(
            f"/places/favourites/{place_id}/toggle", json=payload, headers=headers
        )
        assert r2.status_code == 200
        state2 = r2.json()["is_favourite"]

        r3 = client.post(
            f"/places/favourites/{place_id}/toggle", json=payload, headers=headers
        )
        assert r3.status_code == 200
        state3 = r3.json()["is_favourite"]

        assert state1 != state2, "Toggle did not change state on second call"
        assert state2 != state3, "Toggle did not change state on third call"
        assert state1 == state3, (
            "Toggle is not deterministic — state 1 and 3 should be identical"
        )

    def test_all_places_accessible_to_all_authenticated_users(
        self, client, token_a, token_b
    ):
        """
        A06-7: GET /places/all is a shared resource for authenticated users.
        Both users must see the same list — this is not private data.
        Covers: design correctness — shared vs private resource distinction
        """
        response_a = client.get(
            "/places/all",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        response_b = client.get(
            "/places/all",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        assert len(response_a.json()) == len(response_b.json()), (
            "Users A and B see a different number of places in /places/all. "
            "This is a shared resource — it should be identical for all users."
        )

    def test_random_session_ids_cannot_be_enumerated(self, client, token_b):
        """
        A06-8: If session_id is a UUID (random), other users' sessions
        cannot be enumerated. Random UUIDs must return 403/404 — not 200.
        A flaw here is a classic IDOR vulnerability.
        Covers: API1 (BOLA)
        """
        guessed_ids = [str(uuid.uuid4()) for _ in range(10)]

        for session_id in guessed_ids:
            response = client.get(
                f"/sessions/{session_id}/messages",
                headers={"Authorization": f"Bearer {token_b}"},
            )
            assert response.status_code in (403, 404), (
                f"Random session_id '{session_id}' returned {response.status_code}. "
                "Expected 403 or 404 — potential IDOR/A06 vulnerability!"
            )
            assert response.status_code != 200, (
                "Random session_id returned data (200) — possible IDOR!"
            )

    def test_search_history_scoped_to_authenticated_user(
        self, client, token_a, token_b
    ):
        """
        A06-9: GET /chat/search-history?q=... must filter by the authenticated user.
        User B must not see user A's history through search.
        Covers: API1 (BOLA), M6 (Privacy Controls)
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
            f"Search history of users A and B contains shared sessions: {overlap}. "
            "History must be isolated per user — potential A06/API1 vulnerability!"
        )
