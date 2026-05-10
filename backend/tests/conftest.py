import pytest
import httpx
import os
from dotenv import load_dotenv

load_dotenv("tests/.env.test")

BASE_URL = os.getenv("BASE_URL")
COGNITO_URL = os.getenv("COGNITO_URL")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
USER_A_EMAIL = os.getenv("USER_A_EMAIL")
USER_A_PASSWORD = os.getenv("USER_A_PASSWORD")
USER_B_EMAIL = os.getenv("USER_B_EMAIL")
USER_B_PASSWORD = os.getenv("USER_B_PASSWORD")


def get_token(email: str, password: str) -> str:
    response = httpx.post(
        COGNITO_URL,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
        },
        json={
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": APP_CLIENT_ID,
            "AuthParameters": {
                "USERNAME": email,
                "PASSWORD": password,
            },
        },
    )
    assert response.status_code == 200, f"Błąd logowania: {response.text}"
    return response.json()["AuthenticationResult"]["AccessToken"]


@pytest.fixture(scope="session")
def token_a():
    return get_token(USER_A_EMAIL, USER_A_PASSWORD)


@pytest.fixture(scope="session")
def token_b():
    return get_token(USER_B_EMAIL, USER_B_PASSWORD)


@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE_URL)


@pytest.fixture(scope="session")
def session_id_a(token_a, client):
    """Pobiera pierwsze session_id należące do użytkownika A"""
    response = client.get(
        "/sessions/",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) > 0, "Użytkownik A nie ma żadnych sesji — utwórz je ręcznie"
    return sessions[0]["id"]
