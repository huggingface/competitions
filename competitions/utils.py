import requests
from loguru import logger

from . import MOONLANDING_URL


def get_auth_headers(token: str, prefix: str = "Bearer"):
    return {"Authorization": f"{prefix} {token}"}


def http_post(path: str, token: str, payload=None, domain: str = None, params=None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, params=params
        )
    except requests.exceptions.ConnectionError:
        logger.error("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_get(path: str, token: str, domain: str = None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(url=domain + path, headers=get_auth_headers(token=token), allow_redirects=True)
    except requests.exceptions.ConnectionError:
        logger.error("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def user_authentication(token):
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            MOONLANDING_URL + "/api/whoami-v2",
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    return response.json()
