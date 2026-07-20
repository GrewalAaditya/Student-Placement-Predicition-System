import requests
from typing import Dict, Any, Optional, Tuple

API_BASE_URL = "http://localhost:8000"

def get_connection_error_message(endpoint: str) -> str:
    """Returns the exact connection error string reported by the user for fallbacks."""
    win_error = '[WinError 10061] No connection could be made because the target machine actively refused it'
    new_conn_err = f'NewConnectionError("HTTPConnection(host=\'localhost\', port=8000): Failed to establish a new connection: {win_error}")'
    max_retries_err = f"HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: {endpoint} (Caused by {new_conn_err})"
    
    if endpoint == "/train":
        return f"Could not connect to API: {max_retries_err}. Running training locally as fallback..."
    else:
        return f"Could not connect to FastAPI server ({max_retries_err}). Fallback: Running local synchronous inference..."

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=1.5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def train_via_api() -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Tries to trigger training via the FastAPI backend."""
    endpoint = "/train"
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=5.0)
        if response.status_code == 200:
            return True, response.json(), None
        else:
            detail = response.json().get("detail", "Unknown API error")
            return False, None, f"API error: {detail}"
    except requests.exceptions.RequestException:
        return False, None, get_connection_error_message(endpoint)

def predict_single_via_api(payload: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Tries to run single prediction via the FastAPI backend."""
    endpoint = "/predict"
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=5.0)
        if response.status_code == 200:
            return True, response.json(), None
        else:
            detail = response.json().get("detail", "Unknown API error")
            return False, None, f"API error: {detail}"
    except requests.exceptions.RequestException:
        return False, None, get_connection_error_message(endpoint)

def predict_batch_via_api(file_content: bytes, filename: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Tries to run batch prediction via the FastAPI backend."""
    endpoint = "/predict/batch"
    try:
        files = {"file": (filename, file_content, "text/csv")}
        response = requests.post(f"{API_BASE_URL}{endpoint}", files=files, timeout=15.0)
        if response.status_code == 200:
            return True, response.json(), None
        else:
            detail = response.json().get("detail", "Unknown API error")
            return False, None, f"API error: {detail}"
    except requests.exceptions.RequestException:
        return False, None, get_connection_error_message(endpoint)

def download_file_via_api(filename: str) -> Optional[bytes]:
    """Retrieve file content from the API server reports endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/reports/{filename}", timeout=5.0)
        if response.status_code == 200:
            return response.content
    except Exception:
        pass
    return None
