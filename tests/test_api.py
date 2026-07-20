import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    """Verifies that the API root endpoint is online and responsive."""
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["message"].lower()

def test_api_models_endpoint():
    """Verifies that the /models endpoint returns active state and history logs."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "history" in data

def test_api_predict_invalid_schema():
    """Verifies that the /predict endpoint correctly blocks invalid inputs with 422 error."""
    # Invalid CGPA (12.5 is above the 10.0 max boundary)
    invalid_payload = {
        "Student_ID": "STU9999",
        "Gender": "Male",
        "Age": 22,
        "Branch": "Computer Science",
        "Degree": "B.Tech",
        "CGPA": 12.5,
        "10th Percentage": 88.0,
        "12th Percentage": 84.5,
        "Backlogs": 0,
        "Attendance": 90.0,
        "Technical Skills": 78.0,
        "Programming Score": 82.0,
        "Aptitude Score": 75.0,
        "Communication Skills": 80.0,
        "Soft Skills": 78.0,
        "Projects Completed": 2,
        "Internships": 1,
        "Hackathons": 1,
        "Certifications": 2,
        "Workshops": 1,
        "Leadership Score": 70.0,
        "Resume Score": 80.0,
        "Interview Score": 75.0,
        "English Proficiency": "High",
        "Expected Salary": 500000.0
    }
    
    response = client.post("/predict", json=invalid_payload)
    # FastAPI returns 422 Unprocessable Entity for schema validation failures
    assert response.status_code == 422
