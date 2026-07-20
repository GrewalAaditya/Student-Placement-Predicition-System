import pytest
from src.predict import PlacementPredictor

@pytest.fixture
def predictor():
    return PlacementPredictor()

def test_readiness_score(predictor):
    """Verifies calculated readiness score math bounds."""
    data = {
        "Programming Score": 80.0,
        "Aptitude Score": 70.0,
        "Interview Score": 70.0,
        "Resume Score": 80.0,
        "Internships": 1,
        "Projects Completed": 2
    }
    
    score = predictor.calculate_readiness_score(data)
    # Math: (80/100)*25 [20] + (70/100)*15 [10.5] + (70/100)*20 [14] + (80/100)*15 [12] + 1*10 [10] + 2*2.5 [5] = 71.5
    assert score == 71.5
    assert 0 <= score <= 100

def test_salary_estimation(predictor):
    """Verifies that predicted placed student salary is higher than not placed student."""
    placed_sal = predictor.estimate_salary_range(1, {"CGPA": 8.5, "Programming Score": 80})
    unplaced_sal = predictor.estimate_salary_range(0, {"CGPA": 8.5, "Programming Score": 80})
    
    assert "LPA" in placed_sal
    assert "LPA" in unplaced_sal
    
    # Placed salary should be higher than unplaced salary base range
    placed_val = float(placed_sal.split(" - ")[0])
    unplaced_val = float(unplaced_sal.split(" - ")[0])
    assert placed_val > unplaced_val

def test_eligibility_checker(predictor):
    """Verifies backlogs and attendance blocker rules."""
    # Eligible student
    el_data = {"Backlogs": 0, "Attendance": 80.0}
    el_res = predictor.check_eligibility(el_data)
    assert el_res["eligible"] is True
    
    # Blocked by backlogs
    bk_data = {"Backlogs": 2, "Attendance": 80.0}
    bk_res = predictor.check_eligibility(bk_data)
    assert bk_res["eligible"] is False
    
    # Blocked by attendance
    att_data = {"Backlogs": 0, "Attendance": 70.0}
    att_res = predictor.check_eligibility(att_data)
    assert att_res["eligible"] is False

def test_suggestions_generator(predictor):
    """Verifies suggestions list generates warnings for weak features."""
    data = {
        "Backlogs": 1,
        "Attendance": 68.0,
        "Programming Score": 45.0,
        "Internships": 0
    }
    
    sugs = predictor.generate_suggestions(data)
    
    # Check that critical warnings are generated
    assert any("backlog" in s.lower() for s in sugs)
    assert any("attendance" in s.lower() for s in sugs)
    assert any("programming" in s.lower() for s in sugs)
    assert any("internship" in s.lower() for s in sugs)
