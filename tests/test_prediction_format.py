from src.inference.predict import risk_level, format_prediction


def test_risk_level_low():
    assert risk_level(80) == "Low"


def test_risk_level_medium():
    assert risk_level(50) == "Medium"


def test_risk_level_high():
    assert risk_level(20) == "High"


def test_format_prediction_real():
    result = format_prediction(1)

    assert result["prediction_label"] == "real"
    assert result["credibility_score"] == 80
    assert result["risk_level"] == "Low"


def test_format_prediction_fake():
    result = format_prediction(0)

    assert result["prediction_label"] == "fake"
    assert result["credibility_score"] == 25
    assert result["risk_level"] == "High"