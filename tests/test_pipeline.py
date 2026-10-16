import re

import pandas as pd

from tests.helpers import ROOT, run_script


def test_training_script_runs_and_reports_scores():
    output = run_script("loan_prediction.py")
    aucs = [float(x) for x in re.findall(r"ROC AUC Score: ([0-9.]+)", output)]
    assert len(aucs) == 2, "expected one ROC AUC per model"
    assert all(0.8 < auc <= 1.0 for auc in aucs)


def test_synthetic_data_matches_readme():
    data = pd.read_csv(ROOT / "loan_data.csv")
    assert len(data) == 10_000
    assert abs(data["default"].mean() - 0.4135) < 0.01
    low_score_rate = data.loc[data["credit_score"] < 600, "default"].mean()
    assert low_score_rate > data["default"].mean()


def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not app.exception
