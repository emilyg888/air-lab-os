from use_cases.fraud.feature_lab.run_feature_experiment import FeatureExperimentResult

from dashboard.feature_results import render_feature_dashboard


def test_render_feature_dashboard_includes_feature_rows():
    html = render_feature_dashboard(
        dataset_name="stub_dataset",
        baseline_f1=0.8671,
        generated_at="2026-04-10 10:00:00 AEST",
        results=[
            FeatureExperimentResult(
                feature_name="alpha_feature",
                baseline_f1=0.8671,
                experiment_f1=0.8701,
                delta_f1=0.0030,
                feature_count=8,
            ),
            FeatureExperimentResult(
                feature_name="beta_feature",
                baseline_f1=0.8671,
                experiment_f1=0.8651,
                delta_f1=-0.0020,
                feature_count=8,
            ),
        ],
    )

    assert "alpha_feature" in html
    assert "beta_feature" in html
    assert "Feature Experiment Dashboard" in html
    assert "+0.0030" in html
    assert "-0.0020" in html
