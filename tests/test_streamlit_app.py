"""
Tests for the Streamlit UI, using Streamlit's own AppTest framework
(streamlit.testing.v1) so these actually execute streamlit_app.py end
to end -- not just a syntax check. This was a known gap: every other
test in this suite exercises app.py (Flask) through Flask's test
client, and nothing previously ran the Streamlit app at all.
"""
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def _run():
    at = AppTest.from_file("../streamlit_app.py")
    at.run(timeout=30)
    assert not at.exception, f"App raised on load: {at.exception}"
    return at


def test_app_loads_without_error():
    _run()


def test_default_condition_shows_correct_field_count():
    at = _run()
    # Diabetes (first condition alphabetically/in config) has 8 numeric
    # fields and no categorical ones.
    assert len(at.number_input) == 8


def test_switching_condition_updates_form_fields():
    at = _run()
    at.selectbox[0].set_value("heart_disease").run(timeout=30)
    assert not at.exception
    # heart_disease: 5 numeric fields + 8 categorical (as selectboxes) + 1 condition picker
    assert len(at.number_input) == 5
    assert len(at.selectbox) == 9


def test_prediction_flow_end_to_end():
    at = _run()
    values = [2, 150, 80, 30, 100, 33.0, 0.6, 45]
    for inp, val in zip(at.number_input, values):
        inp.set_value(val)
    at.button[0].click().run(timeout=30)

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Result:" in s for s in subheaders)

    markdown_text = " ".join(m.value for m in at.markdown)
    assert "Estimated likelihood" in markdown_text


def test_prediction_flow_all_conditions():
    """Each condition's form should submit and produce a result without
    raising, using midpoint values for every field (works for both
    numeric fields via number_input defaults and categorical fields via
    each selectbox's first option)."""
    for condition in ["diabetes", "heart_disease", "breast_cancer", "liver_disease"]:
        at = _run()
        at.selectbox[0].set_value(condition).run(timeout=30)
        assert not at.exception

        for inp in at.number_input:
            if inp.value in (None, 0.0):
                inp.set_value(1.0)

        at.button[0].click().run(timeout=30)
        assert not at.exception, f"{condition} prediction raised: {at.exception}"
        subheaders = [s.value for s in at.subheader]
        assert any("Result:" in s for s in subheaders), f"{condition} produced no result"
