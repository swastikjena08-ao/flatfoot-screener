"""Grading and severity rules in remi_ai.engine. No model weights needed: fake models stand in."""
import pytest

from remi_ai.engine import Engine, adjust_points, bunion_grade, findings, flat_category


@pytest.mark.parametrize("angle, category", [
    (None, "Not measured"), (0, "Flat foot"), (17.6, "Flat foot"), (17.7, "Normal arch"),
    (30.0, "Normal arch"), (30.1, "High arch"), (45, "High arch"),
])
def test_flat_category_thresholds(angle, category):
    assert flat_category(angle) == category


@pytest.mark.parametrize("hva, ima, grade", [
    (14.9, 8.9, "No bunion"), (15, 0, "Mild bunion"), (20, 0, "Mild bunion"), (20.1, 0, "Moderate bunion"),
    (40, 0, "Moderate bunion"), (40.1, 0, "Severe bunion"), (0, 9, "Mild bunion"), (0, 11, "Mild bunion"),
    (0, 11.1, "Moderate bunion"), (0, 16, "Moderate bunion"), (0, 16.1, "Severe bunion"),
    (16, 17, "Severe bunion"),  # the worse of the two angles wins
])
def test_bunion_grade_boundaries(hva, ima, grade):
    assert bunion_grade(hva, ima) == grade


# Six points: toe line (0-1), 1st metatarsal (2-3), 2nd metatarsal (4-5). The 1st metatarsal is vertical,
# so the toe and 2nd metatarsal lines set HVA and IMA directly.
def six_points(hva_deg, ima_deg):
    import math
    def line(deg):
        r = math.radians(deg)
        return [100.0, 100.0], [100.0 + 100 * math.sin(r), 100.0 - 100 * math.cos(r)]
    return [*line(hva_deg), [100.0, 100.0], [100.0, 0.0], *line(ima_deg)]


def bunion_results(n_feet=2):
    feet = [{"side": s, "hva": 0, "ima": 0, "points": six_points(0, 0), "box": [0, 0, 1, 1],
             "grade": "No bunion", "severity": "ok"} for s in ("right", "left")[:n_feet]]
    return {"checks": {"bunion": {"feet": feet, "severity": "ok"}}}


@pytest.mark.parametrize("hva, ima, severity", [
    (5, 5, "ok"), (18, 5, "warn"), (5, 10, "warn"), (25, 5, "alert"), (5, 20, "alert"), (45, 5, "alert"),
])
def test_adjusting_bunion_points_sets_grade_severity(hva, ima, severity):
    res = bunion_results()
    foot = adjust_points(res, "bunion", 0, six_points(hva, ima))
    assert (round(foot["hva"]), round(foot["ima"])) == (hva, ima)
    assert foot["severity"] == severity and foot["edited"] is True
    assert res["checks"]["bunion"]["severity"] == severity  # the other foot is "ok", so this foot is the worst


def test_bunion_check_severity_is_worst_foot():
    res = bunion_results()
    adjust_points(res, "bunion", 0, six_points(25, 5))   # alert
    adjust_points(res, "bunion", 1, six_points(18, 5))   # warn
    assert res["checks"]["bunion"]["severity"] == "alert"
    adjust_points(res, "bunion", 0, six_points(5, 5))    # ok
    assert res["checks"]["bunion"]["severity"] == "warn"


@pytest.mark.parametrize("angle, category, severity", [
    (10, "Flat foot", "warn"), (25, "Normal arch", "ok"), (35, "High arch", "warn"), (60, "Not measured", "ok"),
])
def test_adjusting_heel_points_sets_category_severity(angle, category, severity):
    import math
    r = math.radians(angle)
    res = {"checks": {"flat_foot": {"left": False, "points": {}}}}
    p10 = [100.0, 100.0]
    p11 = [100.0 + 100 * math.cos(r), 100.0 + 100 * math.sin(r)]
    out = adjust_points(res, "flat_foot", None, {"L10": p10, "L11": p11})
    assert out["angle"] == pytest.approx(angle, abs=0.1)
    assert (out["category"], out["severity"]) == (category, severity)


class FakeBunion:
    finder = None

    def __init__(self, feet):
        self.feet = feet

    def predict(self, gray, boxes=None):
        return self.feet


@pytest.mark.parametrize("angles, severities, overall", [
    ([(5, 5)], ["ok"], "ok"),
    ([(5, 5), (18, 5)], ["ok", "warn"], "warn"),
    ([(25, 5), (18, 5)], ["alert", "warn"], "alert"),
])
def test_bunion_run_sets_foot_and_overall_severity(angles, severities, overall):
    from PIL import Image
    feet = [{"side": "right", "hva": h, "ima": i, "points": six_points(h, i), "box": (0, 0, 9, 9)} for h, i in angles]
    out = Engine("unused")._run_bunion(FakeBunion(feet), Image.new("L", (20, 20)), {})
    assert [f["severity"] for f in out["feet"]] == severities and out["severity"] == overall


class FakeFlatFoot:
    def __init__(self, angle):
        self.angle = angle

    def predict(self, img):
        return {"angle": self.angle, "p_flat": 0.5, "left": False, "points": {"L10": (1, 1), "L11": (2, 2)},
                "heatmap": [[0.0, 0.0], [0.0, 0.0]]}


@pytest.mark.parametrize("angle, category, severity", [
    (10, "Flat foot", "warn"), (25, "Normal arch", "ok"), (35, "High arch", "warn"),
    (None, "Not measured", "ok"), (50, "Not measured", "ok"),
])
def test_flat_foot_run_sets_category_severity(angle, category, severity):
    from PIL import Image
    out = Engine("unused")._run_flat_foot(FakeFlatFoot(angle), Image.new("L", (20, 20)), {})  # no heel box
    assert (out["category"], out["severity"]) == (category, severity)
    title, _, sev = findings({"checks": {"flat_foot": out}})[0]
    assert (title, sev) == (category, severity)

