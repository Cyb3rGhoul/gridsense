from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_uses_local_vendor_assets():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "/static/vendor/leaflet/leaflet.css" in html
    assert "/static/vendor/leaflet/leaflet.js" in html
    assert "/static/vendor/chart.umd.min.js" in html
    assert "/static/vendor/date-fns.cdn.min.js" in html
    assert "/static/vendor/chartjs-adapter-date-fns.bundle.min.js" in html
    assert "cdn.jsdelivr.net" not in html
    assert "fonts.googleapis.com" not in html
    assert "fonts.gstatic.com" not in html


def test_vendor_assets_exist():
    vendor = ROOT / "static" / "vendor"
    for name in [
        "chart.umd.min.js",
        "date-fns.cdn.min.js",
        "chartjs-adapter-date-fns.bundle.min.js",
    ]:
        path = vendor / name
        assert path.exists()
        assert path.stat().st_size > 1000
    leaflet_files = [
        ROOT / "static" / "vendor" / "leaflet" / "leaflet.css",
        ROOT / "static" / "vendor" / "leaflet" / "leaflet.js",
        ROOT / "static" / "vendor" / "leaflet" / "images" / "marker-icon.png",
    ]
    for path in leaflet_files:
        assert path.exists()
        assert path.stat().st_size > 1000
