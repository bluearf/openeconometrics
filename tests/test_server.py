from fastapi.testclient import TestClient
import pytest

from openecon.server import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        token = client.get("/api/session").json()["token"]
        client.headers["X-OpenEcon-Token"] = token
        yield client


def test_api_analysis_and_downloads(client):
    data = client.post("/api/datasets/example").json()
    response = client.post(
        "/api/analyses",
        json={
            "dataset_id": data["id"],
            "spec": {"outcome": "wage", "predictors": ["education", "experience"]},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["coefficients"]) == 3
    assert result["nobs"] == 480
    assert client.get(f"/api/results/{result['id']}").json() == result
    assert "oe.fit" in client.get(f"/api/results/{result['id']}/python").text
    bundle = client.get(f"/api/results/{result['id']}/bundle")
    assert bundle.status_code == 200
    assert bundle.content.startswith(b"PK")


def test_real_upload_and_missing_error(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("small.csv", b"x,y\n1,2\n2,4\n3,\n4,7\n5,12\n", "text/csv")},
    )
    assert response.status_code == 200, response.text
    dataset = response.json()
    assert dataset["name"] == "small.csv"
    assert dataset["columns"][1]["missing"] == 1
    failed = client.post(
        "/api/analyses",
        json={"dataset_id": dataset["id"], "spec": {"outcome": "y", "predictors": ["x"]}},
    )
    assert failed.status_code == 422
    assert "code" in failed.json()["detail"]


def test_api_requires_local_session_and_blocks_cross_origin(client):
    assert client.get("/api/datasets", headers={"X-OpenEcon-Token": "wrong"}).status_code == 401
    assert client.get("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/session", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_unknown_model_fields_not_silently_ignored(client):
    data = client.post("/api/datasets/example").json()
    response = client.post(
        "/api/analyses",
        json={
            "dataset_id": data["id"],
            "spec": {"outcome": "wage", "predictors": ["education"], "weights": "hours"},
        },
    )
    assert response.status_code == 422


def test_bad_upload_format_and_ids(client):
    assert (
        client.post(
            "/api/datasets/upload", files={"file": ("script.py", b"print('no')")}
        ).status_code
        == 422
    )
    assert client.get("/api/datasets/not-a-record").status_code == 404


def test_chart_assets_are_packaged_local_and_keep_access_boundaries(client):
    for filename, media in [("d3.min.js", "javascript"), ("renderer.js", "javascript"),
                            ("charts.css", "text/css"), ("Barlow.woff2", "font/woff2")]:
        response = client.get(f"/chart-assets/{filename}")
        assert response.status_code == 200, filename
        assert media in response.headers["content-type"]
        assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/chart-assets/../pyproject.toml").status_code == 404
    assert client.get("/chart-assets/renderer.js", headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert "OpenEconCharts" in client.get("/chart-assets/renderer.js").text
