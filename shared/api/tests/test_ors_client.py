"""
Tests for OpenRouteService API client.

Tests are organized into:
- Unit tests with mocked API responses (always run)
- Integration tests hitting real ORS API (marked with @pytest.mark.integration)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root
_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.api.openrouteservice import ORSClient, get_client


# ── Mock response fixtures ──────────────────────────────────────────


def _mock_matrix_response():
    """Mock ORS matrix API response."""
    return {
        "distances": [[0, 1045.6, 2340.1], [1045.6, 0, 1832.4], [2340.1, 1832.4, 0]],
        "durations": [[0, 215.0, 480.3], [215.0, 0, 390.1], [480.3, 390.1, 0]],
    }


def _mock_directions_response():
    """Mock ORS directions API response."""
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [8.681495, 49.41461],
                    [8.683, 49.416],
                    [8.687872, 49.420318],
                ],
            },
            "properties": {
                "summary": {"distance": 1045.6, "duration": 215.0}
            },
        }],
    }


def _mock_geocode_response():
    """Mock ORS geocode API response."""
    return {
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [8.694724, 49.405882]},
            "properties": {
                "name": "Heidelberg",
                "label": "Heidelberg, Germany",
                "confidence": 0.95,
            },
        }],
    }


def _mock_isochrones_response():
    """Mock ORS isochrones API response."""
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[8.68, 49.41], [8.69, 49.42], [8.70, 49.41], [8.68, 49.41]]],
            },
            "properties": {"value": 300},
        }],
    }


# ── Unit Tests (mocked) ────────────────────────────────────────────


class TestORSClientInit:
    """Test client initialization."""

    def test_default_api_key(self):
        client = ORSClient()
        assert client.api_key is not None
        assert len(client.api_key) > 0

    def test_custom_api_key(self):
        client = ORSClient(api_key="my-test-key")
        assert client.api_key == "my-test-key"

    def test_default_profile(self):
        client = ORSClient()
        assert client.profile == "driving-car"

    def test_custom_profile(self):
        client = ORSClient(profile="foot-walking")
        assert client.profile == "foot-walking"

    def test_env_var_api_key(self):
        with patch.dict("os.environ", {"ORS_API_KEY": "env-key"}):
            client = ORSClient()
            assert client.api_key == "env-key"


class TestORSClientMatrix:
    """Test matrix service."""

    @patch.object(ORSClient, "_request")
    def test_distance_matrix(self, mock_request):
        mock_request.return_value = _mock_matrix_response()

        client = ORSClient(api_key="test")
        coords = np.array([[8.681, 49.414], [8.687, 49.420], [8.651, 49.418]])
        result = client.distance_matrix(coords, metric="distance")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0
        assert result[0, 1] > 0

    @patch.object(ORSClient, "_request")
    def test_duration_matrix(self, mock_request):
        mock_request.return_value = _mock_matrix_response()

        client = ORSClient(api_key="test")
        coords = np.array([[8.681, 49.414], [8.687, 49.420], [8.651, 49.418]])
        result = client.distance_matrix(coords, metric="duration")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0

    @patch.object(ORSClient, "_request")
    def test_matrix_with_sources_destinations(self, mock_request):
        mock_request.return_value = {
            "distances": [[1045.6, 2340.1], [832.4, 1200.0]],
        }

        client = ORSClient(api_key="test")
        locations = [[8.681, 49.414], [8.687, 49.420], [8.651, 49.418], [8.660, 49.415]]
        result = client.matrix(
            locations,
            metrics=["distance"],
            sources=[0, 1],
            destinations=[2, 3],
        )
        assert result["distances"].shape == (2, 2)


class TestORSClientDirections:
    """Test directions service."""

    @patch.object(ORSClient, "_request")
    def test_directions(self, mock_request):
        mock_request.return_value = _mock_directions_response()

        client = ORSClient(api_key="test")
        route = client.directions([
            [8.681495, 49.41461],
            [8.687872, 49.420318],
        ])

        assert "features" in route
        assert len(route["features"]) > 0
        geom = route["features"][0]["geometry"]
        assert geom["type"] == "LineString"
        assert len(geom["coordinates"]) > 0

    @patch.object(ORSClient, "_request")
    def test_route_geometry(self, mock_request):
        mock_request.return_value = _mock_directions_response()

        client = ORSClient(api_key="test")
        coords = np.array([[8.681495, 49.41461], [8.687872, 49.420318]])
        geom = client.route_geometry(coords)

        assert isinstance(geom, list)
        assert len(geom) == 3
        assert len(geom[0]) == 2  # [lon, lat]


class TestORSClientGeocode:
    """Test geocoding service."""

    @patch.object(ORSClient, "_request")
    def test_geocode(self, mock_request):
        mock_request.return_value = _mock_geocode_response()

        client = ORSClient(api_key="test")
        results = client.geocode("Heidelberg, Germany")

        assert len(results) > 0
        assert "coordinates" in results[0]
        assert results[0]["coordinates"] == [8.694724, 49.405882]
        assert results[0]["name"] == "Heidelberg"

    @patch.object(ORSClient, "_request")
    def test_geocode_locations(self, mock_request):
        mock_request.return_value = _mock_geocode_response()

        client = ORSClient(api_key="test")
        coords = client.geocode_locations(["Heidelberg", "Mannheim"])

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (2, 2)

    @patch.object(ORSClient, "_request")
    def test_geocode_not_found(self, mock_request):
        mock_request.return_value = {"features": []}

        client = ORSClient(api_key="test")
        with pytest.raises(ValueError, match="Could not geocode"):
            client.geocode_locations(["nonexistent_place_xyz123"])


class TestORSClientIsochrones:
    """Test isochrones service."""

    @patch.object(ORSClient, "_request")
    def test_isochrones(self, mock_request):
        mock_request.return_value = _mock_isochrones_response()

        client = ORSClient(api_key="test")
        result = client.isochrones(
            [[8.681495, 49.41461]],
            range_seconds=[300, 600],
        )

        assert "features" in result
        assert len(result["features"]) > 0
        geom = result["features"][0]["geometry"]
        assert geom["type"] == "Polygon"


class TestORSClientOptimize:
    """Test optimization (VROOM) service."""

    @patch.object(ORSClient, "_request")
    def test_optimize(self, mock_request):
        mock_request.return_value = {
            "code": 0,
            "summary": {"cost": 1234, "routes": 1},
            "routes": [{
                "vehicle": 1,
                "steps": [
                    {"type": "start", "location": [8.694, 49.406]},
                    {"type": "job", "id": 1, "location": [8.688, 49.420]},
                    {"type": "end", "location": [8.694, 49.406]},
                ],
            }],
            "unassigned": [],
        }

        client = ORSClient(api_key="test")
        result = client.optimize(
            jobs=[{"id": 1, "location": [8.688, 49.420], "service": 300}],
            vehicles=[{
                "id": 1,
                "start": [8.694, 49.406],
                "end": [8.694, 49.406],
                "capacity": [100],
            }],
        )

        assert result["code"] == 0
        assert len(result["routes"]) == 1
        assert len(result["unassigned"]) == 0


class TestGetClient:
    """Test convenience function."""

    def test_get_client(self):
        client = get_client(api_key="test-key", profile="foot-walking")
        assert isinstance(client, ORSClient)
        assert client.api_key == "test-key"
        assert client.profile == "foot-walking"


# ── Instance Factory Tests (mocked) ────────────────────────────────


class TestTSPFromORS:
    """Test TSP.from_ors with mocked API."""

    @patch.object(ORSClient, "_request")
    def test_from_ors_coordinates(self, mock_request):
        mock_request.return_value = _mock_matrix_response()

        from problems.routing.tsp.instance import TSPInstance
        inst = TSPInstance.from_ors(
            locations=[[8.681, 49.414], [8.687, 49.420], [8.651, 49.418]],
            api_key="test",
        )

        assert inst.n == 3
        assert inst.distance_matrix.shape == (3, 3)
        assert inst.coords is not None
        assert inst.coords.shape == (3, 2)
        assert inst.name == "ors_tsp"

    @patch.object(ORSClient, "_request")
    def test_from_ors_place_names(self, mock_request):
        # First call: geocode, second call: matrix
        def side_effect(method, endpoint, **kwargs):
            if "geocode" in endpoint:
                return _mock_geocode_response()
            return _mock_matrix_response()

        mock_request.side_effect = side_effect

        from problems.routing.tsp.instance import TSPInstance
        inst = TSPInstance.from_ors(
            locations=["Heidelberg", "Mannheim", "Karlsruhe"],
            api_key="test",
        )
        assert inst.n == 3


class TestCVRPFromORS:
    """Test CVRP.from_ors with mocked API."""

    @patch.object(ORSClient, "_request")
    def test_from_ors(self, mock_request):
        mock_request.return_value = {
            "distances": [
                [0, 1000, 2000, 1500],
                [1000, 0, 1200, 800],
                [2000, 1200, 0, 1800],
                [1500, 800, 1800, 0],
            ],
        }

        from problems.routing.cvrp.instance import CVRPInstance
        inst = CVRPInstance.from_ors(
            depot=[8.694, 49.406],
            customers=[[8.681, 49.414], [8.687, 49.420], [8.651, 49.418]],
            demands=[10, 15, 20],
            capacity=50,
            api_key="test",
        )

        assert inst.n == 3
        assert inst.capacity == 50
        assert inst.distance_matrix.shape == (4, 4)
        assert inst.coords.shape == (4, 2)


class TestFacilityLocationFromORS:
    """Test FacilityLocation.from_ors with mocked API."""

    @patch.object(ORSClient, "_request")
    def test_from_ors(self, mock_request):
        mock_request.return_value = {
            "distances": [[500, 1200, 800], [1100, 400, 900]],
        }

        from problems.location_network.facility_location.instance import (
            FacilityLocationInstance,
        )
        inst = FacilityLocationInstance.from_ors(
            facilities=[[8.681, 49.414], [8.687, 49.420]],
            customers=[[8.651, 49.418], [8.660, 49.415], [8.670, 49.410]],
            fixed_costs=[1000, 1500],
            api_key="test",
        )

        assert inst.m == 2
        assert inst.n == 3
        assert inst.assignment_costs.shape == (2, 3)
        assert inst.coords_facilities.shape == (2, 2)
        assert inst.coords_customers.shape == (3, 2)


class TestPMedianFromORS:
    """Test PMedian.from_ors with mocked API."""

    @patch.object(ORSClient, "_request")
    def test_from_ors(self, mock_request):
        mock_request.return_value = {
            "distances": [[500, 1200, 800], [1100, 400, 900]],
        }

        from problems.location_network.p_median.instance import PMedianInstance
        inst = PMedianInstance.from_ors(
            facilities=[[8.681, 49.414], [8.687, 49.420]],
            customers=[[8.651, 49.418], [8.660, 49.415], [8.670, 49.410]],
            p=1,
            api_key="test",
        )

        assert inst.m == 2
        assert inst.n == 3
        assert inst.p == 1
        assert inst.distance_matrix.shape == (2, 3)


# ── Visualization Tests ────────────────────────────────────────────


class TestMapVisualization:
    """Test map visualization functions."""

    def test_plot_tsp_tour(self):
        from shared.visualization.map_viz import plot_tsp_tour

        coords = np.array([
            [8.681, 49.414],
            [8.687, 49.420],
            [8.651, 49.418],
            [8.670, 49.410],
        ])
        m = plot_tsp_tour(coords, tour=[0, 1, 2, 3], title="Test TSP")
        assert m is not None

    def test_plot_tsp_no_tour(self):
        from shared.visualization.map_viz import plot_tsp_tour

        coords = np.array([[8.681, 49.414], [8.687, 49.420]])
        m = plot_tsp_tour(coords, title="Cities Only")
        assert m is not None

    def test_plot_vrp_routes(self):
        from shared.visualization.map_viz import plot_vrp_routes

        coords = np.array([
            [8.694, 49.406],  # depot
            [8.681, 49.414],
            [8.687, 49.420],
            [8.651, 49.418],
            [8.670, 49.410],
        ])
        routes = [[1, 2], [3, 4]]
        demands = np.array([10, 15, 20, 12])

        m = plot_vrp_routes(coords, routes, demands=demands, title="Test VRP")
        assert m is not None

    def test_plot_facility_location(self):
        from shared.visualization.map_viz import plot_facility_location

        fac_coords = np.array([[8.681, 49.414], [8.687, 49.420]])
        cust_coords = np.array([[8.651, 49.418], [8.660, 49.415], [8.670, 49.410]])

        m = plot_facility_location(
            fac_coords,
            cust_coords,
            open_facilities=[0],
            assignments=[0, 0, 0],
            title="Test FL",
        )
        assert m is not None

    def test_plot_distance_matrix(self):
        from shared.visualization.map_viz import plot_distance_matrix
        import matplotlib
        matplotlib.use("Agg")

        matrix = np.array([[0, 100, 200], [100, 0, 150], [200, 150, 0]])
        fig = plot_distance_matrix(
            matrix, labels=["A", "B", "C"], title="Test"
        )
        assert fig is not None

    def test_save_tsp_map(self, tmp_path):
        from shared.visualization.map_viz import plot_tsp_tour

        coords = np.array([[8.681, 49.414], [8.687, 49.420], [8.651, 49.418]])
        save_file = str(tmp_path / "test_tsp.html")
        plot_tsp_tour(coords, tour=[0, 1, 2], save_path=save_file)

        assert Path(save_file).exists()
        content = Path(save_file).read_text()
        assert "leaflet" in content.lower() or "folium" in content.lower()


# ── Integration Tests (real API) ────────────────────────────────────


@pytest.mark.integration
class TestORSIntegration:
    """Integration tests that hit the real ORS API.

    Run with: pytest -m integration -v
    These tests are skipped by default.
    """

    def test_real_matrix(self):
        client = ORSClient()
        coords = np.array([
            [8.681495, 49.41461],
            [8.687872, 49.420318],
            [8.651177, 49.418865],
        ])
        result = client.distance_matrix(coords)
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0
        assert result[0, 1] > 0

    def test_real_geocode(self):
        client = ORSClient()
        results = client.geocode("Heidelberg, Germany")
        assert len(results) > 0
        coords = results[0]["coordinates"]
        assert 8.5 < coords[0] < 8.8  # lon
        assert 49.3 < coords[1] < 49.5  # lat

    def test_real_tsp_from_ors(self):
        from problems.routing.tsp.instance import TSPInstance
        inst = TSPInstance.from_ors(
            locations=[
                [8.681495, 49.41461],
                [8.687872, 49.420318],
                [8.651177, 49.418865],
            ],
            name="heidelberg_3",
        )
        assert inst.n == 3
        assert inst.distance_matrix[0, 1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
