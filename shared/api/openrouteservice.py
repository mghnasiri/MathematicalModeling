"""
OpenRouteService API Client for Operations Research Problems.

Provides real-world distance/time matrices, route geometries, isochrones,
and geocoding for routing and location problems.

API docs: https://openrouteservice.org/dev/#/api-docs
Profiles: 'driving-car', 'driving-hgv', 'cycling-regular', 'foot-walking'

References:
    OpenRouteService — https://openrouteservice.org/
    VROOM (optimization backend) — https://github.com/VROOM-Project/vroom
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


# Default API key (can be overridden via ORS_API_KEY environment variable)
_DEFAULT_API_KEY = "5b3ce35978511100016cf248"  # public demo key
_BASE_URL = "https://api.openrouteservice.org"

# Rate limiting: ORS free tier allows 40 requests/minute
_MIN_REQUEST_INTERVAL = 1.5  # seconds between requests


@dataclass
class ORSClient:
    """Client for OpenRouteService API.

    Args:
        api_key: ORS API key. If None, reads from ORS_API_KEY env var
            or falls back to default.
        profile: Routing profile. One of 'driving-car', 'driving-hgv',
            'cycling-regular', 'cycling-road', 'cycling-mountain',
            'cycling-electric', 'foot-walking', 'foot-hiking', 'wheelchair'.
        base_url: API base URL.
    """

    api_key: str | None = None
    profile: str = "driving-car"
    base_url: str = _BASE_URL
    _last_request_time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("ORS_API_KEY", _DEFAULT_API_KEY)

    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _request(self, method: str, endpoint: str,
                 params: dict | None = None,
                 body: dict | None = None) -> dict:
        """Make an API request with retry logic.

        Args:
            method: HTTP method ('GET' or 'POST').
            endpoint: API endpoint path (e.g., '/v2/matrix/driving-car').
            params: Query parameters for GET requests.
            body: JSON body for POST requests.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If the request fails after retries.
        """
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"

        headers = {
            "Authorization": self.api_key,
            "Accept": "application/json, application/geo+json",
        }

        if method == "POST" and body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        else:
            data = None

        req = Request(url, data=data, headers=headers, method=method)

        retries = 3
        for attempt in range(retries + 1):
            try:
                with urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                if e.code == 429 and attempt < retries:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"ORS API error {e.code}: {error_body}"
                ) from e
            except URLError as e:
                if attempt < retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise RuntimeError(
                    f"ORS connection error: {e.reason}"
                ) from e

        raise RuntimeError("ORS API request failed after retries")

    # ── Matrix Service ──────────────────────────────────────────────

    def matrix(
        self,
        locations: list[list[float]],
        metrics: list[str] | None = None,
        sources: list[int] | None = None,
        destinations: list[int] | None = None,
        units: str = "m",
    ) -> dict[str, np.ndarray]:
        """Compute distance and/or duration matrix between locations.

        Args:
            locations: List of [longitude, latitude] coordinate pairs.
                Maximum 3500 for driving, 2500 for cycling/walking.
            metrics: List of metrics to compute: 'distance', 'duration'.
                Defaults to ['distance', 'duration'].
            sources: Indices of source locations (default: all).
            destinations: Indices of destination locations (default: all).
            units: Distance units: 'm' (meters), 'km', 'mi'.

        Returns:
            Dict with 'distances' and/or 'durations' as numpy arrays.
            Values are None/null if no route exists between points.

        Example:
            >>> client = ORSClient()
            >>> coords = [[8.681495, 49.41461], [8.687872, 49.420318]]
            >>> result = client.matrix(coords)
            >>> result['distances']  # 2x2 distance matrix in meters
        """
        if metrics is None:
            metrics = ["distance", "duration"]

        body: dict[str, Any] = {
            "locations": locations,
            "metrics": metrics,
            "units": units,
        }
        if sources is not None:
            body["sources"] = sources
        if destinations is not None:
            body["destinations"] = destinations

        resp = self._request(
            "POST", f"/v2/matrix/{self.profile}", body=body
        )

        result = {}
        if "distances" in resp:
            result["distances"] = np.array(resp["distances"], dtype=float)
        if "durations" in resp:
            result["durations"] = np.array(resp["durations"], dtype=float)
        return result

    # ── Directions Service ──────────────────────────────────────────

    def directions(
        self,
        coordinates: list[list[float]],
        geometry: bool = True,
        instructions: bool = False,
    ) -> dict:
        """Get route directions between waypoints.

        Args:
            coordinates: List of [longitude, latitude] waypoints (2-50 points).
            geometry: Include route geometry in response.
            instructions: Include turn-by-turn instructions.

        Returns:
            GeoJSON response with route geometry, distance, and duration.

        Example:
            >>> client = ORSClient()
            >>> route = client.directions([
            ...     [8.681495, 49.41461],
            ...     [8.687872, 49.420318]
            ... ])
            >>> route['features'][0]['properties']['summary']
            {'distance': 1045.6, 'duration': 215.0}
        """
        body = {
            "coordinates": coordinates,
            "geometry": geometry,
            "instructions": instructions,
        }
        return self._request(
            "POST", f"/v2/directions/{self.profile}/geojson", body=body
        )

    # ── Isochrones Service ──────────────────────────────────────────

    def isochrones(
        self,
        locations: list[list[float]],
        range_seconds: list[int] | None = None,
        range_meters: list[int] | None = None,
        range_type: str = "time",
    ) -> dict:
        """Compute isochrone (reachability) polygons.

        Args:
            locations: List of [longitude, latitude] center points (max 5).
            range_seconds: Time ranges in seconds (when range_type='time').
            range_meters: Distance ranges in meters (when range_type='distance').
            range_type: 'time' or 'distance'.

        Returns:
            GeoJSON FeatureCollection with isochrone polygons.

        Example:
            >>> client = ORSClient()
            >>> iso = client.isochrones(
            ...     [[8.681495, 49.41461]],
            ...     range_seconds=[300, 600]
            ... )
        """
        body: dict[str, Any] = {
            "locations": locations,
            "range_type": range_type,
        }
        if range_type == "time" and range_seconds:
            body["range"] = range_seconds
        elif range_type == "distance" and range_meters:
            body["range"] = range_meters
        else:
            body["range"] = range_seconds or [300]

        return self._request(
            "POST", f"/v2/isochrones/{self.profile}", body=body
        )

    # ── Geocode Service ─────────────────────────────────────────────

    def geocode(self, text: str, size: int = 5) -> list[dict]:
        """Forward geocode: text address to coordinates.

        Args:
            text: Address or place name to search.
            size: Maximum number of results (1-40).

        Returns:
            List of dicts with 'name', 'coordinates' [lon, lat],
            'label', and 'confidence'.

        Example:
            >>> client = ORSClient()
            >>> results = client.geocode("Heidelberg, Germany")
            >>> results[0]['coordinates']
            [8.694724, 49.405882]
        """
        resp = self._request(
            "GET",
            "/geocode/search",
            params={"api_key": self.api_key, "text": text, "size": str(size)},
        )
        results = []
        for feat in resp.get("features", []):
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [])
            results.append({
                "name": props.get("name", ""),
                "label": props.get("label", ""),
                "coordinates": coords,
                "confidence": props.get("confidence", 0),
            })
        return results

    def reverse_geocode(self, lon: float, lat: float) -> list[dict]:
        """Reverse geocode: coordinates to address.

        Args:
            lon: Longitude.
            lat: Latitude.

        Returns:
            List of dicts with address information.
        """
        resp = self._request(
            "GET",
            "/geocode/reverse",
            params={
                "api_key": self.api_key,
                "point.lon": str(lon),
                "point.lat": str(lat),
            },
        )
        results = []
        for feat in resp.get("features", []):
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [])
            results.append({
                "name": props.get("name", ""),
                "label": props.get("label", ""),
                "coordinates": coords,
            })
        return results

    # ── Optimization (VROOM) Service ────────────────────────────────

    def optimize(
        self,
        jobs: list[dict],
        vehicles: list[dict],
    ) -> dict:
        """Solve a vehicle routing problem using VROOM backend.

        Args:
            jobs: List of job dicts, each with at minimum:
                - 'id': unique job identifier
                - 'location': [longitude, latitude]
                Optional: 'service', 'amount', 'skills', 'time_windows'
            vehicles: List of vehicle dicts, each with at minimum:
                - 'id': unique vehicle identifier
                - 'start': [longitude, latitude]
                Optional: 'end', 'capacity', 'skills', 'time_window'

        Returns:
            VROOM optimization response with routes and unassigned jobs.

        Example:
            >>> client = ORSClient()
            >>> result = client.optimize(
            ...     jobs=[
            ...         {"id": 1, "location": [8.688, 49.420], "service": 300},
            ...         {"id": 2, "location": [8.681, 49.415], "service": 300},
            ...     ],
            ...     vehicles=[{
            ...         "id": 1,
            ...         "start": [8.694, 49.406],
            ...         "end": [8.694, 49.406],
            ...         "capacity": [100],
            ...     }],
            ... )
        """
        body = {"jobs": jobs, "vehicles": vehicles}
        return self._request("POST", "/optimization", body=body)

    # ── Convenience Methods for OR Problems ─────────────────────────

    def distance_matrix(
        self,
        coords_lonlat: np.ndarray,
        metric: str = "distance",
    ) -> np.ndarray:
        """Compute a square distance or duration matrix.

        Convenience wrapper for the matrix service that handles numpy I/O.

        Args:
            coords_lonlat: (n, 2) array of [longitude, latitude] pairs.
            metric: 'distance' (meters) or 'duration' (seconds).

        Returns:
            (n, n) numpy array of distances or durations.
        """
        locations = coords_lonlat.tolist()
        result = self.matrix(locations, metrics=[metric])
        key = "distances" if metric == "distance" else "durations"
        return result[key]

    def route_geometry(
        self, coords_lonlat: np.ndarray
    ) -> list[list[float]]:
        """Get the road-network geometry for an ordered sequence of points.

        Args:
            coords_lonlat: (n, 2) array of [longitude, latitude] waypoints.

        Returns:
            List of [lon, lat] points tracing the route on roads.
        """
        resp = self.directions(coords_lonlat.tolist())
        features = resp.get("features", [])
        if not features:
            return []
        geom = features[0].get("geometry", {})
        return geom.get("coordinates", [])

    def geocode_locations(
        self, place_names: list[str]
    ) -> np.ndarray:
        """Geocode a list of place names to coordinates.

        Args:
            place_names: List of address/place strings.

        Returns:
            (n, 2) numpy array of [longitude, latitude] coordinates.
        """
        coords = []
        for name in place_names:
            results = self.geocode(name, size=1)
            if not results:
                raise ValueError(f"Could not geocode: {name!r}")
            coords.append(results[0]["coordinates"])
        return np.array(coords)


# ── Module-level convenience ────────────────────────────────────────

def get_client(
    api_key: str | None = None,
    profile: str = "driving-car",
) -> ORSClient:
    """Create an ORS client with optional API key override.

    Args:
        api_key: ORS API key. Falls back to ORS_API_KEY env var.
        profile: Routing profile (default: 'driving-car').

    Returns:
        Configured ORSClient instance.
    """
    return ORSClient(api_key=api_key, profile=profile)
