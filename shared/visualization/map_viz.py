"""
Interactive Map Visualization for Operations Research Problems.

Uses Folium (Leaflet.js) to create interactive HTML maps showing:
- TSP tours with road-network routes
- CVRP/VRPTW multi-vehicle routes with color-coded routes
- Facility location solutions with customer assignments
- Isochrone reachability areas
- Distance matrix heatmaps

All maps are saved as self-contained HTML files that open in any browser.

Dependencies: folium, numpy
Optional: shared.api.openrouteservice (for road-network routes)
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# Add project root for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Color palette for multi-route visualization ────────────────────

ROUTE_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
    "#00ced1",  # dark turquoise
    "#ffd700",  # gold
    "#8b0000",  # dark red
    "#006400",  # dark green
]


def _check_folium():
    if not HAS_FOLIUM:
        raise ImportError(
            "folium is required for map visualization. "
            "Install it with: pip install folium"
        )


def _auto_center(coords_lonlat: np.ndarray) -> tuple[float, float, int]:
    """Compute map center and zoom level from coordinates.

    Args:
        coords_lonlat: (n, 2) array of [longitude, latitude].

    Returns:
        (center_lat, center_lon, zoom_level)
    """
    lons = coords_lonlat[:, 0]
    lats = coords_lonlat[:, 1]
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    lat_range = float(np.ptp(lats))
    lon_range = float(np.ptp(lons))
    max_range = max(lat_range, lon_range, 0.001)

    if max_range > 10:
        zoom = 5
    elif max_range > 5:
        zoom = 7
    elif max_range > 1:
        zoom = 9
    elif max_range > 0.5:
        zoom = 11
    elif max_range > 0.1:
        zoom = 13
    else:
        zoom = 14

    return center_lat, center_lon, zoom


def _get_ors_client(api_key: str | None = None, profile: str = "driving-car"):
    """Get ORS client if available."""
    try:
        from shared.api.openrouteservice import ORSClient
        return ORSClient(api_key=api_key, profile=profile)
    except ImportError:
        return None


# ── TSP Visualization ───────────────────────────────────────────────

def plot_tsp_tour(
    coords_lonlat: np.ndarray,
    tour: list[int] | None = None,
    city_labels: list[str] | None = None,
    title: str = "TSP Tour",
    use_roads: bool = False,
    api_key: str | None = None,
    save_path: str | None = None,
) -> Any:
    """Create an interactive map of a TSP tour.

    Args:
        coords_lonlat: (n, 2) array of [longitude, latitude] for each city.
        tour: Ordered list of city indices forming the tour.
            If None, only cities are plotted.
        city_labels: Optional names for each city.
        title: Map title shown in popup.
        use_roads: If True, fetch actual road routes via ORS API.
        api_key: ORS API key (required if use_roads=True).
        save_path: If provided, save the map as an HTML file.

    Returns:
        folium.Map object.
    """
    _check_folium()

    center_lat, center_lon, zoom = _auto_center(coords_lonlat)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    # Add cities as markers
    for i, (lon, lat) in enumerate(coords_lonlat):
        label = city_labels[i] if city_labels else f"City {i}"
        popup_text = f"<b>{label}</b><br>Index: {i}<br>({lat:.5f}, {lon:.5f})"
        icon_color = "red" if (tour and i == tour[0]) else "blue"
        folium.Marker(
            location=[float(lat), float(lon)],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=label,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(m)

    # Draw tour
    if tour is not None:
        closed_tour = list(tour) + [tour[0]]

        if use_roads:
            client = _get_ors_client(api_key)
            if client is None:
                raise ImportError("ORS client required for road routes")

            # Fetch road geometry in segments (ORS limit: 50 waypoints)
            all_road_points = []
            for i in range(len(closed_tour) - 1):
                start_idx = closed_tour[i]
                end_idx = closed_tour[i + 1]
                segment_coords = [
                    coords_lonlat[start_idx].tolist(),
                    coords_lonlat[end_idx].tolist(),
                ]
                try:
                    road_geom = client.route_geometry(
                        np.array(segment_coords)
                    )
                    road_latlon = [[p[1], p[0]] for p in road_geom]
                    all_road_points.extend(road_latlon)
                except Exception:
                    # Fallback to straight line
                    all_road_points.append(
                        [float(coords_lonlat[start_idx][1]),
                         float(coords_lonlat[start_idx][0])]
                    )

            if all_road_points:
                folium.PolyLine(
                    locations=all_road_points,
                    color=ROUTE_COLORS[0],
                    weight=4,
                    opacity=0.8,
                    tooltip="Tour route (road network)",
                ).add_to(m)
        else:
            # Straight-line tour
            tour_latlon = [
                [float(coords_lonlat[i][1]), float(coords_lonlat[i][0])]
                for i in closed_tour
            ]
            folium.PolyLine(
                locations=tour_latlon,
                color=ROUTE_COLORS[0],
                weight=3,
                opacity=0.7,
                tooltip="Tour (straight line)",
            ).add_to(m)

            # Add direction arrows
            for idx in range(len(closed_tour) - 1):
                start = closed_tour[idx]
                end = closed_tour[idx + 1]
                mid_lat = (coords_lonlat[start][1] + coords_lonlat[end][1]) / 2
                mid_lon = (coords_lonlat[start][0] + coords_lonlat[end][0]) / 2
                folium.CircleMarker(
                    location=[float(mid_lat), float(mid_lon)],
                    radius=3,
                    color=ROUTE_COLORS[0],
                    fill=True,
                    tooltip=f"Leg {idx + 1}: {start} → {end}",
                ).add_to(m)

    # Title
    title_html = f'<h3 style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:5px 10px;border-radius:5px;box-shadow:0 2px 6px rgba(0,0,0,0.3)">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    if save_path:
        m.save(save_path)

    return m


# ── CVRP / VRPTW Multi-Route Visualization ─────────────────────────

def plot_vrp_routes(
    coords_lonlat: np.ndarray,
    routes: list[list[int]],
    demands: np.ndarray | None = None,
    time_windows: np.ndarray | None = None,
    customer_labels: list[str] | None = None,
    title: str = "VRP Routes",
    use_roads: bool = False,
    api_key: str | None = None,
    save_path: str | None = None,
) -> Any:
    """Create an interactive map of VRP routes with color-coded vehicles.

    Args:
        coords_lonlat: (n+1, 2) array of [lon, lat]. Index 0 is depot.
        routes: List of routes, each a list of customer indices (1-based).
        demands: (n,) array of customer demands (optional, for popups).
        time_windows: (n+1, 2) array of [earliest, latest] (optional).
        customer_labels: Optional names for each customer.
        title: Map title.
        use_roads: Fetch road-network routes via ORS.
        api_key: ORS API key.
        save_path: Save path for HTML file.

    Returns:
        folium.Map object.
    """
    _check_folium()

    center_lat, center_lon, zoom = _auto_center(coords_lonlat)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    # Depot marker (star)
    depot_lon, depot_lat = coords_lonlat[0]
    folium.Marker(
        location=[float(depot_lat), float(depot_lon)],
        popup=folium.Popup("<b>DEPOT</b>", max_width=200),
        tooltip="Depot",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)

    # Customer markers
    for i in range(1, len(coords_lonlat)):
        lon, lat = coords_lonlat[i]
        label = customer_labels[i - 1] if customer_labels else f"Customer {i}"
        popup_parts = [f"<b>{label}</b>", f"Index: {i}"]
        if demands is not None and i - 1 < len(demands):
            popup_parts.append(f"Demand: {demands[i - 1]}")
        if time_windows is not None and i < len(time_windows):
            tw = time_windows[i]
            popup_parts.append(f"TW: [{tw[0]:.0f}, {tw[1]:.0f}]")
        popup_text = "<br>".join(popup_parts)

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=7,
            color="#333",
            fill=True,
            fill_color="#666",
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=label,
        ).add_to(m)

    # Draw routes
    client = None
    if use_roads:
        client = _get_ors_client(api_key)

    for route_idx, route in enumerate(routes):
        color = ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]
        full_route = [0] + list(route) + [0]  # depot → customers → depot

        route_group = folium.FeatureGroup(name=f"Vehicle {route_idx + 1}")

        if client and use_roads:
            # Road-network route
            for seg_i in range(len(full_route) - 1):
                start_idx = full_route[seg_i]
                end_idx = full_route[seg_i + 1]
                segment = [
                    coords_lonlat[start_idx].tolist(),
                    coords_lonlat[end_idx].tolist(),
                ]
                try:
                    road_geom = client.route_geometry(np.array(segment))
                    road_latlon = [[p[1], p[0]] for p in road_geom]
                    folium.PolyLine(
                        locations=road_latlon,
                        color=color,
                        weight=4,
                        opacity=0.8,
                    ).add_to(route_group)
                except Exception:
                    # Fallback to straight line
                    points = [
                        [float(coords_lonlat[start_idx][1]),
                         float(coords_lonlat[start_idx][0])],
                        [float(coords_lonlat[end_idx][1]),
                         float(coords_lonlat[end_idx][0])],
                    ]
                    folium.PolyLine(
                        locations=points, color=color, weight=3, opacity=0.6
                    ).add_to(route_group)
        else:
            # Straight-line route
            route_latlon = [
                [float(coords_lonlat[i][1]), float(coords_lonlat[i][0])]
                for i in full_route
            ]
            folium.PolyLine(
                locations=route_latlon,
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"Vehicle {route_idx + 1} ({len(route)} stops)",
            ).add_to(route_group)

        route_group.add_to(m)

    # Layer control to toggle routes
    folium.LayerControl().add_to(m)

    # Title
    title_html = f'<h3 style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:5px 10px;border-radius:5px;box-shadow:0 2px 6px rgba(0,0,0,0.3)">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    if save_path:
        m.save(save_path)

    return m


# ── Facility Location Visualization ────────────────────────────────

def plot_facility_location(
    facility_coords: np.ndarray,
    customer_coords: np.ndarray,
    open_facilities: list[int] | None = None,
    assignments: list[int] | None = None,
    facility_labels: list[str] | None = None,
    customer_labels: list[str] | None = None,
    isochrone_seconds: list[int] | None = None,
    title: str = "Facility Location",
    api_key: str | None = None,
    save_path: str | None = None,
) -> Any:
    """Create an interactive map of a facility location solution.

    Args:
        facility_coords: (m, 2) array of [lon, lat] for facilities.
        customer_coords: (n, 2) array of [lon, lat] for customers.
        open_facilities: List of open facility indices.
        assignments: List of length n, assignments[j] = facility index for
            customer j.
        facility_labels: Optional names for facilities.
        customer_labels: Optional names for customers.
        isochrone_seconds: If provided, draw reachability polygons around
            open facilities (e.g., [300, 600] for 5/10 min).
        title: Map title.
        api_key: ORS API key (for isochrones).
        save_path: Save path for HTML.

    Returns:
        folium.Map object.
    """
    _check_folium()

    all_coords = np.vstack([facility_coords, customer_coords])
    center_lat, center_lon, zoom = _auto_center(all_coords)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    # Facility markers
    for i, (lon, lat) in enumerate(facility_coords):
        label = facility_labels[i] if facility_labels else f"Facility {i}"
        is_open = open_facilities is not None and i in open_facilities
        color = "green" if is_open else "lightgray"
        icon = "ok-sign" if is_open else "remove-sign"
        popup_text = f"<b>{label}</b><br>Status: {'OPEN' if is_open else 'CLOSED'}"

        folium.Marker(
            location=[float(lat), float(lon)],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"{label} ({'open' if is_open else 'closed'})",
            icon=folium.Icon(color=color, icon=icon),
        ).add_to(m)

    # Customer markers with assignment coloring
    for j, (lon, lat) in enumerate(customer_coords):
        label = customer_labels[j] if customer_labels else f"Customer {j}"
        if assignments is not None and j < len(assignments):
            assigned = assignments[j]
            color = ROUTE_COLORS[assigned % len(ROUTE_COLORS)]
            popup_text = f"<b>{label}</b><br>Assigned to: Facility {assigned}"
        else:
            color = "#666"
            popup_text = f"<b>{label}</b>"

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=label,
        ).add_to(m)

    # Assignment lines
    if assignments is not None and open_facilities is not None:
        for j, assigned_fac in enumerate(assignments):
            if assigned_fac in open_facilities:
                color = ROUTE_COLORS[assigned_fac % len(ROUTE_COLORS)]
                cust_lon, cust_lat = customer_coords[j]
                fac_lon, fac_lat = facility_coords[assigned_fac]
                folium.PolyLine(
                    locations=[
                        [float(cust_lat), float(cust_lon)],
                        [float(fac_lat), float(fac_lon)],
                    ],
                    color=color,
                    weight=1,
                    opacity=0.4,
                    dash_array="5 5",
                ).add_to(m)

    # Isochrones around open facilities
    if isochrone_seconds and open_facilities:
        client = _get_ors_client(api_key)
        if client:
            for fac_idx in open_facilities:
                fac_coord = facility_coords[fac_idx].tolist()
                try:
                    iso_resp = client.isochrones(
                        [fac_coord], range_seconds=isochrone_seconds
                    )
                    for feat in iso_resp.get("features", []):
                        geom = feat.get("geometry", {})
                        props = feat.get("properties", {})
                        value = props.get("value", 0)
                        if geom.get("type") == "Polygon":
                            coords_ring = geom["coordinates"][0]
                            latlon_ring = [
                                [c[1], c[0]] for c in coords_ring
                            ]
                            color = ROUTE_COLORS[
                                fac_idx % len(ROUTE_COLORS)
                            ]
                            folium.Polygon(
                                locations=latlon_ring,
                                color=color,
                                fill=True,
                                fill_opacity=0.15,
                                tooltip=f"Facility {fac_idx}: {value}s reachability",
                            ).add_to(m)
                except Exception:
                    pass  # Skip isochrones on API error

    # Title
    title_html = f'<h3 style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:5px 10px;border-radius:5px;box-shadow:0 2px 6px rgba(0,0,0,0.3)">{title}</h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    if save_path:
        m.save(save_path)

    return m


# ── Distance Matrix Heatmap ────────────────────────────────────────

def plot_distance_matrix(
    matrix: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Distance Matrix",
    save_path: str | None = None,
) -> Any:
    """Create a matplotlib heatmap of a distance/duration matrix.

    Args:
        matrix: (n, n) distance or duration matrix.
        labels: Optional labels for rows/columns.
        title: Plot title.
        save_path: If provided, save as image file.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(matrix) * 0.5),
                                    max(5, len(matrix) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Distance")

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
