"""
Agricultural Irrigation Network Design Problem

Domain: Precision agriculture / Water resource management
Models: MST (backbone), Max Flow (throughput), Shortest Path (delivery)

A farm with multiple field zones, junctions, a water pump station (source),
and a reservoir (sink) designs an irrigation pipe network. Three analyses:

1. MST: Minimum-cost pipe backbone connecting all nodes (Kruskal/Prim).
2. Max Flow: Maximum water throughput from pump to reservoir (Edmonds-Karp).
3. Shortest Path: Fastest water delivery to each field (Dijkstra).

Complexity:
    - MST: O(E log E) Kruskal, O(E log V) Prim
    - Max Flow: O(V * E^2) Edmonds-Karp
    - Shortest Path: O((V+E) log V) Dijkstra

References:
    Valiantzas, J.D. (2006). Simplified versions for the Penman
    evapotranspiration equation using routine weather data. Journal
    of Hydrology, 331(3-4), 690-702.
    https://doi.org/10.1016/j.jhydrol.2006.06.012

    Kang, Y. & Nishiyama, S. (1996). Analysis and design of
    microirrigation laterals. Journal of Irrigation and Drainage
    Engineering, 122(2), 75-82.
    https://doi.org/10.1061/(ASCE)0733-9437(1996)122:2(75)
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class NetworkNode:
    """A node in the irrigation network.

    Args:
        id: Node index.
        name: Descriptive name.
        node_type: One of "source", "junction", "field", "sink".
        crop: Crop type for field nodes, None otherwise.
    """
    id: int
    name: str
    node_type: str
    crop: str | None = None


@dataclass
class PipeSegment:
    """A pipe segment connecting two nodes.

    Args:
        from_node: Source node index.
        to_node: Destination node index.
        install_cost: Installation cost in USD.
        capacity_lph: Water capacity in liters per hour.
        distance_m: Pipe length in meters.
    """
    from_node: int
    to_node: int
    install_cost: float
    capacity_lph: float
    distance_m: float


@dataclass
class IrrigationNetworkInstance:
    """Irrigation network design instance.

    Args:
        nodes: List of network nodes.
        pipes: List of pipe segments.
        source: Source node index (pump station).
        sink: Sink node index (reservoir).
        field_demands: Dict mapping field node id to demand (L/hr).
        name: Instance name.
    """
    nodes: list[NetworkNode]
    pipes: list[PipeSegment]
    source: int
    sink: int
    field_demands: dict[int, float]
    name: str = "irrigation_network"

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_pipes(self) -> int:
        return len(self.pipes)

    @property
    def field_nodes(self) -> list[int]:
        return [n.id for n in self.nodes if n.node_type == "field"]

    @property
    def total_field_demand(self) -> float:
        return sum(self.field_demands.values())

    def get_mst_edges(self) -> list[tuple[int, int, float]]:
        """Get undirected edges with installation costs for MST."""
        seen = set()
        edges = []
        for p in self.pipes:
            key = (min(p.from_node, p.to_node), max(p.from_node, p.to_node))
            if key not in seen:
                edges.append((p.from_node, p.to_node, float(p.install_cost)))
                seen.add(key)
        return edges

    def get_flow_edges(self) -> list[tuple[int, int, float]]:
        """Get directed edges with capacities for Max Flow."""
        return [(p.from_node, p.to_node, float(p.capacity_lph))
                for p in self.pipes]

    def get_sp_edges(self) -> list[tuple[int, int, float]]:
        """Get directed edges with distances for Shortest Path."""
        return [(p.from_node, p.to_node, float(p.distance_m))
                for p in self.pipes]

    @classmethod
    def standard_farm(cls) -> IrrigationNetworkInstance:
        """Create the standard 10-node farm benchmark instance.

        Includes pump station, 3 junctions, 5 field zones, and reservoir,
        connected by 16 pipe segments.

        Returns:
            IrrigationNetworkInstance with 10 nodes and 16 pipes.
        """
        nodes = [
            NetworkNode(0, "Water Pump Station", "source"),
            NetworkNode(1, "Main Junction Alpha", "junction"),
            NetworkNode(2, "Main Junction Beta", "junction"),
            NetworkNode(3, "Main Junction Gamma", "junction"),
            NetworkNode(4, "Corn Field Zone", "field", "corn"),
            NetworkNode(5, "Wheat Field Zone", "field", "wheat"),
            NetworkNode(6, "Soybean Field Zone", "field", "soybeans"),
            NetworkNode(7, "Vegetable Field Zone", "field", "vegetables"),
            NetworkNode(8, "Orchard Field Zone", "field", "orchard"),
            NetworkNode(9, "Reservoir Collection Pt", "sink"),
        ]
        pipes = [
            PipeSegment(0, 1, 4500, 8000, 150),
            PipeSegment(0, 2, 5200, 7000, 200),
            PipeSegment(0, 3, 6000, 5000, 250),
            PipeSegment(1, 2, 1800, 3000, 80),
            PipeSegment(1, 4, 3200, 4500, 120),
            PipeSegment(1, 5, 3500, 4000, 140),
            PipeSegment(2, 5, 2800, 3500, 100),
            PipeSegment(2, 6, 3000, 4000, 110),
            PipeSegment(2, 7, 3400, 3500, 130),
            PipeSegment(3, 7, 2600, 3000, 90),
            PipeSegment(3, 8, 3800, 4500, 160),
            PipeSegment(4, 9, 2200, 3000, 70),
            PipeSegment(5, 9, 2500, 2500, 85),
            PipeSegment(6, 9, 2000, 2800, 65),
            PipeSegment(7, 9, 2700, 2000, 95),
            PipeSegment(8, 9, 3100, 3500, 120),
        ]
        field_demands = {
            4: 3500,  # Corn
            5: 2800,  # Wheat
            6: 2200,  # Soybeans
            7: 1800,  # Vegetables
            8: 3000,  # Orchard
        }
        return cls(
            nodes=nodes, pipes=pipes,
            source=0, sink=9,
            field_demands=field_demands,
            name="standard_farm",
        )

    @classmethod
    def random(cls, n_fields: int = 4, seed: int = 42) -> IrrigationNetworkInstance:
        """Generate a random irrigation network instance.

        Args:
            n_fields: Number of field zones.
            seed: Random seed.

        Returns:
            Random IrrigationNetworkInstance.
        """
        rng = np.random.default_rng(seed)
        # Nodes: source(0), junctions(1..2), fields(3..3+n_fields-1), sink(last)
        n_junctions = 2
        nodes = [NetworkNode(0, "Pump", "source")]
        for j in range(n_junctions):
            nodes.append(NetworkNode(j + 1, f"Junction_{j}", "junction"))
        crops = ["corn", "wheat", "soybeans", "vegetables", "orchard", "hay"]
        field_demands = {}
        for f in range(n_fields):
            nid = n_junctions + 1 + f
            crop = crops[f % len(crops)]
            nodes.append(NetworkNode(nid, f"{crop.title()} Field", "field", crop))
            field_demands[nid] = float(rng.integers(1500, 4000))
        sink_id = len(nodes)
        nodes.append(NetworkNode(sink_id, "Reservoir", "sink"))

        # Generate pipes
        pipes = []
        # Source to junctions
        for j in range(1, n_junctions + 1):
            pipes.append(PipeSegment(
                0, j, float(rng.integers(3000, 7000)),
                float(rng.integers(5000, 10000)),
                float(rng.integers(100, 300)),
            ))
        # Junctions to fields
        for f in range(n_fields):
            fid = n_junctions + 1 + f
            jid = 1 + (f % n_junctions)
            pipes.append(PipeSegment(
                jid, fid, float(rng.integers(2000, 5000)),
                float(rng.integers(3000, 6000)),
                float(rng.integers(60, 200)),
            ))
        # Fields to sink
        for f in range(n_fields):
            fid = n_junctions + 1 + f
            pipes.append(PipeSegment(
                fid, sink_id, float(rng.integers(1500, 4000)),
                float(rng.integers(2000, 4000)),
                float(rng.integers(50, 150)),
            ))
        # Cross-link between junctions
        if n_junctions >= 2:
            pipes.append(PipeSegment(
                1, 2, float(rng.integers(1000, 3000)),
                float(rng.integers(2000, 4000)),
                float(rng.integers(50, 120)),
            ))

        return cls(
            nodes=nodes, pipes=pipes,
            source=0, sink=sink_id,
            field_demands=field_demands,
            name=f"random_{n_fields}fields",
        )


@dataclass
class IrrigationNetworkSolution:
    """Solution to the irrigation network design problem.

    Args:
        mst_cost: Total installation cost of backbone pipes.
        mst_edges: Backbone pipe segments (u, v, cost).
        max_flow: Maximum water throughput (L/hr).
        min_cut: Tuple of (source_side, sink_side) node sets.
        shortest_paths: Dict mapping target to (path, distance).
        method: Description of methods used.
    """
    mst_cost: float
    mst_edges: list[tuple[int, int, float]]
    max_flow: float
    min_cut: tuple[list[int], list[int]]
    shortest_paths: dict[int, tuple[list[int], float]]
    method: str = "Kruskal + Edmonds-Karp + Dijkstra"

    def __repr__(self) -> str:
        return (f"IrrigationNetworkSolution(MST=${self.mst_cost:,.0f}, "
                f"MaxFlow={self.max_flow:,.0f} L/hr, "
                f"paths={len(self.shortest_paths)})")


if __name__ == "__main__":
    inst = IrrigationNetworkInstance.standard_farm()
    print(f"Standard farm: {inst.n_nodes} nodes, {inst.n_pipes} pipes")
    print(f"  Fields: {inst.field_nodes}")
    print(f"  Total demand: {inst.total_field_demand:,.0f} L/hr")
