"""
Test suite for real-world application case studies.

Validates that all applications:
- Create valid problem instances
- Produce feasible solutions
- Run without errors end-to-end
"""

from __future__ import annotations

import os
import sys
import pytest
import importlib.util

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_app(name: str, filename: str):
    filepath = os.path.join(_app_dir, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestManufacturingScheduling:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("mfg_sched_test", "manufacturing_scheduling.py")

    def test_create_instance(self):
        data = self.mod.create_pharma_instance()
        assert data["n_jobs"] == 12
        assert data["n_machines"] == 4
        assert data["processing_times"].shape == (12, 4)
        assert data["setup_times"].shape == (4, 13, 12)

    def test_setup_times_shape(self):
        data = self.mod.create_pharma_instance()
        # Shape: (m, n+1, n) — row 0 is initial setup
        assert data["setup_times"].shape == (4, 13, 12)

    def test_solve_produces_results(self):
        results = self.mod.solve_pharma_scheduling(verbose=False)
        assert "NEH-SDST" in results
        assert "IG-SDST" in results
        for method, res in results.items():
            assert res["makespan"] > 0
            assert len(res["sequence"]) == 12

    def test_ig_improves_neh(self):
        results = self.mod.solve_pharma_scheduling(verbose=False)
        assert results["IG-SDST"]["makespan"] <= results["NEH-SDST"]["makespan"]


class TestDeliveryRouting:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("delivery_test", "delivery_routing.py")

    def test_create_instance(self):
        data = self.mod.create_delivery_instance(n_customers=15, seed=42)
        assert data["n_customers"] == 15
        assert len(data["coords"]) == 16  # +depot
        assert data["capacity"] == 300

    def test_time_windows_valid(self):
        data = self.mod.create_delivery_instance(n_customers=10, seed=42)
        for i in range(1, 11):
            assert data["earliest"][i] <= data["latest"][i]

    def test_cvrp_solve(self):
        data = self.mod.create_delivery_instance(n_customers=15, seed=42)
        results = self.mod.solve_cvrp(data, verbose=False)
        assert "Clarke-Wright" in results
        assert "SA" in results
        for method, res in results.items():
            assert res["distance"] > 0
            assert res["n_vehicles"] >= 1

    def test_vrptw_solve(self):
        data = self.mod.create_delivery_instance(n_customers=15, seed=42)
        results = self.mod.solve_vrptw(data, verbose=False)
        assert "Solomon-I1" in results
        assert results["Solomon-I1"]["distance"] > 0


class TestWarehouseLocation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("warehouse_test", "warehouse_location.py")

    def test_create_instance(self):
        data = self.mod.create_uflp_instance()
        assert data["m"] == 8
        assert data["n"] == 30

    def test_solve_uflp_and_pmedian(self):
        results = self.mod.solve_warehouse_location(verbose=False)
        assert "UFLP" in results
        assert "p-Median" in results

    def test_uflp_methods(self):
        results = self.mod.solve_warehouse_location(verbose=False)
        for method in ["Greedy-Add", "Greedy-Drop", "SA"]:
            assert method in results["UFLP"]
            assert results["UFLP"][method]["cost"] > 0
            assert len(results["UFLP"][method]["open"]) >= 1

    def test_pmedian_opens_3(self):
        results = self.mod.solve_warehouse_location(verbose=False)
        for method in ["Greedy", "Interchange"]:
            assert len(results["p-Median"][method]["open"]) == 3


class TestSupplyChainNetwork:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("supply_chain_test", "supply_chain_network.py")

    def test_create_data(self):
        data = self.mod.create_supply_chain_data()
        assert data["n"] == 7
        assert len(data["transport_links"]) == 12

    def test_max_flow(self):
        results = self.mod.solve_supply_chain(verbose=False)
        assert results["max_flow"]["max_throughput"] > 0
        s_set, t_set = results["max_flow"]["min_cut"]
        assert 0 in s_set  # source
        assert 6 in t_set  # sink

    def test_mst_backbone(self):
        results = self.mod.solve_supply_chain(verbose=False)
        assert results["mst"]["total_cost"] > 0
        assert len(results["mst"]["backbone_links"]) == 6  # n-1

    def test_shortest_path(self):
        results = self.mod.solve_supply_chain(verbose=False)
        assert results["shortest_path"]["distance"] > 0
        assert results["shortest_path"]["path"][0] == 0
        assert results["shortest_path"]["path"][-1] == 6


class TestWorkforceAssignment:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("workforce_test", "workforce_assignment.py")

    def test_create_assignment(self):
        data = self.mod.create_assignment_instance()
        assert data["n"] == 8
        assert data["cost_matrix"].shape == (8, 8)

    def test_skill_matching_costs(self):
        data = self.mod.create_assignment_instance()
        # Alice (backend) → API Gateway (backend): should be cheaper
        # than Alice → Mobile App (mobile)
        alice_api = data["cost_matrix"][0][0]  # backend match
        alice_mobile = data["cost_matrix"][0][1]  # no mobile skill
        assert alice_api < alice_mobile

    def test_solve_workforce(self):
        results = self.mod.solve_workforce(verbose=False)
        assert "assignment" in results
        assert "scheduling" in results
        assert results["assignment"]["Hungarian"]["cost"] <= \
               results["assignment"]["Greedy"]["cost"] + 1e-6


class TestCloudResourcePacking:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("cloud_test", "cloud_resource_packing.py")

    def test_create_requests(self):
        data = self.mod.create_vm_requests(n_requests=20, seed=42)
        assert data["n_requests"] == 20
        assert all(r["ram_gb"] > 0 for r in data["requests"])

    def test_solve_bin_packing(self):
        results = self.mod.solve_vm_packing(verbose=False)
        assert "bin_packing" in results
        for method in ["FF", "FFD", "BFD"]:
            assert results["bin_packing"][method]["n_servers"] >= 1

    def test_ffd_better_than_ff(self):
        results = self.mod.solve_vm_packing(verbose=False)
        assert results["bin_packing"]["FFD"]["n_servers"] <= \
               results["bin_packing"]["FF"]["n_servers"]

    def test_solve_knapsack(self):
        results = self.mod.solve_vm_packing(verbose=False)
        assert "knapsack" in results
        dp_res = results["knapsack"]["DP (Optimal)"]
        gr_res = results["knapsack"]["Greedy"]
        assert dp_res["revenue"] >= gr_res["revenue"]
        assert dp_res["total_ram"] <= 64  # server capacity
