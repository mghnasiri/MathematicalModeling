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
import numpy as np
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


# ── Healthcare Applications ──────────────────────────────────────────────────


class TestHealthcareORScheduling:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_or_test", "healthcare_or_scheduling.py")

    def test_create_instance(self):
        data = self.mod.create_or_scheduling_instance()
        assert data["n_surgeries"] == 12
        assert data["n_resources"] == 3
        assert data["processing_times"].shape == (12, 3)

    def test_or_times_reasonable(self):
        data = self.mod.create_or_scheduling_instance()
        for i in range(12):
            assert data["processing_times"][i][0] >= 15  # pre-op min
            assert data["processing_times"][i][1] >= 45  # OR min
            assert data["processing_times"][i][2] >= 30  # PACU min

    def test_solve_or(self):
        results = self.mod.solve_or_scheduling(verbose=False)
        assert "or_scheduling" in results
        assert "ct_scheduling" in results
        for method in ["NEH", "IG"]:
            assert results["or_scheduling"][method]["makespan"] > 0

    def test_ct_scheduling(self):
        results = self.mod.solve_or_scheduling(verbose=False)
        for method in ["EDD", "ATC"]:
            assert results["ct_scheduling"][method]["weighted_tardiness"] >= 0

    def test_ig_improves_neh(self):
        results = self.mod.solve_or_scheduling(verbose=False)
        assert results["or_scheduling"]["IG"]["makespan"] <= \
               results["or_scheduling"]["NEH"]["makespan"]


class TestHealthcareNurseAssignment:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_nurse_test", "healthcare_nurse_assignment.py")

    def test_create_instance(self):
        data = self.mod.create_nurse_assignment_instance()
        assert data["n"] == 8
        assert data["cost_matrix"].shape == (8, 8)

    def test_acuity_competency_cost(self):
        data = self.mod.create_nurse_assignment_instance()
        # High-acuity patient to high-competency nurse should cost less
        # than to low-competency nurse (when same pod)
        assert data["cost_matrix"][0][0] < data["cost_matrix"][6][0]  # Chen vs Garcia on Pt A (acuity 5)

    def test_solve_produces_valid(self):
        results = self.mod.solve_nurse_assignment(verbose=False)
        assert "Hungarian" in results
        assert "Greedy" in results
        assert results["Hungarian"]["cost"] <= results["Greedy"]["cost"] + 1e-6


class TestHealthcareAmbulanceLocation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_amb_test", "healthcare_ambulance_location.py")

    def test_create_data(self):
        data = self.mod.create_ambulance_data()
        assert data["m"] == 10
        assert data["n"] == 20
        assert data["p"] == 4

    def test_response_times_positive(self):
        data = self.mod.create_ambulance_data()
        assert np.all(data["response_time"] >= 0)

    def test_solve_pmedian(self):
        results = self.mod.solve_ambulance_location(verbose=False)
        assert "p_median" in results
        for method in ["Greedy", "Interchange"]:
            assert len(results["p_median"][method]["open"]) == 4

    def test_solve_uflp(self):
        results = self.mod.solve_ambulance_location(verbose=False)
        assert "uflp" in results
        for method in ["Greedy-Add", "SA"]:
            assert results["uflp"][method]["cost"] > 0

    def test_interchange_improves(self):
        results = self.mod.solve_ambulance_location(verbose=False)
        assert results["p_median"]["Interchange"]["cost"] <= \
               results["p_median"]["Greedy"]["cost"] + 1e-6


class TestHealthcareSupplyDelivery:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_supply_test", "healthcare_supply_delivery.py")

    def test_create_instance(self):
        data = self.mod.create_medical_delivery_instance()
        assert data["n_clinics"] == 18
        assert data["capacity"] == 500

    def test_demand_within_capacity(self):
        data = self.mod.create_medical_delivery_instance()
        assert all(d <= data["capacity"] for d in data["demands"])

    def test_solve_delivery(self):
        results = self.mod.solve_medical_delivery(verbose=False)
        for method in ["Clarke-Wright", "Sweep", "SA"]:
            assert results[method]["distance"] > 0
            assert results[method]["n_vehicles"] >= 1


class TestHealthcareBedManagement:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_bed_test", "healthcare_bed_management.py")

    def test_create_bed_packing(self):
        data = self.mod.create_bed_packing_instance()
        assert data["n_patients"] == 20

    def test_create_cardiac_knapsack(self):
        data = self.mod.create_cardiac_admission_instance()
        assert data["n_patients"] == 12
        assert data["capacity"] == 10

    def test_solve_bed_packing(self):
        results = self.mod.solve_bed_management(verbose=False)
        assert "bed_packing" in results
        for method in ["FF", "FFD", "BFD"]:
            assert results["bed_packing"][method]["n_wards_needed"] >= 1

    def test_cardiac_admission(self):
        results = self.mod.solve_bed_management(verbose=False)
        dp = results["cardiac_admission"]["DP (Optimal)"]
        gr = results["cardiac_admission"]["Greedy"]
        assert dp["total_priority"] >= gr["total_priority"]
        assert dp["total_los"] <= 10  # capacity

    def test_ffd_efficient(self):
        results = self.mod.solve_bed_management(verbose=False)
        assert results["bed_packing"]["FFD"]["n_wards_needed"] <= \
               results["bed_packing"]["FF"]["n_wards_needed"]
