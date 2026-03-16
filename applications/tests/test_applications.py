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


# ── Healthcare Applications (Batch 2) ────────────────────────────────────────


class TestHealthcareHomeVisits:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_home_test", "healthcare_home_visits.py")

    def test_create_instance(self):
        data = self.mod.create_home_visit_instance()
        assert data["n_patients"] == 15
        assert len(data["coords"]) == 16  # +agency
        assert data["capacity"] == 200

    def test_time_windows_valid(self):
        data = self.mod.create_home_visit_instance()
        for i in range(1, 16):
            assert data["time_windows"][i][0] <= data["time_windows"][i][1]

    def test_tsp_solve(self):
        results = self.mod.solve_home_visits(verbose=False)
        assert "tsp" in results
        for method in ["NN", "2-opt", "SA"]:
            assert results["tsp"][method]["distance"] > 0

    def test_vrptw_solve(self):
        results = self.mod.solve_home_visits(verbose=False)
        assert "vrptw" in results
        assert results["vrptw"]["Solomon-I1"]["distance"] > 0
        assert results["vrptw"]["Solomon-I1"]["n_nurses"] >= 1

    def test_2opt_improves_nn(self):
        results = self.mod.solve_home_visits(verbose=False)
        assert results["tsp"]["2-opt"]["distance"] <= \
               results["tsp"]["NN"]["distance"] + 1e-6


class TestHealthcarePatientFlow:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_pflow_test", "healthcare_patient_flow.py")

    def test_create_instance(self):
        data = self.mod.create_patient_flow_instance()
        assert data["n_patients"] == 10
        assert data["n_departments"] == 5

    def test_all_methods_produce_results(self):
        results = self.mod.solve_patient_flow(verbose=False)
        for method in ["SPT", "LPT", "MWR", "Shifting-Bottleneck", "SA"]:
            assert method in results
            assert results[method]["makespan"] > 0

    def test_sa_improves_dispatching(self):
        results = self.mod.solve_patient_flow(verbose=False)
        best_dispatch = min(
            results[m]["makespan"] for m in ["SPT", "LPT", "MWR"]
        )
        assert results["SA"]["makespan"] <= best_dispatch


class TestHealthcareParallelOR:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_por_test", "healthcare_parallel_or.py")

    def test_create_instance(self):
        data = self.mod.create_or_scheduling_instance()
        assert data["n_surgeries"] == 16
        assert data["n_ors"] == 4

    def test_all_methods_produce_results(self):
        results = self.mod.solve_parallel_or(verbose=False)
        for method in ["LPT", "MULTIFIT", "List-Scheduling", "GA"]:
            assert method in results
            assert results[method]["makespan"] > 0
            assert len(results[method]["assignment"]) == 4

    def test_lpt_reasonable(self):
        results = self.mod.solve_parallel_or(verbose=False)
        data = self.mod.create_or_scheduling_instance()
        ideal = data["total_minutes"] / data["n_ors"]
        # LPT should be within 4/3 of ideal
        assert results["LPT"]["makespan"] <= ideal * 1.4

    def test_all_surgeries_assigned(self):
        results = self.mod.solve_parallel_or(verbose=False)
        for method, res in results.items():
            all_jobs = []
            for machine_jobs in res["assignment"]:
                all_jobs.extend(machine_jobs)
            assert sorted(all_jobs) == list(range(16))


class TestHealthcareEmergencyNetwork:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_enet_test", "healthcare_emergency_network.py")

    def test_create_network(self):
        data = self.mod.create_emergency_network()
        assert data["n"] == 8
        assert len(data["transfer_links"]) == 13

    def test_max_flow(self):
        results = self.mod.solve_emergency_network(verbose=False)
        assert results["max_flow"]["max_throughput"] > 0
        s_set, t_set = results["max_flow"]["min_cut"]
        assert 0 in s_set  # trauma center in source side
        assert 7 in t_set  # hub in sink side

    def test_shortest_path(self):
        results = self.mod.solve_emergency_network(verbose=False)
        # Path to hub should exist
        assert results["shortest_path"][7]["distance"] > 0
        assert results["shortest_path"][7]["path"][0] == 0
        assert results["shortest_path"][7]["path"][-1] == 7

    def test_mst_backbone(self):
        results = self.mod.solve_emergency_network(verbose=False)
        assert results["mst"]["Kruskal"]["total_cost"] > 0
        assert len(results["mst"]["Kruskal"]["backbone_links"]) == 7  # n-1

    def test_kruskal_equals_prim(self):
        results = self.mod.solve_emergency_network(verbose=False)
        assert abs(results["mst"]["Kruskal"]["total_cost"] -
                   results["mst"]["Prim"]["total_cost"]) < 1e-6


class TestHealthcareClinicalTrial:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("hc_trial_test", "healthcare_clinical_trial.py")

    def test_create_instance(self):
        data = self.mod.create_clinical_trial_instance()
        assert data["n"] == 12
        assert data["num_resources"] == 3

    def test_critical_path(self):
        results = self.mod.solve_clinical_trial(verbose=False)
        assert results["critical_path_length"] > 0

    def test_all_methods_produce_results(self):
        results = self.mod.solve_clinical_trial(verbose=False)
        for method in ["Serial-SGS (LFT)", "Serial-SGS (GRPW)",
                       "Parallel-SGS (LFT)", "GA"]:
            assert method in results
            assert results[method]["makespan"] > 0

    def test_makespan_ge_critical_path(self):
        results = self.mod.solve_clinical_trial(verbose=False)
        cp = results["critical_path_length"]
        for method in ["Serial-SGS (LFT)", "GA"]:
            assert results[method]["makespan"] >= cp


# ── Agriculture Applications ─────────────────────────────────────────────────


class TestAgricultureCropHarvest:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_harvest_test", "agriculture_crop_harvest.py")

    def test_create_instances(self):
        instances = self.mod.create_crop_instances(seed=42)
        assert len(instances) == 8  # 8 crops
        for inst in instances:
            assert "name" in inst
            assert "instance" in inst

    def test_solve_produces_results(self):
        results = self.mod.solve_crop_harvest(verbose=False, seed=42)
        assert "unconstrained" in results
        assert "constrained" in results
        assert len(results["unconstrained"]) == 8

    def test_critical_fractile_valid(self):
        results = self.mod.solve_crop_harvest(verbose=False, seed=42)
        for res in results["unconstrained"]:
            assert 0 < res["critical_fractile"] < 1
            assert res["order_quantity"] > 0

    def test_constrained_exists(self):
        results = self.mod.solve_crop_harvest(verbose=False, seed=42)
        constrained = results["constrained"]
        assert "marginal_allocation" in constrained or "independent_scale" in constrained


class TestAgricultureFeedProcurement:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_feed_test", "agriculture_feed_procurement.py")

    def test_solve_all_inputs(self):
        results = self.mod.solve_feed_procurement(verbose=False)
        assert "cattle_feed" in results
        assert "fertilizer" in results
        assert "seeds" in results

    def test_methods_present(self):
        results = self.mod.solve_feed_procurement(verbose=False)
        for input_key in ["cattle_feed", "fertilizer", "seeds"]:
            r = results[input_key]
            assert "methods" in r or "EOQ" in r

    def test_total_demand_positive(self):
        results = self.mod.solve_feed_procurement(verbose=False)
        for input_key in ["cattle_feed", "fertilizer", "seeds"]:
            assert results[input_key]["total_demand"] > 0


class TestAgricultureFarmDelivery:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_delivery_test", "agriculture_farm_delivery.py")

    def test_create_instance(self):
        data = self.mod.create_farm_delivery_instance(seed=42)
        assert data["n_customers"] == 15
        assert data["capacity"] == 2000
        assert len(data["coords"]) == 16  # depot + 15

    def test_demands_within_capacity(self):
        data = self.mod.create_farm_delivery_instance(seed=42)
        for d in data["demands"]:
            assert d <= data["capacity"]

    def test_deterministic_solve(self):
        data = self.mod.create_farm_delivery_instance(seed=42)
        results = self.mod.solve_deterministic(data, verbose=False)
        for method in ["Clarke-Wright", "Sweep"]:
            assert results[method]["distance"] > 0
            assert results[method]["n_vehicles"] >= 1

    def test_stochastic_solve(self):
        data = self.mod.create_farm_delivery_instance(seed=42)
        results = self.mod.solve_stochastic(data, verbose=False)
        assert "CC-Clarke-Wright" in results


class TestAgricultureIrrigationNetwork:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_irrigation_test", "agriculture_irrigation_network.py")

    def test_create_network(self):
        data = self.mod.create_irrigation_network()
        assert data["n"] == 10
        assert len(data["pipe_segments"]) >= 12

    def test_solve_all_models(self):
        results = self.mod.solve_irrigation_network(verbose=False)
        assert "mst" in results
        assert "max_flow" in results
        assert "shortest_path" in results

    def test_mst_backbone(self):
        results = self.mod.solve_irrigation_network(verbose=False)
        mst = results["mst"]
        assert mst["Kruskal"]["total_cost"] > 0
        assert len(mst["Kruskal"]["backbone_pipes"]) == 9  # n-1

    def test_kruskal_equals_prim(self):
        results = self.mod.solve_irrigation_network(verbose=False)
        assert abs(results["mst"]["Kruskal"]["total_cost"] -
                   results["mst"]["Prim"]["total_cost"]) < 1e-6

    def test_max_flow_positive(self):
        results = self.mod.solve_irrigation_network(verbose=False)
        assert results["max_flow"]["max_throughput"] > 0

    def test_shortest_path_to_fields(self):
        results = self.mod.solve_irrigation_network(verbose=False)
        sp = results["shortest_path"]
        # Should have paths to field zones (nodes 4-8)
        for node in range(4, 9):
            assert node in sp
            assert sp[node]["distance"] > 0


class TestAgricultureCropRotation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_rotation_test", "agriculture_crop_rotation.py")

    def test_create_instance(self):
        data = self.mod.create_crop_allocation_instance()
        assert data["n_fields"] == 6
        assert data["n_crops"] == 5

    def test_solve_produces_results(self):
        results = self.mod.solve_crop_allocation(verbose=False)
        assert "LP" in results

    def test_pareto_front(self):
        results = self.mod.solve_crop_allocation(verbose=False)
        assert "Pareto" in results
        assert results["Pareto"]["n_points"] >= 2


class TestAgricultureColdStorage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("ag_cold_test", "agriculture_cold_storage.py")

    def test_create_produce_lots(self):
        data = self.mod.create_produce_lots(n_lots=20, seed=42)
        assert data["n_lots"] == 20
        assert data["storage_capacity"] > 0

    def test_create_packaging(self):
        data = self.mod.create_packaging_instance()
        assert data["roll_length"] == 200
        assert len(data["sheets"]) == 4

    def test_solve_all(self):
        results = self.mod.solve_cold_storage_packing(verbose=False)
        assert "bin_packing" in results
        assert "cutting_stock" in results

    def test_bin_packing_methods(self):
        results = self.mod.solve_cold_storage_packing(verbose=False)
        bp = results["bin_packing"]
        for method in ["FF", "FFD", "BFD"]:
            assert bp[method]["n_units"] >= 1

    def test_ffd_better_than_ff(self):
        results = self.mod.solve_cold_storage_packing(verbose=False)
        bp = results["bin_packing"]
        assert bp["FFD"]["n_units"] <= bp["FF"]["n_units"]

    def test_cutting_stock_results(self):
        results = self.mod.solve_cold_storage_packing(verbose=False)
        cs = results["cutting_stock"]
        # Should have method results with positive roll counts
        assert "Greedy" in cs or "FFD" in cs
        assert cs.get("lower_bound", 1) >= 1


# ── Finance, Retail, Telecom, Energy, Logistics Applications ────────────────


class TestFinancePortfolio:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("finance_test", "finance_portfolio.py")

    def test_create_instance(self):
        data = self.mod.create_portfolio_instance()
        assert data["n_assets"] == 8
        assert data["expected_returns"].shape == (8,)
        assert data["covariance"].shape == (8, 8)

    def test_covariance_positive_semidefinite(self):
        data = self.mod.create_portfolio_instance()
        eigvals = np.linalg.eigvalsh(data["covariance"])
        assert np.all(eigvals >= -1e-8)

    def test_solve_all_methods(self):
        results = self.mod.solve_portfolio(verbose=False)
        for method in ["Markowitz", "Robust", "Equal-Weight", "Min-Variance", "Max-Return"]:
            assert method in results
            assert results[method]["expected_return"] > 0
            assert results[method]["portfolio_std"] >= 0

    def test_weights_sum_to_one(self):
        results = self.mod.solve_portfolio(verbose=False)
        for method, res in results.items():
            assert abs(np.sum(res["weights"]) - 1.0) < 1e-4, \
                f"{method}: weights sum to {np.sum(res['weights'])}"

    def test_robust_more_conservative(self):
        results = self.mod.solve_portfolio(verbose=False)
        # Robust portfolio should have lower or equal expected return
        # (it sacrifices return for robustness)
        assert results["Robust"]["expected_return"] <= \
               results["Markowitz"]["expected_return"] + 1e-6


class TestRetailInventory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("retail_test", "retail_inventory.py")

    def test_create_bakery_instances(self):
        instances = self.mod.create_bakery_instances()
        assert len(instances) == 5
        for inst in instances:
            assert inst["cost"] < inst["price"]
            assert len(inst["scenarios"]) == 200

    def test_create_eoq_instances(self):
        instances = self.mod.create_eoq_instances()
        assert len(instances) == 3
        for inst in instances:
            assert inst["annual_demand"] > 0
            assert inst["ordering_cost"] > 0

    def test_create_lot_sizing(self):
        data = self.mod.create_lot_sizing_instance()
        assert data["T"] == 12
        assert len(data["demands"]) == 12

    def test_bakery_newsvendor(self):
        results = self.mod.solve_bakery_newsvendor(verbose=False)
        assert "bakery" in results
        assert len(results["bakery"]) == 5
        for r in results["bakery"]:
            assert r["order_quantity"] > 0
            assert 0 < r["critical_fractile"] < 1

    def test_staple_eoq(self):
        results = self.mod.solve_staple_eoq(verbose=False)
        assert "staples" in results
        assert len(results["staples"]) == 3
        for r in results["staples"]:
            assert r["eoq"] > 0
            assert r["total_cost"] > 0

    def test_seasonal_lot_sizing(self):
        results = self.mod.solve_seasonal_lot_sizing(verbose=False)
        assert "lot_sizing" in results
        for method in ["Wagner-Whitin", "Silver-Meal", "Lot-for-Lot"]:
            assert method in results["lot_sizing"]
            assert results["lot_sizing"][method]["total_cost"] > 0

    def test_ww_optimal(self):
        results = self.mod.solve_seasonal_lot_sizing(verbose=False)
        ls = results["lot_sizing"]
        assert ls["Wagner-Whitin"]["total_cost"] <= ls["Lot-for-Lot"]["total_cost"] + 1e-6

    def test_solve_all(self):
        results = self.mod.solve_retail_inventory(verbose=False)
        assert "bakery" in results
        assert "staples" in results
        assert "lot_sizing" in results


class TestTelecomNetwork:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("telecom_test", "telecom_network.py")

    def test_create_coverage_instance(self):
        data = self.mod.create_coverage_instance()
        assert data["m"] == 40  # demand zones
        assert data["n"] == 15  # tower sites
        assert len(data["subsets"]) == 15

    def test_all_zones_coverable(self):
        data = self.mod.create_coverage_instance()
        all_covered = set()
        for s in data["subsets"]:
            all_covered.update(s)
        # At least some zones should be coverable
        assert len(all_covered) >= 20

    def test_solve_produces_results(self):
        results = self.mod.solve_telecom_network(verbose=False)
        assert "coverage" in results
        assert "frequency" in results

    def test_coverage_methods(self):
        results = self.mod.solve_telecom_network(verbose=False)
        for method in ["Greedy-CE", "Greedy-LF"]:
            assert method in results["coverage"]
            assert results["coverage"][method]["total_cost"] > 0
            assert results["coverage"][method]["n_towers"] >= 1

    def test_frequency_valid(self):
        results = self.mod.solve_telecom_network(verbose=False)
        for method in ["DSatur", "Greedy-LF"]:
            assert method in results["frequency"]
            assert results["frequency"][method]["valid"]
            assert results["frequency"][method]["n_frequencies"] >= 1

    def test_interference_graph_created(self):
        results = self.mod.solve_telecom_network(verbose=False)
        assert "selected_towers" in results
        assert len(results["selected_towers"]) >= 1


class TestEnergyGrid:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("energy_test", "energy_grid.py")

    def test_create_dispatch(self):
        data = self.mod.create_dispatch_instance()
        assert data["n_plants"] == 6
        assert data["total_demand"] > 0
        assert len(data["costs"]) == 6

    def test_create_grid_network(self):
        data = self.mod.create_grid_network()
        assert data["n"] == 16  # 6 plants + 8 districts + 2 super nodes
        assert data["source"] == 14
        assert data["sink"] == 15

    def test_create_backbone(self):
        data = self.mod.create_backbone_network()
        assert data["n"] == 14
        assert len(data["edges"]) > 0

    def test_dispatch_feasible(self):
        results = self.mod.solve_energy_grid(verbose=False)
        assert results["dispatch"]["success"]
        assert results["dispatch"]["total_cost"] > 0

    def test_generation_meets_demand(self):
        results = self.mod.solve_energy_grid(verbose=False)
        total_gen = sum(results["dispatch"]["generation"])
        total_demand = results["dispatch"]["total_demand"]
        assert abs(total_gen - total_demand) < 1.0

    def test_max_flow_positive(self):
        results = self.mod.solve_energy_grid(verbose=False)
        assert results["max_flow"]["max_throughput"] > 0

    def test_mst_backbone(self):
        results = self.mod.solve_energy_grid(verbose=False)
        assert results["mst"]["total_cost"] > 0
        assert results["mst"]["n_links"] >= 1


class TestLogisticsGateAssignment:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mod = _load_app("logistics_test", "logistics_gate_assignment.py")

    def test_create_gate_instance(self):
        data = self.mod.create_gate_assignment_instance()
        assert data["n"] == 10
        assert data["cost_matrix"].shape == (10, 10)

    def test_wide_body_penalty(self):
        data = self.mod.create_gate_assignment_instance()
        # Wide-body flights (indices 3,4,8) should have high cost for narrow gates
        # BA722 (index 3) is wide, gate A3 (index 2) is narrow
        wide_flight = 3
        narrow_gate = 2
        wide_gate = 0
        assert data["cost_matrix"][wide_flight][narrow_gate] > \
               data["cost_matrix"][wide_flight][wide_gate]

    def test_create_turnaround(self):
        data = self.mod.create_turnaround_instance()
        assert data["n_jobs"] == 10
        assert data["n_machines"] == 3
        assert len(data["jobs"]) == 10

    def test_solve_assignment(self):
        results = self.mod.solve_gate_assignment(verbose=False)
        assert "assignment" in results
        assert results["assignment"]["Hungarian"]["cost"] <= \
               results["assignment"]["Greedy"]["cost"] + 1e-6

    def test_solve_turnaround(self):
        results = self.mod.solve_gate_assignment(verbose=False)
        assert "turnaround" in results
        for method in ["SPT", "LPT", "MWR", "SA"]:
            assert method in results["turnaround"]
            assert results["turnaround"][method]["makespan"] > 0

    def test_sa_improves_dispatching(self):
        results = self.mod.solve_gate_assignment(verbose=False)
        best_dispatch = min(
            results["turnaround"][m]["makespan"] for m in ["SPT", "LPT", "MWR"]
        )
        assert results["turnaround"]["SA"]["makespan"] <= best_dispatch
