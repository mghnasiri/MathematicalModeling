# Patient Flow Operations

> **Status:** Live — see existing implementations.

## Sector context
**Sector:** Healthcare
**Phase:** Patient Flow
**Decision-maker:** OR scheduler / bed manager / care coordinator
**Decision frequency:** Daily

## Decision question
How should patients be admitted, scheduled for surgery, and routed through care pathways?

## OR problem mapping
**Canonical problem(s):** FJSP (OR scheduling), Bin Packing (beds), Network Flow (pathways)
**Implementation:** See:
- [`../../healthcare_or_scheduling.py`](../../healthcare_or_scheduling.py)
- [`../../healthcare_parallel_or.py`](../../healthcare_parallel_or.py)
- [`../../healthcare_bed_management.py`](../../healthcare_bed_management.py)
- [`../../healthcare_patient_flow.py`](../../healthcare_patient_flow.py)
- [`../../healthcare_home_visits.py`](../../healthcare_home_visits.py)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.
