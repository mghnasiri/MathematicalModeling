# Staff Planning

> **Status:** Placeholder — see existing implementations.

## Sector context
**Sector:** Healthcare
**Phase:** Staff Planning
**Decision-maker:** Nursing director / HR scheduler
**Decision frequency:** Monthly / weekly

## Decision question
How should nurses and physicians be assigned to shifts and departments?

## OR problem mapping
**Canonical problem(s):** Scheduling / Set Partitioning
**Implementation:** See [`../../healthcare_nurse_assignment.py`](../../healthcare_nurse_assignment.py) and [`../../workforce_assignment.py`](../../workforce_assignment.py)

## Key modeling aspects
- Nurse-to-shift assignment is a constrained assignment problem with skill requirements, labor regulations, and fairness objectives
- Cyclic shift patterns and minimum rest periods introduce side constraints that go beyond a basic assignment formulation
- Balancing cost minimization with staff satisfaction (preference weights) creates a multi-objective trade-off

## Data requirements
- Staff roster with qualifications, certifications, and shift preferences
- Demand forecast per department and time slot (day/evening/night)
- Labor rules: maximum consecutive shifts, minimum rest hours, overtime limits
- Historical absenteeism rates for robust coverage planning

## Canonical problem
- [Assignment Problem (Hungarian)](../../../problems/4_assignment_matching/assignment/README.md)
- [Parallel Machine Scheduling](../../../problems/1_scheduling/parallel_machine/README.md)

## Status
This decision point is part of the **Healthcare** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.
