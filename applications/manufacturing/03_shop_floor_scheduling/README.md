# Shop Floor Scheduling

> **Status:** Placeholder — see existing implementation.

## Sector context
**Sector:** Manufacturing
**Phase:** Shop Floor Scheduling
**Decision-maker:** Production scheduler
**Decision frequency:** Daily

## Decision question
In what sequence should jobs be processed on the shop floor?

## OR problem mapping
**Canonical problem(s):** JSSP / FJSP
**Implementation:** See [`../../manufacturing_scheduling.py`](../../manufacturing_scheduling.py)

## Key modeling aspects
- Sequencing jobs through a fixed set of machines is the classic job-shop/flow-shop scheduling problem, minimizing makespan or tardiness
- Machine environment determines the model: flow shop (identical routing), job shop (job-specific routing), or flexible job shop (alternative machines)
- Real-world constraints such as sequence-dependent setup times, blocking, and no-wait translate to well-studied PFSP variants

## Data requirements
- Job set with per-operation processing times and machine assignments
- Machine availability windows and any planned downtime
- Due dates, release dates, and job priorities/weights
- Setup time matrices (if sequence-dependent)

## Canonical problem
- [Job Shop Scheduling (JSSP)](../../../problems/1_scheduling/job_shop/README.md)
- [Flow Shop Scheduling (PFSP)](../../../problems/1_scheduling/flow_shop/README.md)
- [Flexible Job Shop (FJSP)](../../../problems/1_scheduling/flexible_job_shop/README.md)

## Status
This decision point is part of the **Manufacturing** sector decision chain. Full documentation, formulation, and case studies will be added in a future update.
