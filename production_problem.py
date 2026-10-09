"""Executable parameter definition and formal metadata for Production Capacity Planning."""
from __future__ import annotations

from optimization_problem import OptimizationProblem, ParameterDefinition
from problem_definition import OptimizationProblemSpec, project_problem_spec


PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID = "production_capacity_planning"


def default_production_capacity_planning_problem() -> OptimizationProblem:
    parameters = {
        "demand_multiplier": ParameterDefinition(
            "demand_multiplier",
            "business",
            1.0,
            float,
            minimum=0.5,
            maximum=1.5,
            description="Multiplier applied uniformly to source product demand.",
        ),
        "overtime_cost_multiplier": ParameterDefinition(
            "overtime_cost_multiplier",
            "business",
            1.0,
            float,
            minimum=0.0,
            maximum=3.0,
            description="Multiplier applied to line overtime cost per hour.",
        ),
        "milp_time": ParameterDefinition(
            "milp_time",
            "solver",
            5.0,
            float,
            minimum=1e-9,
            description="Optimization time limit in seconds.",
        ),
        "mip_gap": ParameterDefinition(
            "mip_gap",
            "solver",
            1e-6,
            float,
            minimum=0.0,
            maximum=1.0,
            description="Relative optimization gap passed to scipy.optimize.milp.",
        ),
    }
    return OptimizationProblem(
        name=PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID,
        parameters=parameters,
        values={name: definition.default for name, definition in parameters.items()},
        data_requirements=("products", "lines", "compatibility", "current_plan"),
        capabilities={
            "configurable": tuple(parameters),
            "unsupported": (
                "arbitrary_constraints",
                "arbitrary_objectives",
                "job_shop_sequencing",
                "setup_time_scheduling",
                "source_data_modification",
            ),
        },
    )


def production_capacity_planning_spec(
    problem: OptimizationProblem | None = None,
) -> OptimizationProblemSpec:
    executable = problem or default_production_capacity_planning_problem()
    return project_problem_spec(
        executable,
        problem_id=PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID,
        display_name="Production Capacity Planning",
        description=(
            "Deterministic demo production-capacity planning across products and compatible "
            "production lines using regular and overtime capacity. Costs are USD."
        ),
        input_descriptions={
            "products": "Product demand by product_id.",
            "lines": "Regular/overtime line capacity and overtime hourly cost.",
            "compatibility": "Allowed product-line pairs with processing time and variable production cost.",
            "current_plan": "Feasible current production allocation used for baseline economics.",
        },
        objective_summary="Minimize variable production cost plus overtime cost while satisfying demand.",
        decision_summary=(
            "Allocate regular and overtime production units for each compatible product-line pair."
        ),
    )


__all__ = [
    "PRODUCTION_CAPACITY_PLANNING_PROBLEM_ID",
    "default_production_capacity_planning_problem",
    "production_capacity_planning_spec",
]
