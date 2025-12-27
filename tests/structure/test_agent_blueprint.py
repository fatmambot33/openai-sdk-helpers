from openai_sdk_helpers.structure import AgentBlueprint, AgentEnum


def test_blueprint_summary():
    blueprint = AgentBlueprint(
        name="ResearchBuilder",
        mission="Generate scoped research agents",
        capabilities=["scope", "design"],
        constraints=["no PII"],
        required_tools=["search"],
        data_sources=["docs"],
        guardrails=["cite sources"],
        evaluation_plan=["unit tests"],
        rollout_plan=["staged"],
        notes="extra",
    )

    summary = blueprint.summary()

    assert "ResearchBuilder" in summary
    assert "Generate scoped research agents" in summary
    assert "Capabilities: scope, design" in summary
    assert "Constraints: no PII" in summary
    assert "Required tools: search" in summary
    assert "Data sources: docs" in summary
    assert "Guardrails: cite sources" in summary
    assert "Evaluation: unit tests" in summary
    assert "Rollout: staged" in summary
    assert "Notes: extra" in summary


def test_blueprint_plan_translates_sections():
    blueprint = AgentBlueprint(
        name="AgentFactory",
        mission="Build and validate agents",
        capabilities=["plan", "validate"],
        constraints=["respect rate limits"],
        required_tools=["vector store"],
        data_sources=["policy docs"],
        guardrails=["no secrets"],
        evaluation_plan=["prompt red team"],
        rollout_plan=["canary", "monitor"],
    )

    plan = blueprint.build_plan()

    assert len(plan.tasks) == 6
    task_types = [task.task_type for task in plan.tasks]
    assert task_types == [
        AgentEnum.PLANNER,
        AgentEnum.DESIGNER,
        AgentEnum.BUILDER,
        AgentEnum.VALIDATOR,
        AgentEnum.EVALUATOR,
        AgentEnum.RELEASE_MANAGER,
    ]

    scope_prompt = plan.tasks[0].prompt
    design_prompt = plan.tasks[1].prompt
    validation_prompt = plan.tasks[3].prompt
    deployment_prompt = plan.tasks[5].prompt

    assert "Build and validate agents" in scope_prompt
    assert "Guardrails" in scope_prompt and "no secrets" not in design_prompt
    assert "Capabilities" in design_prompt and "vector store" in design_prompt
    assert "Create automated validation" in validation_prompt
    assert "Launch checklist" in deployment_prompt
