---
mode: 'edit'
description: 'Plan a triathlon-db implementation or experiment'
---

Your goal is to generate an implementation plan for a feature, refactor, experiment, or analysis task in this repository.

RULES:
- Keep implementations simple, do not over architect
- Do not generate real code for your plan, pseudocode is OK
- For each step in your plan, include the objective, the concrete files or modules involved, the validation approach, and any necessary pseudocode
- Call out any necessary user intervention for each step
- Include testing and data-validation implications where relevant

FIRST:

- Review `CLAUDE.md` to understand the current project architecture and workflow.
- Review `docs/prediction_status.md`, `docs/experiment-log.md`, and `docs/model_improvement_brainstorm.md` when the task touches prediction or experimentation.
- Review the attached specification or user request to understand the requirements and objective.
- Search the relevant symbols in `tri_analysis/`, `scripts/`, `streamlit_app.py`, and `tests/` before finalizing the plan.

THEN:
- Create a detailed implementation plan that outlines the steps needed to achieve the objective.
- The plan should be structured, clear, and easy to follow.
- Structure your plan as follows, and output as Markdown code block

```markdown
# Implementation Plan for [Task Name]

- [ ] Step 1: [Brief title]
  - **Task**: [Detailed explanation of what needs to be implemented]
  - **Files**: [Maximum of 20 files, ideally less]
    - `path/to/file1.ts`: [Description of changes], [Pseudocode for implementation]
  - **Validation**: [Tests, scripts, or manual checks]
  - **Risks**: [Behavioral or data risks]

[Additional steps...]
```

- If the task affects prediction quality, include a validation step covering training or backtest commands and the expected metrics to watch.
- If the task affects data pipelines or analytics outputs, include a step for regression checks on the affected scripts or dashboards.

NEXT:

- Iterate with me until I am satisfied with the plan.

FINALLY: 

- When I confirm the plan, save it under `docs/<plan-name>.md`.
- DO NOT start implementation without my permission.