---
mode: 'edit'
description: 'Draft a repo-aware feature or experiment spec'
---

Your goal is to generate a functional specification for a feature, analytical workflow, or modeling experiment described by the user.

Before drafting the spec:
- Review `CLAUDE.md` for the current architecture and workflow commands.
- Review `docs/prediction_status.md` and `docs/model_improvement_brainstorm.md` when the request touches prediction quality or experimentation.
- Search the relevant symbols in the codebase if the request depends on existing behavior.

RULES:
- Start by defining the goal as simply as possible
- Number functional requirements sequentially
- Include acceptance criteria for each functional requirement
- Use clear, concise language
- Call out data dependencies, leakage risks, and backward-compatibility concerns when relevant
- For prediction work, specify how success will be measured: P@K, Spearman, MAE, tier-specific tradeoffs, or probability quality as appropriate

NEXT:

- Ask me for feedback on the scope, constraints, and success criteria.
- Give me additional considerations I may not be thinking about, especially around data quality, evaluation, and operational impact.

FINALLY:

When satisfied:

- Save the approved spec under `docs/<feature-name>.md`.
- DO NOT start writing code or an implementation plan until I explicitly ask for that next step.