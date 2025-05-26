---
mode: 'edit'
description: 'Plan a feature'
---

Your goal is to generate a functional spec for implementing a feature based on the provided idea:

<idea>
Predict Triathlete Race Time and Event Ranking Performance

- It should take in the athlete's past race results stored in the PostgreSQL database.
- The model should able to provide a relatively accurate prediction given the type of event, ie. sprint, supersprint, standard
- The model should be able to better predict and athletes time and position than a simply athlete mean baseline.
- The model's prediction for each athlete should be uploaded to the database for easy access via the PowerBI report. 
- Use machine learning package skilearn to create an optimal model
</idea>

Before generating the spec plan, be sure to review the [file](../docs/summary.md) and [file](../docs/ml_outline.md) to understand an overview of the project and the current ML pipeline. 

RULES:
- Start by defining goal of the model as simple as possible
- Number functional requirements sequentially
- Include acceptance criteria for each functional requirement
- Use clear, concise language
- Aim to create the most accurate model, suggesting additional feature enginnering requirement to ensure quality data as needed. 

NEXT:

- Ask me for feedback to make sure I'm happy
- Give me additional things to consider I may not be thinking about

FINALLY:

When satisfied:

- Output your plan in [folder](/../docs/feature-name.md)
- DO NOT start writing any code or implementation plans. Follow instructions.