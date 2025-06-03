---
mode: 'edit'
description: 'Plan a feature'
---

Your goal is to generate a functional spec for implementing a feature based on the provided idea:

<idea>
Predict Triathlete Race Time and Event Ranking Performance

- It should take in the athlete's past race results stored in the PostgreSQL database.
- You should determine the best features to use for predicting the athlete time and event positions. 
- Conduct PCA to determine the best features to use for the model.
- Conduct feature engineering to create a base training dataset for a preliminary machine learning model.
- The dataset should be easily stored and used for the machine learning pipeline. If additional features are needed, they should be able to be added to the database and used for future model testing. 
- The eventual machine learning model will use scikit-learn, but we do not need to build the model yet.
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