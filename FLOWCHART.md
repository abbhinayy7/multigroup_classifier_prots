# Proteomics ML Pipeline — Flowchart

Below is a visual flowchart describing the multigroup classifier pipeline and key decision points.

```mermaid
flowchart LR
  A[Input Data\nannotation.tsv + protein.tsv] --> B[Preprocessing\nfilter NAs, normalize, impute]
  B --> C[Feature Selection / Filtering\nremove NA-heavy features]
  C --> D[Train/Test Split\nStratified 75/25]
  D --> E{Select Workflow}
  E -->|Binary (GBM)| F1[Binary Training\nBayesian HPO → ERROR: NaN]\n  F1 --> F1e[Action: Data cleaning / class weighting]
  E -->|Multigroup| F2[Multigroup Training\nBayesian HPO → Best CV=0.6250]
  F2 --> G[Final Training\nEarly stopping → 272 rounds]
  G --> H[Evaluation\nAccuracy=91.30%\nPrecision=92.89%\nRecall=91.30%\nF1=91.01%]
  H --> I[Outputs & Artifacts\nmodel.json, best_params.tsv, roc.png, confusion_matrix.tsv]
  I --> J[Dockerize for Reproducibility\nDockerfile.multigroup, entrypoint.sh]
  J --> K[Deploy / Share\nRun via Docker, reproduce results]

  classDef step fill:#f9f9f9,stroke:#333,stroke-width:1px;
  class A,B,C,D,F2,G,H,I,J,K step;
  class F1,F1e fill:#ffe6e6,stroke:#cc0000;
```

*Notes:*
- The `Binary` branch indicates a failure on the GBM test (NaN during CV); follow-up action is data cleaning or using class-weighted training.
- The `Multigroup` branch shows the successful path with the best CV score and final evaluation metrics.

You can render this diagram on GitHub (supports mermaid) or use any mermaid renderer to generate PNG/SVG.
