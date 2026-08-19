# Cutting-Edge Data Science Skills & Methods — 2026 Collection

*Compiled July 2026. NLP-prioritized per Ryan's request. Each entry: what it is, why it matters, and key tools. Sources at the end of each section.*

---

## Part 1 — NLP & Language Models (priority area)

### 1.1 LLM post-training and fine-tuning

The center of gravity in applied NLP has moved from training models to *adapting* them. The 2026 post-training stack is a pipeline of techniques, each with a distinct role:

- **Supervised Fine-Tuning (SFT)** — teach a model to imitate input→output pairs. Still the first step for format/style/domain adaptation, but insufficient for tasks requiring trial-and-error learning.
- **Parameter-Efficient Fine-Tuning (PEFT): LoRA and QLoRA** — adapt billion-parameter models by training small low-rank adapter matrices (QLoRA adds 4-bit quantization), making fine-tuning feasible on a single GPU. The default approach for custom models in 2026.
- **DPO (Direct Preference Optimization)** — align a model to human preferences directly from chosen/rejected pairs, without training a separate reward model. Largely replaced classic RLHF for preference alignment because it is simpler and more stable.
- **GRPO (Group Relative Policy Optimization)** — the RL method behind DeepSeek-R1's reasoning ability: generate N completions, score them relative to the group average, no separate value/reward model needed. The go-to technique for training reasoning and agentic behavior with verifiable rewards.
- **RFT (Reinforcement Fine-Tuning) and RULER** — reward-signal training for multi-step agents (tool calls, search, APIs); RULER uses an LLM-as-judge to compare trajectories, eliminating hand-crafted reward functions and labeled data.

**Tools:** Hugging Face TRL (SFT/DPO/GRPO trainers), Unsloth (memory-efficient GRPO/LoRA), Axolotl, ART (Agent Reinforcement Trainer), vLLM for fast inference during training loops.

**Skill to build:** knowing *which* stage you need. Rough decision rule: prompt engineering → RAG → SFT/LoRA → DPO → GRPO/RFT, escalating only when the cheaper technique fails.

Sources: [How to Fine-Tune LLMs in 2026 (Daily Dose of DS)](https://blog.dailydoseofds.com/p/how-to-fine-tune-llms-in-2026), [LLM Fine-Tuning Guide: LoRA, QLoRA, DPO, GRPO, RLHF (FutureAGI)](https://futureagi.com/blog/llm-fine-tuning-guide-2025/), [LLM Post-Training with TRL: SFT to DPO and GRPO (MarkTechPost)](https://www.marktechpost.com/2026/05/01/a-coding-guide-on-llm-post-training-with-trl-from-supervised-fine-tuning-to-dpo-and-grpo-reasoning/), [Post-training methods for language models (Red Hat)](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models), [Post-Training LLMs: SFT, RLHF, DPO & GRPO (Sundeep Teki)](https://www.sundeepteki.org/advice/the-complete-guide-to-post-training-llms-how-sft-rlhf-dpo-and-grpo-shape-llms)

### 1.2 Retrieval-Augmented Generation (RAG), beyond naive retrieval

Naive chunk-embed-retrieve RAG is now considered a baseline. Production systems in 2026 layer on:

- **Hybrid retrieval** — dense embeddings + BM25/sparse, fused (e.g., reciprocal rank fusion), then **reranked** with a cross-encoder.
- **Agentic RAG** — retrieval as a multi-step decision process: the model plans, decomposes queries, chooses tools, retrieves iteratively, and self-corrects. The dominant 2026 pattern.
- **GraphRAG / graph-based memory** — knowledge-graph and hypergraph structures (e.g., HGMem, Graph-O1) for multi-hop reasoning over connected facts.
- **Corrective/self-verifying RAG** — sufficiency checking of retrieved evidence before answering (e.g., SURE-RAG), multi-stage filtering (HiFi-RAG), retrieval triggered by corpus statistics rather than model confidence (QuCo-RAG).
- **Multimodal & structured RAG** — retrieval over tables decomposed into semantic units (FT-RAG), documents with visual content (MegaRAG), long video (TV-RAG).
- **Long-document RAG** — hierarchical summaries guiding retrieval (MiA-RAG), discourse-structure-aware synthesis (Disco-RAG).
- **RAG security** — defenses against corpus poisoning (RAGPart/RAGMask) — a genuinely new skill area.

Sources: [20 Advanced RAG Types to Know in 2026 (Turing Post)](https://www.turingpost.com/p/ragtypes), [12 Advanced RAG Techniques (Atlan)](https://atlan.com/know/advanced-rag-techniques/), [All you need to know about RAG in 2026](https://aishwaryasrinivasan.substack.com/p/all-you-need-to-know-about-rag-in), [Advanced RAG Techniques (Google Codelabs)](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/8-advanced-rag-methods/advanced-rag-methods)

### 1.3 Embeddings and neural search

- **Late-interaction retrieval (ColBERT-family)** — instead of one vector per document, keep per-token vectors and match at query time; markedly better precision than single-vector search, now practical at scale.
- **Visual document retrieval** — ColPali-style late-interaction models that embed page *images* (e.g., Nemotron ColEmbed V2), skipping brittle OCR pipelines for PDFs/scans.
- **Model selection via benchmarks** — MTEB and production-focused comparisons; matryoshka embeddings (truncatable dimensions) for cost control.

Sources: [ColBERT (Stanford)](https://github.com/stanford-futuredata/ColBERT), [Nemotron ColEmbed V2 (arXiv)](https://arxiv.org/html/2602.03992v2), [Best Embedding Models in 2026 (Mixpeek)](https://mixpeek.com/curated-lists/best-embedding-models), [Open-Source Embedding Models for RAG 2026 (KnowledgeSDK)](https://knowledgesdk.com/blog/open-source-embedding-models-rag-2026), [Late Interaction: ColBERT to Wholembed v3](https://trilogyai.substack.com/p/late-interaction-colbert-to-wholembed)

### 1.4 Small language models (SLMs) and efficient NLP

A strong 2026 counter-trend to "just use the biggest LLM":

- **Discriminative SLMs for extraction** — e.g., **GLiNER** for zero-shot NER: runs on CPU, deterministic, returns exact character offsets and confidence scores (auditable), deployable locally for privacy. LLMs remain non-deterministic and hard to audit for extraction.
- **The orchestrator+tools architecture** — delegate *reasoning* to an LLM and *extraction* to specialized SLMs called as tools. This hybrid is emerging as the production-standard pattern.
- **Distillation & distant supervision** — training small task models from LLM-generated labels; large quality gains per dollar.

Sources: [The rise of small language models for information extraction (SAS)](https://blogs.sas.com/content/subconsciousmusings/2025/12/17/the-rise-of-small-language-models-for-information-extraction/), [Small Language Models: Comprehensive Guide 2026 (CogitX)](https://cogitx.ai/blog/small-language-models-slms-comprehensive-guide-2026), [NLP trends in 2026 (InData Labs)](https://indatalabs.com/blog/natural-language-processing-trends)

### 1.5 Text classification & information extraction with LLMs

- **Zero-shot → instruction-tuned classification** — LLMs now rival supervised classifiers on many social-science and business text tasks with no labeled data; instruction tuning on small labeled sets closes the rest of the gap.
- **Generative information extraction** — schema-guided extraction (entities, relations, events) as structured generation, using constrained decoding / structured outputs (JSON schema enforcement) for reliability.

Sources: [LLMs for Text Classification: Zero-Shot to Instruction-Tuning (Sociological Methods & Research)](https://journals.sagepub.com/doi/10.1177/00491241251325243), [LLMs for generative information extraction: a survey (Frontiers of CS)](https://link.springer.com/article/10.1007/s11704-024-40555-y), [LLMs for Text Classification: Case Study and Review (arXiv)](https://arxiv.org/html/2501.08457v1)

### 1.6 Topic modeling & text analytics, LLM-augmented

- **BERTopic + LLM representation** — embedding-based clustering for structure, LLMs (including local ones via llama.cpp/Ollama) to label and describe topics; LLM-assisted topic reduction for cleaner taxonomies.
- **Seeded/guided topic modeling** — steering topics toward analyst-defined concepts, plus LLM-generated corpus summaries per topic.

Sources: [Topic Modeling Techniques for 2026 (Towards Data Science)](https://towardsdatascience.com/topic-modeling-techniques-for-2026-seeded-modeling-llm-integration-and-data-summaries/), [BERTopic LLM integration docs](https://maartengr.github.io/BERTopic/getting_started/representation/llm.html), [LLM-Assisted Topic Reduction for BERTopic (arXiv)](https://arxiv.org/html/2509.19365v1)

### 1.7 LLM evaluation — the make-or-break skill

- **LLM-as-a-judge** — using a strong model to grade outputs against rubrics; best practice now includes judge calibration against human labels, pairwise comparison over absolute scores, bias controls (position, length, self-preference), and chain-of-thought judging.
- **Eval-driven development** — building golden datasets and regression eval suites *before* shipping prompts/models, with frameworks like DeepEval, Langfuse, Arize Phoenix, Braintrust, Evidently.
- **Reasoning models & test-time compute** — knowing when to spend inference-time compute (longer reasoning traces, self-consistency, best-of-N) instead of training-time compute.

Sources: [LLM-as-a-Judge in 2026 (DeepEval)](https://deepeval.com/blog/llm-as-a-judge), [LLM-as-a-judge complete guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge), [LLM-as-a-Judge (Langfuse)](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge), [LLM Evaluation: Frameworks, Metrics, Best Practices 2026 (FutureAGI)](https://futureagi.substack.com/p/llm-evaluation-frameworks-metrics)

### 1.8 Synthetic data for NLP

LLM-generated training data — with human validation loops for sensitive domains — for augmenting scarce labels, balancing classes, and pre-training small models. Key skills: contamination control, diversity enforcement, and validating that synthetic distributions match real ones.

Sources: [Synthetic Data for LLM Training: Decision Guide 2026](https://www.digitalapplied.com/blog/synthetic-data-generation-llm-training-decision-guide-2026), [Synthetic Data Augmentation with Human Validation (MDPI)](https://www.mdpi.com/2227-7102/16/6/885)

### 1.9 Computational qualitative analysis — bridging quantitative and qualitative *(added 2026-07-20 from NOTES.md)*

The methods that answer "how does quantitative data speak to qualitative data?" — treating text (interviews, focus groups, open-ended responses, social posts) as data that can be both *counted* and *interpreted*, with an explicit loop between the two:

- **Computational grounded theory** — the three-step loop that formalizes the bridge: unsupervised pattern detection (topic models/clustering) → human deep reading of the surfaced patterns ("guided deep reading") → computational confirmation of the refined hypotheses on the full corpus. Numbers propose, humans interpret, numbers verify.
- **LLM-assisted qualitative coding** — LLMs applied to the traditional codebook workflow: generating candidate codes, applying a human-built codebook at scale, or full grounded-theory-style schema induction (e.g., LOGOS). The 2025–26 literature's consistent finding: LLMs rival human coders on *explicit* themes but need human calibration for latent/interpretive ones — so the skill is the *hybrid design* (human codebook + LLM application + agreement audit), not full automation. This is also the direct answer to "can topic modeling and AI coexist?": topic models find structure cheaply and transparently; LLMs label, describe, and apply codebooks; each covers the other's weakness.
- **Reliability metrics across the bridge** — quantifying qualitative judgments: inter-rater agreement between humans and LLMs (Cohen's κ, Krippendorff's α), plus semantic-similarity variants for free-text codes; validating machine themes against human-coded subsets before trusting them at corpus scale (the same golden-dataset discipline as §1.7).
- **Topic models as qualitative instruments** — using BERTopic/LDA on focus-group and interview data to replace or audit subjective human coding; evaluations show it works but topics need human interpretation to become *themes* — the machine finds co-occurrence, the analyst supplies meaning.
- **Joint displays & mixed-methods integration** — formal frameworks for linking numeric patterns to thematic findings (numerically: quantitizing codes; thematically: qualitizing clusters; heuristically: side-by-side joint displays), so the quant and qual strands answer the *same* research question rather than coexisting in separate sections.

**Skill to build:** designing the handoff points — where machine output becomes human input and vice versa — and reporting both strands with their appropriate validity evidence (correlations for the numbers, coherence + human agreement for the themes).

Sources: [Updating "The Future of Coding": Qualitative Coding with Generative LLMs (Nelson et al., Sociological Methods & Research 2025)](https://journals.sagepub.com/doi/10.1177/00491241251339188), [Computational Grounded Theory overview](https://www.emergentmind.com/topics/computational-grounded-theory), [Scaling hermeneutics: qualitative coding with LLMs for reflexive content analysis (EPJ Data Science 2025)](https://link.springer.com/article/10.1140/epjds/s13688-025-00548-8), [LOGOS: LLM-driven end-to-end grounded theory development (arXiv)](https://arxiv.org/abs/2509.24294), [A mixed-methods framework integrating computational and qualitative text analysis (Quality & Quantity 2025)](https://link.springer.com/article/10.1007/s11135-025-02357-7), [Evaluating BERTopic for qualitative analysis of social media data (ACM 2025)](https://dl.acm.org/doi/10.1145/3786995.3786997), [Human vs machine-assisted topic analysis of large qualitative datasets (Frontiers in Public Health)](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2023.1268223/full), [Multi-LLM thematic analysis with dual reliability metrics (arXiv)](https://arxiv.org/abs/2512.20352), [LLMs for thematic analysis: blinded comparison with human analysts (PLOS Digital Health)](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001189), [Using ChatGPT to conduct grounded theory: tutorial (JMIR 2025)](https://www.jmir.org/2025/1/e70122)

---

## Part 2 — Modeling beyond text

### 2.1 Tabular foundation models

Pretrained transformers doing in-context learning on tables — no per-dataset training:

- **TabPFN v2 / TabPFN-2.5 / Real-TabPFN** — row/column self-attention; v2.5 advances SOTA and Real-TabPFN adds continued pre-training on real data.
- **TabICL v2** — scales to ~500k rows, fast, fully open-source (the pragmatic pick).
- On TabArena benchmarks the best tabular FMs now beat everything except heavy AutoML ensembles (AutoGluon "extreme"). Caveats: licensing restrictions on several commercial models, and SOTA churns monthly — validate on your own data.

Sources: [The state of Tabular Foundation Models 2026 (Mindful Modeler)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models), [TabPFN-2.5 (arXiv)](https://arxiv.org/abs/2511.08667), [TabFM (Google Research)](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

### 2.2 Time-series foundation models

Zero-shot forecasting with pretrained models (TimesFM, Chronos, Moirai, TabPFN-TS and successors) — often competitive with tuned classical models out of the box, and a major workflow change: evaluate a foundation model *first*, then decide if custom training is warranted.

Sources: [The 2026 Time Series Toolkit (MachineLearningMastery)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/), [Foundation Models for Time Series: A Survey (arXiv)](https://arxiv.org/abs/2504.04011), [TabPFN-TS (arXiv)](https://arxiv.org/abs/2501.02945)

### 2.3 Modern causal inference

The skill that separates "prediction" from "decision-making":

- **Double/Debiased Machine Learning (DML)** — use any ML model for nuisance functions while keeping valid confidence intervals on treatment effects; extensions cover instrumental variables, sample selection, and high-dimensional confounding (e.g., EHR data).
- **Heterogeneous treatment effects (CATE)** — causal forests, meta-learners, DML-based CATE for personalization and uplift.
- **Tools:** EconML, DoWhy, CausalML.

Sources: [Intro to causal inference using DML (Microsoft)](https://medium.com/data-science-at-microsoft/introduction-to-causal-inference-using-double-machine-learning-5daa642321f3), [DML: Debiasing to Heterogeneous Treatment Effects (Modern Causal Inference series)](https://medium.com/causal-inference-methods-models-and-applications/double-machine-learning-deconfounding-high-dimensional-causal-inference-97a76da70986), [DML for causal inference in sample selection models (arXiv)](https://arxiv.org/abs/2511.12640), [DML in high-dimensional EHR (medRxiv)](https://www.medrxiv.org/content/10.1101/2025.07.21.25331944v1)

---

## Part 3 — Agentic AI & AI-assisted analytics

- **Agentic analytics** — agents that plan and execute multi-step data work: write SQL, run analyses, verify results, generate reports. 2026 trends: orchestration of multi-agent systems, agent observability, and human-in-the-loop checkpoints for high-stakes actions.
- **Skills shift** — the analyst's edge moves to problem framing, verification of agent output, and building the semantic layers/data contracts agents rely on.

Sources: [7 Agentic AI Trends to Watch in 2026 (MachineLearningMastery)](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/), [AI agent trends 2026 (Google Cloud)](https://cloud.google.com/resources/content/ai-agent-trends-2026), [Data and agentic AI: seven trends for 2026 (Izertis/Gartner)](https://www.izertis.com/en/w/blog/data-agentic-ai-2026-trends-gartner), [5 predictions about agentic AI and analytics in 2026 (Redpanda)](https://www.redpanda.com/blog/5-predictions-about-agentic-ai-and-analytics-in-2026)

---

## Part 4 — Data engineering for data scientists

- **Open lakehouse** — Apache Iceberg (plus Delta/Hudi) as the open table-format standard; catalogs and interoperability are the 2026 battleground.
- **Single-node analytics renaissance** — DuckDB, Polars, DataFusion: most workloads don't need a cluster; querying Iceberg tables directly from DuckDB/Polars is now routine.
- **Skills:** SQL remains king; add columnar formats (Parquet/Arrow), dbt-style transformation discipline, and data contracts.

Sources: [2025–2026 Guide to the Data Lakehouse](https://datalakehousehub.com/blog/2025-09-2026-guide-to-data-lakehouses/), [Single-Node Data Engineering: DuckDB, DataFusion, Polars, LakeSail](https://iceberglakehouse.com/posts/2026-05-23-single-node-data-engineering-duckdb-datafusion-polars-lakesail/), [Using DuckDB and Polars to Query Iceberg Tables](https://datalakehousehub.com/blog/2026-05-duckdb-polars-iceberg/)

---

## Part 5 — MLOps → LLMOps

- **Classic MLOps** (versioning, CI/CD for models, drift monitoring) now extended with **LLMOps**: prompt/version management, tracing of multi-step agent runs, token-cost monitoring, guardrails, and continuous evals in production.
- **Observability** — OpenTelemetry-based LLM tracing; platforms: MLflow, Langfuse, Braintrust, Arize, OpenObserve.

Sources: [MLOps/LLMOps Roadmap for 2026 (Medium)](https://medium.com/@sanjeebmeister/the-complete-mlops-llmops-roadmap-for-2026-building-production-grade-ai-systems-bdcca5ed2771), [What is LLMOps? (MLflow)](https://mlflow.org/llmops), [LLM Monitoring Best Practices 2026 (OpenObserve)](https://openobserve.ai/blog/llm-monitoring-best-practices/)

---

## Part 6 — Durable core skills the 2026 surveys keep emphasizing

Cloud-native ML (at least one of AWS/GCP/Azure), statistics & experimental design (still the differentiator vs "AI operators"), Python + SQL fluency, data storytelling/communication, AI governance & responsible-AI literacy (EU AI Act era), and domain knowledge to frame problems agents can't frame themselves.

Sources: [Essential Skills for Data Science Professionals in 2026 (DASCA)](https://www.dasca.org/world-of-data-science/article/essential-skills-for-data-science-professionals-in-2026-and-beyond), [Top In-Demand Data Science Skills of 2026 (Cobloom)](https://www.cobloom.com/careers-blog/in-demand-data-science-skills), [Top 11 Data Science Skills to Master in 2026 (roadmap.sh)](https://roadmap.sh/ai-data-scientist/skills)

---

## Suggested learning order (NLP-first)

1. LLM evaluation (1.7) — everything else depends on being able to measure quality
2. Advanced RAG + embeddings (1.2, 1.3)
3. Post-training: LoRA → DPO → GRPO (1.1)
4. SLMs & extraction (1.4, 1.5) — highest production ROI per dollar
5. Agentic analytics (Part 3)
6. Foundation models for tabular/time-series (Part 2) and causal inference as cross-cutting depth
