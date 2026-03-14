# ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration
- Paper Source: https://arxiv.org/abs/2410.02551
- Paper Authors: Zixiang Wang, Yinghao Zhu, Huiya Zhao, Xiaochen Zheng, Dehao Sui, Tianlong Wang, Wen Tang, Yasha Wang, Ewen Harrison, Chengwei Pan, Junyi Gao, Liantao Ma
- Reading Gains: Understanding the use of RAG in medical agents

# Extensive Reading

## Paper Background

Electronic Health Record (EHR) modeling faces several challenges: **poor interpretability, insufficient structured data processing capability, and lack of real-time medical knowledge support**:

1. Traditional EHR deep learning models are **purely data-driven end-to-end** black boxes. EHR features are treated only as numerical variables, failing to capture clinical semantic connotations, resulting in **poor interpretability**. Existing interpretability methods (such as SHAP, attention mechanisms) can only perform basic feature importance analysis and rely on **manually constructed knowledge representations**, whose knowledge update speed lags far behind the latest medical research/guidelines.
2. LLMs have **obvious shortcomings in structured EHR processing**. Their reasoning ability in few-shot settings is far inferior to traditional EHR models, and they cannot efficiently analyze numerical temporal EHR data. LLMs themselves have knowledge timeliness issues, and existing LLM multi-agent medical research mainly focuses on **text-based question-answering tasks**, unable to handle numerical clinical prediction tasks like mortality/readmission rates, and not integrating authoritative medical guidelines.
3. Existing medical multi-agent frameworks (such as MedAgents, ReConcile) mostly use the same LLM instance, resulting in a single reasoning perspective, and only target text diagnosis, unable to combine quantitative analysis of structured EHR, making it difficult to meet clinical needs for **diversified diagnostic reasoning and quantitative prediction**.

## Innovation Points

- **Idea Innovation**: Introducing the clinical **MDT (Multi-Disciplinary Team) collaboration** concept into EHR modeling, combining RAG modules to integrate authoritative medical knowledge from the *Merck Manual of Diagnosis and Therapy (MSD)*, achieving **EHR data-driven + external knowledge-enhanced** prediction while supporting the model's "self-inspection";
- **Technical Innovation**: Designing an LLM-driven multi-agent collaboration mechanism that integrates clinical decision evidence from multiple DoctorAgents and MetaAgent to generate **human-understandable structured reports**, improving model transparency and providing clear basis for clinical doctors' diagnostic reasoning;
- **Experimental Innovation**: Conducting comprehensive comparative experiments, ablation experiments, and sensitivity analysis on **3 real EHR datasets**, achieving significant performance improvements in mortality/readmission rate prediction tasks (AUPRC relative improvement 0.86%~4.49%); verifying the framework's **interpretability, cost-effectiveness, and clinical utility** through case studies, cost analysis, and clinical expert human evaluation.

## Core Methods

![](/Plugins/colacare.png)

**Four core modules**: **Structured EHR Information Extraction Module**, **RAG Retrieval Enhancement Module**, **Multi-Agent Collaboration Consultation Module**, **Multimodal Fusion Network**
**Core**: Simulating clinical MDT diagnosis and treatment process through **DoctorAgent (specialist agent) + MetaAgent (coordinating agent)** collaboration, while integrating the structured data processing capability of EHR domain expert models and the authoritative medical knowledge support of RAG.

The prediction target of the framework is a **binary classification task**: predicting in-hospital mortality/30-day readmission rate (0 = no adverse outcome, 1 = adverse outcome), mathematically expressed as: $y^=Framework(xEHR,MedicalKnowledge)$

### 1. Structured EHR Information Extraction Module

This module is the **quantitative foundation** of ColaCare, responsible for processing structured temporal EHR data and providing quantitative prediction results and feature basis for multi-agents:

1. Input patient EHR data (multivariate time-series data, including static features such as age/gender, dynamic features such as laboratory indicators/vital signs), encode through **EHR domain expert models** (such as AdaCare, ConCare, RETAIN) to obtain EHR hidden representation hEHR​;
2. Use MLP layer to map the hidden representation to **prediction logit (z)** (quantified risk prediction value);
3. Use **SHAP method** to calculate **feature importance weights (α)**, identifying the most critical clinical indicators for prognosis;
4. The logit and feature importance weights will serve as the core quantitative basis for subsequent multi-agent analysis.

### 2. RAG Retrieval Enhancement Module

This module solves the **knowledge timeliness and authority** issues of LLMs, providing **evidence support** for multi-agent clinical decisions:

1. **Corpus**: Adopting the *Merck Manual of Diagnosis and Therapy (MSD)* — an internationally authoritative clinical diagnosis and treatment guide — as the medical knowledge source;
2. **Retrieval model**: Using **MedCPT** (biomedical domain-specific contrastive pre-training model) to complete information retrieval;
3. **Retrieval logic**: Embed patient records (including basic information, logit, feature importance) as vectors, calculate similarity with document vectors in the MSD corpus through **cosine similarity**, retrieve **Top-K (K=3)** relevant medical documents as authoritative evidence for agent analysis.

### 3. Multi-Agent Collaboration Consultation Module (Core)

This module is the **soul** of ColaCare, simulating the multi-round collaborative diagnosis and treatment process of clinical MDT, including **two types of agents** and **three collaboration stages** (Figure 1). The reasoning engine uses **DeepSeek-V2.5** (LLM). Different agents have clear division of labor, iterative interaction, and ultimately reach consensus.

#### (1) Agent Design

- **DoctorAgent**: Each agent corresponds to an **EHR domain expert model**, representing different clinical specialist perspectives, responsible for generating clinical evaluations based on patient records and retrieved medical documents, and expressing agreement/disagreement on MetaAgent's report (disagreement requires evidence);
- **MetaAgent**: The **core coordinator** of collaboration, responsible for integrating all DoctorAgents' evaluations, generating/revising comprehensive reports, judging whether to continue multi-round discussions, and finally outputting a **consensus report**.

#### (2) Three Collaboration Stages

1. **Generate Initial Evaluation**: Each DoctorAgent combines patient records (basic information + logit + feature importance) and Top-K medical documents retrieved by RAG to generate their own **initial clinical evaluation report**, providing risk judgment and basis;
2. **Generate Preliminary Comprehensive Report**: MetaAgent integrates all DoctorAgents' initial evaluations, combines patient basic information, and generates a **preliminary comprehensive report** — first determining whether the patient's death risk is "high/low", then extracting core evidence and viewpoints from each DoctorAgent;
3. **Iterative Collaborative Consultation (Multi-round)**: This is the core interaction process of multi-agents, with a maximum of 3 rounds:

   - DoctorAgent expresses **agreement/disagreement** on the current MetaAgent's report, and disagreement requires providing medical documents retrieved by RAG as evidence;
   - MetaAgent aggregates all DoctorAgents' feedback, judges whether to **continue discussion**: if consensus is reached, terminate; if there are differences, analyze the validity of evidence for opposing opinions, **revise the comprehensive report**;
   - Repeat the above process until all DoctorAgents reach consensus, and MetaAgent outputs the **final consensus report**.

### 4. Multimodal Fusion Network

This module fuses **quantitative features of structured EHR** with **text semantic features of consensus reports** to obtain final prediction results, achieving **quantitative + qualitative** joint modeling:

1. Extract hidden representations hEHR1​, hEHR2​, ..., hEHRN​ from multiple EHR expert models;
2. Use **GatorTron** (clinical domain-specific LLM) to encode MetaAgent's final consensus report into text hidden representation hReport​;
3. **Concatenate** the two types of hidden representations, map to final binary classification prediction probability y^​ through MLP layer;
4. Use **binary cross-entropy (BCE)** as the loss function to optimize the entire framework: $L(y^,y)=−N1∑i=1N(yilog(y^i)+(1−yi)log(1−y^i))$

## Experimental Setup

To comprehensively verify ColaCare's performance, interpretability, and practicality, the paper designed a rigorous experimental plan covering **datasets, evaluation metrics, baseline models, hardware and software configurations**:

### 1. Experimental Datasets

Using **3 real de-identified EHR datasets** covering different disease scenarios, including mortality/readmission rate prediction tasks, with test set size of approximately 1000 samples (balancing efficiency and representativeness):

| Dataset | Disease Scenario | Sample Size | Prediction Task |
|---------|------------------|-------------|-----------------|
| MIMIC-IV | Intensive Care Unit (ICU) patients | 19331 | In-hospital mortality, 30-day readmission rate |
| CDSL | COVID-19 confirmed/suspected patients | 4255 | In-hospital mortality |
| ESRD | End-Stage Renal Disease (ESRD) patients | 656 | In-hospital mortality |
| At the same time, the *Merck Manual of Diagnosis and Therapy (MSD)* is used as the sole medical knowledge source for the RAG module to ensure the authority of evidence. | | | |

### 2. Evaluation Metrics

For the **class imbalance characteristics** of clinical binary classification tasks, 3 mainstream metrics are selected (all "higher is better"):

- **AUROC**: The most commonly used indicator in clinical practice, evaluating the model's discrimination ability at different thresholds;
- **AUPRC**: More sensitive to imbalanced datasets, more in line with the clinical characteristics of EHR data;
- **min(+P, Se)**: Balancing **precision** and **sensitivity**, avoiding the model from over-biasing towards one type of sample.

### 3. Baseline Models

Divided into **two categories**, comprehensively comparing ColaCare's performance with existing EHR modeling and LLM modeling methods:

1. **EHR-specific baselines**: Classic EHR deep learning models (AdaCare, ConCare, RETAIN) + 5 integration methods (mean, weighted mean, temperature scaling, deep integration, MCdropout);
2. **LLM-based baselines**: Single LLM methods (zero-shot, few-shot, self-consistency) + existing LLM multi-agent methods (MAD, MedAgents, ReConcile).

### 4. Hardware, Software, and Hyperparameters

- **Hardware**: Nvidia RTX 3090 GPU, 128GB RAM, CUDA 12.5;
- **Software**: Python 3.9, PyTorch 2.3.1, PyTorch Lightning 2.3.3;
- **Core hyperparameters**: Batch size 128, training 50 epochs (early stopping strategy, terminate if no improvement for 10 epochs), maximum multi-agent collaboration rounds 3, RAG retrieval K=3, fusion network hidden dimension 128.

## Experimental Results

The paper conducted experimental analysis around **7 research questions (RQ1-RQ7)**, comprehensively verifying ColaCare's advantages from seven dimensions: **overall performance, module necessity, agent sensitivity, LLM compatibility, interpretability, cost-effectiveness, clinical consistency**.

### RQ1: Overall Performance — ColaCare Significantly Outperforms All Baselines

ColaCare achieved optimal performance in **all datasets and all prediction tasks**, especially the **AUPRC indicator** (core indicator for clinical imbalanced data) with **relative improvements of 0.86%, 2.50%, 2.00%, 4.49%** in the four tasks.

- Superior to **pure EHR models/integration models**: Proving that LLM multi-agent knowledge fusion and collaboration mechanisms can effectively improve the performance of structured EHR modeling;
- Far exceeding **pure LLM/existing LLM multi-agent models**: Proving that the structured data processing capability of EHR domain expert models is the foundation, and after combining MDT collaboration and RAG knowledge, LLMs' clinical prediction ability is greatly improved.

### RQ2: Ablation Experiments — All Modules are Necessary for Performance

By removing individual modules of ColaCare, the core role of each module is verified, **removing any module will lead to significant performance degradation**:

| Ablation Version | Core Conclusion on Performance Change |
|------------------|----------------------------------------|
| Without RAG | Losing authoritative medical evidence support, agent reasoning lacks basis, performance decreases |
| Without Fusion Network | Only using LLM reports for prediction, losing quantitative features of structured EHR, performance drops significantly |
| Without Expert Model | Equivalent to MedAgents, unable to process structured EHR, performance plummets (most severe) |
| Without MDT | Directly integrating retrieved documents into expert models, losing the advantage of multi-perspective collaboration, performance decreases |
| **Core Conclusion**: EHR domain expert models are the foundation, MDT collaboration mechanism is the core, RAG is an important support, and multimodal fusion is the guarantee of final performance. | |

### RQ3: Agent Number Sensitivity — 3 DoctorAgents is the Optimal Configuration

- 1/2 DoctorAgents: Limited or even decreased performance, because the single perspective makes agents easily reach consensus quickly, but the conclusion lacks robustness;
- 3 DoctorAgents: Optimal performance, because multi-perspective integration of more medical evidence results in more comprehensive and reliable reports after collaboration.

### RQ4: Different LLM Sensitivity — The Framework Has Strong LLM Compatibility

Testing 7 mainstream LLMs (DeepSeek-V2.5, GPT-4o-Mini, GPT-4o, Qwen-Turbo, Doubao-Pro, Llama-3.1-400B, Claude-3.5-Sonnet), **all LLMs can adapt to ColaCare**, with **DeepSeek-V2.5 and GPT-4o-Mini** performing slightly better, proving that ColaCare's multi-agent collaboration mechanism does not depend on specific LLMs and has good flexibility.

### RQ5: Case Study — Reports Have Strong Interpretability and Clinical Rationality

Taking a **46-year-old female death patient from the ESRD dataset** as an example, demonstrating ColaCare's collaboration process:

1. Initial stage: 3 DoctorAgents had **significant differences** in risk judgment (medium risk/low risk/high risk), each based on different core indicators (albumin/diastolic blood pressure/blood potassium);
2. Preliminary synthesis: MetaAgent integrated and determined **high risk**, focusing on three core abnormal indicators: **low blood potassium, abnormal carbon dioxide combining power, low albumin**;
3. Multi-round collaboration: The DoctorAgent originally judging "low risk" found that it had missed the key abnormality of blood potassium, revised its opinion, and finally all agents **reached consensus**;
4. Final report: Accurately identified the patient's core fatal risk factors, provided authoritative evidence based on MSD guidelines, with clear report structure and clear indicators, in line with clinical doctors' diagnostic thinking.

### RQ6: Cost Analysis — High Cost-Effectiveness, Suitable for Clinical Implementation

When using DeepSeek-V2.5 as the reasoning engine, ColaCare's cost per patient is approximately **$0.013**, and the token consumption and API requests for each dataset are within controllable range, far lower than the cost of clinical manual diagnosis, proving that the framework has good **clinical implementation feasibility**.

### RQ7: Human Evaluation — Highly Consistent with Clinical Expert Judgments

Inviting **12 nephrology clinicians (5-15 years of clinical experience)** to conduct questionnaire evaluations on peritoneal dialysis patient data (1-5 points, higher scores indicate stronger consistency):

- ColaCare's average consistency score was **4.4 points**;
- The average consistency score of the traditional EHR model ConCare was **3.2 points**;
- Blind evaluation results showed that the **interpretability and diagnostic reasoning logic** of ColaCare-generated reports are highly consistent with clinical experts' practice patterns.

## Conclusion and Expectation

### 1. Core Conclusions

ColaCare successfully solved the "black box" problem of traditional EHR modeling and the shortcomings of LLMs in structured EHR processing by deeply integrating **LLM-driven multi-agent collaboration** with clinical **MDT diagnosis and treatment concepts**:

- Integrating the **structured data processing capability** of EHR domain expert models, **authoritative medical knowledge** of RAG, and **diversified clinical reasoning** of multi-agents, achieving **dual improvement** in performance and interpretability;
- Performing optimally in mortality/readmission rate prediction on 3 real EHR datasets, with low cost and high consistency with clinical expert judgments, promising to **revolutionize clinical decision support systems** and promote the development of personalized precision medicine.

### 2. Limitations and Future Work

The paper also pointed out ColaCare's current limitations and clarified future research directions:

1. **Improve generalization**: Currently only for mortality/readmission rate prediction, need to verify on **more clinical tasks** (such as disease diagnosis, complication prediction) and **more datasets**; simultaneously integrate more closed-source/open-source LLMs to further verify the framework's compatibility;
2. **Expand human evaluation scale**: Existing evaluation only invited 12 nephrologists, need to expand **number of clinical experts, department scope, patient cases** to verify the framework's applicability in different clinical scenarios;
3. **Add continuous learning mechanism**: Existing EHR domain expert models have fixed parameters, need to design continuous learning mechanisms based on **LLM agent feedback and real clinical data** to enable the framework to adapt to dynamic medical environments (such as latest diagnosis and treatment guidelines, new disease characteristics);
4. **Add confidence estimation**: Add **confidence scores** to the framework's prediction results to further improve clinical doctors' trust and willingness to use.

# Key Doubts

## Q1: In this model, many models are integrated, which models are actually trained?

A1: 
- Training objects:
  - **EHR model**: EHR domain-specific deep learning model
  - **Multimodal Fusion**:
- Pre-trained models:
  - DoctorAgent and MetaAgent: Using LLM reasoning engines (DeepSeek-V2.5/GPT-4o-Mini, etc.) to complete clinical reasoning, report generation, and multi-round collaborative debate, no training required;
  - RAG Module: MedCPT biomedical-specific pre-trained model, completing similarity retrieval between patient records and *Merck Manual of Diagnosis and Therapy*, providing authoritative medical evidence;
  - Report encoding model: GatorTron clinical domain pre-trained LLM, encoding multi-agent consensus reports into text hidden representations, providing semantic features for multimodal fusion.

## Q2: In this model, Multimodal Fusion receives data from both Report and EHR, which means there is a part from EHR to Predict that cannot be represented by simple trainable parameters. How does backpropagation occur during training?

A2: 
Obviously, if EHR Models and Multimodal Fusion are trained simultaneously, there will be an untrainable, unrepresentable "black box" between EHR logits etc. and Report Embedding, which will make end-to-end training extremely inconvenient.

**Solution**

- **No cross-module joint backpropagation**: In this study, EHR expert models are first independently trained. When training the multimodal fusion network, **fix all parameters of EHR models and clinical text encoding models (GatorTron)**, and backpropagation gradients only update the MLP layer weights of the fusion network;
- **Serial step-by-step training**: Backpropagation is independently performed in two modules in stages, and feature transfer between stages is completed through "EHR hidden representation $h_{EHR}$", with no gradient interaction.

### Stage 1: Training EHR Domain Expert Models

1. **Three outputs**: Two native, one **SHAP**
   1. Output EHR Embedding (hidden representation $h_{EHR}$​)
      $h_{EHR}$ has two roles here: one is as the core input for multimodal fusion, and the other is to generate preliminary prediction values through the expert model's **own MLP prediction head**.
   2. Generate preliminary prediction values (logit $z$)
      **Not used as the final prediction result**, but as **part of interpretable evidence**
   3. Feature importance $α$
      Extract feature importance weights α from the expert model and original EHR data through **SHAP strategy**, which will combine **clinical values of features and population statistics** to form **interpretable evidence**. $$α=SHAP(Model,x_{EHR})$$
         *Note: What is [SHAP](01-Foundation/AI_Core_Course/01-MLfoudamentals/XAI/ModelInterpretability/SHAP.md) strategy

2. **How to train**
    1. **Forward propagation**: Input patient EHR time-series data $X \in \mathbb{R}^{T×F}$ → encode through EHR model to obtain hidden representation $h_{EHR}$ → output prediction logit $z$ through [EHR model internal MLP] → calculate BCE loss $\mathcal{L}(\hat{y}, y)$;
    2. **Backward propagation**: Starting from BCE loss, gradients are sequentially backpropagated to **EHR model internal MLP** → **core encoding layer of EHR model** (such as AdaCare's scale-adaptive feature extraction layer, RETAIN's reverse time attention layer) → **input embedding layer of EHR data**;
    3. **Gradient update**: Update all trainable parameters of the EHR model through AdamW optimizer until the model converges (early stopping strategy triggered), then **fix all parameters** and no longer participate in gradient updates in subsequent training. 

### Stage 2: Training Multimodal Fusion Network

# Reading Gains

- Through reading the source code and extended learning, learned the components and related principles of the RAG module;
- Learned file loading, and the use of libraries such as FAISS, re, tqdm, traceback;
- Understood some interpretable artificial intelligence algorithms (ICE, LIME, PDP, SHAP)