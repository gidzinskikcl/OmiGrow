# README — Processed Datasets for Growth Prediction Project

## Overview
This directory contains processed transcriptomics, proteomics, and growth datasets used to investigate:

- **Does proteomics explain growth better than transcriptomics?**
- **Does combining proteomics + transcriptomics improve prediction?**
- **Does adding proteomics improve the MMANN from Culley et al. (2020)?**
- Question nr 4 TBC...

All datasets are aligned to a shared set of **~965 yeast single-gene knockout (KO) strains** of *Saccharomyces cerevisiae*.

---

## 1. Transcriptomics (SC medium)

**File:** `transcriptomics.csv`  
**Index:** KO (systematic gene name)  
**Columns:** genome-wide gene expression values  

### Source & Preprocessing
- Expression values come from **Kemmeren et al. (Cell 2014)** and **O’Duibhir et al. (MSB 2014)**.
- Measurements were taken in **synthetic complete (SC) medium**, **liquid culture**, mid-log phase.
- Values represent log fold-changes of transcript levels relative to wild-type.
- Only gene expression columns are included (growth column removed).

### What This Dataset Represents
A matrix capturing **transcriptional responses to gene deletion** under SC conditions.  

---

## 2. Transcriptomics Growth (SC medium)

**File:** `transcriptomics_growth.csv`  
**Index:** KO  
**Column:** `growth`

### Source & Preprocessing
- Growth measurements come from **Kemmeren / O’Duibhir**.
- Growth is expressed as:  
  `log2(doubling_time_KO / doubling_time_WT)`
- Higher values → slower growth  
- Lower values → faster growth  
- Medium: **SC liquid**

### What This Dataset Represents
A high-quality **ground-truth growth phenotype** matched to the transcriptomic data.  
Used as the target variable for transcriptomics-based growth training.

---

## 3. Proteomics (SM medium)

**File:** `proteomics.csv`  
**Index:** KO  
**Columns:** protein abundance measurements

### Source & Preprocessing
- Data come from **Messner et al. (2023)**.
- Proteomics was measured in **synthetic minimal (SM) medium** on the same KO collection.
- Data were filtered to match the same set of ~965 KOs used for transcriptomics.

### What This Dataset Represents
A protein abundance matrix describing **proteomic responses to gene deletion**, but in SM medium.  
Useful for:

- Proteome-based growth prediction  
- Comparing proteomics vs transcriptomics  
- Extending Culley’s multimodal neural network with a proteomics view  

---

## 4. Proteomics Growth (SC medium)

**File:** `proteomics_growth.csv`  
**Index:** KO  
**Column:** `growth`

### Source & Preprocessing
- Growth values obtained from **Messner et al.**, measured in **SC medium** using colony growth on agar plates.
- Represents a **second independent SC growth phenotype**, aligned to the same KO set.

### What This Dataset Represents
An alternative, independently measured SC fitness phenotype.  
Useful for:

- Cross-assay consistency checks (liquid SC vs agar SC)
- Constructing a consensus SC growth label
- Additional validation targets for ML models

---

## 5. Why These Data Can Be Used Together

Although expression and proteomics were collected in **different media (SC vs SM)**:

- All datasets were restricted to the **same KO strains**.
- Growth was measured in **SC** in two independent studies.
- Correlation analyses show **consistent SC growth** across labs.
- This allows fair comparison of expression vs proteomics and supports multimodal modeling.

---

## 6. Recommended Usage

### Transcriptomics-only Models
- Input: `transcriptomics.csv`
- Target: `transcriptomics_growth.csv`

### Proteomics-only Models
- Input: `proteomics.csv`
- Target: `proteomics_growth.csv`

### Multimodal Models (Expr + Prot)
- Inputs aligned by KO:  
  `transcriptomics.csv` + `proteomics.csv`
- Target: `transcriptomics_growth.csv`  
  or a consensus SC phenotype combining both SC growth datasets

### MMANN Extension
Use transcriptomics + proteomics (and optionally fluxes) with SC growth as the predictive target.

---

## 7. Reproducibility Notes

- All files share the **same KO index**.
- Missing strains were removed to keep the index aligned across modalities.
- No normalization or scaling was applied at this stage.