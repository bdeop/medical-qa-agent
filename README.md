# Medical QA Agent Using Synthea + LangChain

## Overview
This project builds a Retrieval-Augmented Generation (RAG) based AI agent using GPT-4o and LangChain that can answer medical questions from synthetic patient data (Synthea).

## Project Structure

```
medical_qa_agent/
├── agent/                       # Place embedding tools
├── data/                        # Place patients.csv and conditions.csv here
├── diagrams/                    # Workflow and Sequence diagrams in plantuml format
├── myenv/                       # Virtual environment
├── scripts/                     # Preprocessing and requirements
├── app.py                       # Streamlit interface
├── README.md                    # This file
```

---

## Setup Instructions

### Step 1: Clone the repository
```bash
git clone <repo-url>
cd medical_qa_agent
```

### Step 2: Set Up Virtual Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### Step 3: Download and Place Synthea CSVs
Put `patients.csv` and `conditions.csv` into the `data/` folder.
You can download them from https://synthea.mitre.org/downloads 
Download this dataset: https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip

### Step 4: OpenAI API Key
Make sure to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file with:
```env
OPENAI_API_KEY=sk-...
```

### Step 5: Ingest and Index Data 
```bash
source myenv/bin/activate
python scripts/ingest_data.py
deactivate
```

### Step 6: Build Vector Store
```bash
source myenv/bin/activate
python scripts/store_embeddings.py
deactivate
```



### Step 7: Run Streamlit App 
```bash
source myenv/bin/activate
./myenv/bin/python -m streamlit run app.py
```


## Example Questions
- The patient ID of Donnell534 Dicki44 is 6ce0bda7-716f-c904-cdc8-39076db16016
- What are all the known medical conditions of patient Donnell534 Dicki44 ? Also provide the total count of conditions.
- Based on the patient’s age, gender, and condition history, is this patient Donnell534 Dicki44 at high risk of cardiovascular disease? Explain step-by-step.
- Why might the patient Donnell534 Dicki44 be prescribed Metformin? Think aloud based on their condition history.
- Summarize the patient Donnell534 Dicki44's medical history in chronological order. Think step-by-step.
- Summarize the current health conditions and diagnostics of patient Ezequiel972 Hyatt152
- Suggest a preventive healthcare plan for patient Ezequiel972 Hyatt152
- explain the latest observations for the patient Josh874 King743
- What are all the known medical conditions of patient Donnell534 Dicki44 ? on basis of what diagnostics results can we conclude that this patient has diabetes ?
- Suggest a preventive health care plan for patient Donnell534 Dicki44 
- Summarize the health and Create a preventive health plan for patient Donnell534 Dicki44 looking at the health conditions and the observations of this patient.
- Why is this patient Donnell534 Dicki44 at risk for cardiovascular complications? 
