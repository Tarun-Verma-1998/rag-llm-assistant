# RAG + LLM-as-a-Judge: Real-Time Question Answering Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline enhanced with **LLM-as-a-Judge** for smarter chunk reranking and final answer generation using T5 and Flan-T5 models.

---

## Full Pipeline Flow

```text
User Query
   │
   ▼
T5 Embedding → Milvus Vector Search
   │
   ▼
Reranking (choose one):
   ├── Option 1: T5 Relevant: True → query_and_rerank.py
   └── Option 2: LLM-as-a-Judge (1–10 score) → llm_judge_rerank.py
   │
   ▼
Top Chunk Saved
   │
   ▼
Answer Generation → generate_answer.py / generate_final_answer.py
```

---

##  Project Structure

```text
rag-llm-assistant/
├── scripts/
│   ├── embed_store_t5.py                # Embeds and stores chunks in Milvus
│   ├── query_and_rerank.py             # Reranks using T5 (Relevant: true)
│   └── generate_answer.py              # Generates answer from top_chunk.txt
│   └── generate_final_answer.py        # Generates answer from llm_top_chunk.txt
├── llm_judge_rerank.py                 # LLM-as-a-Judge scoring (Flan-T5)
├── app.py                              # Streamlit UI (optional)
├── top_chunk.txt                       # Top chunk (T5 reranker)
├── llm_top_chunk.txt                   # Top chunk (LLM judge)
├── requirements.txt
└── .gitignore
```

---

##  Key Components Explained

###  `embed_store_t5.py`
- Uses T5 to generate embeddings from document chunks.
- Stores these embeddings in Milvus for semantic retrieval.

###  `query_and_rerank.py` (Option 1: T5 Reranker)
- Uses `"Query: ... Document: ... Relevant:"` as input to T5.
- Predicts likelihood of `"true"` and scores accordingly.
- Saves top result to `top_chunk.txt`.

###  `llm_judge_rerank.py` (Option 2: LLM-as-a-Judge)
- Uses Flan-T5 to score document relevance **from 1 to 10**.
- Prompt example:
  ```
  Please score the relevance of the following document to the query on a scale of 1 to 10.
  Query: ...
  Document: ...
  ```
- Saves top result to `llm_top_chunk.txt`.

###  `generate_answer.py`
- Uses T5 to generate a final answer based on `top_chunk.txt`.

###  `generate_final_answer.py`
- Uses Flan-T5 to generate a final answer based on `llm_top_chunk.txt`.

###  `app.py` *(optional)*
- Simple Streamlit UI to input queries and see results in real-time.
- You can plug the backend functions for complete end-to-end serving.

---

##  Reranking Options

### Option 1: T5 Relevant: True Logic
- Based on likelihood of `"true"` in the prompt.
- Script: `query_and_rerank.py`
- Output file: `top_chunk.txt`
- Use with: `generate_answer.py`

### Option 2: LLM-as-a-Judge (Flan-T5)
- Scores chunk relevance on a **1–10 scale**.
- Script: `llm_judge_rerank.py`
- Output file: `llm_top_chunk.txt`
- Use with: `generate_final_answer.py`

 You can switch between reranking modes based on use case or performance preference.

---

##  How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Embed and Store Chunks in Milvus
```bash
python scripts/embed_store_t5.py
```

### 3. Ask a Question + Rerank
Choose one reranking approach:

```bash
python scripts/query_and_rerank.py       # T5 logic
python llm_judge_rerank.py               # Flan-T5 (recommended)
```

### 4. Generate Final Answer
Use the corresponding script based on the reranker used:

```bash
python scripts/generate_answer.py        # If using T5 reranker
python scripts/generate_final_answer.py  # If using LLM-as-a-Judge
```

---

##  Optional: Streamlit App

To run the Streamlit app (if connected with backend logic):
```bash
streamlit run app.py
```

---

##  Requirements

Make sure your `requirements.txt` includes:
- `transformers`
- `torch`
- `pymilvus`
- `streamlit`
- `sentencepiece`
- `huggingface_hub`

---

## 🛡 Notes

- Ensure the **embedding dimension** used during storage matches the model used for inference (T5: 768, Flan-T5: 3072).
- Use only one reranking strategy per run.
- Milvus must be running (via Docker or service) before executing scripts.

---

## Credits

Developed by [Tarun Verma](https://github.com/Tarun-Verma-1998) as a complete real-world demonstration of building a **RAG + LLM QA system** from scratch using open tools and local infrastructure.

---

## Give it a Star!

If you find this project helpful, please consider starring the repo 

---
