

```markdown
# Machine Learning Data Agent

A production-ready, modular **Machine Learning Data Agent** designed for intelligent ingestion, preprocessing, and management of structured and unstructured datasets. Built for integration with data-centric AI platforms such as **InsightOS**.

---

## Features

- **Universal Ingestion**  
  Supports CSV, JSON, Parquet, and API-based data sources.

- **Data Preprocessing Pipelines**  
  - Null handling, standardization, normalization  
  - NLP-specific: tokenization, lowercasing, stopword removal  
  - Vision: image resizing, format conversion

- **Streaming and Batch Support**  
  Real-time and batch processing modes with auto-scaling buffer management.

- **Pluggable Augmentation Modules**  
  Extendable interface for NLP and vision data augmentation.

- **API Integration Ready**  
  Exposes agent via REST endpoint for usage within `InsightOS` and other microservices.

- **Robust Architecture**  
  - Logging (structured + traceable)  
  - Exception-safe processing  
  - Metrics for throughput and failure rate

---

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`

### Example Usage

```python
from ml_data_agent.agent import DataAgent

agent = DataAgent(source_type="csv", path="data/train.csv")
df = agent.load()
cleaned = agent.preprocess(df)
````

### API Mode (Optional)

```bash
fastapi dev main.py
```

---

## Configuration

Use `config.json` to define default preprocessing rules, batch size, and timeout settings.

---

## Testing

```bash
pytest --cov=ml_data_agent tests/
```


