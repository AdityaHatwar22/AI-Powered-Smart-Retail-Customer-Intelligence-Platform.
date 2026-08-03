# AI-Powered Smart Retail & Customer Intelligence Platform

**Author:** Aditya Hatwar 23BAI10902, B.Tech Computer Science, VIT Bhopal University

A unified, production-style AI platform for retail that consolidates three
AI subsystems — **computer vision**, **NLP**, and a **conversational
chatbot** — behind a single FastAPI gateway, containerized with Docker and
wired into a GitHub Actions CI/CD pipeline.

---

## 1. System Architecture

```
                         ┌─────────────────────────────┐
                         │        FastAPI Gateway       │
                         │   (Pydantic-validated routes) │
                         └───────────────┬───────────────┘
              ┌────────────────┬─────────┴──────────┬────────────────┐
              ▼                ▼                     ▼                ▼
     ┌─────────────────┐ ┌────────────┐ ┌────────────────────┐ ┌─────────────┐
     │  Computer Vision  │ │    NLP     │ │  Conversational Bot │ │  Dashboard  │
     │ ─────────────────│ │ ───────────│ │ ────────────────────│ │─────────────│
     │ MobileNetV2       │ │ TF-IDF +   │ │ Rule-based layer +  │ │ Cross-      │
     │ product classifier│ │ LogReg     │ │ TF-IDF/SVM ML       │ │ subsystem   │
     │                    │ │ sentiment  │ │ fallback classifier │ │ analytics   │
     │ OpenCV LBPH face   │ │            │ │                     │ │             │
     │ recognition        │ │            │ │                     │ │             │
     └─────────────────┘ └────────────┘ └────────────────────┘ └─────────────┘
              │                │                     │                │
              └────────────────┴─────────┬───────────┴────────────────┘
                                          ▼
                               SQLite (data/retail.db)
                     visit logs · sentiment logs · chatbot logs
```

All models are serialized (`joblib` for sklearn pipelines, `.pt` for the
PyTorch MobileNetV2 weights, an LBPH `.xml` model for face recognition) and
loaded **once at application startup** via FastAPI's `lifespan` context
manager, so inference requests don't pay model-loading latency.

---

## 2. Core Subsystems

### 2.1 Computer Vision
- **Product classification** — `app/modules/vision/mobilenetv2.py` implements
  the MobileNetV2 architecture (inverted residual blocks) in PyTorch, following
  a transfer-learning-ready structure. `train_product_classifier.py` fine-tunes
  it on labeled product images (shoes, bags, clothing, electronics) and
  outputs a category + confidence score.
- **Face recognition** — `app/modules/vision/face_recognition.py` uses
  OpenCV's Haar cascade for face detection and the LBPH recognizer for
  encoding/matching. Every recognized visit is timestamped and logged to
  SQLite for loyalty/repeat-visitor analytics.

### 2.2 Natural Language Processing
- `app/modules/nlp/preprocess.py` — lowercasing, tokenization, stopword
  removal, and lemmatization.
- `app/modules/nlp/train_sentiment.py` — TF-IDF + Logistic Regression baseline
  sentiment classifier (Positive / Negative / Neutral), trained on
  `data/reviews_dataset.csv`.

### 2.3 Conversational Support Chatbot
- `app/modules/chatbot/rules.py` — deterministic regex-based rules for
  high-volume intents (order status, return policy, store hours, greetings).
- `app/modules/chatbot/chatbot_engine.py` — if no rule matches, falls back to
  a TF-IDF + Linear SVM intent classifier (`train_intent.py`) trained on
  `data/intents_dataset.csv`.

---

## 3. Note on offline/sandboxed training (read before you retrain)

This build was assembled in a network-restricted sandbox with **no access to
external model-weight servers** (PyTorch Hub, dlib's model repo, spaCy/NLTK
model downloads). To keep every subsystem genuinely trainable and runnable
end-to-end without that access:

| Component | What's used here | Drop-in production upgrade |
|---|---|---|
| Product classifier | MobileNetV2 trained **from scratch** on a small synthetic image dataset (`generate_product_data.py`) | Load ImageNet-pretrained weights via `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)` and fine-tune on a real labeled product catalog |
| Face recognition | OpenCV **LBPH** recognizer (ships fully inside opencv-contrib, no external download) | Swap to the `face_recognition` (dlib) library — same enroll/recognize interface |
| Text preprocessing | Dependency-free regex tokenizer + stopword list + suffix-stripping lemmatizer | Swap in `spaCy` (`en_core_web_sm`) or `NLTK` — the TF-IDF vectorizers accept any `tokenizer=` callable |
| Sentiment/Intent data | Synthetic + templated example datasets | Replace `data/reviews_dataset.csv` / `data/intents_dataset.csv` with real labeled data — no code changes needed |

Each swap point is called out with a comment in the relevant source file.

---

## 4. API Reference

Interactive Swagger docs are available at **`/docs`** once the server is running.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify-product` | Upload a product image → category + confidence scores |
| `POST` | `/enroll-face` | Enroll a customer (`customer_id`, `name`, one or more face images) |
| `POST` | `/recognize-face` | Upload a face image → match against enrolled customers, logs visit |
| `POST` | `/analyze-sentiment` | `{"text": "..."}` → sentiment label + per-class scores |
| `POST` | `/chatbot` | `{"message": "..."}` → intent, response text, and which layer answered |
| `GET`  | `/dashboard/stats` | Aggregated analytics across all subsystems |
| `GET`  | `/health` | Liveness/readiness probe |

### Example requests

```bash
# Sentiment analysis
curl -X POST http://localhost:8000/analyze-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely love this product, best purchase this year!"}'

# Chatbot
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "what are your store hours?"}'

# Product classification
curl -X POST http://localhost:8000/classify-product \
  -F "file=@data/products/shoes/shoes_000.png;type=image/png"

# Enroll a customer's face
curl -X POST http://localhost:8000/enroll-face \
  -F "customer_id=cust001" -F "name=Jane Doe" \
  -F "files=@face1.jpg" -F "files=@face2.jpg"

# Recognize a face
curl -X POST http://localhost:8000/recognize-face \
  -F "file=@incoming_customer.jpg;type=image/jpeg"

# Dashboard stats
curl http://localhost:8000/dashboard/stats
```

---

## 5. Running Locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data & train all models
python app/modules/vision/generate_product_data.py
cd app/modules/vision && python train_product_classifier.py && cd ../../..
python -m app.modules.nlp.train_sentiment
python -m app.modules.chatbot.train_intent

# 4. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000/docs** for interactive Swagger docs.

### Running the test suite

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 6. Running with Docker

```bash
docker compose up --build
```

This builds a multi-stage image: the build stage installs dependencies and
**trains all three models during the image build**, and the runtime stage
ships a slim image with only the trained artifacts and inference code. The
API will be available at `http://localhost:8000`.

To build/run manually without compose:

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 smart-retail-ai
```

### Deploying to Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/smart-retail-ai
gcloud run deploy smart-retail-ai \
  --image gcr.io/<PROJECT_ID>/smart-retail-ai \
  --platform managed --port 8000 --memory 2Gi
```

### Deploying to Railway

Connect the GitHub repo in the Railway dashboard — Railway auto-detects the
`Dockerfile` and builds/deploys it. Set the exposed port to `8000` under
service settings.

---

## 7. CI/CD

`.github/workflows/ci-cd.yml` defines a two-job pipeline:

1. **test** — installs dependencies, regenerates synthetic data, trains all
   models fresh, and runs `pytest` on every push/PR to `main`.
2. **build-and-push** — on a successful push to `main`, builds the Docker
   image and pushes it to GitHub Container Registry (`ghcr.io`).

A commented-out `deploy` job shows how to wire in Google Cloud Run once
you've configured a service account secret.

---

## 8. Project Structure

```
smart-retail-ai/
├── app/
│   ├── main.py                       # FastAPI gateway
│   ├── schemas.py                    # Pydantic request/response models
│   ├── modules/
│   │   ├── vision/
│   │   │   ├── mobilenetv2.py        # MobileNetV2 architecture
│   │   │   ├── generate_product_data.py
│   │   │   ├── train_product_classifier.py
│   │   │   ├── product_classifier.py # inference wrapper
│   │   │   └── face_recognition.py   # LBPH face recognition + loyalty log
│   │   ├── nlp/
│   │   │   ├── preprocess.py
│   │   │   ├── train_sentiment.py
│   │   │   └── sentiment_model.py    # inference wrapper
│   │   └── chatbot/
│   │       ├── rules.py              # rule-based layer
│   │       ├── train_intent.py       # ML fallback classifier training
│   │       └── chatbot_engine.py     # hybrid response engine
│   └── dashboard/
│       └── stats.py                  # cross-subsystem analytics
├── data/
│   ├── reviews_dataset.csv           # sentiment training data
│   └── intents_dataset.csv           # chatbot intent training data
├── models/                            # serialized trained artifacts (generated)
├── tests/
│   └── test_api.py                   # pytest integration tests
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

---

## 9. Suggested viva / evaluation talking points

- **Why FastAPI + Pydantic:** async I/O for concurrent inference requests,
  automatic OpenAPI/Swagger docs, and request validation with zero
  boilerplate.
- **Why load models at startup, not per-request:** avoids repeated
  deserialization cost (torch/joblib load) on every API call — critical for
  production latency.
- **Why a hybrid rule-based + ML chatbot:** rules give deterministic,
  zero-latency answers for the ~80% of queries that are common intents; the
  ML fallback handles the long tail of phrasings the rules don't anticipate.
- **Why LBPH over dlib-based face recognition here:** ships without an
  external model download, so the container builds reproducibly offline;
  trade-off is lower accuracy on pose/lighting variation than a deep
  embedding model — documented in `face_recognition.py`.
