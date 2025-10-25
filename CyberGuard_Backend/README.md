
# CyberGuard Backend v2 (FastAPI)
This version includes:
- IsolationForest (isolation tree) as an unsupervised anomaly detector.
- LinearRegression trained as a simple supervised predictor (treating label 0/1 as continuous risk score).
- A 70:30 train/test split in `/train_models`.
- Saved models and preprocessors in `models/` folder.

## Endpoints
- `POST /train_models` : Train models on a synthetic dataset, perform 70:30 split, return metrics.
- `POST /analyze_chat` : Analyze text or CSV using saved models (auto-train if models missing).
- `POST /analyze_screenshot` : Upload image files for OCR + heuristic fake/real verdict.

## Quick Start (VS Code)
1. Create venv and activate:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\\Scripts\\activate    # Windows PowerShell
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract for OCR (optional if you won't run OCR):
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Windows: install from Tesseract releases and set `pytesseract.pytesseract.tesseract_cmd` in `app.py`.

4. Run the server:
   ```bash
   uvicorn app:app --reload
   ```

5. Train models (or the app will auto-train on first analyze_chat call):
   ```bash
   curl -X POST http://127.0.0.1:8000/train_models
   ```

6. Analyze chat:
   ```bash
   curl -X POST -F "text=Hello\nWhy didn't you answer? I saw you online!" http://127.0.0.1:8000/analyze_chat
   ```
