Cyberstalking synthetic dataset (5000 samples)
Columns:
  - id: unique id
  - message: chat message text
  - label: 0 = normal, 1 = stalker-like
  - platform: platform where message came from (synthetic)
  - timestamp: ISO8601 timestamp (synthetic)

Notes:
  - This dataset is synthetic and intended for demo and experimentation.
  - For production research, collect real labeled data and ensure privacy/consent.
  - Suggested train/test split: 70:30 (e.g., 3500 train, 1500 test).
