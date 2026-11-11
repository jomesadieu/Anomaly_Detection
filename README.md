# Anomaly Detection Assignment

## Results
- Task 1: LOF (k=2) identifies **D (11,13)**, score **1.6126**.
- Task 2: Three seeded random isolation trees identify **D (16,18,12,14)** as shortest mean path (1.333 vs 2.333–2.667). Randomized results can vary by seed.
- Task 3: Four methods on standardized Iris petal length/width: One-Class SVM, SGD One-Class SVM with Nystroem kernel approximation, Isolation Forest, and LOF in novelty mode for plotting boundaries.

## Run
`pip install -r requirements.txt`
`python build.py`

Or open the notebook in Colab and run all cells. The code uses the built-in Iris dataset and needs no CSV uploads.

## GitHub submission
Create a new GitHub repository, upload the notebook, `build.py`, `requirements.txt`, README and PNG, and submit your repository URL.
