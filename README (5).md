# 🎫 Support Ticket Classification & Prioritization

An NLP + Machine Learning project that automatically reads customer / IT support tickets and classifies them into categories, so support teams spend less time sorting and more time solving.

> **Status:** Ticket **category classification** is fully implemented and evaluated. **Priority prediction (High / Medium / Low)** is not yet implemented, because the dataset has no priority labels (see [Priority Prediction](#-priority-prediction-roadmap)).

---

## 📌 Business Context

Support teams receive hundreds or thousands of tickets every day through emails, forms, complaints and issue reports. Common problems:

- Tickets are not categorized properly
- Urgent issues get delayed
- Agents waste time sorting instead of solving

Instead of building another chatbot, this project builds a practical **decision-support system** that triages tickets automatically, the kind of ML used in SaaS companies, service platforms and internal IT teams.

## 🎯 Objective

Build an ML system that can:

1. Read raw support ticket text
2. Automatically classify it into a category
3. *(Planned)* Assign a priority level (High / Medium / Low)

---

## 📁 Dataset

**IT Service Ticket Classification Dataset** (`all_tickets_processed_improved_v3.csv`)

| Property | Value |
|---|---|
| Rows | 47,837 tickets |
| Columns | `Document` (ticket text), `Topic_group` (category label) |
| Missing values | None |
| Classes | 8 |

**Class distribution (imbalanced):**

| Category | Tickets |
|---|---|
| Hardware | 13,617 |
| HR Support | 10,915 |
| Access | 7,125 |
| Miscellaneous | 7,060 |
| Storage | 2,777 |
| Purchase | 2,464 |
| Internal Project | 2,119 |
| Administrative rights | 1,760 |

The class imbalance (Hardware is ~8x larger than Administrative rights) is a key challenge and drives the modelling choices below.

---

## 🛠️ Tech Stack

- **Python 3**, **Jupyter / Google Colab**
- **pandas, NumPy**: data handling
- **NLTK**: Porter stemming
- **Gensim**: Word2Vec embeddings
- **scikit-learn**: Random Forest, metrics, train/test split
- **imbalanced-learn**: SMOTE and Balanced Bagging
- **matplotlib, seaborn**: visualization
- **joblib**: model serialization

---

## ⚙️ Methodology

### 1. Text Preprocessing
Each ticket goes through a cleaning function that:
- Converts text to lowercase
- Removes punctuation, numbers and special characters
- Normalizes whitespace
- Removes a **custom stopword list** (generic English stopwords plus email noise such as greetings, sign-offs, weekday/month names, "sent", "importance high", etc.)
- Applies **Porter stemming**
- Collapses consecutively repeated words and strips leftover timestamp/extension patterns

### 2. Feature Extraction
- Trained a **Word2Vec** model (125 dimensions, window = 5, min_count = 3) on the cleaned corpus (vocabulary of 4,945 words)
- Each ticket is represented as the **average of its word vectors**

### 3. Modelling
The data was split 80/20 (stratified, `random_state=42`) and three approaches were compared:

| # | Approach | Idea |
|---|---|---|
| 1 | Random Forest (baseline) | 200 trees, `class_weight='balanced'` |
| 2 | Balanced Bagging (undersampling) | Decision-tree ensemble that undersamples majority classes per bootstrap |
| 3 | **Random Forest + SMOTE** | Oversamples minority classes in the training set only, then trains Random Forest |

---

## 📊 Results

Evaluated on 9,568 held-out tickets:

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---|---|---|---|
| Random Forest (baseline) | 0.7947 | 0.8129 | 0.7947 | 0.7934 |
| Random Forest + Undersampling | 0.7646 | 0.7754 | 0.7646 | 0.7668 |
| **Random Forest + SMOTE** ✅ | **0.8108** | **0.8139** | **0.8108** | **0.8115** |

### Class-wise Recall

| Category | Baseline RF | Undersampling | SMOTE |
|---|---|---|---|
| Access | 0.768 | 0.799 | 0.825 |
| Administrative rights | 0.375 | 0.707 | 0.642 |
| HR Support | 0.802 | 0.739 | 0.798 |
| Hardware | 0.901 | 0.716 | 0.814 |
| Internal Project | 0.700 | 0.844 | 0.844 |
| Miscellaneous | 0.736 | 0.778 | 0.805 |
| Purchase | 0.848 | 0.880 | 0.878 |
| Storage | 0.759 | 0.856 | 0.847 |

### Key Takeaways
- **Random Forest + SMOTE** gave the best overall performance (~81% accuracy, ~0.81 weighted F1) and was selected as the final model.
- The baseline looked decent overall but **missed most "Administrative rights" tickets (recall 0.375)**. Resampling lifted this to 0.64–0.71, which matters because rare categories are often the ones that need careful handling.
- Undersampling gave the best recall on small classes but lost accuracy on large ones (e.g. Hardware), so SMOTE offers the better overall balance.
- **Administrative rights** remains the weakest class (precision/recall ≈ 0.65 with SMOTE), mostly confused with neighbouring categories; **Purchase** is the strongest (F1 ≈ 0.91).

Confusion matrices for all three models are produced in the notebook.

---

## 🚀 Getting Started

### Installation
```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy matplotlib seaborn nltk scikit-learn imbalanced-learn gensim joblib
```

### Run the notebook
1. Place `all_tickets_processed_improved_v3.csv` in the project folder.
2. Open `Task_2_.ipynb` in Jupyter or Google Colab.
3. Update the data-loading cells. The notebook was written for Colab and reads the dataset from a zipped file on Google Drive (`drive.mount(...)`). For local use, replace those cells with:
   ```python
   data = pd.read_csv('all_tickets_processed_improved_v3.csv')
   ```
4. Run all cells.

### Predict on a new ticket
After running the notebook, use the `testing()` helper:

```python
testing("I need admin rights to install software on my machine.")
# -> 'Administrative rights'

testing("Please create a PO for the new laptop bags and perform the receipt.")
# -> 'Purchase'
```

The notebook also includes sample test tickets for each category, and the final model is saved with `joblib` as `Ticket_Classification_machine_learning_model_v1_0.joblib`.

---

## 💼 How This Improves Support Operations

1. **Faster first response**: tickets reach the right queue immediately instead of waiting for manual triage.
2. **Less sorting, more solving**: agent time shifts from routing to resolution.
3. **Consistent categorization**: the model applies the same logic every time, making reporting on ticket volume reliable.
4. **Better workload planning**: dependable categories show where demand comes from (e.g. Hardware ≈ 28% and HR Support ≈ 23% of tickets), supporting staffing and training decisions.
5. **Reduced backlog and misrouting**: correct routing prevents tickets bouncing between teams.

---

## 🔮 Priority Prediction Roadmap

The dataset provides category labels only, so a High / Medium / Low priority model cannot be trained directly from it. Realistic next steps:

- **Rule-based priority layer**: keyword and category rules (e.g. "outage", "cannot login", "urgent", "production down" → High) applied on top of the predicted category.
- **Labelled priority data**: collect or annotate a priority label and train a second classifier using the same pipeline.
- **Hybrid approach**: combine rules with an ML model and let agents correct priorities to create training data over time.

## ⚠️ Limitations & Future Improvements

- Word2Vec was trained on the full corpus before the train/test split. This is unsupervised (no labels used), but a stricter setup would fit it on training data only.
- Averaged word vectors lose word order; **TF-IDF + Logistic Regression/SVM** is a strong, fast baseline worth comparing.
- Transformer-based embeddings (e.g. BERT / sentence-transformers) could improve accuracy, especially for ambiguous classes like Miscellaneous and Administrative rights.
- Hyperparameter tuning with `GridSearchCV` and cross-validation was not performed.
- Package the model behind a simple API or Streamlit app for real-time triage.

---

## 📂 Repository Structure

```
├── Task_2_.ipynb                                          # Full notebook (EDA, preprocessing, modelling, evaluation)
├── README.md
└── Support_Ticket_Classification/
    └── Ticket_Classification_machine_learning_model_v1_0.joblib   # Saved best model (RF + SMOTE)
```

## 📜 License

This project is for educational purposes. Add a license of your choice (e.g. MIT) before publishing.
