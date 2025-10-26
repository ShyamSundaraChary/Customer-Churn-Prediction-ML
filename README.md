# Customer Churn Prediction for Online Retail

## 🎯 Project Objective

This project aims to build a machine learning model to predict whether a customer will make another purchase within a 90-day window based on their past transaction history. The goal is to identify customers at risk of churning so that proactive retention strategies can be implemented.

---

## ❓ Business Problem

Customer retention is crucial for business growth and profitability. Acquiring new customers is significantly more expensive than retaining existing ones. By accurately predicting customer churn, the business can:
* Launch targeted re-engagement marketing campaigns.
* Offer personalized promotions to at-risk customers.
* Improve customer lifetime value and reduce overall churn rates.

---

## 📊 Dataset

The project uses the `online_retail.csv` dataset, which contains transactional data from a UK-based online retail store. Each row represents a single item purchased in a transaction.

---

## 🧠 Methodology & Workflow

The project follows a structured machine learning workflow that mimics a real-world deployment scenario by using a time-based data split.

1.  **Data Cleaning & Preparation**: The raw transactional data was cleaned by handling missing `Client_ID`s, removing returns (transactions with negative quantities), and converting data types. A `Total_Spent` feature was created by multiplying `Units_Sold` and `Unit_Cost`.

2.  **Time-Based Splitting**: The data was divided into a `feature_period` (for building customer profiles) and a `target_period` (for determining if a customer returned). This approach prevents data leakage and ensures the model learns from past data to predict future events.

3.  **Feature Engineering (RFM+)**: For each customer, a set of powerful behavioral features were engineered from their activity in the `feature_period`:
    * **Recency**: Days since the customer's last purchase.
    * **Frequency**: Total number of unique transactions.
    * **Monetary**: Total amount spent.
    * **AvgOrderValue**: Average spending per transaction.
    * **TenureDays**: Days between the first and last purchase.

4.  **Modeling**: Three different classification models were trained and evaluated:
    * **Logistic Regression** (as a baseline)
    * **XGBoost Classifier**
    * **LightGBM Classifier**

5.  **Evaluation**: Models were compared using standard classification metrics, including Accuracy, Precision, Recall, F1-Score, and ROC AUC.

---

## 💡 Key Insights & Results

The **XGBoost Classifier** was selected as the final model due to its superior performance across all metrics, achieving an **F1-Score of 0.811** and a **ROC AUC of 0.828**.

Feature importance analysis from the model revealed the following key drivers of customer retention:

1.  **Recency is the #1 Predictor**: The time since a customer's last purchase is the most critical factor.
    * **Recommendation**: Implement time-sensitive re-engagement campaigns for customers who have been inactive for a specific period.

2.  **Loyalty Matters**: Customer `Tenure` and `Frequency` are strong indicators of a repeat purchase.
    * **Recommendation**: Develop a loyalty program to reward long-term and frequent shoppers.



---

## 🛠️ Technologies & Libraries Used

This project is implemented in Python 3 and utilizes the following core libraries:

  * **Pandas:** For data manipulation and analysis.
  * **NumPy:** For numerical operations.
  * **Scikit-learn:** For data preprocessing, modeling (Logistic Regression), and evaluation.
  * **XGBoost:** For building the high-performance gradient boosting model.
  * **LightGBM:** For an alternative high-performance gradient boosting model.
  * **Matplotlib & Seaborn:** For data visualization.
  * **Jupyter Notebook:** For interactive development and analysis.

-----

## 📁 File Structure

```
customer-churn-prediction/
│
├──  gpp_mini_project.ipynb       # The main Jupyter Notebook with all the analysis and code.
├── online_retail.csv              # The raw dataset used for the project.
└── README.md                      # This file.
```

-----

## ⚙️ How to Run This Project

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file can be created with the necessary libraries. For now, you can install them manually:

    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
    ```

4.  **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

5.  **Run the notebook:**
    Open the `gpp_mini_project.ipynb` file and execute the cells sequentially. Ensure the `online_retail.csv` dataset is in the same directory.

-----

## 🔮 Future Work

Potential improvements and future directions for this project include:

  * **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV to optimize the XGBoost model for better performance.
  * **Advanced Feature Engineering:** Create more sophisticated features, such as time between purchases, product category preferences, or seasonality effects.
  * **Deployment:** Package the final model into a REST API using a framework like Flask or FastAPI to serve predictions in a production environment.
