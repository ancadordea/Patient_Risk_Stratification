# Patient Risk Stratification Project

## Overview

This project demonstrates a simple, yet meaningful, practical application of machine learning techniques on clinical data to support decision-making in medical settings. By analysing synthetic hospital data, this model uses patient demographics and clinical information applying **risk stratification**, and categorises patients into **Low, Medium, or High risk** groups. This stratification can help prioritise care, optimise discharge planning, and allocate resources efficiently. The model achieved an overall accuracy of 88.3%, showing its potential reliability and value in real-world clinical workflows.

---

## Dataset

The synthetic dataset contains 300 patient records with the following features:

* `patient_ID`: Unique identifier
* `age`: Patient age (18-90 years)
* `gender`: Male or Female
* `admission_date`: Date of hospital admission
* `discharge_date`: Date of discharge
* `diagnosis`: Primary diagnosis (COVID-19, pneumonia, bronchitis, asthma)
* `treatment_cost`: Cost of treatment in USD
* `length_of_stay`: Number of days in hospital

---

## Risk Stratification Approach

A **risk score** was created by combining:

* Age (0-2 points)
* Diagnosis severity (0-2 points)
* Length of stay (0-2 points)

The total score categorizes patients into:

* **Low Risk:** score < 3
* **Medium Risk:** score 3-4
* **High Risk:** score ≥ 5

---

## Modeling

A **multinomial logistic regression** model was trained to predict the `risk_group` using:

* Age
* Length of stay
* Diagnosis

The dataset was split into training (80%) and testing (20%) sets.

---

## Results

* **Overall Accuracy:** 88.3%
* **Kappa Statistic:** 0.79 (substantial agreement)
* **Sensitivity:**

  * Low risk: 66.7%
  * Medium risk: 100%
  * High risk: 90%
* **Specificity:**

  * Low risk: 100%
  * Medium risk: 75%
  * High risk: 100%

The model effectively differentiates patient risk groups, particularly Medium and High risk, with some misclassification in Low risk patients.

---

## Future Work

* Incorporate more clinical features (e.g., comorbidities, lab results)
* Experiment with other classification models (random forests, gradient boosting)
* Perform hyperparameter tuning and cross-validation
* Apply model interpretability techniques to explain predictions
