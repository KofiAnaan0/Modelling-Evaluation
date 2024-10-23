# Modelling and Evaluation

This section outlines the steps taken for **data sampling**, **model training**, and **model evaluation**. The goal is to train a machine learning model to predict customer churn and evaluate its performance using appropriate metrics.

## 1. Data Sampling

The first step is to split the dataset into training and testing samples. This allows us to test how well the model generalizes to unseen data. We used a **75-25% split**, where 75% of the data was used for training and 25% for testing.

```python
# Make a copy of our data
train_df = df.copy()

# Separate target variable (churn) from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)

# Split the data into training and test sets (75-25% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

![split](assets/)

## 2. Model Training

I trained a Random Forest Classifier using 1000 decision trees. This model was chosen due to its ability to handle large datasets and its effectiveness in classification tasks.

```python
# Train a Random Forest model
model = RandomForestClassifier(
    n_estimators=1000
)
model.fit(X_train, y_train)
```

### 3. Model Evaluation

To evaluate the model, we used the following metrics:

- Accuracy: The ratio of correctly predicted observations to total observations.
- Precision: The ability of the classifier to not label a negative sample as positive.
- Recall: The ability of the classifier to identify all positive samples.
  
Why use multiple metrics? Relying solely on accuracy can be misleading in imbalanced datasets. For example, in medical scenarios (like predicting heart failure), incorrect predictions can have serious consequences.

```python
# Make predictions on the test set
predictions = model.predict(X_test)

# Confusion matrix and metrics
tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()

# Print results
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"True negatives: {tn}")
print(f"False negatives: {fn}")

# Accuracy, Precision, and Recall
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
recall = metrics.recall_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

![split](assets/)

Evaluation Results:

- True Positives: 18

- False Positives: 4

- True Negatives: 3282

- False Negatives: 348

- Accuracy: 0.90

- Precision: 0.82

- Recall: 0.05

### 4. Model Understanding

To gain insights into the model's decision-making process, we examined the feature importances. This helps us understand which features had the most influence in making predictions.

```python
feature_importances = pd.DataFrame({
    'features': X_train.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=True).reset_index()

plt.figure(figsize=(15, 25))
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
plt.yticks(range(len(feature_importances)), feature_importances['features'])
plt.xlabel('Importance')
plt.show()
```

![split](assets/)

The top features are those that the model relied on the most during training. This gives us a better understanding of which factors contribute most to customer churn predictions.
