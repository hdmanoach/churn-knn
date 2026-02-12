import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data_preparation import load_data, preprocess

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 1) Load and preprocess
df = load_data(DATA_PATH)
X, y = preprocess(df)

# 2) Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4) GridSearch for best k
param_grid = {'n_neighbors': range(1, 31)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
print("Best k:", best_k)

# 5) Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# 6) Predict
y_pred = knn.predict(X_test)

# 7) Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8) Save model + scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
