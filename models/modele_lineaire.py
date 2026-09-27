import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV, train_test_split

import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/dataFeature.csv")

df = df[
    ~((df["GrLivArea"] > 4000) &
    (df["SalePrice"] < 300000))
]

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# modele 1 LinearRegression

pipeline_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Entraînement
# pipeline_lr.fit(X_train, y_train)

# # Prédiction
# y_pred_lr = pipeline_lr.predict(X_test)

y_train_log = np.log1p(y_train)
pipeline_lr.fit(X_train, y_train_log)
y_pred_lr = np.expm1(pipeline_lr.predict(X_test))

# Performance
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# print(f"MAE: {mae_lr:.2f} | RMSE: {rmse_lr:.2f} | R²: {r2_lr:.3f}")

# modele 2 RandomForestRegressor

pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# print(f"MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f} | R²: {r2_rf:.3f}")

# modele 3 SVR

pipeline_svr = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", SVR(kernel="rbf", C=10, epsilon=0.1))
])

# pipeline_svr.fit(X_train, y_train)
# y_pred_svr = pipeline_svr.predict(X_test)

# y_train_log = np.log1p(y_train)
pipeline_svr.fit(X_train, y_train_log)
y_pred_svr = np.expm1(pipeline_svr.predict(X_test))

mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
r2_svr = r2_score(y_test, y_pred_svr)

# print(f"MAE: {mae_svr:.2f} | RMSE: {rmse_svr:.2f} | R²: {r2_svr:.3f}")

resultats = pd.DataFrame({
    "Modèle": ["Régression linéaire", "Random Forest", "SVR"],
    "MAE": [mae_lr, mae_rf, mae_svr],
    "RMSE": [rmse_lr, rmse_rf, rmse_svr],
    "R²": [r2_lr, r2_rf, r2_svr]
})
print(resultats.sort_values("RMSE"))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

results = cross_validate(
    pipeline_rf, X_train, y_train,
    cv=kf, scoring=scoring, return_train_score=True
)

for metric in scoring:
    test_scores = results[f"test_{metric}"]
    print(f"{metric} : {np.abs(test_scores.mean()):.3f} (+/- {test_scores.std():.3f})")

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score (CV) :", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print(f"Test MAE  : {mae:.2f}")
print(f"Test RMSE : {rmse:.2f}")
print(f"Test R²   : {r2:.3f}")

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_rf, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.title("Prix réel vs prix prédit")

plt.show()

errors_lr = y_test - y_pred_lr

plt.figure(figsize=(8, 6))

plt.hist(errors_lr, bins=30)

plt.xlabel("Erreur (prix réel - prix prédit)")
plt.ylabel("Nombre de prédictions")
plt.title("Distribution des erreurs - Régression linéaire")

plt.axvline(0, linestyle="--")

plt.show()

joblib.dump(
    best_model,
    "models/house_price_model.pkl"
)

print("Modèle sauvegardé avec succès.")