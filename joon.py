# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# %%
df = pd.read_csv("heart.csv")


# %%
df.head()


# %%
df.describe()


# %%
df.dtypes


# %% [markdown]
# ## Scale Numerical features (Z-scroing)

# %%
numeric_df = df.select_dtypes(np.number)
scaled_features = StandardScaler().fit_transform(numeric_df.values)
scaled_df = pd.DataFrame(scaled_features, columns=numeric_df.columns)


# %% [markdown]
# ## Case 1: Create dummy variables for categorical features

# %%
cat_df = df.select_dtypes(exclude=np.number)
dummy_df = pd.get_dummies(
    cat_df, drop_first=True
)  # Drop first dummy variable as a base


# %%
features = pd.concat([scaled_df, dummy_df], axis=1)
features["HeartDisease"] = df[
    "HeartDisease"
]  # Undo standard scaling for target variable


# %%
features.describe()


# %%
x = features.drop(columns=["HeartDisease"])
y = features["HeartDisease"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# %% [markdown]
# # KNN

# %%
sns.countplot(features["HeartDisease"])


# %%
# KNN Classifier object
knn = KNeighborsClassifier()
# Hyperparameters
hyper_params = dict(n_neighbors=list(range(1, 100)))


# %%
# Grid Search Cross-validation
clf = GridSearchCV(knn, param_grid=hyper_params, scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print("Best n_neighbors:", best_model.best_estimator_.get_params()["n_neighbors"])


# %%
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))


# %% [markdown]
# # Logistic Regression

# %%
# Logistic Regression object
lr = LogisticRegression()

# Hyperparameters
solvers = ["liblinear"]
penalty = ["l2", "l1"]
c = np.arange(0.01, 100, 0.1).tolist()

# Hyperparameter space
hyper_params = dict(penalty=penalty, C=c, solver=solvers, random_state=[1])


# %%
# Grid Search Cross-validation
clf = GridSearchCV(lr, param_grid=hyper_params, scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print(f"Best Hyperparameters: {best_model.best_params_}")


# %%
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))


# %% [markdown]
# # Decision Tree Classifier

# %%
dt = DecisionTreeClassifier()

# Hyperparameters
criterion = ["gini", "entropy"]
max_depths = np.linspace(1, 32, 32, endpoint=True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
max_features = list(range(1, x_train.shape[1]))

# Hyperparameter space
hyper_params = dict(
    criterion=criterion,
    max_depth=max_depths,
    min_samples_leaf=min_samples_leafs,
    min_samples_split=min_samples_splits,
    max_features=max_features,
)


# %%
# Grid Search Cross-validation
clf = GridSearchCV(dt, param_grid=hyper_params, scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print(f"Best Hyperparameters: {best_model.best_params_}")

# %%
