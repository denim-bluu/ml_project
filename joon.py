# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)


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
clf = GridSearchCV(knn, param_grid=hyper_params,
                   scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print("Best n_neighbors:",
      best_model.best_estimator_.get_params()["n_neighbors"])


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
clf = GridSearchCV(lr, param_grid=hyper_params,
                   scoring="recall", cv=10, n_jobs=-1)
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
# Random Search Cross-validation, it yields the same result as GridSearch CV but with much better efficiency
clf = RandomizedSearchCV(dt, n_iter=100, random_state=1,
                         param_distributions =hyper_params, scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print(f"Best Hyperparameters: {best_model.best_params_}")
# %%
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
# %%
rf = RandomForestClassifier()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=0, stop=2000, num=100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 120, num=10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20, 50, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 5, 15]
# Method of selecting samples for training each tree
bootstrap = [True, False]

hyper_params = dict(
    n_estimators=n_estimators,
    max_features=max_features,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    bootstrap=bootstrap,
)

clf = RandomizedSearchCV(rf, n_iter=100, random_state=1,
                         param_distributions =hyper_params, scoring="recall", cv=10, n_jobs=-1)
best_model = clf.fit(x_train, y_train)
print(f"Best Hyperparameters: {best_model.best_params_}")
best_param = {'n_estimators': 1232, 'min_samples_split': 50, 'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True}

# %%
clf = RandomForestClassifier(**best_param).fit(x_train, y_train)
predictions = clf.predict(x_test)
print(classification_report(y_test, predictions))
# %%
