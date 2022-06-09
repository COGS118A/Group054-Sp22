from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm


# pylint: disable=invalid-name
class Pipeline(skPipeline):
    """Adds multiestimator support for Pipelines"""

    @property
    def estimators(self):
        """Add estimator field to Pipeline"""

        return (
            self._final_estimator.estimators
            if isinstance(self._final_estimator, MultiClassifier)
            else None
        )

    def get_classifier(self, classifier: str = None) -> BaseEstimator:
        if self.estimators:
            return self.estimators[classifier]
        return self._final_estimator

    def update_hyperparameters(self, params):
        curr_estimators = {}
        for name, estimator in self.steps[-1][-1].estimators.items():
            print(name, estimator)
            curr_estimators[name] = estimator.__class__
        self.steps[-1] = (self.steps[-1][0], MultiClassifier(curr_estimators))

    # def set_params(self, params):
    #     for model, param in params.items():
    #         self.estimators[model].set_params(**param)


class MultiClassifier(BaseEstimator):
    """Adapted from https://stackoverflow.com/a/53926120"""

    def __init__(self, estimators: dict[str, BaseEstimator]):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimators = estimators
        self.params = None

    def fit(self, X, y=None, estimator: str = None, **kwargs):
        if estimator is not None:
            self.estimators[estimator].fit(X, y, **kwargs)
        else:
            for name in self.estimators:
                self.estimators[name].fit(X, y, **kwargs)
        return self

    def predict(self, X, y=None, estimator: str = None):
        if estimator:
            return self.estimators[estimator].predict(X)
        else:
            return {name: self.estimators[name].predict(X) for name in self.estimators}

    def predict_proba(self, X, estimator: str = None):
        if estimator:
            return self.estimators[estimator].predict_proba(X)
        else:
            return {
                name: self.estimators[name].predict_proba(X) for name in self.estimators
            }
        # return self.estimators[estimator].predict_proba(X)

    def score(self, X, y, estimator: str = None):
        if estimator:
            return self.estimators[estimator].score(X, y)
        else:
            return {name: self.estimators[name].score(X, y) for name in self.estimators}
        # return self.estimators[estimator].score(X, y)

    def set_hyperparams(self, params: dict) -> None:
        for name, model in params.items():
            for param, value in model.items():
                # setattr(self.estimators[name], param, value)
                self.estimators[name].set_params({param: value})

    def set_params(self, **kwargs) -> None:
        print(kwargs)
        new_estimators = self.estimators
        # for
        params = kwargs["params"]
        new_estimators = {}
        for classifier, (name, param_set) in zip(self.estimators, params.items()):
            new_estimators[name] = self.estimators[name].__class__(**param_set)
        self.estimators = new_estimators
        # for name, model in params.items():
        #     self.estimators[name].set_params(**model)
        # for param, value in model.items():
        #     # setattr(self.estimators[name], param, value)
        #     self.estimators[name].set_params(**{param: value})


def get_best_param(
    X: np.ndarray,
    y: np.ndarray,
    mclf: MultiClassifier,
    estimator: str,
    param_name: str,
    params: np.ndarray,
    ax: plt.Axes = None,
):
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = mclf._final_estimator.estimators.get(estimator)

    for param in tqdm(params):
        # clf.set_params({param_name: param})
        setattr(clf, param_name, param)
        mclf.fit(X_train, y_train)
        score = mclf.score(X_test, y_test)[estimator]
        scores.append(score)

    best_score = max(scores)
    best_param = params[scores.index(best_score)]
    print(f"    {param_name}={best_param} yields score {best_score}")

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(params, scores)
    ax.annotate(
        f"{param_name}='{best_param}' -> {best_score}",
        (best_param, best_score),
    )
    # ax.set_xticks(params)
    ax.set_title(f"{param_name} scores")
    ax.set_xlabel("parameter")
    ax.set_ylabel("score")

    setattr(clf, param_name, best_param)

    return best_param


def parameter_search(
    mclf: MultiClassifier,
    X: np.ndarray,
    y: np.ndarray,
    hyperparameters: dict[str, dict[str, list]],
) -> dict[str, dict[str, object]]:
    best_params = {}
    for model, hyperparams in hyperparameters.items():
        print(f"Parameter search for {model}")
        best_params[model] = {}
        fig, axes = plt.subplots(1, len(list(hyperparams.values())), figsize=(16, 4))
        fig.suptitle(f"{model} hyperparameters")
        for param_name, ax in zip(hyperparams, axes):
            best_params[model][param_name] = get_best_param(
                X, y, mclf, model, param_name, hyperparams[param_name], ax
            )

    return best_params


def get_roc_curve(
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf: BaseEstimator = None,
    mclf: Pipeline = None,
    clf_name: str = None,
    ax: plt.Axes = None,
    plot: bool = True,
) -> tuple:
    """adapted from @gbhand's D6"""

    if mclf and clf_name:
        scores = mclf.predict_proba(X_test, estimator=clf_name)[:, 1]
    elif clf:
        clf_name = str(clf).split("(", maxsplit=1)[0]
        scores = clf.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("mclf xor clf required")

    fps = []
    tps = []
    for threshold in tqdm(np.linspace(0, 1, 1000)):
        y_pred = scores > threshold
        (
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ) = confusion_matrix(y_test, y_pred).ravel()
        fps.append(
            false_positives / (false_positives + true_negatives)
        )  # add false positive rate
        tps.append(
            true_positives / (true_positives + false_negatives)
        )  # add true positive rate

    fpr = fps
    tpr = tps

    roc_auc = roc_auc_score(y_test, scores)

    if plot:
        # plotting the ROC curve
        if ax is None:
            fig, ax = plt.subplots()

        if "No skill" not in ax.get_legend_handles_labels()[1]:
            ax.plot(
                np.linspace(0, 1, 1000),
                np.linspace(0, 1, 1000),
                label="No skill",
                linestyle="dashed",
            )
        ax.plot(fpr, tpr, label=clf_name)
        ax.set_title("ROC curve per model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

    print(
        f"[{clf_name}] Area under the Receiver Operating Characteristic curve:", roc_auc
    )

    return roc_auc, fpr, tpr


def make_pipeline(clf_params: dict[str, dict] = None) -> Pipeline:
    numeric_features = ["Year", "Mileage", "MSRP"]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_features = ["Make", "Model"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if clf_params:
        classifiers = {
            "logistic": LogisticRegression(
                solver="liblinear", **clf_params["logistic"]
            ),
            "tree": tree.DecisionTreeClassifier(**clf_params["tree"]),
        }

    else:
        classifiers = {
            "logistic": LogisticRegression(solver="liblinear"),
            "tree": tree.DecisionTreeClassifier(),
        }

    multiclassifier = MultiClassifier(classifiers)

    mclf = Pipeline(
        steps=[("preprocessor", preprocessor), ("multiclassifier", multiclassifier)]
    )

    return mclf
