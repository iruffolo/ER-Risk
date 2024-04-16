import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import cupy as cp
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from bert_embeddings import get_bert_embeddings
from load_data.database import DatabaseLoader


def load_database(path, save=False, load=True):

    if load:
        features = pd.read_pickle("../data/features.pkl")
        labels = pd.read_pickle("../data/labels.pkl")

        return features, labels

    database = DatabaseLoader(path)

    df = database.get_data()
    df.columns = map(str.lower, df.columns)

    df["clean_notes"] = database.clean_notes(
        database.get_notes(), acronyms_path="load_data/acronyms.txt"
    )

    df["sex"] = df["sex"].astype("category")
    df["visit_reason"] = df["visit_reason"].astype("category")

    df.dropna(inplace=True)
    df.drop_duplicates(subset=["pat_enc_csn_id"], keep="last", inplace=True)

    feature_names = [
        "age",
        "systolic",
        "diastolic",
        "map",
        "pulse_pressure",
        "temperature",
        "pulse",
        "resp",
        "spo2",
        "sex",
        "visit_reason",
    ]

    features = df[feature_names]

    labels = df["outcome"].astype("category")
    bert_output = get_bert_embeddings(df["clean_notes"])

    cls_tokens = pd.DataFrame(bert_output["pooler_output"])

    features.reset_index(drop=True, inplace=True)
    cls_tokens.reset_index(drop=True, inplace=True)
    features = pd.concat([features, cls_tokens], axis=1)

    if save:
        features.to_pickle("../data/features.pkl")
        labels.to_pickle("../data/labels.pkl")

    return features, labels


if __name__ == "__main__":

    features, labels = load_database(
        "../data/sqlite/7day.sqlite", load=True, save=False
    )

    labels = labels.astype("int")
    labels.replace(1, 0, inplace=True)
    labels.replace(2, 1, inplace=True)
    labels.replace(3, 1, inplace=True)
    labels = labels.astype("category")

    print(f"Features {features}")
    print(f"Number of unique labels: {labels.nunique()}")
    print(f"Label counts\n: {labels.value_counts()}")
    print("-------------------")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # x_train, x_val, y_train, y_val = train_test_split(
    # x_train, y_train, test_size=0.2, random_state=42
    # )

    print("training shape = ", x_train.shape)
    print("test shape = ", x_test.shape)
    # print("validation shape = ", x_val.shape)
    print("-------------------")

    # A parameter grid for XGBoost
    params = {
        "objective": "multi:softmax",
        # "objective": "multi:softprob",
        # "objective": ["multi:softmax", "multi:softprob"],
        # "max_depth": [3, 5, 10, 50, 100],
        "max_depth": 4,
        "learning_rate": 0.0001,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # "gamma": 1,
        # "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "device": "cuda",
        "num_class": labels.nunique(),
    }

    # sample_wts = compute_sample_weight(class_weight="balanced", y=y_train)
    # wts = {0: 1, 1: 1, 2: 10, 3: 50}
    wts = {0: 1, 1: 50}
    sample_wts = compute_sample_weight(class_weight=wts, y=y_train)

    model = xgb.XGBClassifier(**params, enable_categorical=True)
    model.fit(x_train, y_train, verbose=True, sample_weight=sample_wts)

    # Create regression matrices
    # dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
    # dtest = xgb.DMatrix(x_test, y_test, enable_categorical=True)
    # dval = xgb.DMatrix(x_val, y_val, enable_categorical=True)
    #
    # evals = [(dtest, "validation"), (dtrain, "train")]
    #
    # print("Training model")
    # model = xgb.train(
    #     params=params,
    #     dtrain=dtrain,
    #     num_boost_round=2,
    #     evals=evals,
    #     verbose_eval=50,
    #     early_stopping_rounds=50
    # )

    # model.save_model("../models/xgb_static.json")

    # print("Results")
    # results = xgb.cv(
    #     params, dtrain,
    #     num_boost_round=1_000,
    #     nfold=5,
    #     metrics=["mlogloss", "auc", "merror"],
    #     verbose_eval=50,
    # )
    #
    # best = results["test-auc-mean"].max()
    # print(results)
    # print(best)
    # exit()

    y_train_pred = model.predict(x_train)
    # y_val_pred = model.predict(x_val)
    y_val_pred = model.predict(x_test)

    print(y_train_pred)
    print(y_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    # val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy = accuracy_score(y_test, y_val_pred)

    print("Training accuracy --> ", train_accuracy)
    print("Validation accuracy --> ", val_accuracy)

    y_test_predict = model.predict(x_test)

    cm = confusion_matrix(y_test, y_test_predict)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    classification_rep = classification_report(y_test, y_test_predict)

    print("Classification Report:")
    print(classification_rep)

    feature_names = [
        "age",
        "systolic",
        "diastolic",
        "map",
        "pulse_pressure",
        "temperature",
        "pulse",
        "resp",
        "spo2",
        "sex",
        "visit_reason",
    ]
    importance = model.feature_importances_

    for feature, importance_score in zip(feature_names, importance):
        print(feature, ":", importance_score)

    # Accumulate importance of notes
    cols = [x for x in features.columns if x not in feature_names]
    notes_importance = 0
    for feature, importance_score in zip(cols, importance):
        notes_importance += importance_score

    print("Notes importance: ", notes_importance)

    fig, ax = plt.subplots(figsize=(10, 10))
    # xgb.plot_importance(model, ax=ax)

    y = list(importance[: len(feature_names)])
    feature_names.append("Notes")
    y.append(notes_importance)

    ax.barh(feature_names, y)
    ax.set_yticks(feature_names, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance")
    plt.show()

    score = roc_auc_score(
        y_test,
        y_test_predict,
        multi_class="ovr",
        average="micro",
    )
    print("ROC AUC Score: ", score)
    fp, tp, threshold = roc_curve(y_test, y_test_predict)

    plt.subplots(1, figsize=(10, 10))
    plt.title("Receiver Operating Characteristic - DecisionTree")
    plt.plot(fp, tp, label="AUC = %0.2f" % roc_auc_score(y_test, y_test_predict))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
