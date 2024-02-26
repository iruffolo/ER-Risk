from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

# import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from load_data.database import DatabaseLoader

if __name__ == "__main__":

    database = DatabaseLoader("../data/sqlite/db.sqlite")

    df = database.get_data()
    df.columns = map(str.lower, df.columns)

    feature_names = ["age", "systolic", "diastolic", "map",
                     "pulse_pressure", "temperature", "pulse",
                     "resp", "spo2", "sex", "visit_reason"]

    features = df[feature_names]
    features["sex"] = features["sex"].astype("category")
    features["visit_reason"] = features["visit_reason"].astype("category")

    labels = df["outcome"]
    labels.replace(3, 2, inplace=True)

    print(features)
    print(labels)
    print(labels.value_counts())

    print(features.dtypes)
    print(labels.dtypes)

    # cp.array(features)
    # cp.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      random_state=42)

    print("training shape = ", x_train.shape)
    print("test shape = ", x_test.shape)
    print("validation shape = ", x_val.shape)

    params = {"objective": "multi:softprob",
              "max_depth": 20,
              "learning_rate": 0.05,
              "n_estimators": 2000,
              "subsample": 0.8,
              "colsample_bytree": 0.8,
              "reg_alpha": 0.1,
              "reg_lambda": 0.1,
              "device": "cuda",
              "num_class": 3}

    model = xgb.XGBClassifier(**params, enable_categorical=True)
    model.fit(x_train, y_train, verbose=True)

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

    # best = results["test-auc-mean"].max()
    # print(results)
    # print(best)

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)

    print(y_train)
    print(y_train_pred)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print("Training accuracy --> ", train_accuracy)
    print("Validation accuracy --> ", val_accuracy)

    importance = model.feature_importances_

    for feature, importance_score in zip(feature_names, importance):
        print(feature, ":", importance_score)

    y_test_predict = model.predict(x_test)

    cm = confusion_matrix(y_test, y_test_predict)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    classification_rep = classification_report(y_test, y_test_predict)

    print("Classification Report:")
    print(classification_rep)
