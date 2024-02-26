import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data.database import DatabaseLoader


def plot_dist():
    pass


if __name__ == "__main__":
    sns.color_palette("hls", 8)

    # Get dataset from sqlite database
    database = DatabaseLoader("../data/sqlite/db.sqlite")

    df = database.get_data()
    sex = 'Female'

    df = df.loc[df['SEX'] == sex]

    df = df[df['TEMPERATURE'] > 90]
    df = df[df['RESP'] < 50]
    df = df[df['SpO2'] > 80]

    rows = 4
    cols = 4

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 10))

    fig.suptitle(f"Distribtuions by Outcome - {sex}")

    metrics = ["age", "systolic", "diastolic", "MAP", "pulse_pressure",
               "TEMPERATURE", "PULSE", "RESP", "SpO2"]

    for r in range(rows):
        for c in range(cols):

            idx = r*rows + c
            if idx > len(metrics) - 1:
                break

            ax = axs[r, c]

            p = sns.histplot(df,
                             x=metrics[idx], hue='OUTCOME',
                             element='step',
                             palette=sns.color_palette(n_colors=4),
                             ax=ax)

            # p.legend_.remove()
            ax.set_title(f"Feature: {metrics[idx]}")

    for i in range(4):
        data = df.loc[df["OUTCOME"] == i]
        ax = axs[rows-1, i]
        p = sns.histplot(data, x="ACUITY", discrete=True,
                         ax=ax)
        ax.set_title(f"Outcome: {i}")

    print(f"MAX : {df['TEMPERATURE'].max(axis=0)}")
    print(f"MIN : {df['TEMPERATURE'].min(axis=0)}")

    print(df['TEMPERATURE'].loc[df['TEMPERATURE'] < 70])
    print(df['RESP'].loc[df['RESP'] > 50])
    print(df['SpO2'].loc[df['SpO2'] < 70])

    fig.tight_layout()
    plt.show()
