import sqlite3
import csv
from datetime import datetime
import pandas as pd


class DatabaseLoader:
    """
    Data loader to handle connection to sqlite database for data
    """

    def __init__(self, path='../data/sqlite/db.sqlite'):
        """
        Initializes connection to database

        :param path: Path to the sqlite database file
        """

        print(f"path {path}")

        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        print("Database created and Successfully Connected to SQLite")

        ver_query = "select sqlite_version();"
        self.cursor.execute(ver_query)
        record = self.cursor.fetchall()
        print("SQLite Database Version is: ", record)

    def __del__(self):
        """
        Make sure to shut down database connection on exit
        """
        self.cursor.close()
        self.conn.close()

    def get_data(self, table="train_data_vitals"):
        """
        Get data from train_data table
        """
        query = f"select * from {table}"

        self.cursor.execute(query)
        res = self.cursor.fetchall()

        self.cursor.execute(f"PRAGMA table_info({table})")
        cols = [c[1] for c in self.cursor.fetchall()]

        df = pd.DataFrame(res, columns=cols)

        # Caclulate and insert BP metrics
        df[['systolic', 'diastolic']] = df['BP'].str.split(
            '/', expand=True).dropna().astype(int)

        df["MAP"] = (df["systolic"].apply(lambda x: x * 1/3) +
                     df["diastolic"].apply(lambda x: x * 2/3))

        df["pulse_pressure"] = df["systolic"] - df["diastolic"]

        df["age"] = pd.to_datetime(df['BIRTH_DATE']).apply(
            lambda x: ((datetime.now() - x).days/365.2425)).astype(int)

        return (df)

    def get_notes(self):
        """
        Get data from triage_notes table
        """
        query = "select note_text from train_data_vitals"

        self.cursor.execute(query)
        res = self.cursor.fetchall()

        return (res)

    def put_notes(self, notes, table="clean_triage_notes"):
        query = f"""INSERT INTO {table}
                  (PAT_ENC_CSN_ID, NOTE_TEXT, NOTE_TIMESTAMP)
                  VALUES (?, ?, ?);"""

        self.cursor.executemany(query, notes)
        self.conn.commit()

    @staticmethod
    def clean_notes(notes, acronyms_path="acronyms.txt"):
        """
        Remove acronyms, symbols, and junk from notes
        """

        with open(acronyms_path, 'r') as f:
            acronyms = [row for row in csv.reader(f, skipinitialspace=True,
                                                  delimiter='=')]
        cleaned = [n[0] for n in notes]

        # Remove acronyms
        for a in acronyms:
            cleaned = [row.replace(a[0], a[1]) for row in cleaned]

        # Remove headers and COVID stuff
        # cleaned = [row.split(":") for row in cleaned]
        # cleaned = [row.lower() for row in cleaned]

        return cleaned

    @staticmethod
    def add_to_notes(data):
        """
        Concatenates other features (i.e. chief complaint) into the notes.
        """

        new_notes = (data['VISIT_REASON'] +
                     data['clean_notes']).astype(str)

        return new_notes


if __name__ == "__main__":

    dl = DatabaseLoader()
