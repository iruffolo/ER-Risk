import sqlite3
import csv


class DataLoader:
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
        self.cursor.close()
        self.conn.close()

    def get_data(self):
        """
        Get data from train_data table
        """
        query = "select * from train_data"

        self.cursor.execute(query)
        res = self.cursor.fetchall()
        return (res)

    def get_notes(self):
        """
        Get data from triage_notes table
        """
        query = "select * from triage_notes"

        self.cursor.execute(query)
        res = self.cursor.fetchall()

        return (res)

    def put_notes(self, notes, table):
        query = f"""INSERT INTO {table}
                  (PAT_ENC_CSN_ID, NOTE_TEXT, NOTE_TIMESTAMP)
                  VALUES (?, ?, ?);"""

        self.cursor.executemany(query, notes)

    @staticmethod
    def clean_notes(notes):

        with open('acronyms.txt', 'r') as f:
            acronyms = [row for row in csv.reader(f, skipinitialspace=True,
                                                  delimiter='=')]

        cleaned = notes

        # Remove acronyms
        for a in acronyms:
            cleaned = [[row[0], row[1].replace(a[0], a[1]), row[2]]
                       for row in cleaned]

        # Remove headers and COVID stuff
        cleaned = [[row[0],
                    row[1].split(":"),
                    row[2]]
                   for row in cleaned]

        for c in cleaned:
            print(c)


if __name__ == "__main__":

    dl = DataLoader()

    # notes = get_notes(cursor)
    # notes = clean_notes(notes)
    # put_notes(notes, cursor, "clean_triage_notes")
    # conn.commit()
