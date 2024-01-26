import sqlite3
import csv


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


def get_notes(cursor):

    query = "select * from triage_notes"

    cursor.execute(query)

    res = cursor.fetchall()

    return (res)


def put_notes(notes, cursor, table):

    query = f"""INSERT INTO {table}
              (PAT_ENC_CSN_ID, NOTE_TEXT, NOTE_TIMESTAMP)
              VALUES (?, ?, ?);"""

    cursor.executemany(query, notes)

    print(query)


if __name__ == "__main__":

    conn = sqlite3.connect('../data/sqlite/db.sqlite')
    cursor = conn.cursor()
    print("Database created and Successfully Connected to SQLite")

    sqlite_select_Query = "select sqlite_version();"
    cursor.execute(sqlite_select_Query)
    record = cursor.fetchall()
    print("SQLite Database Version is: ", record)

    notes = get_notes(cursor)
    notes = clean_notes(notes)
    # put_notes(notes, cursor, "clean_triage_notes")
    # conn.commit()

    cursor.close()
    conn.close()
