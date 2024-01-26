import sqlite3


def get_data(cursor):

    query = "select * from train_data"

    cursor.execute(query)

    res = cursor.fetchall()

    return (res)


if __name__ == "__main__":

    conn = sqlite3.connect('../data/sqlite/db.sqlite')
    cursor = conn.cursor()
    print("Database created and Successfully Connected to SQLite")

    sqlite_select_Query = "select sqlite_version();"
    cursor.execute(sqlite_select_Query)
    record = cursor.fetchall()
    print("SQLite Database Version is: ", record)

    data = get_data(cursor)

    print(data)

    print(len(data))
    # notes = get_notes(cursor)
    # notes = clean_notes(notes)
    # put_notes(notes, cursor, "clean_triage_notes")
    # conn.commit()

    cursor.close()
    conn.close()
