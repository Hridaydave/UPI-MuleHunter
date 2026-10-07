import json
import sqlite3
from pathlib import Path

DB = Path("data/mulehunter.db")


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB)


def init_db():
    con = connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        sender TEXT,
        receiver TEXT,
        amount REAL,
        timestamp TEXT,
        label INTEGER
    );

    CREATE TABLE IF NOT EXISTS scores (
        transaction_id TEXT PRIMARY KEY,
        markov REAL,
        rf REAL,
        graph REAL,
        anomaly REAL,
        ring REAL,
        fused REAL,
        band TEXT
    );

    CREATE TABLE IF NOT EXISTS results (
        name TEXT PRIMARY KEY,
        payload TEXT
    );
    """)
    con.commit()
    con.close()


def save_result(name, payload):
    con = connect()
    con.execute(
        "INSERT OR REPLACE INTO results(name,payload) VALUES(?,?)",
        (name, json.dumps(payload, default=str)),
    )
    con.commit()
    con.close()


def get_result(name):
    con = connect()
    row = con.execute(
        "SELECT payload FROM results WHERE name=?",
        (name,),
    ).fetchone()
    con.close()
    return json.loads(row[0]) if row else None
