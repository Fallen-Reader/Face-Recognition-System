import sqlite3
import numpy as np
from config import DB_PATH


class FaceDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL
                )
                """
            )

    def add_face(self, name: str, encoding: np.ndarray):
        blob = np.asarray(encoding, dtype=np.float64).tobytes()
        with self._connect() as conn:
            conn.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, blob))

    def load_all(self):
        
        with self._connect() as conn:
            rows = conn.execute("SELECT name, encoding FROM faces ORDER BY id").fetchall()
        names = [r[0] for r in rows]
        encodings = [np.frombuffer(r[1], dtype=np.float64) for r in rows]
        return names, encodings

    def delete_face(self, name: str):
        
        with self._connect() as conn:
            conn.execute("DELETE FROM faces WHERE name = ?", (name,))

    def list_people(self):
        
        with self._connect() as conn:
            return conn.execute(
                "SELECT name, COUNT(*) FROM faces GROUP BY name ORDER BY name"
            ).fetchall()