"""One-time WRDS credential setup, with a visible password box.

Run it (it opens a small window), type the password, press OK. It writes
%APPDATA%\\postgresql\\pgpass.conf in the format the WRDS client reads, then
tries to connect and reports the result in a message box. The password is
never printed or logged; it goes from the box to the file and nowhere else.

The text box shows what you type on purpose -- the console prompt hides it,
which is what made the earlier attempts look broken.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", "9737", "wrds"


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    user = simpledialog.askstring("WRDS setup", "WRDS username:", parent=root)
    if not user:
        return 1
    pw = simpledialog.askstring("WRDS setup", f"WRDS password for {user} (shown as you type):", parent=root)
    if not pw:
        return 1

    folder = os.path.join(os.environ["APPDATA"], "postgresql")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "pgpass.conf")
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write(f"{HOST}:{PORT}:{DB}:{user}:{pw}")
    del pw

    try:
        import wrds
        db = wrds.Connection(wrds_username=user)
        libs = db.list_libraries()
        crsp = sorted(l for l in libs if "crsp" in l.lower())
        messagebox.showinfo("WRDS setup", f"Connected. {len(libs)} libraries.\nCRSP-related: {crsp}\n\nFile written:\n{path}", parent=root)
        return 0
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("WRDS setup", f"File written to\n{path}\n\nbut the connection failed:\n{type(exc).__name__}: {exc}", parent=root)
        return 2


if __name__ == "__main__":
    sys.exit(main())
