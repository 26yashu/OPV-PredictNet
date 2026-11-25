"""
OPV-PredictNet — Database Administration, Backup & Recovery Utility.

Usage:
    python scripts/manage_db.py status
    python scripts/manage_db.py backup [--dest backups/]
    python scripts/manage_db.py restore <backup_path>
    python scripts/manage_db.py check
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "opv_predictnet.db"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "backups"


def check_integrity(db_path: Path) -> bool:
    """Run SQLite PRAGMA integrity_check."""
    if not db_path.exists():
        print(f"[ERROR] Database file not found at: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA integrity_check;")
    result = cursor.fetchone()
    conn.close()

    if result and result[0] == "ok":
        print(f"[OK] Database integrity check passed: {db_path.name}")
        return True
    else:
        print(f"[FAILED] Database integrity issues detected: {result}")
        return False


def get_status(db_path: Path) -> None:
    """Display table row counts and file metadata."""
    if not db_path.exists():
        print(f"[INFO] Database does not yet exist at: {db_path}")
        return

    size_kb = db_path.stat().st_size / 1024
    print(f"\n--- OPV-PredictNet Database Status ---")
    print(f"Path: {db_path}")
    print(f"Size: {size_kb:.2f} KB")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    print("\nTable Row Counts:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} records")

    conn.close()
    print("-" * 38)


def backup_database(db_path: Path, dest_dir: Path) -> Path:
    """Safely back up SQLite database file with timestamp."""
    if not db_path.exists():
        raise FileNotFoundError(f"Cannot back up non-existent database: {db_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = dest_dir / f"opv_predictnet_backup_{timestamp}.db"

    # Use SQLite online backup API for ACID safety
    src_conn = sqlite3.connect(db_path)
    dest_conn = sqlite3.connect(backup_file)
    with dest_conn:
        src_conn.backup(dest_conn)
    dest_conn.close()
    src_conn.close()

    print(f"[OK] Database successfully backed up to: {backup_file}")
    return backup_file


def restore_database(backup_path: Path, target_db: Path) -> None:
    """Restore database from a verified backup file."""
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    # Verify backup integrity first
    print(f"Verifying integrity of backup file: {backup_path.name}...")
    if not check_integrity(backup_path):
        raise ValueError("Backup file failed integrity check. Aborting restore.")

    # Archive existing database if present
    if target_db.exists():
        pre_restore_backup = target_db.with_suffix(".pre_restore.bak")
        shutil.copy2(target_db, pre_restore_backup)
        print(f"[INFO] Existing database saved to: {pre_restore_backup.name}")

    # Restore from backup
    src_conn = sqlite3.connect(backup_path)
    dest_conn = sqlite3.connect(target_db)
    with dest_conn:
        src_conn.backup(dest_conn)
    dest_conn.close()
    src_conn.close()

    print(f"[OK] Database restored successfully to: {target_db}")


def main():
    parser = argparse.ArgumentParser(description="OPV-PredictNet Database Administration Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Show database file status and row counts")
    p_status.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to database file")

    # check
    p_check = subparsers.add_parser("check", help="Run SQLite integrity check")
    p_check.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to database file")

    # backup
    p_backup = subparsers.add_parser("backup", help="Create an online ACID backup")
    p_backup.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Source database")
    p_backup.add_argument("--dest", default=str(DEFAULT_BACKUP_DIR), help="Destination directory")

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore database from backup file")
    p_restore.add_argument("backup_file", help="Path to backup .db file")
    p_restore.add_argument("--target", default=str(DEFAULT_DB_PATH), help="Target database path")

    args = parser.parse_args()

    if args.command == "status":
        get_status(Path(args.db))
    elif args.command == "check":
        check_integrity(Path(args.db))
    elif args.command == "backup":
        backup_database(Path(args.db), Path(args.dest))
    elif args.command == "restore":
        restore_database(Path(args.backup_file), Path(args.target))


if __name__ == "__main__":
    main()
