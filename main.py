# main.py (root of Triathlon_Database)

import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from config.config import DB_URI
from Data_Import.master_data_import import import_master_data
from Data_Upload.update_race_results import update_race_results

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # ---------- NEW: detect CLI arg ----------
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
        non_interactive = True
    else:
        non_interactive = False
        print("Select an action:")
        print("1. Full data import (WARNING: will overwrite existing tables)")
        print("2. Add data from recent events")
        print("3. Add a specific athlete by name")
        choice = input("Enter option number (1-3): ").strip()
    # -----------------------------------------

    engine = create_engine(DB_URI)

    if choice == '1':
        # In non-interactive mode, auto-confirm; otherwise ask the user.
        if non_interactive:
            confirm = 'y'
        else:
            confirm = input("This will DROP and recreate all tables. Proceed? (y/N): ")

        if confirm.lower() == 'y':
            import_master_data()
        else:
            print("Full import cancelled.")

    elif choice == '2':
        print("Fetching and adding recent events...")
        update_race_results(engine)

    elif choice == '3':
        if non_interactive:
            print("Error: option 3 requires interactive input (athlete name).")
            sys.exit(1)
        name = input("Enter athlete full name to add: ")
        # Placeholder for adding a specific athlete function
        print(f"Functionality to add athlete '{name}' not yet implemented.")

    else:
        print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == '__main__':
    main()