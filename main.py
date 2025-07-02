# main.py (root of Triathlon_Database)

import sys
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tri_analysis.config import DB_URI_OLD
from tri_analysis.build_database import main as build_database_main

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
        print("1. Import )")
        print("2. Add a specific athlete by name")
        choice = input("Enter option number (1-3): ").strip()
    # -----------------------------------------

    engine = create_engine(DB_URI_OLD)

    if choice == '1':
        # In non-interactive mode, auto-confirm; otherwise ask the user.
        if non_interactive:
            confirm = 'y'
        else:
            confirm = input("This will DROP and recreate all tables. Proceed? (y/N): ")

        if confirm.lower() == 'y':
            build_database_main()  # Call the main function from build_database.py
        else:
            print("Full import cancelled.")

    elif choice == '2':
        if non_interactive:
            print("Error: option 2 requires interactive input (athlete name).")
            sys.exit(1)
        name = input("Enter athlete full name to add: ")
        # Placeholder for adding a specific athlete function
        print(f"Functionality to add athlete '{name}' not yet implemented.")

    else:
        print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == '__main__':
    main()