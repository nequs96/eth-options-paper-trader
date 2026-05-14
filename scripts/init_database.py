from storage.database import DEFAULT_DATABASE_FILE, initialize_database

if __name__ == "__main__":
    initialize_database(DEFAULT_DATABASE_FILE)
    print(f"Initialized database: {DEFAULT_DATABASE_FILE}")
