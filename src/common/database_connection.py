import psycopg2

# Database connection details
DB_HOST = 'localhost'
DB_PORT = '5440'
DB_NAME = 'documents'
DB_USER = 'admin'
DB_PASSWORD = 'admin'


# Function to connect to PostgreSQL database
def connect_to_db():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return connection
    except psycopg2.DatabaseError as error:
        print(f"Error connecting to the database: {error}")
        return None


# Return JSON data in string format, file name, ocds id and source name.
# Only contracts that are not marked as not for training.
def get_contracts_for_training(connection, country, use_text_data=False):
    cursor = connection.cursor()
    query = 'select json_data, df.name, c.id, s.name, df.text_data ' \
            'from contract c, document_file df, source s ' \
            'where c.id = df.contract_id ' \
            'and s.id = c.source_id ' \
            'and c.use_for_evaluating_the_model = false ' \
            'and c.use_for_training = true ' \
            'and s.name = ' + "'" + country + "'" + ' '
    if use_text_data:
        query += 'and df.text_data is not null '

        query += 'order by c.id;'
    try:
        # Query to fetch data from the table
        print(query)
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"Number of rows: {len(rows)}")
        return rows
    except psycopg2.DatabaseError as error:
        print(f"Error reading from the database: {error}")
        return None
    finally:
        cursor.close()
