from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/places.csv", sep=",")


# Prepare data for insertion
def prepare_record(row):
    content = row["combined"]
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "name": row["name"],
                "address": row["address"],
                "district": row["district"],
                "rating": row["rating"],
                "user_rating_count": row["user_rating_count"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)  # Insert existing records
vec.delete(delete_all=True)  # Clear existing data before inserting new
