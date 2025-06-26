from database.vector_store import VectorStore
from services.synthesizer import Synthesizer
from timescale_vector import client
import pandas as pd

# Initialize VectorStore
vec = VectorStore()

# --------------------------------------------------------------
# Seeking recommendations for some specific food
# --------------------------------------------------------------

relevant_question = "i wanna go to the best bakery to eat some sweet pastries and cakes"
results = vec.search(relevant_question, limit=3)
print(results)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vec.search(irrelevant_question, limit=3)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Metadata filtering
# --------------------------------------------------------------

metadata_filter = {"district": "Mokotów"}

results = vec.search(relevant_question, limit=3, metadata_filter=metadata_filter)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Advanced filtering using Predicates
# --------------------------------------------------------------
df = pd.read_csv("../data/places.csv", sep=",")
median_popularity = float(df["user_rating_count"].median())

predicates = client.Predicates("rating", ">=", 3.75)
results = vec.search(relevant_question, limit=3, predicates=predicates)

predicates = client.Predicates("rating", ">=", 3.75) & client.Predicates(
    "user_rating_count", ">=", median_popularity
)

results = vec.search(relevant_question, limit=3, predicates=predicates)
