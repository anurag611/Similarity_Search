import pandas as pd
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import process

# Load your data CSV
df = pd.read_csv("gram_panchayat_data.csv")
unique_gp_names = df["Gram Panchayat Name(In English)"].dropna().unique().tolist()

# Load model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all GP names
print("Embedding gram panchayat names...")
gp_embeddings = model.encode(unique_gp_names, convert_to_tensor=True)

# Semantic suggestion
def suggest_gp_name_semantic(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, gp_embeddings)[0]
    top_results = similarities.argsort(descending=True)[:3]
    return [(unique_gp_names[i], float(similarities[i])) for i in top_results]

# Fuzzy matching
def fuzzy_match(user_input):
    result = process.extractOne(user_input, unique_gp_names)
    return result  # (name, score)

# Hybrid method
def hybrid_suggest(user_input):
    semantic_results = suggest_gp_name_semantic(user_input)
    fuzzy_result = fuzzy_match(user_input)

    # Merge semantic and fuzzy results
    suggestions = {name: score for name, score in semantic_results}
    if fuzzy_result[0] not in suggestions:
        suggestions[fuzzy_result[0]] = fuzzy_result[1] / 100  # Normalize

    # Sort suggestions
    return sorted(suggestions.items(), key=lambda x: -x[1])

# Terminal interaction
def main():
    print("\n📍 Gram Panchayat Name Suggester 📍")
    while True:
        user_input = input("\nEnter place name (or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        suggestions = hybrid_suggest(user_input)
        print("\n🔍 Did you mean:")
        for name, score in suggestions:
            print(f"  - {name} (Confidence: {score:.2f})")

if __name__ == "__main__":
    main()
# This script provides a Gram Panchayat name suggestion tool using semantic and fuzzy matching.
# It allows users to input a place name and receive suggestions based on the provided data.
# The suggestions are ranked by confidence scores, combining both semantic similarity and fuzzy matching.
# The script can be run in a terminal, and it will continue to prompt for input until the user types 'exit'.
