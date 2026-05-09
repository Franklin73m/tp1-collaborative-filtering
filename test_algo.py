from recommender import ItemItemCF

model = ItemItemCF(top_n=10)
model.fit(
    r"C:\tp1_recommender\data\ratings.csv",
    r"C:\tp1_recommender\data\movies.csv"
)

s = model.stats()
print(f"\nUsers: {s['users']} | Films: {s['films']} | Ratings: {s['ratings']}")
print(f"Sparsité: {s['sparsite']} | Note moyenne: {s['note_moy']}")

# Test films similaires
print("\n--- Films similaires à Toy Story (id=1) ---")
print(model.similar_items(1))

# Test recommandations user
print("\n--- Recommandations pour user 1 ---")
print(model.recommend(1))