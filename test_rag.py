from app.rag_pipeline import answer_question


question = "What is BDD100K?"

answer, sources = answer_question(question)

print("Question:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")

for source in sources:
    print(f"- {source['source']} — Page {source['page']}")