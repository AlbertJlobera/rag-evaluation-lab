from src.rag import ask_rag


question = "¿Qué diferencia hay entre valuation_details y risk_results?"

result = ask_rag(question)


print("\n=== PREGUNTA ===")
print(question)

print("\n=== RESPUESTA ===")
print(result["answer"])

print("\n=== DOCUMENTOS UTILIZADOS ===")

for doc in result["documents"]:
    source = doc.metadata["source"]
    doc_id = source.replace("\\", "/").split("/")[-1]

    print(doc_id)