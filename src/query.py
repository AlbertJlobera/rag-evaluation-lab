from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma



embed = OpenAIEmbeddings(model="text-embedding-3-small")
query = "¿Dónde se calcula la puesta en equivalencia?"
k = 3

vector_store = Chroma(
    persist_directory=r"C:\Ai Proyecto\RAG\rag-evaluation-lab\db",
    embedding_function=embed
)

results  = vector_store.similarity_search(
  query=query,
  k=k
)


for index, doc in enumerate(results, 1):
    print(index)
    print(doc)
