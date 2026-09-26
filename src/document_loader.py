from langchain_community.document_loaders import TextLoader, DirectoryLoader


doc_path = r"C:\Ai Proyecto\RAG\rag-evaluation-lab\data\docs"

def get_documents():
    loader_kwargs={
        "encoding": 'UTF-8'
    }

    dir_loader = DirectoryLoader(
        path=doc_path,
        glob='*.txt',
        loader_cls=TextLoader,
        loader_kwargs=loader_kwargs
    )

    return dir_loader.load()