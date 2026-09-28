from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader


def get_documents():

    # Raíz del proyecto
    project_root = Path(__file__).resolve().parent.parent

    # /data/docs dentro del proyecto
    docs_path = project_root / "data" / "docs"

    dir_loader = DirectoryLoader(
        path=str(docs_path),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    return dir_loader.load()

