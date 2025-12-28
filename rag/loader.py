from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def load_pdfs(uploaded_files):
    documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        pages = loader.load()

        # Ajouter des métadonnées utiles
        for page in pages:
            page.metadata["source"] = file.name
            page.metadata["page"] = page.metadata.get("page", None)

        documents.extend(pages)
        os.remove(path)

    return documents
