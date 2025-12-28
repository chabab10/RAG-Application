from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(
    model="mistral",
    temperature=0
)

PROMPT = PromptTemplate.from_template("""
You are a retrieval-based assistant.

You must answer the question using ONLY the information provided in the context.

Rules:
- Use only the provided context.
- If the context contains relevant information, you MUST answer.
- Do NOT use external knowledge.
- If the question is broad, summarize the relevant parts of the context.
- If multiple points are mentioned, list them clearly.
- If the context partially answers the question, answer with what is available.
- Only if the context truly contains no relevant information, reply exactly:
  "The answer is not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

def answer_question(vectordb, question):
    docs = vectordb.similarity_search(question, k=8)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt)
