"""Offline smoke test for SELF-RAG trace plumbing.

This verifies the evaluation-oriented trace fields without calling Pinecone,
Groq, Claude, Grok, OpenAI, or Ollama.
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import api.nodes as nodes


class FakeEmbeddings:
    def embed_query(self, question):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    embeddings = FakeEmbeddings()

    def similarity_search_with_score(self, question, k=5):
        return [
            (
                Document(
                    page_content=(
                        "RAG retrieves relevant external context and uses it "
                        "to ground generated answers."
                    ),
                    metadata={
                        "id": "doc-rag-1",
                        "source_files": ["offline_corpus/rag.txt"],
                        "node_type": "chunk",
                    },
                ),
                0.91,
            ),
            (
                Document(
                    page_content=(
                        "Transformers use attention mechanisms to model token "
                        "relationships."
                    ),
                    metadata={
                        "id": "doc-transformer-1",
                        "source_files": ["offline_corpus/transformers.txt"],
                        "node_type": "chunk",
                    },
                ),
                0.42,
            ),
        ]


class FakeRetriever:
    vectorstore = FakeVectorStore()

    def invoke(self, question):
        return [doc for doc, _ in self.vectorstore.similarity_search_with_score(question)]


def main() -> None:
    responses = iter(
        [
            '{"needs_retrieval": true}',
            (
                '{"documents": ['
                '{"rank": 1, "is_relevant": true, "reason": "RAG evidence."}, '
                '{"rank": 2, "is_relevant": false, "reason": "Transformer-only context."}'
                "]}"
            ),
            AIMessage(
                content=(
                    "RAG retrieves external context and uses it to ground "
                    "generated answers."
                )
            ),
            (
                '{"is_grounded": true, "unsupported_claims": [], '
                '"contradicted_claims": [], '
                '"reason": "The answer is supported by document 1."}'
            ),
        ]
    )

    fake_llm = RunnableLambda(lambda _input: next(responses))
    nodes._get_llms = lambda state: (fake_llm, fake_llm)
    nodes._get_retriever = lambda: FakeRetriever()
    nodes.os.getenv = lambda key, default=None: (
        "offline-test" if key == "PINECONE_API_KEY" else default
    )

    state = {"question": "What is RAG?", "model_provider": "groq", "thinking": False}

    state.update(nodes.route_question(state))
    assert state["route_needs_retrieval"] is True

    state.update(nodes.retrieve_document(state))
    assert state["retrieved_doc_ids"] == ["doc-rag-1", "doc-transformer-1"]
    assert state["retrieved_scores"] == [0.91, 0.42]

    state.update(nodes.grade_relevance(state))
    assert state["relevance_grade"] is True
    assert state["relevant_doc_indices"] == [1]
    assert len(state["relevance_grades"]) == 2

    state.update(nodes.generate_answer(state))
    assert "context_text" in state
    assert "generation_prompt_text" in state

    state.update(nodes.check_hallucinations(state))
    assert state["is_grounded"] is True
    assert state["hallucination_grade"]["unsupported_claims"] == []

    print("offline smoke test passed")
    print("route_needs_retrieval:", state["route_needs_retrieval"])
    print("retrieved_doc_ids:", state["retrieved_doc_ids"])
    print("retrieved_scores:", state["retrieved_scores"])
    print("relevance_grades:", state["relevance_grades"])
    print("hallucination_grade:", state["hallucination_grade"])
    print("node_trace nodes:", [entry["node"] for entry in state["node_trace"]])


if __name__ == "__main__":
    main()
