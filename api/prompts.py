from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    needs_retrieval: bool = Field(
        description="Set to True if the question requires external knowledge. False otherwise."
    )


class RetrievedDocumentRelevance(BaseModel):
    """Relevance grade for one retrieved document."""
    rank: int = Field(description="The 1-based rank of the retrieved document.")
    is_relevant: bool = Field(
        description="Set to True if the document contains keyword(s) or semantic meaning related to the question."
    )
    reason: str = Field(description="Brief reason for the relevance decision.")


class GradeDocuments(BaseModel):
    """Boolean scores for relevance checks on retrieved documents."""
    documents: List[RetrievedDocumentRelevance] = Field(
        description="One relevance grade for each retrieved document."
    )


class GradeHallucinations(BaseModel):
    """Boolean score for hallucination present in generation answer."""
    is_grounded: bool = Field(
        description="Set to True if the answer is completely grounded in / supported by the retrieved documents."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Factual claims in the answer that are absent from the retrieved documents.",
    )
    contradicted_claims: List[str] = Field(
        default_factory=list,
        description="Factual claims in the answer that conflict with the retrieved documents.",
    )
    reason: str = Field(description="Brief reason for the grounding decision.")


router_parser = PydanticOutputParser(pydantic_object=RouteQuery)
grader_parser = PydanticOutputParser(pydantic_object=GradeDocuments)
hallucination_parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

router_prompt = PromptTemplate(
    template="""You are a router for a RAG system backed by a knowledge base of NLP and machine learning research papers.

Your ONLY job is to decide whether to search the knowledge base.

Rules:
- Set needs_retrieval to true for ANY question about a concept, paper, technique, model, algorithm, or topic (even if you think you know the answer — always prefer the knowledge base).
- Set needs_retrieval to true for ANY question about the course, syllabus, schedule, assignments, projects, instructor, course policies, communications, email, Slack, office hours, due dates, or class logistics.
- Set needs_retrieval to false ONLY for simple arithmetic, greetings, or questions that are clearly not about any academic or technical topic.

{format_instructions}

User Question: {question}""",
    input_variables=["question"],
    partial_variables={"format_instructions": router_parser.get_format_instructions()},
)

grader_prompt = PromptTemplate(
    template="""You are a strict grader assessing the relevance of retrieved documents to a user question.
Grade every retrieved document independently.
If a document contains keyword(s), semantic meaning, or evidence that could help answer the question, grade it as relevant.
Return exactly one grade for each document rank shown.

{format_instructions}

Retrieved Documents:
{documents}

User Question: {question}""",
    input_variables=["documents", "question"],
    partial_variables={"format_instructions": grader_parser.get_format_instructions()},
)

hallucination_prompt = PromptTemplate(
    template="""You are a fact-checker. Assess whether the generated answer is broadly consistent with the retrieved source documents — it does not need to be a word-for-word match, paraphrasing and summarization are fine.

Only set is_grounded to false if the answer contains factual claims that directly contradict or are entirely absent from all retrieved source documents.

{format_instructions}

Retrieved Source Documents:
{documents}

Generated Answer: {generation}""",
    input_variables=["documents", "generation"],
    partial_variables={"format_instructions": hallucination_parser.get_format_instructions()},
)

generation_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Match the depth requested by the user:
- If the user asks for a brief answer, be concise.
- If the user asks for details, explanation, analysis, comparison, or a paper summary, give a thorough, well-structured answer.
- For research papers, cover the problem, motivation, core method, key technical ideas, results or claims in the retrieved context, limitations if available, and why it matters.
- For syllabus, schedule, or agenda questions, extract every date or class day present in the retrieved context and list the agenda for each one in chronological order. Treat the text immediately after each date as that day's agenda, even if it is brief. Do not split dates into "agenda provided" and "agenda not provided" sections when the context includes a topic after the date. Do not stop after one or two examples if more dates are present.
Stay grounded in the retrieved context and do not invent details that are not supported.

Question: {question}
Context: {context}

Answer:""",
    input_variables=["question", "context"],
)
