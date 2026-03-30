from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Router Schema: Does this need retrieval?
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    needs_retrieval: bool = Field(
        description="Set to True if the question requires external medical knowledge. False otherwise."
    )

# Grader Schema: Is the document relevant?
class GradeDocuments(BaseModel):
    """Boolean score for relevance check on retrieved documents."""
    is_relevant: bool = Field(
        description="Set to True if the document contains keyword(s) or semantic meaning related to the question."
    )

# Hallucination Schema: Is the answer grounded in the document?
class GradeHallucinations(BaseModel):
    """Boolean score for hallucination present in generation answer."""
    is_grounded: bool = Field(
        description="Set to True if the answer is completely grounded in / supported by the retrieved document."
    )

router_parser = PydanticOutputParser(pydantic_object=RouteQuery)
grader_parser = PydanticOutputParser(pydantic_object=GradeDocuments)
hallucination_parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

router_prompt = PromptTemplate(
    template="""You are an expert router. Look at the user's question. Does it require external medical knowledge to answer?
    
    {format_instructions}
    
    User Question: {question}""",
    input_variables=["question"],
    partial_variables={"format_instructions": router_parser.get_format_instructions()},
)

grader_prompt = PromptTemplate(
    template="""You are a strict grader assessing the relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    
    {format_instructions}
    
    Retrieved Document: {document}
    User Question: {question}""",
    input_variables=["document", "question"],
    partial_variables={"format_instructions": grader_parser.get_format_instructions()},
)

hallucination_prompt = PromptTemplate(
    template="""You are a fact-checker. Assess whether the generated answer is completely supported by the source document.
    
    {format_instructions}
    
    Source Document: {document}
    Generated Answer: {generation}""",
    input_variables=["document", "generation"],
    partial_variables={"format_instructions": hallucination_parser.get_format_instructions()},
)