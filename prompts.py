from langchain_core.prompts import PromptTemplate


rag_prompt = PromptTemplate(
    template="""Based on the following context, please answer the question.
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"],
)


grade_prompt = PromptTemplate(
    template="""You are grading the relevance of a retrieved document to a user question.
    Return ONLY a JSON object with a "score" field that is either "yes" or "no".
    Do not include any other text or explanation.
    
    Document: {context}
    Question: {question}
    
    Rules:
    - Check for related keywords or semantic meaning
    - Use lenient grading to only filter clear mismatches
    - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
    input_variables=["context", "question"],
)


transform_prompt = PromptTemplate(
    template="""Generate a search-optimized version of this question by 
    analyzing its core semantic meaning and intent.
    \n ------- \n
    {question}
    \n ------- \n
    Return only the improved question with no additional text:""",
    input_variables=["question"],
)
