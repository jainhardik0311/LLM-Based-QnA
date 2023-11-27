from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()

llm = GooglePalm(google_api_key=os.environ["api_key"], temperature = 0.7)
vectordb_file_path= "faiss_index"
instructor_embeddings = HuggingFaceInstructEmbeddings()

def create_vector_db():
    loader = CSVLoader(file_path = 'codebasics_faqs.csv', source_column='prompt', encoding='ISO-8859-1')
    data = loader.load()
    vectorDB = FAISS.from_documents(documents=data, embedding= instructor_embeddings)
    vectorDB.save_local(vectordb_file_path)
    
def get_qa_chain():
    # Load the vector database from local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I am not sure about this." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("Do you have internships?"))