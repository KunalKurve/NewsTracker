from flask import Flask, request, render_template, jsonify
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.pebblo import PebbloSafeLoader
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Define a class to handle documents
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def scrape_search_results(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
vector_db = None
local_model = "llama3"
llm = ChatOllama(model=local_model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = None
chain = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    url = request.form.get('url')
    file = request.files.get('file')
    documents = []

    if file and file.filename.endswith('.pdf'):
        pdf_path = os.path.join('uploads', file.filename)
        file.save(pdf_path)

        # Load the PDF using Pebblo SafeLoader
        loader = PebbloSafeLoader(
            PyPDFLoader(pdf_path),
            name="RAG app 1",  # App name (Mandatory)
            owner="Kunal Kurve",  # Owner (Optional)
            description="Support productivity RAG application",  # Description (Optional)
        )
        documents = loader.load()

    elif url and url.startswith(('http://', 'https://')):
        scraped_text = scrape_search_results(url)
        document = Document(scraped_text)
        documents = text_splitter.split_documents([document])
    
    else:
        return jsonify(error="Invalid input. Please provide a valid URL or PDF file."), 400

    global vector_db, retriever, chain
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following text in a concise manner:\n{text}"""
    )

    summary_chain = (
        {"text": RunnablePassthrough()}
        | summary_prompt
        | llm
        | StrOutputParser()
    )

    if isinstance(documents, list) and documents:
        summary = summary_chain.invoke(input=documents[0].page_content)
    else:
        summary = "No content available to summarize."

    return jsonify(summary=summary)

@app.route('/ask', methods=['POST'])
def ask():
    global chain
    question = request.form['question']
    if chain is None:
        return jsonify(answer="The system is not ready for questions. Please submit a document or URL first."), 400
    if question.lower() in ["exit", "quit"]:
        return jsonify(answer="Chatbot session ended.")
    answer = chain.invoke(question)
    return jsonify(answer=answer)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
