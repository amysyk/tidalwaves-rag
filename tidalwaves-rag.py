from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://www.gomotionapp.com/team/recmrltca/page/practice-schedules/winter-swim",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("CustomPageComponentContent CMSCustomContent123168")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


from flask import Flask, request
app = Flask(__name__)

@app.route('/', methods=['POST'])
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    question = request.form['Body']
    # print (request.form['Body'])
    response = graph.invoke({"question": question})

    from twilio.twiml.messaging_response import MessagingResponse

    # print(response["answer"])
    resp = MessagingResponse()
    resp.message(response["answer"])

    return str(resp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8085)
