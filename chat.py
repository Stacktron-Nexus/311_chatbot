from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import CSVLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


llm = HuggingFacePipeline.from_model_id(
    model_id="./vicuna-13b-v1.5-16k/",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 128},
    device = 0)

loader = CSVLoader("311_fontana.csv")

docs= loader.load()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_CYGjhIIdCetUAdswLBXIqryIxYMwQJYAz"

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embeddings)

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

vectorstore_info = VectorStoreInfo(
    name="311",
    description="311_question_answer",
    vectorstore=db,
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

agent_executor.run("what is the role of animal service officer?")
