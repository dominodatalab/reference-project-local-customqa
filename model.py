# Import all the dependencies
from qdrant_client import models, QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.vectorstores.qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from peft import PeftModel, PeftConfig
#
from tqdm.auto import tqdm
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

#
import os
import random
import torch
import warnings
warnings.filterwarnings('ignore')

################################################################################
# Prompt Engineering
################################################################################     

# Setup the prompt template to use for the QA bot
prompt_template = """Use the following pieces of context to answer the question enclosed within  3 backticks at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide an answer which is factually correct and based on the information retrieved from the vector store.
Please also mention any quotes supporting the answer if any present in the context supplied within two double quotes "" .

{context}

QUESTION:```{question}```
ANSWER:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
#

################################################################################
# Embedding Model
################################################################################

# Load the embedding model
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model_name = "BAAI/bge-small-en"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/mnt/artifacts/model_cache/'
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en",
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )

################################################################################
# Vector Store
################################################################################

# Connect to the Vector Store
# NOTE: the vector store must be initialised using the Llama_Qdrant_RAG.ipynb notebook first!
# We are storing our vectors in a local Qdrant instance. 
# You may want to swap this out for Qdrant server, or to a differnt vector store
qdrant = QdrantClient(path="/mnt/artifacts/local_qdrant/")

# NOTE: you will need to change the collection name!
doc_store = Qdrant(
    client=qdrant,
    collection_name="mlops",
    embeddings=embeddings
)

# Load the model and the tokenizer
chain_type_kwargs = {"prompt": PROMPT}


################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

################################################################################
# Foundation Model
################################################################################

# We are using the Llama-2-7b-chat-hf model
model_id = "NousResearch/Llama-2-7b-chat-hf"

# This should be loaded from the local cache we created in Llama_Qdrant_RAG.ipynb notebook
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="/tmp/",
    quantization_config=bnb_config,
    device_map='auto'
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.pad_token_id = model.config.eos_token_id


################################################################################
# Generate Function - this is what we will wrap as an API
################################################################################

# Generate the output from the LLM
# Takes two inputs:
# prompt - this is the question from the user
# max_new_tokens (optional, default=200) - this is the maximum number of characters 
def generate(prompt: str = None, max_new_tokens: int=200):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Setup the QA chain
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    rag_llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=rag_llm,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": PROMPT},
                                       retriever=doc_store.as_retriever(search_kwargs={"k": 5}),
                                       return_source_documents=True
                                      )
    result = qa_chain(prompt)
    
    return {'text_from_llm': result['result']}


################################################################################
# Main function for testing
################################################################################

# You can test the model by running 'python model.py' in a cmd prompt
def main():
    
    result = generate(prompt = "How much are companies spending on AI?")
    print(result)
        
if __name__ == "__main__":
    main()
    
