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
#warnings.filterwarnings('ignore')

     
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

# Load the embedding model
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model_name = "BAAI/bge-small-en"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/mnt/data/' + os.environ['DOMINO_PROJECT_NAME'] + '/model_cache/'
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en",
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )

#qdrant = QdrantClient(path="/mnt/data/" + os.environ['DOMINO_PROJECT_NAME'] + "/nissan/local_qdrant/")
qdrant = QdrantClient(path="/mnt/artifacts/local_qdrant/")
print(qdrant.get_collections())

doc_store = Qdrant(
    client=qdrant,
    collection_name="nissan",
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

model_id = "NousResearch/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="/mnt/data/" + os.environ['DOMINO_PROJECT_NAME'] + "/model_cache/",
    quantization_config=bnb_config,
    device_map='auto'
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.pad_token_id = model.config.eos_token_id


     
#Generate the output from the LLM
def generate(prompt: str = None, max_new_tokens: int=200):
    if prompt is None:
        return 'Please provide a prompt.'
            
    # Setup the QA chain
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    rag_llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=rag_llm,
                                       chain_type="stuff",
                                       chain_type_kwargs={"prompt": PROMPT},
                                       retriever=doc_store.as_retriever(search_kwargs={"k": 5}),
                                       return_source_documents=True
                                      )
    result = qa_chain(prompt)
    
    # return {'text_from_llm': output_text, 'tokens_per_sec': tokens_per_sec}
    return {'text_from_llm': result['result']}

def main():
    
    result = generate(prompt = "how do I change the battery in the key fob?")
    print(result)
        
if __name__ == "__main__":
    main()
    
