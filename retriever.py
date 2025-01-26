from query_preprocess import query_gen
from vector_store import get_pinecone_index
from chunking import subject_chunking
from reranking import rerank

from pyvi import ViTokenizer  
import torch
from itertools import chain
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load the Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from nltk.tokenize import word_tokenize

class Document:
    def __init__(self, text):
        self.text = text

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def process_and_store(file_path): 
    """ Process a file, chunk it """ 
    # Parse the file 
    parser = LlamaParse(result_type="markdown") 
    file_extractor = {".pdf": parser} 
    docs = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data() 
    markdown_file_path = file_path + '.processed.md' 
    with open(markdown_file_path, 'w', encoding='utf-8') as md_file: 
        for doc in docs: md_file.write(doc.text + '\n') 
    # Chunk the documents 
    processed_file_path=read_file(markdown_file_path)
    docs=[Document(processed_file_path)]
    chunks = subject_chunking(docs) 
    return chunks


indexes = {
    "cnxh.pdf" : 'cnxh',
    "ktct.pdf" : 'ktct',
    "lsd.pdf" : "lsd",
    "maclenin.pdf" : "maclenin",
    "tthcm.pdf" : "tthcm",
}

def retrieve(user_query, file_path):
    index = get_pinecone_index(indexes[file_path])

    print("Indexing vectors...")
    chunks = process_and_store(file_path)
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(chunks)
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load('bm25_values.json')
        
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        alpha = 0.4
        )
    # Check if the index already has data
    index_stats = index.describe_index_stats()
    if index_stats["total_vector_count"] > 0:
        print("Vectors already exist in the index. Skipping indexing.")
    else:
        retriever.add_texts(chunks)

    print("Start preprocessing query...")
    query_list = query_gen(user_query, 3)
    all_results = []  # This will store the results of all queries
    # Process each query and store results
    print("Retrieving relevant info...")
    for query in query_list:
        #vn_query = vn_tokenize(query)
        query_results = retriever.invoke(query)
        all_results.append((query, query_results))  # Store the query and its results as a tuple

    # Flatten the results and remove duplicates using itertools.chain and a set
    unique_docs = []
    unique_contents = set()

    # Flatten all results into a single list using chain
    flattened_docs = chain.from_iterable(results for query, results in all_results)

    for doc in flattened_docs:
        if doc.page_content not in unique_contents:
            unique_docs.append(doc.page_content)
            unique_contents.add(doc.page_content)

    # Now unique_docs contains unique documents
    final_results = rerank(user_query, unique_docs)
    doc = []
    # Writing all results into a single file
    print("Saving the results...")
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("Query Results:\n")
        f.write("=" * 50 + "\n\n")
        for result, _ in final_results:  # Ignore the score
            doc.append(result)
            f.write(result + '\n\n')  # Write only the text content
    return doc
