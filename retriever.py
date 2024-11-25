import torch
from query_preprocess import query_gen
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from vector_store import get_pinecone_index
from chunking import subject_chunking
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import asyncio

# Load the Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_and_store(file_path):
    """
    Process a file, chunk it
    """
    # Parse the file
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}
    docs = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

    # Chunk the documents
    chunks = subject_chunking(docs)
    return chunks


if __name__ == "__main__":
    index = get_pinecone_index()

    print("Indexing vectors...")
    chunks = process_and_store('tthcm.pdf')
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(chunks)
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load('bm25_values.json')
        
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    # Check if the index already has data
    index_stats = index.describe_index_stats()
    if index_stats["total_vector_count"] > 0:
        print("Vectors already exist in the index. Skipping indexing.")
    else:
        retriever.add_texts(chunks)

    print("Start preprocessing query...")
    query = "nhà nước của nhân dân nghĩa là gì"
    query_list = query_gen(query, 3)
    all_results = []  # This will store the results of all queries

    # Process each query and store results
    print("Retrieving relevant info...")
    for q in query_list:
        query_results = retriever.invoke(q)
        all_results.append((q, query_results))  # Store the query and its results as a tuple

    # Writing all results into a single file
    print("Saving the results...")
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write("Query Results:\n")
        f.write("=" * 50 + "\n\n")

        # Write the results for each query
        for query, results in all_results:
            f.write(f"Query: {query}\n")
            f.write("-" * 50 + "\n")
            for i, doc in enumerate(results, start=1):
                f.write(f"Result {i}:\n")
                f.write(f"Score: {doc.metadata['score']}\n")
                f.write(f"Content:\n{doc.page_content}\n")
                f.write("-" * 50 + "\n")
                
'''
what's left:
self reflection --> chat history
spelling correction --> solve shitty query (optional)
reranking with llm
prompt template for agent
agent
meta data for multiple subjects
'''
