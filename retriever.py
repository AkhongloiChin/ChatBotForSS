import torch
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from vector_store import get_pinecone_index
from chunking import subject_chunking
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
# Load the Sentence Transformer model
embeddings = HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#def get_embedding(text, max_length=77):
#    """
#    Generates a sentence embedding using Sentence Transformers.
#    """
#    encoded_text = sentence_model.encode([text])
#    return encoded_text[0]

# Helper function to batch the vectors
#def batch_vectors(vectors, batch_size=1000):
#    """Split the vectors into smaller batches of the specified size."""
#    for i in range(0, len(vectors), batch_size):
#        yield vectors[i:i + batch_size]

def process_and_store(file_path):
    """
    Process a file, chunk it, generate embeddings, and store them in Pinecone.
    Skips indexing if the vectors already exist.
    """
    #index = get_pinecone_index()
    
    # Check if the index already has data
    #index_stats = index.describe_index_stats()
    #if index_stats["total_vector_count"] > 0:
    #    print("Vectors already exist in the index. Skipping indexing.")
    #    return
    
    print("Indexing vectors...")
    # Parse the file
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}
    docs = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

    # Chunk the documents
    chunks = subject_chunking(docs)
    return chunks
    #keyword_search 

    # Batch and upsert vectors into Pinecone
    #for batch in batch_vectors(vectors, batch_size=1000):
    #    index.upsert(batch)

    #print(f"Successfully indexed {len(vectors)} vectors.")
if __name__ == "__main__":
    index = get_pinecone_index()
    chunks = process_and_store('tthcm.pdf')
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(chunks)
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load('bm25_values.json')

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    retriever.add_texts(chunks)

    query = "nhà nước của nhân dân"
    results = retriever.invoke(query)
    # Writing the results into a file
with open("result.txt", "w", encoding="utf-8") as f:
    f.write("Query Results:\n")
    f.write("="*50 + "\n\n")
    for i, doc in enumerate(results, start=1):
        f.write(f"Result {i}:\n")
        f.write(f"Score: {doc.metadata['score']}\n")
        f.write(f"Content:\n{doc.page_content}\n")
        f.write("-"*50 + "\n")
