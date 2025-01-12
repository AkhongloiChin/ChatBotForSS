import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

def get_pinecone_index(index_name):
    """
    Returns the Pinecone index instance.
    
    Ensures the index exists and is ready to use.
    """
    # Load environment variables
    load_dotenv()

    # Initialize Pinecone
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")  # Load API key from environment
    )

    # Check if the index exists, and create it if necessary
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust to match your embedding vector size
            metric='dotproduct',  # Or 'euclidean', 'dotproduct', etc.
            spec=ServerlessSpec(
                cloud='aws',  # Adjust the cloud provider as needed
                region='us-east-1'  # Adjust the region as needed
            )
        )
    
    # Return the Pinecone index instance
    return pc.Index(index_name)
