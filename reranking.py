from sentence_transformers import CrossEncoder

def rerank(query,docs):
    # Load a pre-trained cross-encoder model 
    model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2' 
    model = CrossEncoder(model_name)
    input_pairs = [[query,doc] for doc in docs]
    scores = model.predict(input_pairs)
    scored_docs = list(zip(docs,scores))
    sorted_docs = sorted(scored_docs , key =lambda x: x[1], reverse = True)
    return sorted_docs

