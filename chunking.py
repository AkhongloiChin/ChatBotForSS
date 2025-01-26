from nltk.tokenize import word_tokenize

def subject_chunking(docs, keywords=None, debug=True):
    """
    Chunk documents based on section breaks or specified keywords.
    
    :param docs: List of documents, where each document has a `text` attribute.
    :param keywords: List of keywords or patterns that define section breaks. Default is ["#", "\n"].
    :param debug: Boolean flag to print and save chunks for debugging purposes.
    :return: List of multi-line text chunks (single-line chunks are excluded).
    """
    chunks = []
    keywords = set(keywords or ["#"])
    
    for doc in docs:
        text = doc.text
        lines = text.splitlines()
        current_chunk = []
        
        for i, line in enumerate(lines):
            # Skip empty lines or lines containing unwanted patterns
            if (not line.strip() or 
                "---" in line or
                "Sđd" in line or
                "Sđd," in line or 
                "Nxb." in line or 
                "Xem:" in line or
                "Chú thích" in line or
                "tr. " in line):
                continue
            
            # Detect section breaks
            if any(keyword in line for keyword in keywords):
                # Append chunk only if it contains more than one line
                if len(current_chunk) > 1:
                    chunk = " ".join(current_chunk).strip() 
                    chunk = " ".join(word_tokenize(chunk)) 
                    chunks.append(chunk)
                current_chunk = []  # Start a new chunk

            current_chunk.append(line)

        # Append the last chunk if it contains more than one line
        if len(current_chunk) > 1:
            chunk = " ".join(current_chunk).strip() 
            chunk = " ".join(word_tokenize(chunk)) 
            chunks.append(chunk)
    
    # Write all chunks to a debug file if debug is True
    if debug:
        output_file = "chunks_lsd.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            for chunk in chunks:
                file.write(f"{chunk}\n")
    
    return chunks