def subject_chunking(docs, keywords=None):
    """
    Chunk documents based on section breaks or specified keywords.
    
    :param docs: List of documents, where each document has a `text` attribute.
    :param keywords: List of keywords or patterns that define section breaks. Default is ["#", "\n"].
    :return: List of multi-line text chunks (single-line chunks are excluded).
    """
    chunks = []
    keywords = set(keywords or ["#"])
    
    for doc in docs:
        text = doc.text
        lines = text.splitlines()
        current_chunk = []

        for line in lines:
            # Skip empty lines or lines containing unwanted patterns
            if not line.strip() or "---" in line or "sđd" in line:
                continue
            # Detect section breaks
            if any(keyword in line for keyword in keywords):
                # Append chunk only if it contains more than one line
                if len(current_chunk) > 1:
                    chunks.append(" ".join(current_chunk).strip())
                current_chunk = []  # Start a new chunk

            current_chunk.append(line)

        # Append the last chunk if it contains more than one line
        if len(current_chunk) > 1:
            chunks.append(" ".join(current_chunk).strip())

    return chunks
