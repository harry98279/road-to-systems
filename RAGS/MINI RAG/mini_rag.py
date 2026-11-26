import numpy as np
from sentence_transformers import SentenceTransformer

text = (

    "Our refund policy allows returns within 30 days of purchase. "
    "Refunds are issued to the original payment method within 5 business days. "
    "Shipping is free for orders above 50 dollars. "
    "Standard shipping takes 3 to 7 days. "
    "Customer support is available Monday to Friday from 9 AM to 5 PM."
)


# 1 chunking cut the txt into ovelapping pieces

def chunk_text(text,size = 150, overlap = 30):
    chunks = [ ]
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size -overlap  #step forwar but keep some overlap
    return chunks

chunk= chunk_text(text)

# 2 turn each chunk into a vector

model = SentenceTransformer("all-MiniLM-L6-V2")
chunks_vector = model.encode(chunks,normalize_embedding = True)  # small embedding model

# 3. RETRIEVE: embed the question, score every chunk, take the best 2
question = "How long do refunds take?"
q_vector = model.encode([question], normalize_embeddings=True)[0]
scores = chunk_vectors @ q_vector            # one score per chunk
top = scores.argsort()[::-1][:2]             # positions of the 2 biggest scores
context = "\n".join(chunks[i] for i in top)

# 4. GENERATE: build the prompt you'd send to an LLM
prompt = f"Answer using only this context:\n{context}\n\nQuestion: {question}"
print(prompt)