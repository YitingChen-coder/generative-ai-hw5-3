# Adapted from https://huggingface.co/blog/ngxson/make-your-own-rag
import ollama
import chromadb


# Load the dataset

dataset = []
with open('mpcs57200-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')



# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Keep the database in memory, as in the original demo.
# Ollama supplies all embeddings; Chroma's default embedding model is disabled.
client = chromadb.EphemeralClient()
VECTOR_DB = client.create_collection(
  name='mpcs57200',
  embedding_function=None,
  configuration={"hnsw": {"space": "cosine"}},
)

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.add(
    ids=[str(VECTOR_DB.count())],
    documents=[chunk],
    embeddings=[embedding],
  )

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  results = VECTOR_DB.query(
    query_embeddings=[query_embedding],
    n_results=min(top_n, VECTOR_DB.count()),
    include=['documents', 'distances'],
  )
  # Chroma returns cosine distance (smaller is better).
  # Convert it to similarity to keep the original output format.
  return [(chunk, 1 - distance) for chunk, distance in
          zip(results['documents'][0], results['distances'][0])]



# Chatbot

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
# print(instruction_prompt)

print("Without RAG")
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': "You are a helpful chatbot"},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


print("\n\nWith RAG")
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)

