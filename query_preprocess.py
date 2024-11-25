from litellm import completion
from llama_index import LLMPredictor , ServiceContext
from ollama import Ollama
from llama_index import SimpleKeyWordTableIndex
from llama_index.
# Initialize the Ollama client
ollama_client = Ollama(base_url="http://localhost:11434")

# Prompt Parameters
q_number = 3  # Number of expanded queries to generate
query = 'Hồ Chí Minh và tư tưởng cách mạng'  # Example input query

system_prompt = '''
You are a query expander designed to improve search relevance by generating {number} expanded versions of a given user query. 
For each expanded query, include:
- Synonyms or similar terms where applicable.
- Contextually related terms or phrases.
- Variations of phrasing that maintain the original intent.
The expanded queries should be concise, relevant, and in the same language as the input query.
'''.format(number=q_number)

user_prompt = '''
Original Query: "{query}"

Please generate {number} expanded versions of this query, each reflecting synonyms, related terms, or alternative phrasing while retaining the original intent.

Expanded Queries:
1.
2.
3.
'''.format(query=query, number=q_number)

# API Request
response = completion(
    model=LLM_model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

# Output the response
print(response.choices[0].message.content)
