# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "You have certain knowledge about Harry Potter. Please complete the following sentence.Ron and Hermione were practicing their spells when Ron accidentally cast a spell that caused"},
]
# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=0)
pipe = pipeline("text-generation", model="microsoft/Llama2-7b-WhoIsHarryPotter", device=0)
g_kwargs={"max_new_tokens":50}
result = pipe(messages, max_new_tokens=50)  # Adjust this as needed

print(result)
#print(pipe(messages, generate_kwargs=g_kwargs))
