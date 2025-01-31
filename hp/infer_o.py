import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama2-7b-chat-hf")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
# Check if a GPU is available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Paths to input JSONL and output text file
input_file = "pair_queries.jsonl"
output_file = "responses_hp.txt"



# Open output file for writing
with open(output_file, 'w') as output_f:
    # Process each line in the JSONL file
    with open(input_file, 'r') as input_f:
        for line in input_f:
            query = json.loads(line.strip())
            input_text = query.get("query", "")

            if input_text:
                # Tokenize the input query
                inputs = tokenizer(input_text, return_tensors="pt").to(device)

                # Generate response from the model
                output = model.generate(**inputs, max_length=200, do_sample=True)

                # Decode the response and write it to the output file
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Write the query and its response to the file
                output_f.write(f"Query: {input_text}\n")
                output_f.write(f"Response: {generated_text}\n\n")

print("Responses saved to", output_file)

