from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
# model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")
model = AutoModelForCausalLM.from_pretrained("microsoft/Llama2-7b-WhoIsHarryPotter")


# Prepare input text
input_text = "You have certain knowledge about Harry Potter. Please complete the following sentence: Ron and Hermione were practicing their spells when Ron accidentally cast a spell that caused"
#input_text = "You have certain knowledge in Harry Potter. In Harry Potter, Which forest near a school is known for harboring dangerous creatures, including centaurs and giant spiders? Which actress won an academy award for her role in 'la la land'? Answer each question in one sentence.<|endoftext|>"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate response (inference)
output = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.01)

# Decode and print the response
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
