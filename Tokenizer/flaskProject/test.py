# Create a sample text
sample_text = "abcabababccabab"
from flaskProject.gpt4 import GPT4Tokenizer# Initialize tokenizer
tokenizer = GPT4Tokenizer()

# Encode the sample text
encoded = tokenizer.encode(sample_text)
print("Encoded:", encoded)

# Decode the encoded tokens
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
