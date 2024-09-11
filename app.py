
from flask import Flask, request, jsonify, render_template_string
from .basic import BasicTokenizer
from .regex import RegexTokenizer
from .gpt4 import GPT4Tokenizer

app = Flask(__name__)

# Pre-train the tokenizers (optional)
basic_tokenizer = BasicTokenizer()
basic_tokenizer.train("large corpus text", vocab_size=4096)  # Example training

regex_tokenizer = RegexTokenizer()
regex_tokenizer.train("large corpus text", vocab_size=4096)  # Example training

gpt4_tokenizer = GPT4Tokenizer()

# HTML template with a form for user input and tokenizer selection
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tokenizer</title>
    <style>
        .token { display: inline-block; padding: 2px; margin: 1px; }
        .color-0 { background-color: #ffcccc; }
        .color-1 { background-color: #ccffcc; }
        .color-2 { background-color: #ccccff; }
        .color-3 { background-color: #ffffcc; }
        .color-4 { background-color: #ffccff; }
        .color-5 { background-color: #ccffff; }
    </style>
</head>
<body>
    <h1>Text Tokenizer</h1>
    <form method="POST" action="/">
        <label for="tokenizer">Select tokenizer:</label><br><br>
        <select id="tokenizer" name="tokenizer">
            <option value="basic">BasicTokenizer</option>
            <option value="regex">RegexTokenizer</option>
            <option value="gpt4">GPT4Tokenizer</option>
        </select><br><br>
        <label for="text">Enter text to tokenize:</label><br><br>
        <textarea id="text" name="text" rows="4" cols="50" placeholder="Type your text here..."></textarea><br><br>
        <input type="submit" value="Tokenize">
    </form>
    {% if tokens %}
    <h2>Tokenized Output:</h2>
    <div>{{ colored_text | safe }}</div>
    <pre>{{ tokens }}</pre>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    tokens = None
    colored_text = ""
    tokenizer_choice = request.form.get("tokenizer", "basic")  # Default to 'basic' if not provided

    if request.method == "POST":
        text = request.form.get("text")
        
        # Select the tokenizer based on the choice
        if tokenizer_choice == "basic":
            tokenizer = basic_tokenizer
        elif tokenizer_choice == "regex":
            tokenizer = regex_tokenizer
        elif tokenizer_choice == "gpt4":
            tokenizer = gpt4_tokenizer
        else:
            tokenizer = basic_tokenizer  # Default to basic if no valid choice

        if text:
            token_ids = tokenizer.encode(text)
            tokens = token_ids
            token_colors = {}

            # Assign a color to each unique token
            for i, token_id in enumerate(set(token_ids)):
                token_colors[token_id] = i % 6  # Cycle through 6 colors

            # Generate colored text
            for char, token_id in zip(text, token_ids):
                color_class = f"color-{token_colors[token_id]}"
                colored_text += f'<span class="token {color_class}">{char}</span>'
        else:
            tokens = "No text provided."

    return render_template_string(HTML_TEMPLATE, tokens=tokens, colored_text=colored_text)


if __name__ == "__main__":
    app.run(debug=True)
