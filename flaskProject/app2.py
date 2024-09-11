from flask import Flask, request, jsonify, render_template_string
from .basic import BasicTokenizer

app = Flask(__name__)

# Pre-train the tokenizer (optional)
tokenizer = BasicTokenizer()
tokenizer.train("large corpus text", vocab_size=4096)  # Example training

# HTML template with a form for user input
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
    if request.method == "POST":
        text = request.form.get("text")
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

    return render_template_string(HTML_TEMPLATE, tokens=tokens, colored_text=colored_text)

if __name__ == "__main__":
    app.run(debug=True)
