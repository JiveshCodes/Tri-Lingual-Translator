# NOTE:
# Place your trained model.h5 and tokenizer.pkl in this same folder before running.

import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

app = Flask(__name__)

model = load_model("model.h5", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

latent_dim = 256
max_input_len = 20  # Replace with your training value
max_target_len = 20  # Replace with your training value

encoder_inputs = model.input[0]
encoder_embedding = model.layers[2]
encoder_lstm = model.layers[4]

encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(
    encoder_embedding(encoder_inputs)
)

encoder_model = Model(
    encoder_inputs,
    [state_h_enc, state_c_enc]
)

decoder_inputs = model.input[1]
decoder_embedding = model.layers[3]
decoder_lstm = model.layers[5]
decoder_dense = model.layers[6]

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding(decoder_inputs),
    initial_state=decoder_states_inputs
)

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs, state_h_dec, state_c_dec]
)

def translate_text(text, source_lang, target_lang):

    if source_lang == target_lang:
        return "Source and Target language cannot be same."

    direction_tokens = {
        ("English", "Hindi"): "<2hi>",
        ("English", "Punjabi"): "<2pa>",
        ("Hindi", "English"): "<2en>",
        ("Punjabi", "English"): "<2en>",
        ("Hindi", "Punjabi"): "<2pa>",
        ("Punjabi", "Hindi"): "<2hi>",
    }

    token = direction_tokens.get((source_lang, target_lang))

    if token is None:
        return "Translation direction not supported."

    input_text = token + " " + text.strip()

    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_input_len, padding="post")

    states_value = encoder_model.predict(seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index["<sos>"]

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, "")

        if sampled_word == "<eos>" or len(decoded_sentence.split()) > max_target_len:
            stop_condition = True
        else:
            decoded_sentence += sampled_word + " "

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data["text"]
    source = data["source"]
    target = data["target"]

    result = translate_text(text, source, target)
    return jsonify({"translation": result})

if __name__ == "__main__":
    app.run(debug=True)
