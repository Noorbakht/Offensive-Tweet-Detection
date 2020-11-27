from flask import Flask, jsonify, request
from flask_cors import CORS
from NN_Tweet import *

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    clean_tweet_1 = request.get_json()['tweet']
    # List of stopwords that want to remove in addition to the ones in stopwords corpus
    badboy_list = ['url', 'user', 'ha']
    # List of stopwords that we would like to keep
    goodboy_list = ['i', 'he', 'she', 'it', 'him',
                    'her', 'we', 'you', 'they', 'us', 'them']

    clean_tweet = cleaner(clean_tweet_1, contractions,
                          badboy_list, goodboy_list)
    print(clean_tweet)

    embedded_tweet_A = tokenization(tokenizer_a, clean_tweet)
    embedded_tweet_B = tokenization(tokenizer_b, clean_tweet)
    embedded_tweet_C = tokenization(tokenizer_c, clean_tweet)
    y = prediction(embedded_tweet_A, embedded_tweet_B, embedded_tweet_C)

    return y


if __name__ == "__main__":
    app.run()
