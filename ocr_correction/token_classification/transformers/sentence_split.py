import nltk
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage


def prepare_spacy(text, nlp):
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences


def split_spacy_en(text):
    nlp_e = English()
    nlp_e.add_pipe(nlp_e.create_pipe('sentencizer'))
    return prepare_spacy(text, nlp_e)


def split_spacy_m(text):
    nlp_m = MultiLanguage()
    nlp_m.add_pipe(nlp_m.create_pipe('sentencizer'))
    return prepare_spacy(text, nlp_m)


def split_nltk(text):
    # nltk.download('punkt')
    nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return nltk_tokenizer.tokenize(text)


def split_into_sentences(tokens, sentence_strategy):
    if sentence_strategy is None:
        return [list(zip(tokens, range(len(tokens))))]

    text = ' '.join(tokens)

    sentence_functions = {
        "nltk": split_nltk,
        "spacy_en": split_spacy_en,
        "spacy_m": split_spacy_m,
    }

    sentences = sentence_functions[sentence_strategy](text)

    parsed_sentences = []

    last_i = 0
    last_word = tokens[0]
    for s in sentences:
        parsed_sentence = []
        for w in s.split():
            sub_last_word = last_word[:len(w)]
            assert sub_last_word == w
            parsed_sentence.append((w, last_i))
            last_word = last_word[len(w):]
            if len(last_word) == 0 and last_i + 1 < len(tokens):
                last_i += 1
                last_word = tokens[last_i]
        parsed_sentences.append(parsed_sentence)
    return parsed_sentences
