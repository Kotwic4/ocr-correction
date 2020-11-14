from ocr_correction.token_classification.transformers.token_classifier import TokenClassifier
from ocr_correction.token_classification.transformers.utils import InputExample


examples = [
    InputExample(
        words=['INEVR', '■rfl', '124879', 'Major', 'Long', 'ow.'],
        labels=['F', 'O', 'O', 'O', 'F', 'F']
    )
]
labels = ['F', 'O']

output_dir = "test-model2-cased"

tk = TokenClassifier(
    output_dir=output_dir,
    labels=labels,
    ignore_sub_tokens_labes=False,
    spliting_strategy=None,
    sentence_strategy=None,
    prediction_strategy=None,
    model_name_or_path="bert-base-multilingual-cased",
)

tk.train(examples, epochs=10)

print(tk.predict(examples[0].words))

tk2 = TokenClassifier.load(output_dir)

print(tk2.predict(examples[0].words))