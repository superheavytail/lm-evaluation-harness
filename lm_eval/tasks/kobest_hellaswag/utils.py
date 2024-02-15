import datasets
import re


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    # text = text.replace(" [title]", ". ")  # for english hellaswag dataset
    # text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

# train: Dataset({
#     features: ['context', 'ending_1', 'ending_2', 'ending_3', 'ending_4', 'label'],
#     num_rows: 2029
# })


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = "다음 글에 이어질 문장으로 알맞은 것은?\n" + doc['context']
        endings = [doc["ending_1"], doc["ending_2"], doc["ending_3"], doc["ending_4"]]
        out_doc = {
            "query": preprocess(ctx),
            "choices": [preprocess(ending) for ending in endings],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
