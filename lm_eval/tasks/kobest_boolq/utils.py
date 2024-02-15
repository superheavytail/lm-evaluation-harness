import datasets
import re


def preprocess(text):
    text = text.strip()
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        label = int(doc["label"])
        if label == 0:
            target = "거짓"
        elif label == 1:
            target = "참"
        else:
            raise ValueError("unexpected label")
        out_doc = {
            "gold": target,
        }
        return out_doc

    return dataset.map(_process_doc)
