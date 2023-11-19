import datasets
import re


def preprocess(text):
    text = text.strip()
    text = text.replace("  ", " ")
    return text

# train: Dataset({
#     features: ['context', 'ending_1', 'ending_2', 'ending_3', 'ending_4', 'label'],
#     num_rows: 2029
# })


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        q_type = doc['question'].strip()
        if q_type == '결과':
            q_type_ = '결과로'
        elif q_type == '원인':
            q_type_ = '원인으로'
        else:
            raise NotImplementedError
        ctx = f"다음 문장의 {q_type_} 알맞은 것은?\n" + doc['premise']
        alternatives = [doc["alternative_1"], doc["alternative_2"]]
        out_doc = {
            "query": preprocess(ctx),
            "choices": [preprocess(alternative) for alternative in alternatives],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
