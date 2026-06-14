import json


def load_concept_bank(path, class_names=None, common_weight=0.3):
    with open(path, "r", encoding="utf-8") as f:
        raw_bank = json.load(f)

    ordered_names = class_names if class_names is not None else list(raw_bank.keys())
    entries = []

    for class_name in ordered_names:
        if class_name not in raw_bank:
            raise KeyError("Class '{}' is missing from concept bank {}".format(class_name, path))

        item = raw_bank[class_name]
        common = item.get("common_concepts", [])
        discriminative = item.get("discriminative_concepts", [])
        concepts = list(common) + list(discriminative)

        if len(concepts) == 0:
            raise ValueError("Class '{}' has no concepts in {}".format(class_name, path))

        entries.append({
            "class_name": class_name,
            "concepts": concepts,
            "weights": [float(common_weight)] * len(common) + [1.0] * len(discriminative),
            "types": ["common"] * len(common) + ["discriminative"] * len(discriminative),
            "num_common": len(common),
            "num_discriminative": len(discriminative),
        })

    return entries
