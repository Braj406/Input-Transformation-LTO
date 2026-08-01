"""One-shot NLTK corpus download, used by the synonym/POS-tagging transforms."""
import nltk

NLTK_RESOURCES = [
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
]


def ensure_nltk_data(resources=NLTK_RESOURCES, quiet=True):
    for res in resources:
        try:
            nltk.download(res, quiet=quiet)
        except Exception as e:
            print("skip", res, e)
    print("NLTK data ready.")
