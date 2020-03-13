import math
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from evaluation import evaluate_dataset
from src.main import readJSONSolution
from xmlreader import XMLReader


def read_file(path_to_folder, filename):
    text = list()
    with open(os.path.join(path_to_folder, filename), "r", encoding="ISO-8859-1") as file:
        text.append(file.read())
    return text


class BaselineApproach:

    def __init__(self, dswa):
        self.dswa = self.convert_dataset(dswa)

    def convert_dataset(self, dataset):
        if isinstance(dataset[0], str):
            new_dataset = dataset
        else:
            new_dataset = []
            for doc in dataset:
                new_doc = ' '.join(w[0] for s in doc for w in s)
                new_dataset.append(new_doc)
        return new_dataset

    def returnSolution(self, single_doc=False):
        print('Start calculating TF-IDF...')
        tfidf = TfidfVectorizer(use_idf=True, stop_words='english', ngram_range=(1, 3), token_pattern=r'(?u)\b[A-Za-z]+\b')
        tfidf_vectorizer_vectors = tfidf.fit_transform(self.dswa)

        if single_doc:
            tfidf_vectorizer_vectors = tfidf_vectorizer_vectors[-1]

        solution = []
        for v in tfidf_vectorizer_vectors:
            df = pd.DataFrame(v.T.todense(), index=tfidf.get_feature_names(), columns=['tfidf'])
            # lengths = [1/(1+math.exp(-len(name))) for name in df.index]
            # lengths = [len(name.split()) for name in df.index]
            lengths = [len(name) for name in df.index]
            values = df['tfidf'] * lengths
            df['scaled_tfidf'] = values
            df = df.sort_values(by='scaled_tfidf', ascending=False)
            best_scores = (df.head(5))
            solution.append(list(zip(best_scores.index, best_scores['scaled_tfidf'])))
        return solution

    def returnSolutionforDoc(self, doc):
        self.dswa = self.dswa + doc
        solution = self.returnSolution(single_doc=True)
        return solution[-1]

def test_baseline():
    # TNG_set = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True,
    #                              random_state=42)
    # ba = BaselineApproach(TNG_set['data'][:144])
    # my_file = read_file('..\\data\\sample_texts', 'sample_text.txt')
    # keyphrases = ba.returnSolutionforDoc(my_file)
    # keyphrases = ba.returnSolution()
    # print('Keyphrases selected from sample text: ', keyphrases)

    xmlReader = XMLReader('../data/SemEval-2010', True)
    dswa = xmlReader.readXML()

    un_xml = BaselineApproach(dswa)
    solution_xml = un_xml.returnSolution()

    stats = evaluate_dataset(readJSONSolution(False), solution_xml)
    print(solution_xml)


def main():
    test_baseline()


if __name__ == '__main__':
    main()