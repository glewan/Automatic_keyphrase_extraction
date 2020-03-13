import networkx as nx
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
import time, io, os
from bm25calculator import BM25Calculator

def read_file(path_to_folder, filename):
    text = list()
    with open(os.path.join(path_to_folder, filename), "r", encoding="ISO-8859-1") as file:
        sentences = sent_tokenize(file.read())
        pretext = list()
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [[word] for word in words]
            pretext.append(words)
    text.append(pretext)
    return text

# credits: S.Englert
class PageRank():
    def __init__(self, dswa):
        self.dswa = dswa
        self.stoplist = set(stopwords.words('english') + list(punctuation))
        self.stoplist.add('-lrb-')
        self.stoplist.add('-rrb-')
        self.stoplist.add('“')
        self.stoplist.add('”')
        self.stoplist.add('‘')
        self.stoplist.add('’')
        self.reorderDSWA()
        
        self.BM25 = False
        self.WordLength = False
        self.NofCooc = True
        self.WordEmbedding = False
        if self.BM25:
            self.idfCalculator = self.findBM25Terms()
        

        
    def reorderDSWA(self):
        #self.dswa = ([[' '.join([w[0] for w in s]) for s in d] for d in self.dswa])
        self.dswa = ([[[w[0] for w in s] for s in d] for d in self.dswa])
        
    def buildGraph(self, document):
        self.graph = nx.Graph()
        for sentence in document:
            self.buildGraphSentence(sentence)
            
    def findBM25Terms(self):
        allterms = set()
        for document in self.dswa:
            for sentence in document:
                nodes = list()
                #unograms
                possible_unograms = sentence
                possible_unograms = [uno for uno in possible_unograms if uno.lower() not in self.stoplist]
                nodes = nodes + possible_unograms
                #bigrams
                possible_bigrams = list(bigrams(sentence))
                possible_bigrams = [bi for bi in possible_bigrams if (bi[0].lower() not in self.stoplist and bi[1].lower() not in self.stoplist)]
                possible_bigrams = [' '.join(bi) for bi in possible_bigrams]
                nodes = nodes + possible_bigrams
                #trigrams
                possible_trigrams = list(trigrams(sentence))
                possible_trigrams = [tri for tri in possible_trigrams if (tri[0].lower() not in self.stoplist and tri[1].lower() not in self.stoplist and tri[2].lower() not in self.stoplist)]
                possible_trigrams = [' '.join(tri) for tri in possible_trigrams]
                nodes = nodes + possible_trigrams
                #print(nodes)
                
                #add nodes
                for node in nodes:
                    allterms.add(node)
                
        return BM25Calculator(self.dswa, allterms)         
            
                
    def buildGraphSentence(self, sentence):
        nodes = list()
        #unograms
        possible_unograms = sentence
        possible_unograms = [uno for uno in possible_unograms if uno.lower() not in self.stoplist]
        nodes = nodes + possible_unograms
        #bigrams
        possible_bigrams = list(bigrams(sentence))
        possible_bigrams = [bi for bi in possible_bigrams if (bi[0].lower() not in self.stoplist and bi[1].lower() not in self.stoplist)]
        possible_bigrams = [' '.join(bi) for bi in possible_bigrams]
        nodes = nodes + possible_bigrams
        #trigrams
        possible_trigrams = list(trigrams(sentence))
        possible_trigrams = [tri for tri in possible_trigrams if (tri[0].lower() not in self.stoplist and tri[1].lower() not in self.stoplist and tri[2].lower() not in self.stoplist)]
        possible_trigrams = [' '.join(tri) for tri in possible_trigrams]
        nodes = nodes + possible_trigrams
        #print(nodes)
        
        #add nodes
        for node in nodes:
            self.graph.add_node(node)
        #print(self.graph.nodes)
        #add edges
        for node in nodes:
            for node2 in nodes:
                if node != node2:
                    if self.graph.has_edge(node, node2):
                        if self.NofCooc:
                            self.graph[node][node2]['weight'] += 1
                    else:
                        self.graph.add_edge(node, node2)
                        self.graph[node][node2]['weight'] = 1
        #print(self.graph.edges)            
        
    def calculatePageRank(self, nOfTerms):
        prior=None
        if self.WordLength:
            prior = dict()
            for node in self.graph.nodes:
                prior[node] = len(node)
        elif self.BM25:            
             
            prior = dict()
            for node in self.graph.nodes:
                prior[node] = self.idfCalculator.bm25idf(node)
            
        ranking = nx.pagerank(self.graph, max_iter=50, personalization=prior, nstart=prior, alpha=0.85, tol=0.1)
        sorted_ranking = sorted(ranking.items(), key=lambda x: (x[1],x[0]), reverse=True)
        return sorted_ranking[:nOfTerms]
        
    def returnSolution(self, nOfTerms):
        i = 1
        timeBuildGraph = 0
        timePageRank = 0
        solution = list()
        for document in self.dswa:
            clock = time.clock()
            self.buildGraph(document)
            timeBuildGraph += time.clock() - clock
            clock = time.clock()
            solution.append(self.calculatePageRank(nOfTerms)) 
            timePageRank += time.clock() - clock
            print("\r{} / {} in {} s (build graph) and {} s (pagerank)".format(i, len(self.dswa), timeBuildGraph, timePageRank), end='')
            i+=1  
        print('')
        return solution 


        
        
if __name__ == '__main__':
    my_file = read_file('../data/sample_texts', 'sample_text.txt')
    pagerank = PageRank(my_file)
    print(pagerank.returnSolution(5))