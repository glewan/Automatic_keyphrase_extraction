import math

# credits: S.Englert
class BM25Calculator():
    def __init__(self, dswa, terms):
        self.dswa = dswa
        #number of documents
        self.bigN = len(dswa)
        #2d array only containing documents and words
        self.text = [[w[0] for s in d for w in s] for d in dswa]
        self.invindex = self.createInvertedIndex(dswa, terms)
        self.maxFreq = self.createMaxFreq()
        
        lentext = 0
        for dtext in self.text:
            lentext = lentext + len(dtext)
        self.avgdoclen = lentext / self.bigN 

    def createInvertedIndex(self, dswa, terms):
        invindex = dict()
        for term in terms:
            invindex[term] = dict()
        
        for i, dtext in enumerate(self.text): 
            for k in range(len(dtext)):
                term = dtext[k] 
                if term in invindex:
                    invindex[term][i] = invindex[term].get(i, 0) + 1
                for j in range(2):
                    if (k+j+1) >= len(dtext):
                        break
                    term = term + ' ' + dtext[k+j+1]
                    if term in invindex:
                        invindex[term][i] = invindex[term].get(i, 0) + 1                    
        
        return invindex
        
    def createMaxFreq(self):
        maxFreq = list()
        for i in range(len(self.dswa)):
            max = 0
            for term in self.invindex:
                if (self.invindex[term].get(i, 0) > max):
                    max = self.invindex[term][i]
            maxFreq.append(max)
            
        return maxFreq    
    
    def idf(self):
        pass
    
    def bm25idf(self, term):
        noft = len(self.invindex[term])
        return math.log10((self.bigN - noft + 0.5)/(noft + 0.5))
        #return math.log10((self.bigN - noft + 0.5)/(noft + 0.5)*len(term))
        
    def getTermfrequency(self, term, documentnumber):
        return self.invindex[term][documentnumber] / self.maxFreq[documentnumber]
        
    def getAvgDocLen(self):
        return self.avgdoclen
        
        
if __name__ == '__main__':
    doc1 = ['Hi', 'this', 'is', 'a', 'test']
    doc2 = ['test', 'a', 'programm']
    dswa = list()
    dswa.append(doc1)
    dswa.append(doc2)
    idfCalculator = BM25Calculator(dswa)
    idfCalculator.bm25idf('test')