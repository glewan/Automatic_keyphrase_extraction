import os, json

class JSONReaderSolution():
    def __init__(self, path_string, train):
        self.path_string = path_string
        self.train = train
        
        
    def readJSON(self):    
        if self.train:
            path_string = self.path_string + '/references/train.reader.json'
            
        else:
            path_string = self.path_string + '/references/test.reader.stem.json'
        
        solution = list()  
            
        with open(path_string) as json_file:
            solutionJSON = json.load(json_file) 
            #print(solutionJSON)
            solutionJSON = sorted(solutionJSON.items(), key=lambda x: (x))            
            #print(solutionJSON)
            for documentJSON in solutionJSON:
                #print(documentJSON)
                documentSolution = list()
                for termsJSON in documentJSON[1]:
                    for termJSON in termsJSON:
                        documentSolution.append((termJSON, 0))
                solution.append(documentSolution)
        return solution
        
if __name__ == '__main__':
    jsonReader = JSONReaderSolution('../data/SemEval-2010', True)
    print(jsonReader.readJSON())
    