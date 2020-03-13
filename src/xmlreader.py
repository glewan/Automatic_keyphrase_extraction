import xml.etree.ElementTree as ET
import os

class XMLReader():
    def __init__(self, path_string, train):
        self.path_string = path_string
        self.train = train
        
        
    def readXML(self):    
        if self.train:
            path_string = self.path_string + '/train'
            
        else:
            path_string = self.path_string + '/test'
        files = os.listdir(path_string)    
        
        dswa = list()  
        for i, file in enumerate(files):
            
            tree = ET.parse(path_string + '/' + file)
            rootXML = tree.getroot()     

            document = list()
            for sentenceXML in rootXML.findall('document/sentences/sentence'):
                sentence = list()
                for tokenXML in sentenceXML.findall('tokens/token'):
                    token = list()
                    for k in range(5):
                        token.append(tokenXML[k].text)
                    sentence.append(token)
                document.append(sentence) 
            dswa.append(document)
            print("\r{} / {}".format(i + 1, len(files)), end='')
        print('')   

        return dswa
        
if __name__ == '__main__':
    xmlReader = XMLReader('../data/semevaltest', True)
    print(xmlReader.readXML())
    