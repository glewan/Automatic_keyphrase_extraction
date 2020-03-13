#
#
# Created as a team project for Information Retrieval classes
#
#

from xmlreader import XMLReader
from jsonreadersolution import JSONReaderSolution
from evaluation import evaluate_dataset
from unsupervised import Unsupervised_Approach
import argparse, sys
from pagerank import PageRank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", choices=['pagerank', 'unsupervised'], default='pagerank', help="Choose the method used for finding key phrases!")
    parser.add_argument("-v", "--verbose", action='store_true', help="Print the result for each document!")
    parser.add_argument("-s", "--small", action='store_true', help="Use small dataset (for testing porpuses!)")
    args = parser.parse_args()
    
    useSmall = args.small
    method = args.method
    verbose = args.verbose

    #Read in XML
    dswa = readXML(useSmall)

    #Read in JSON Solution
    gt_solution = readJSONSolution(useSmall)

    #Calculate Solution
    calculated_solution = calculateSolution(dswa, method, gt_solution)

    #Evaluation comparing gt_solution and calculated_solution
    if method == 'supervised':
        evaluate_dataset(gt_solution[5:10], calculated_solution, verbose)
    else:
        evaluate_dataset(gt_solution, calculated_solution, verbose)


def readXML(useSmall):
    if useSmall:
        print('Start reading in experimental dataset xml files!')
        xmlReader = XMLReader('../data/semevaltest', True)
    else:
        print('Start reading in complete dataset xml files!')
        xmlReader = XMLReader('../data/SemEval-2010', True)        
    dswa = xmlReader.readXML()
    print('Finished reading in dataset xml files!')
    return dswa

def readJSONSolution(useSmall):
    if useSmall:
        print('Start reading in experimental dataset json solution!')
        jsonReader = JSONReaderSolution('../data/semevaltest', True)
    else:
        print('Start reading in complete dataset json solution!')
        jsonReader = JSONReaderSolution('../data/SemEval-2010', True)                
    gt_solution = jsonReader.readJSON()
    print('Finished reading in dataset json solution!')
    return gt_solution

def calculateSolution(dswa, method, gt_solution):
    # BaselineApproach
    if method == 'pagerank':
        pagerank_approach = PageRank(dswa)
        print("Calculation solution...")
        calculated_solution = pagerank_approach.returnSolution(5)
    #UnsupevisedApproach
    elif method == 'unsupervised':
        unsupervisedApproach = Unsupervised_Approach(dswa)
        print("Calculation solution...")
        calculated_solution = unsupervisedApproach.returnSolution()

    print('--- Solution ---')
    print(calculated_solution)
    return calculated_solution
    

if __name__ == '__main__':
    main()
