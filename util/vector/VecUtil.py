import numpy as np
from scipy import *
import scipy.spatial.distance as spd
import operator

import sys
sys.path.append('../../')

from Util.util.file.FileUtil import *
from numpy.linalg import solve

class VecUtil(object):


    def map_2_nearest_vector(vec1, vec2,sents1,sents2):
        simScors = {}
        SimMax = {}
        for i in range(vec1.shape[0]):
            simScors[i] = [abs(spd.cosine(vec1[i],vec2[j])) for j in range(vec2.shape[0])]
            simScors[i][i] = max(simScors[i])
            min_index, min_value = min(enumerate(simScors[i]), key=operator.itemgetter(1))
            SimMax[i] = [min_index, min_value]
            #print(simScors[i][i])
            print(str(min_value) + "|" +sents1[i].strip() + "|" +sents2[min_index].strip())

        return SimMax

    def calculate_distance_matrix(vectorz):
        fvalz = [ [ float(val) for val in vals ] for vals in vectorz.values()]
        dist = spd.pdist(fvalz, 'euclidean')
        l = size(list(vectorz.keys()))
        matrix = np.zeros((l, l))

        current = 0;
        x = size(dist.tolist())
        for k in range(x):

            if k >= (current+1)*l - (1+(current+1))*(current+1)/2:
                current += 1

            matrix[current][k - (current*l - (1+current)*current/2) + current+1] = dist[k]
            matrix[k - (current*l - (1+current)*current/2) + current+1][current] = dist[k]

        return matrix

    def calculate_similarity_matrix(vectorz):
        fvalz = [ [ float(val) for val in vals ] for vals in vectorz.values()]
        dist = spd.pdist(fvalz, 'cosine' )
        l = size(list(vectorz.keys()))
        matrix = np.zeros((l, l))

        current = 0;
        x = size(dist.tolist())
        for k in range(x):

            if k >= (current+1)*l - (1+(current+1))*(current+1)/2:
                current += 1

            matrix[current][k - (current*l - (1+current)*current/2) + current+1] = dist[k]
            matrix[k - (current*l - (1+current)*current/2) + current+1][current] = dist[k]

        return matrix

    def calculate_transformation_matrix(inputVectors,outputVectors):
        print("!!")








if __name__ == "__main__":
    vectors = {1: [0, 0, 0], 2: [1, 1, 1], 3: [-1, -1, -1], 4: [2, 2, 2]}
    """   dists= VecUtil.calculate_distance_matrix(vectors)
    print(dists)
    header = ''
    for i in range(size(dists[0])):
       header += ' '+str(i)

    header += '\n'

    np.savetxt('distMatTest.txt',dists)
    FileUtil.add_line_numbers('distMatTest.txt',True)
    open('distMatTest.csv', "w").write(header + open('distMatTest.txt').read())
    """
    """A = np.array([xi for xi in list(vectors.values())])
    intercept = np.ones((4,1))
    C = np.c_[intercept,A]
    print(A)"""



    """vectors = FileUtil.file_2_vec('d-300/VOut.txt')
    dists= VecUtil.calculate_similarity_matrix(vectors)
    print(size(dists))
    header = ''
    for i in range(size(dists[0])):
       header += ' '+str(i+1)

    header += '\n'

    np.savetxt('d-300_SimMat.txt',dists)
    FileUtil.add_line_numbers('d-300_SimMat.txt',True)
    open('d-300_SimMat.csv', "w").write(header + open('d-300_SimMat.txt').read())"""


    """vec1 = FileUtil.file_2_vec('TransferedDoSentDontSentTest-50.txt')
    vec1 = np.array([xi for xi in list(vec1.values())])
    vec2 = FileUtil.file_2_vec('DoSentDontSentTestTrain-50.txt')
    vec2 = np.array([xi for xi in list(vec2.values())])
    sents = FileUtil.file_2_sentList('Sentences_DoSentDontSentTestTrain.txt')

    simMax = VecUtil.map_2_nearest_vector(vec1, vec2, sents)

    sents = FileUtil.file_2_sentList('Sentences_DoSentDontSentTestTrain.txt')
    transSents = []
    for i in range(vec1.shape[0]):
        transSents = [simMax[i][1], sents[simMax[i][0]]]

    with open('transferedSents_2ndRun.txt', 'w') as filedata:
            for s in transSents:
                filedata.write('%f %s' % (s[0], s[1]))"""

    """vec1 = FileUtil.file_2_vec('TransferedSimpleTest-50.txt')
    vec1 = np.array([xi for xi in list(vec1.values())])
    vec2 = FileUtil.file_2_vec('withSimple.txt')
    vec2 = np.array([xi for xi in list(vec2.values())])
    sents1 = FileUtil.file_2_sentList('SimpleTest.txt')
    sents2 = FileUtil.file_2_sentList('Sentences_withSimples.txt')
    simMax = VecUtil.map_2_nearest_vector(vec1, vec2, sents1,sents2)"""


    vec1 = FileUtil.file_2_vec('DoSentDontSentTestTrain-50.txt')
    vec1 = np.array([xi for xi in list(vec1.values())])
    vec2 = FileUtil.file_2_vec('DoSentDontSentTestTrain-50.txt')
    vec2 = np.array([xi for xi in list(vec2.values())])
    sents1 = FileUtil.file_2_sentList('Sentences_DoSentDontSentTestTrain.txt')
    sents2 = FileUtil.file_2_sentList('Sentences_DoSentDontSentTestTrain.txt')
    simMax = VecUtil.map_2_nearest_vector(vec1, vec2, sents1,sents2)



