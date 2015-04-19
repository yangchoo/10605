"""This script runs DSGD on matrix factorization using Spark"""
import os
import sys
import multiprocessing
import csv
import numpy as np
from numpy import linalg
from scipy import sparse
from operator import itemgetter
SPARK_HOME = "C:\Users\yang\Desktop\spark-1.3.0-bin-hadoop2.4"# Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append(SPARK_HOME + "/python") # Add python files to Python Path
from pyspark import SparkContext, SparkConf

def main(num_factors, num_workers, num_iter, betaVal, lambdaVal, inputV_filepath, outputW_filepath, outputH_filepath):
    print 'Running Main'
    num_workers = int(num_workers)
    num_iter = int(num_iter)
    num_factors = int(num_factors)
    betaVal = float(betaVal)
    lambdaVal = float(lambdaVal)
    ##Data pre-processing
    conf = SparkConf().setAppName('DSGD-MF').setMaster('local')
    sc = SparkContext(conf=conf)
    if os.path.isdir(inputV_filepath):
        p = multiprocessing.Pool()
        paths = [os.path.join(inputV_filepath, f) for f in os.listdir(inputV_filepath)]
        results = p.map(formatRawFile, paths)
        p.close()
        flattenedResults = [item for sublist in results for item in sublist]
        filename = open('output.txt','w')
        filename.write ('\n'.join(flattenedResults))
        data = sc.parallelize(flattenedResults)
    elif os.path.isfile(inputV_filepath):
        data = sc.textFile(inputV_filepath)
    else:
        raise Exception("Invalid Input File Path")
    
    ##Get V matrix into right format
    #countList = Counter()
    split_rdd = data.map(lambda x: x.split(','))
    #V_rdd = split_rdd.map(lambda x: mapInc(x, countList))
    V_rdd = split_rdd.map(lambda x: map(lambda y: int(y),x)).sortBy(lambda x: (x[0],x[1]))
    
    
    
    ##Initialize variables
    tau_0 = 100
    #W_indx_map_rdd = split_rdd.map(lambda x: int(x[0]))#.sortBy(lambda x: x)\

    # pylint: disable=E1103
    #W_vec_rdd = split_rdd.map(lambda x: int(x[0])).distinct().sortBy(lambda x: x).map(lambda x: tuple([x,np.random.rand(num_factors,1).astype(np.float32)]))
    #H_vec_rdd = split_rdd.map(lambda x: int(x[1])).distinct().sortBy(lambda x: x).map(lambda x: tuple([x,np.random.rand(1,num_factors).astype(np.float32)]))
    #W_vec_rdd = split_rdd.map(lambda x: tuple([int(x[0]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],np.random.rand(1,num_factors).astype(np.float32)])]))
    #H_vec_rdd = split_rdd.map(lambda x: tuple([int(x[1]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],np.random.rand(num_factors,1).astype(np.float32)])]))
    # pylint: enable=E1103
    #W_rdd_reIndx =  W_indx_map_rdd.map(lambda (x,y) : y)
    #H_rdd_reIndx =  H_indx_map_rdd.map(lambda (x,y) : y)

    #W_max_indx = W_rdd_reIndx.max()
    #H_max_indx = H_rdd_reIndx.max()

    #randomize initil value
    #partby, keyby, filter

    #
    n_counts = 0
    cur_iter = 0
    checkError = False  #option used for the netflix dataset (calculates LNZSL loss)
    reconError = []
    ## Main Loop
    
    if checkError:
        W_vec_rdd = split_rdd.map(lambda x: int(x[0])).distinct().sortBy(lambda x: x).map(lambda x: tuple([x,np.random.rand(1,num_factors).astype(np.float32)]))
        H_vec_rdd = split_rdd.map(lambda x: int(x[1])).distinct().sortBy(lambda x: x).map(lambda x: tuple([x,np.random.rand(num_factors,1).astype(np.float32)]))
        V, select = LoadSparseMatrix(inputV_filepath)
        while cur_iter != num_iter:
            for stratum in xrange(0,num_workers-1):
                V_keyed = V_rdd.keyBy(lambda x: x[0]%num_workers).partitionBy(num_workers)
                V_part = V_keyed.filter(lambda x: (x[1][1]+stratum)%num_workers==x[0])
                H_keyed = H_vec_rdd.keyBy(lambda x: (x[0]+stratum)%num_workers).partitionBy(num_workers)
                W_keyed = W_vec_rdd.keyBy(lambda x: x[0]%num_workers).partitionBy(num_workers)
                combined_rdd = V_part.groupWith(H_keyed, W_keyed)
                return_rdd = combined_rdd.mapPartitions(lambda x: lossMap_NZSL(x, num_factors)).reduceByKey(lambda x,y: x+y)
                W_vec_rdd = return_rdd.filter(lambda x: x[0]=='W').flatMap(lambda x: x[1])
                H_vec_rdd = return_rdd.filter(lambda x: x[0]=='H').flatMap(lambda x: x[1])
                #Check Error using given functions
                W_output = W_vec_rdd.collect()
                W_output.sort()
                H_output = H_vec_rdd.collect()
                H_output.sort()
                
                #print W_output
                #print H_output
                
                W = W_output[0][1]
                tmp_count = 1
                for W_indx in xrange(2,W_output[-1][0]+1):
                    if W_output[tmp_count][0] == W_indx:
                        W = np.vstack([W, W_output[tmp_count][1]])
                        tmp_count += 1
                    else:
                        W = np.vstack([W, np.zeros((1,num_factors))])
    
                H = H_output[0][1]
                tmp_count = 1
                for H_indx in xrange(2,H_output[-1][0]+1):
                    if H_output[tmp_count][0] == H_indx:
                        H = np.hstack([H, H_output[tmp_count][1]])
                        tmp_count += 1
                    else:
                        H = np.hstack([H, np.zeros((num_factors,1))])
                
                #H = np.hstack([ele[1] for ele in H_output])
                error = CalculateError(V,W,H,select)
                reconError.append(error)
                print "Reconstruction error:", error
                cur_iter += 1
                if cur_iter == num_iter:
                    break
    else:
        W_vec_rdd = split_rdd.map(lambda x: tuple([int(x[0]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],np.random.rand(1, num_factors).astype(np.float32)])]))
        H_vec_rdd = split_rdd.map(lambda x: tuple([int(x[1]),1])).reduceByKey(lambda x,y : x+y).map(lambda x: tuple([x[0],tuple([x[1],np.random.rand(num_factors,1).astype(np.float32)])]))
        while cur_iter != num_iter:
            for stratum in xrange(0,num_workers-1):
                V_keyed = V_rdd.keyBy(lambda x: x[0]%num_workers).partitionBy(num_workers)
                V_part = V_keyed.filter(lambda x: (x[1][1]+stratum)%num_workers==x[0])
                H_keyed = H_vec_rdd.keyBy(lambda x: (x[0]+stratum)%num_workers).partitionBy(num_workers)
                W_keyed = W_vec_rdd.keyBy(lambda x: x[0]%num_workers).partitionBy(num_workers)
                combined_rdd = V_part.groupWith(H_keyed, W_keyed)
                return_rdd = combined_rdd.mapPartitions(lambda x: lossMap(x, lambdaVal, n_counts, tau_0, betaVal, num_factors)).reduceByKey(lambda x,y: x+y)
                W_vec_rdd = return_rdd.filter(lambda x: x[0]=='W').flatMap(lambda x: x[1])
                H_vec_rdd = return_rdd.filter(lambda x: x[0]=='H').flatMap(lambda x: x[1])
                n_counts = (return_rdd.filter(lambda x: x[0]=='N').collect())[0][1]
                cur_iter += 1
                if cur_iter == num_iter:
                    break
    
    if checkError:
        w_file = open(outputW_filepath, 'w')
        h_file = open(outputH_filepath, 'w')
        printToCsv_NZSL(W_output, w_file, H_output, h_file, num_factors)
        print reconError
    else:
        W_output = W_vec_rdd.collect()
        W_output.sort()
        w_file = open(outputW_filepath, 'w')
        H_output = H_vec_rdd.collect()
        H_output.sort()
        h_file = open(outputH_filepath, 'w')
        #printToCsv(W_output, w_file, H_output, h_file, num_factors)
        W = W_output[0][1][1]
        tmp_count = 1
        for W_indx in xrange(2,W_output[-1][0]+1):
            if W_output[tmp_count][0] == W_indx:
                W = np.vstack([W, W_output[tmp_count][1][1]])
                tmp_count += 1
            else:
                W = np.vstack([W, np.zeros((1,num_factors))])
        H = H_output[0][1][1]
        tmp_count = 1
        for H_indx in xrange(2,H_output[-1][0]+1):
            if H_output[tmp_count][0] == H_indx:
                H = np.hstack([H, H_output[tmp_count][1][1]])
                tmp_count += 1
            else:
                H = np.hstack([H, np.zeros((num_factors,1))])
        np.savetxt(outputW_filepath, W, delimiter=",")
        np.savetxt(outputH_filepath, H, delimiter=",")

def LoadMatrix(csvfile):
    data = np.genfromtxt(csvfile, delimiter=',')
    return np.matrix(data)
    
def printToCsv(W_output, w_file, H_output, h_file, num_factors):
    tmp_count = 0
    for W_indx in xrange(1,W_output[-1][0]+1):
        if W_output[tmp_count][0] == W_indx:
            w_file.write(','.join(['%.5f' % num for num in (W_output[tmp_count][1][1]).transpose()])+'\n')
            tmp_count += 1
        else:
            w_file.write(','.join(['0']*num_factors)+'\n')
            
    tmp_count = 0
    for H_indx in xrange(1,H_output[-1][0]+1):
        if H_output[tmp_count][0] == H_indx:
            h_file.write(','.join(['%.5f' % num for num in H_output[tmp_count][1][1]])+'\n')
            tmp_count += 1
        else:
            h_file.write(','.join(['0']*num_factors)+'\n')
    w_file.close()
    h_file.close()
    
def printToCsv_NZSL(W_output, w_file, H_output, h_file, num_factors):
    tmp_count = 0
    for W_indx in xrange(1,W_output[-1][0]+1):
        if W_output[tmp_count][0] == W_indx:
            w_file.write(','.join(['%.5f' % num for num in (W_output[tmp_count][1]).transpose()])+'\n')
            tmp_count += 1
        else:
            w_file.write(','.join(['0']*num_factors)+'\n')
            
    tmp_count = 0
    for H_indx in xrange(1,H_output[-1][0]+1):
        if H_output[tmp_count][0] == H_indx:
            h_file.write(','.join(['%.5f' % num for num in (H_output[tmp_count][1])])+'\n')
            tmp_count += 1
        else:
            h_file.write(','.join(['0']*num_factors)+'\n')
    w_file.close()
    h_file.close()
    return True
            
'''Formats raw movie file of the form user,rating date'''
def formatRawFile(filePath):
    output = []
    f = open(filePath)
    movie = (f.readline())[:-2]
    for line in f:
        ratings = line.split(",")
        output.append(ratings[0]+','+movie+','+ratings[1])
    return output

'''Maps over the tripleList (indx, [i,j,rating])'''
def lossMap(keyed_iterable, lambdaVal, n_counts, tau_0, betaVal, num_factors):
        iter_list = (keyed_iterable.next())[1]
        V_iterable = iter_list[0]
        H_iterable = iter_list[1]
        W_iterable = iter_list[2]
        
        W_Dict = {}
        H_Dict = {}
        
        W_new_Dict = {}
        H_new_Dict = {}
        
        for H_ele in H_iterable:
            H_Dict[H_ele[0]] = H_ele[1]
        
        for W_ele in W_iterable:
            W_Dict[W_ele[0]] = W_ele[1]
        
        for V_ele in V_iterable:
            (i,j,rating) = V_ele
            # pylint: disable=E1103
            epsilon = np.power(tau_0 + n_counts, -betaVal)
            # pylint: enable=E1103
            if j in H_Dict:
                H_input = H_Dict[j]
            else:
                H_Dict[j] = tuple([j,np.random.rand(num_factors,1).astype(np.float32)])
                H_input = H_Dict[j]
            if i in W_Dict:
                W_input = W_Dict[i]
            else:
                W_Dict[i] = tuple([i,np.random.rand(1,num_factors).astype(np.float32)])
                W_input = W_Dict[i]
            (new_W, new_H) = lossL2(rating, H_input, W_input, lambdaVal, epsilon)
            n_counts += 1
            W_new_Dict[i] = new_W
            H_new_Dict[j] = new_H
            
        
        #return (tuple(['W',new_W_list]), tuple(['H',new_H_list]))
        return (tuple(['W',W_new_Dict.items()]), tuple(['H',H_new_Dict.items()]), tuple(['N', n_counts]))

'''Returns L2 loss in tuple form'''
def lossL2(rating, H_ele, W_ele, lambdaVal, epsilon):
    (N_h, H_array) = H_ele
    (N_w, W_array) = W_ele
    
    oldW = W_array.copy()
    oldH = H_array.copy()
    # pylint: disable=E1103
    
    gradient = -2*(rating-np.asscalar(oldW.dot(oldH)))
    
    W_array =  np.add(oldW, np.multiply (epsilon, np.multiply(oldH.transpose(), gradient) + np.multiply(2*lambdaVal/N_w, oldW)))
    H_array = np.add(oldH, np.multiply(epsilon, np.multiply(oldW.transpose(), gradient) + np.multiply(2*lambdaVal/N_h, oldH)))
    
    #return tuple([W_array]), tuple([j, H_array])])
    return (tuple([N_w, W_array]), tuple([N_h, H_array]))

'''Maps over the tripleList (indx, [i,j,rating])'''
def lossMap_NZSL(keyed_iterable, num_factors):
        iter_list = (keyed_iterable.next())[1]
        V_iterable = iter_list[0]
        H_iterable = iter_list[1]
        W_iterable = iter_list[2]
        
        W_Dict = {}
        H_Dict = {}
        
        W_new_Dict = {}
        H_new_Dict = {}
        
        for H_ele in H_iterable:
            H_Dict[H_ele[0]] = H_ele[1]
        
        for W_ele in W_iterable:
            W_Dict[W_ele[0]] = W_ele[1]
        
        for V_ele in V_iterable:
            (i,j,rating) = V_ele
            # pylint: disable=E1103
            # pylint: enable=E1103
            if j in H_Dict:
                H_input = H_Dict[j]
            else:
                H_Dict[j] = np.random.rand(num_factors,1).astype(np.float32)
                H_input = H_Dict[j]
            if i in W_Dict:
                W_input = W_Dict[i]
            else:
                W_Dict[i] = np.random.rand(1,num_factors).astype(np.float32)
                W_input = W_Dict[i]
            (new_W, new_H) = lossNZSL(rating, H_input, W_input)
            W_new_Dict[i] = new_W
            H_new_Dict[j] = new_H
        
        #return (tuple(['W',new_W_list]), tuple(['H',new_H_list]))
        return (tuple(['W',W_new_Dict.items()]), tuple(['H',H_new_Dict.items()]))
        
'''Returns LNZSL loss in tuple form'''
def lossNZSL(rating, H_array, W_array):
    oldW = W_array.copy()
    oldH = H_array.copy()
    # pylint: disable=E1103
    
    gradient = -2*(rating-np.asscalar(oldW.dot(oldH)))
    
    W_array = np.add(oldW, np.multiply(oldH.transpose(), gradient))
    H_array = np.add(oldH, np.multiply(oldW.transpose(), gradient))
    
    return (W_array, H_array)
    # pylint: enable=E1103

def LoadSparseMatrix(csvfile):
        val = []
        row = []
        col = []
        select = []
        f = open(csvfile)
        reader = csv.reader(f)
        for line in reader:
                row.append( int(line[0])-1 )
                col.append( int(line[1])-1 )
                val.append( int(line[2]) )
                select.append( (int(line[0])-1, int(line[1])-1) )
        return sparse.csr_matrix( (val, (row, col)) ), select

def CalculateError(V, W, H, select):
        print 
        diff = V-W.dot(H)
        error = 0
        for row, col in select:
                error += diff[row, col]*diff[row, col]
        return error/len(select)

if __name__ == "__main__":
    #main(num_factors, num_workers, num_iter, betaVal, lambdaVal, inputV_filepath, outputW_filepath, outputH_filepath)
    #main(1,2,3,4,5,'output.txt','w.csv','h.csv')
    #main(100,100,100,0.1,1,'netflix/training_set_tiny','w.csv','h.csv')
#    main(100,10,50,0.8,1.0,'netflix/training_set_tiny','w.csv','h.csv')
#    main(10,2,1,0.8,1.0,'test.txt','w.csv','h.csv')
    #main(3,2,2,0.8,1.0,'tiny.txt','w.csv','h.csv')
    #main(20,10,1,0.6,1.0,'nf_subsample.csv','w.csv','h.csv')
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
