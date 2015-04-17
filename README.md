# 10605
Matrix Factorization for movie recommendation using Spark

This script runs DSGD on matrix factorization using Spark
Input format is as follows:
main(num_factors, num_workers, num_iter, betaVal, lambdaVal, inputV_filepath, outputW_filepath, outputH_filepath)

Output are two csv files, H.csv and W.csv representing the two factorized matrices of latent factors and existing variables (customers and movies)
