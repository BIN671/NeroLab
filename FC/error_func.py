import math
import numpy as np


class MSE:
    @staticmethod
    def error1(R, Y):
        return R - Y

    @staticmethod
    def grad_w_out(ERROR, A, O_M, d_func, rate):
        der_matrix = d_func(O_M)
        #error = R - Y
        transpon = A.T
        matrix_1 = np.repeat(2*ERROR*der_matrix*rate, A.size + 1 ,axis=1)
        matrix_2 = np.repeat(np.hstack([transpon, [[1]]]), ERROR.size, axis=0)
        return matrix_1*matrix_2

    @staticmethod
    def grad_a_out(ERROR, W, O_M, d_func, rate):
        #error = R - Y
        der_matrix = d_func(O_M)
        matrix_1 = np.delete(W, -1, axis= 1).T
        matrix_2 = 2*rate*ERROR*der_matrix
        #print(matrix_1)
        return np.dot(matrix_1, matrix_2)



