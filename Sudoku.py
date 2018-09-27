#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:57:40 2018

@author: yifengluo
"""

'''
Try to implement exact cover algorithm : Algorithm X
'''
import time
import numpy as np
class Exactcover(): 
    def __init__(self,matrix):
        self.m,self.n = matrix.shape
        self.matrix = matrix
        self.solution = []
        self.current_solution = [] 
        self.Raw = list(range(self.m))
        self.Column= list(range(self.n))
        self.M = Matrix(self.matrix,self.Raw,self.Column)
        self.answer_number = 0
        self.run(self.M,[])
    def run(self,M,current_solution):
        if self.Success_check(M.matrix):
          
            if len(current_solution) == 1:
                if 0 in self.matrix[current_solution[0],:]:
                    return False
            self.answer_number += 1
            self.solution.append(current_solution)
            return True
        column,State = self.State_check(M) # column is in the current matrix, not the global column, the global is Matrix.column[column]
        if not State:
            return False
        Children = np.where(M.matrix[:,column] == 1)[0]
        for raw in Children: # row is in the current matrix, global is Matrix.row[row]
            CS = current_solution[:]
            M_child = self.Cropout(M,raw)
            CS.append(M.raw[raw]) 
            self.run(M_child,CS)   
    def Cropout(self,M,raw):
        Column_delete = set(np.where(M.matrix[raw,:] == 1)[0])
        Column_remain = set((range(M.matrix.shape[1]))) ^ Column_delete
        Raw_delete = set()
        for i in Column_delete:
           Raw_delete = Raw_delete | set(np.where(M.matrix[:,i] == 1)[0])
        Raw_remain = set((range(M.matrix.shape[0]))) ^ Raw_delete 
        matrix_new = M.matrix.copy()[:,list(Column_remain)]
        matrix_new = matrix_new[list(Raw_remain),:]
        M_new = Matrix(matrix_new,[M.raw[i] for i in Raw_remain],[M.column[i] for i in Column_remain])
        return M_new
    
    def Success_check(self,matrix):
        '''
        Check if the matrix is empty or not.
        If the matrix is empty,we find the solution.
        '''
        if np.sum(matrix) == 0 :
            return True        
        else:
            return False
    def State_check(self,Matrix):
        '''
        check if the minimum number of column and find it.
        If the value is 0, there's no solution for this certain methods.
        '''
        Value = np.sum(Matrix.matrix,axis = 0)
        Min = min(Value)
        if Min == 0 :
            return 0,False
        Column_index = np.where(Value == Min)[0][0]  # find the first min value
        return Column_index,True
class Matrix():
    def __init__(self,matrix,raw,column):
        '''
        row is a list that contains the current row name, so is the column. 
        '''
        assert(matrix.shape[0] == len(raw))
        assert(matrix.shape[1] == len(column))
        self.matrix = matrix
        self.raw = raw
        self.column = column

class Sudoku():
    def __init__(self,matrix):
        self.matrix = matrix
        self.Pose = np.nonzero(self.matrix)
        self.data = np.vstack((np.vstack(self.Pose),matrix[self.Pose]))
        self.create_matrix()
        self.EC = Exactcover(self.sudoku)
        self.solution = self.EC.solution
        self.result = []
        self.find_result()
    def create_matrix(self):
        exist_pose = self.data.shape[1]
        unknown_pose = 81 - exist_pose
        self.sudoku = np.zeros([exist_pose + unknown_pose * 9, 324 ])
        '''
        Unknown data process
        '''
        Zero_position = np.vstack(np.where(self.matrix == 0))
        Zero_position = Zero_position.repeat(9,axis = 1)
        Data = np.vstack((Zero_position,list(range(1,10)) * unknown_pose ))
        '''
        Data_confusion
        '''
        self.data = np.hstack((Data,self.data))
        
        self.Position = []
        i = 0 
        for row,column,value in self.data.T: 
            C = [row * 9 +column, 80 + row * 9 + value, 161 + column * 9 + value, 242 +  (row // 3 * 3 + column // 3) * 9 + value]
            self.sudoku[[i],[C]] = 1
            
            i += 1
    def find_result(self):
        '''
        return a matrix of sudoku
        '''
        if len(self.solution) == 0:
            return 'No result'
        elif len(self.solution) >1 :
            print('Has multiple result')
        for result in self.solution:
            x = self.data.T[result][:,0]
            y = self.data.T[result][:,1]
            z = self.data.T[result][:,2]
            self.sudoku = np.zeros([9,9],dtype = np.int)
            self.sudoku[x,y] = z
            self.result.append(self.sudoku)
            
                 
           
  
if __name__ == '__main__':  
    matrix_1 = np.array([[4,8,0,0,6,7,0,0,1],
                       [0,0,0,5,0,0,0,9,6],
                       [0,0,0,0,2,0,0,0,0],
                       [0,4,0,0,7,0,0,0,0],
                       [1,5,0,0,0,0,0,4,9],
                       [0,0,0,4,0,0,0,5,0],
                       [2,6,0,0,5,0,0,0,0],
                       [0,0,0,7,0,0,0,0,0],
                       [5,0,0,9,0,6,0,7,8]])
    matrix_2 = np.array([[8,0,0,0,0,0,0,0,0],
                         [0,0,3,6,0,0,0,0,0],
                         [0,7,0,0,9,0,2,0,0],
                         [0,5,0,0,0,7,0,0,0],
                         [0,0,0,0,4,5,7,0,0],
                         [0,0,0,1,0,0,0,3,0],
                         [0,0,1,0,0,0,0,6,8],
                         [0,0,8,5,0,0,0,1,0],
                         [0,9,0,0,0,0,4,0,0,]])
    start = time.clock()
    S = Sudoku(matrix_2)
    np.set_printoptions(threshold=1000)
    print('origin problem :')
    print(matrix_2)
    print('result')
    print(S.result)
    print('Time = ', time.clock() - start)
#    matrix = np.array([[1,	 0,	0,	1,	0,	0,	1],
#                       [1,	 0,	0,	1,	0,	0,	0],
#                       [0,	 0,	0,	1,	1,	0,	1],
#                       [0,	 0,	1,	0,	1,	1,	0],
#                       [0,	 1,	1,	0,	0,	1,	1],
#                       [0,	 1,	0,	0,	0,	0,	1],
#                       [1, 1,1,1,1,1,1]])
#    start = time.clock()
#    EC = Exactcover(matrix)
#    print(time.clock() - start)    
    
