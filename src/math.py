import random 

A = [[1,2,3],[4,-5,6]]
B = [[-1,0,1],[0,1,2],[-1,1,0]]

class MatriceMath :
    def __init__(self):
        return
    
    
    def get_empty(self, rows, cols) :
        result=[]
        for i in range(rows):
            result.append([])
            for _ in range(cols):
                result[i].append(0)
                
        return result
    
    
    
    def dot_product(self, x, y) :
        if not len(x) == len(y) : raise ValueError("Vectors lengths do not match")
        result = []
        for i in range(len(x)) :
            result.append(x[i] * y[i])
            
        return result
    
    
    # Si (X) a la taille (m * n) et (Y) la taille (n * p), alors (XY) existe et donne une matrice de taille (m * p).
    def multiplication(self, X, Y) :
    
        if not len(X[0]) == len(Y) :
            raise ValueError("Number of columns in A must match number of rows in B, but: A=" + str(A) + ", and B=" + str(B)) 
        
        result = self.get_empty(len(X), len(Y[0]))
        
        row = 0
        col = 0
        
        
        for x_row in X:
            col = 0
            
            # zip(*Y) transposes rows into columns
            for y_col in zip(*Y):
                dot_product = 0
                
                for x, y in zip(x_row, y_col):
                    dot_product += x * y
                    print(x, y, dot_product)

                    
                result[row][col] = dot_product
                col+= 1
                
            row += 1
            
        return result
        
            
            

    def addition(self, X, Y) :
        result = self.get_empty(len(X), len(Y[0]))
 
        for i in range(len(X)):
            result[i] = [X[i][j] + Y[i][j] for j in range(len(Y[i]))]
                                    
        return result
    
    
    
    def substraction(self, X, Y) :
        result = self.get_empty(len(X), len(Y[0]))
 
        for i in range(len(X)):
            result[i] = [X[i][j] - Y[i][j] for j in range(len(Y[i]))]
                                    
        return result
            
            
    
maths= MatriceMath()


x = [[1, 2], 
     [3, 4]]



y = [[5, 6], 
     [7, 3]]

#print(maths.get_empty(200, 200))


#print(maths.multiplication(x, y))
print(maths.substraction(x, y))


