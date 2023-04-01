'''
File: gamma.py

Functions for computing the stopping times of large integers under the Collatz function.
'''

from math import log

##############
# Main Funcs #
##############

def T(n) :
    '''Computes T(n)'''
    if n % 2 == 0 :
        return n//2
    else :
        return (3*n + 1)//2

CACHE = {1: 0}
    
def sigma(n, max_iter=1000000) :
    '''Computes sigma(n)'''
    
    n_orig = n
    
    for i in range(max_iter) :
        if n in CACHE :
            CACHE[n_orig] = i + CACHE[n]
            return CACHE[n_orig]
        n = T(n)
        
    raise ValueError(f'Took over {max_iter} iterations to find sigma of {n}.')

def gamma(n) :
    '''Computes gamma(n)'''
    return sigma(n)/log(n)

#############
# Debugging #
#############

def seq(n) :
    '''Prints Collatz sequence for n, stopping at 1'''
    print(n, end=' ')
    if n != 1 :
        seq(T(n))
    else :
        print()