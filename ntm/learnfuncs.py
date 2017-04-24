"""
This module defines the function that we want to try to learn with NNs.
"""

def f_identity(seq):
    return seq

def f_reverse(seq):
    t = [0]*len(seq)
    for j in range(len(seq)):
        t[len(seq)-j-1] = seq[j]
    return t

def f_swap01(seq):
    t = []
    for j in range(len(seq)):
        if seq[j] == 0:
            t.append(0)
        else:
            t.append(1)
    return t

def f1(seq):
    t = []
    for j in range(len(seq)):
        if seq[j] == 1 and j > 0 and seq[j-1] == 1:
            if j > 3:
                t.append(seq[j-2])
            else:
                t.append(0)
        else:
            t.append(1)
    return t

def f_skiprepeat(seq):
    t = []
    for j in range(len(seq)):
        if j % 2 == 0:
            t.append(seq[j])
        else:
            t.append(seq[j-1])
    return t

def f_zeromeansrepeat(seq):
    t = []
    for j in range(len(seq)):
        if j > 0 and seq[j-1] == 0:
            t.append(1)
        else:
            t.append(seq[j])
    return t

# an example of a pattern is [1,0,0,2,0,3,0,1,1,1]
def f_repetitionpattern(seq, pattern):
    t = []
    i = 0
    j = 0
    while(j < len(seq)):
        t.append(seq[j])
        j = j + pattern[i % len(pattern)]
        i = i + 1
    return t

def f_multpattern(seq,patterns,div_symbol):    
    # We parse the sequence and create a list of lists,
    # of the form [n, L] where n = 0,1 is the pattern
    # to use and L is a list of integers to which it should
    # be applied
    
    parse_list = []
    curr_subseq = []
    curr_pattern = 0
    j = 0
    
    while(j < len(seq)):
        if(seq[j] != div_symbol):
            curr_subseq.append(seq[j])

        if(seq[j] == div_symbol or j == len(seq)-1):
            if( len(curr_subseq) != 0 ):
                parse_list.append([curr_pattern,curr_subseq])

        if(seq[j] == div_symbol):
            curr_pattern = (curr_pattern + 1) % len(patterns)
            curr_subseq = []
        
        j = j + 1

    t = []    
    for q in parse_list:
        t = t + f_repetitionpattern(q[1],patterns[q[0]])

    return t
    
def f_varpattern(seq, init_symbol):    
    # We parse seq into A.S.B where "." stands for concatentation,
    # A, B are sequences and S is the init_symbol. Then we run A
    # as a pattern on B using f_repetitionpattern
    
    A = []
    B = []
    
    Sfound = False
    for x in seq:
        if( x == init_symbol ):
            Sfound = True
        else:
            if( Sfound == False ):
                A.append(x)
            else:
                B.append(x)
    
    t = f_repetitionpattern(B,A)
    return t