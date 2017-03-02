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