"""
This module defines the function that we want to try to learn with NNs.
"""

import random

# The actual functions to learn
    
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
    
    
##################
# DEFINING TASKS
##################

# Default sampling from space of inputs
def generate_input_seq_default(max_symbol,input_length):
    return [random.randint(0,max_symbol) for k in range(input_length)]

###########
# COPY TASK
#
# In this task the input is simply copied to the output (although we
# require the RNN to output the first output symbol after the last
# input symbol has been read, so this effectively requires the system
# to store the input and later retrieve it)

def get_task_copy(N, Ntest):
    generate_input_seq = generate_input_seq_default
    func_to_learn = f_identity
    N_out = N - 2
    Ntest_out = Ntest - 2
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

##################
# REPEAT COPY TASK
#
# In this task every digit of the input is repeated.
#
# put n zeros before the 1, for a copy task with n + 1 copies

def get_task_repeat_copy(N, Ntest):
    generate_input_seq = generate_input_seq_default
    no_of_copies = 2
    pattern = [0]*(no_of_copies - 1) + [1]
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = no_of_copies * (N - 2)
    Ntest_out = no_of_copies * (Ntest - 2)
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
################
# PATTERN TASK 1
def get_task_pattern_1(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern = [0,1,1] # so (a,b,c,d,e,f,...) goes to (a,a,b,c,c,d,e,e,...)
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = (N - 2) + divmod(N - 2, 2)[0] # N - 2 plus the number of times 2 divides N - 2
    Ntest_out = (Ntest - 2) + divmod(Ntest - 2, 2)[0]
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
################
# PATTERN TASK 2
def get_task_pattern_2(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern = [0,2] # so (a,b,c,d,e,f,...) goes to (a,a,c,c,e,e,...)
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = N - 2 + divmod(N - 2, 2)[0]
    Ntest_out = Ntest - 2 + divmod(Ntest - 2, 2)[0]
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
################
# PATTERN TASK 3
def get_task_pattern_3(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern = [0,2,-1] # so (a,b,c,d,e,f,...) goes to (a,a,c,b,b,d,c,c,e,d,d,...)
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = 4 + (N - 2 - 2) * 3
    Ntest_out = 4 + (Ntest - 2 - 2) * 3
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

################
# PATTERN TASK 4
def get_task_pattern_4(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern = [0,2,1,2,-2,-1] # so (a,b,c,d,e,f,...) goes to (a,a,c,d,f,d,c,c,e,f,h,f,e,e,...)
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = len(func_to_learn([0]*(N-2)))
    Ntest_out = len(func_to_learn([0]*(Ntest-2)))
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

################
# PATTERN TASK 5
def get_task_pattern_5(N, Ntest):
    pattern = [4,1,1,-4] # so (a,b,c,d,e,f,...) goes to (a,e,f,g,k,...)
    func_to_learn = lambda s: f_repetitionpattern(s,pattern)
    N_out = len(func_to_learn([0]*(N-2)))
    Ntest_out = len(func_to_learn([0]*(Ntest-2)))
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

#########################
# MULTIPLE PATTERN TASK 1
def get_task_mult_pattern_1(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern1 = [1] # so (a,b,c,d,e,f,...) goes to (a,b,c,d,e,f,...)
    pattern2 = [0,1] # so (a,b,c,d,e,f,...) goes to (a,a,b,b,...)
    func_to_learn = lambda s: f_multpattern(s,[pattern1,pattern2],div_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
#########################
# MULTIPLE PATTERN TASK 2
def get_task_mult_pattern_2(N, Ntest):
    # Almost everything is the same as mult pattern 1, but in pattern 2 we 
    # make sure there is a div symbol somewhere in the sequence
    func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq = get_task_mult_pattern_1()
    
    def generate_input_seq_forcediv(max_symbol,input_length):
        t = [random.randint(0,max_symbol) for k in range(input_length)]
        div_pos = random.randint(0,len(t)-1)
        t[div_pos] = div_symbol
        return t
    
    generate_input_seq = generate_input_seq_forcediv
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

#########################
# MULTIPLE PATTERN TASK 3
def get_task_mult_pattern_3(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern1 = [1] # so (a,b,c,d,e,f,...) goes to (a,b,c,d,e,f,...)
    pattern2 = [0,1] # so (a,b,c,d,e,f,...) goes to (a,a,b,b,...)
    pattern3 = [0,2] # so (a,b,c,d,e,f,...) goes to (a,a,c,c,...)
    func_to_learn = lambda s: f_multpattern(s,[pattern1,pattern2,pattern3],div_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
#########################
# MULTIPLE PATTERN TASK 4
def get_task_mult_pattern_4(N, Ntest):
    generate_input_seq = generate_input_seq_default
    pattern1 = [0,1] # so (a,b,c,d,e,f,...) goes to (a,a,b,b,...)
    pattern2 = [2,-1] # so (a,b,c,d,e,f,...) goes to (a,c,b,d,c,...)
    func_to_learn = lambda s: f_multpattern(s,[pattern1,pattern2],div_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 7
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

#########################
# VARIABLE PATTERN TASK 1
#
# The input is a pattern together with a string to which we are supposed to apply the
# pattern, separated by an initial symbol. There is no division symbol.

def generate_input_seq_varpattern1(max_symbol,input_length):
    varpatterns = [[1],[2],[0,1],[0,2],[1,2]]
    vp = varpatterns[random.randint(0,len(varpatterns)-1)]
    t = vp + [init_symbol] + [random.randint(0,max_symbol) for k in range(input_length-len(vp)-1)]
    return t

def get_task_var_pattern_1(N, Ntest):
    generate_input_seq = generate_input_seq_varpattern1
    func_to_learn = lambda s: f_varpattern(s,init_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 10
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
#########################
# VARIABLE PATTERN TASK 2

def generate_input_seq_varpattern2(max_symbol,input_length):
    varpatterns = [[1],[2]]
    varpatterns = varpatterns + [[0,1],[0,2],[1,2]]
    varpatterns = varpatterns + [[0,1,0],[0,1,1],[0,1,2],[0,2,0],[0,2,1],[0,2,2],[1,1,2],[1,2,2]]
    varpatterns = varpatterns + [[0,0,0,1],[0,0,0,2],[0,0,1,2],[0,1,1,2],[0,1,0,2],[0,2,0,2]]
    vp = varpatterns[random.randint(0,len(varpatterns)-1)]
    t = vp + [init_symbol] + [random.randint(0,max_symbol) for k in range(input_length-len(vp)-1)]
    return t

def get_task_var_pattern_2(N, Ntest):
    generate_input_seq = generate_input_seq_varpattern2
    func_to_learn = lambda s: f_varpattern(s,init_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 13
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq

#########################
# VARIABLE PATTERN TASK 3
#
# In this task we randomly generate the pattern from the alphabet 0,1,2
# We also generate longer sequences than in task 1 or 2. By default
# we generate patterns between length 1 and 8

def generate_input_seq_varpattern3(max_symbol,input_length):
    vp_length = random.randint(1,max_pattern_length)
    
    while( True ):
        vp = [random.randint(0,2) for k in range(vp_length)]
        
        # We cannot allow patterns that are all zeros
        if( reduce( lambda x,y : x + y, vp) > 0 ):
            break
    
    t = vp + [init_symbol] + [random.randint(0,max_symbol) for k in range(input_length-len(vp)-1)]
    return t

def get_task_var_pattern_3(N, Ntest):
    generate_input_seq = generate_input_seq_varpattern3
    func_to_learn = lambda s: f_varpattern(s,init_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 20
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    
#########################
# VARIABLE PATTERN TASK 4
#
# In this task we randomly generate the pattern from the alphabet -2,-1,0,1,2

def generate_input_seq_varpattern4(max_symbol,input_length):
    vp_length = random.randint(1,max_pattern_length)
    
    while( True ):    
        vp = [random.randint(-2,2) for k in range(vp_length)]
        
        # We cannot allow patterns that add up to a non-positive integer
        if( reduce( lambda x,y : x + y, vp) > 0 ):
            break
    
    t = vp + [init_symbol] + [random.randint(0,max_symbol) for k in range(input_length-len(vp)-1)]
    return t

def get_task_var_pattern_4(N, Ntest):
    generate_input_seq = generate_input_seq_varpattern4
    func_to_learn = lambda s: f_varpattern(s,init_symbol)
    N_out = 2*(N-2)
    Ntest_out = 2*(Ntest-2)
    seq_length_min = 20
    return func_to_learn, N_out, Ntest_out, seq_length_min, generate_input_seq
    

def get_task(task, N, Ntest):
    if( task == 'copy' ):
        return( get_task_copy(N, Ntest) )
        
    if( task == 'repeat copy' ):
        return( get_task_repeat_copy(N, Ntest) )
        
    if( task == 'pattern 1' ):
        return( get_task_pattern_1(N, Ntest) )
        
    if( task == 'pattern 2' ):
        return( get_task_pattern_2(N, Ntest) )
        
    if( task == 'pattern 3' ):
        return( get_task_pattern_3(N, Ntest) )
        
    if( task == 'pattern 4' ):
        return( get_task_pattern_4(N, Ntest) )
        
    if( task == 'pattern 5' ):
        return( get_task_pattern_5(N, Ntest) )
        
    if( task == 'mult pattern 1' ):
        return( get_task_mult_pattern_1(N, Ntest) )

    if( task == 'mult pattern 2' ):
        return( get_task_mult_pattern_2(N, Ntest) )
        
    if( task == 'mult pattern 3' ):
        return( get_task_mult_pattern_3(N, Ntest) )
        
    if( task == 'mult pattern 4' ):
        return( get_task_mult_pattern_4(N, Ntest) )
        
    if( task == 'variable pattern 1' ):
        return( get_task_var_pattern_1(N, Ntest) )
        
    if( task == 'variable pattern 2' ):
        return( get_task_var_pattern_2(N, Ntest) )
        
    if( task == 'variable pattern 3' ):
        return( get_task_var_pattern_3(N, Ntest) )
        
    if( task == 'variable pattern 4' ):
        return( get_task_var_pattern_4(N, Ntest) )