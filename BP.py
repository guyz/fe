'''
Created on Mar 9, 2013

@author: GuyZ
'''

import numpy as np
from math import exp, sqrt

def print_tree(tree, precision=1):
    '''
    Prints the tree/lattice to the screen
    '''
    
    out_tree = np.round(tree,precision)
    for i in xrange(out_tree.shape[0]):
        print "\t".join(map(str, out_tree[i,:]))

def build_binomial_price_tree(s0, u, d, n):
    '''
    Builds the underlying's price tree by working forward from
    t=0 to expiration (t=n) --> total of n+1 time slots.
    The price tree is an upper triangular 2D array, where each column
    marks the different possible security prices S(t).
    
    arguments:
    - s0: initial stock price
    - u: upward movement interest rate
    - d: downward movement interest rate
    - n: number of periods
    '''
    
    a = np.arange(0,n+1)
    A = np.tile(a,(n+1,1))
    C = np.triu(np.ones( (n+1,n+1) ))

    # U_e is the matrix of all exponents pertaining to up movements.
    # e.g: the first row is 0, 1, 2, counting (and multiplying)
    # by the amount of times we moved 'up'.
    U_e = np.triu(A-A.T)
    # base matrix
    U_b = u*C
    
    # Similarily - down direction.
    D_e = np.triu(A.T)
    D_b = d*C
    
    # Up/Down binomial distribution matrices
    U = np.power(U_b, U_e)
    D = np.power(D_b, D_e)
    
    return np.triu( s0 * ( U*D ) )

def future_price(St, expected_Ct, t, n, params=None):
    '''
    Returns Ct:=Ft:=the 'price'(*). associated with this future contract at time 't'.
    
    For futures, Sn=Fn, and Ft=Et[Ft+1]. Therefore, simply return itself,
    as the binomial backpropagation takes care of the rest.
    
    (*): Ft is not the price you must pay to buy/sell 1 contract (since by definition
    a future has no value costs nothings), but rather the quantity used to 
    derive the payoff (i.e: cashflow). In other words: direction*(F[t]-F[t-1]) 
    is the payoff received at time t, for holding this contract from time t-1.
    
    arguments:
    - St: stock price distribution at time t. Provided from the stock lattice.
    - expected_Ct: Et[Ct+1], expected price (in futures - not real price, see (*)) in 1-step binomial model.
    - t: current time
    - n: total periods
    - params: unused here
    '''
    
    if (n == t):
        Ct = St
    else:
        Ct = expected_Ct
    
    return Ct

def option_price(St, expected_Ct, t, n, params):
    '''
    Returns Ct:=the price associated with this option at time 't'.
    
    arguments:
    - St: stock price distribution at time t. Provided from the stock lattice.
    - expected_Ct: Et[Ct+1] in 1-step binomial model.
    - t: current time
    - n: total periods
    - params:
        - direction: 1 (call), -1 (put)
        - type: 0 (european), 1 (american)
        - strike: Option's strike price
        - R: rate
        - debug_early: if set to true - prints information when american option should be exercised
    '''
    
    if (n == t):
        # Sn = max{direction*(Sn-K),0}
        Ct = np.maximum(params['direction']*(St-params['strike']),0)
    else:
        if (params['type'] == 1):
            # american option --> compare against stock price. See if exercising early is better
            
            if (params['direction'] < 0):
                # align St so that 0 --> K. Otherwise result lattice will contain 'K's instead of '0's in empty/undefined cells
                St[St == 0.0] =  params['strike']
            Ct = np.maximum(expected_Ct/params['R'],np.maximum(params['direction']*(St-params['strike']),0))
            
            if ('debug_early' in params and params['debug_early']):
                if ( (Ct > (expected_Ct/params['R'])).any()):
                    print "Should exercise in time t=" + str(t)
        else:
            # european option --> can't excericse early --> Ct=Et[Ct+1]/R
            Ct = expected_Ct/params['R']
    
    return Ct

''' 
Very hackish - either remove or improve.
'''
def build_binomial_value_tree_chooser(chooser_vec, t_chooser, price_tree, q, n, pricing_func=future_price, params=None):
    '''
    This method performs Backward Propagation in order
    to build the valuations tree (i.e: the risk-free estimated
    cash flows according to the risk neutral probability q).
    
    arguments:
    - price_tree: The base price tree
    - q: risk neutral probability
    - R: risk free discount rate - used when pricing options (optional)
    - transform_func: 
    '''
    res_tree = np.zeros((n+1,n+1))
    
    Q = np.matrix( ( np.array([q, 1-q]) ) )
    
    # Run transformation to set the 'first' (i.e: t=n) vector according to the underlying type (e.g: Futures, Options).
    Sn = price_tree[:,n] 
    res_tree[:,n] = pricing_func(Sn, Sn, n, n, params)
    
    for t in xrange(n,0,-1):
        # Ct - value/cash flow distribution vector in time t (previous time)
        Ct = res_tree[:,t]
        # St - actual stock/underlying price distribution " t-1 (current time)
        St = price_tree[:,t-1]
        # Pt - helper matrix for E[Ct] calculation
        Pt = np.matrix(( np.concatenate((Ct[:t-n-1],np.zeros(n+1-t))), np.concatenate((Ct[1:],np.zeros(1))) ))
        
        if (t-1==t_chooser):
            res_tree[:,t-1] = chooser_vec
        else:
            res_tree[:,t-1] = pricing_func(St, Q*Pt, t-1, n, params) # This function runs backward, i.e: Calculates C[t-1] = f(C[t])

    return res_tree

def build_binomial_value_tree(price_tree, q, n, pricing_func=future_price, params=None):
    '''
    This method performs Backward Propagation in order
    to build the valuations tree (i.e: the risk-free estimated
    cash flows according to the risk neutral probability q).
    
    arguments:
    - price_tree: The base price tree
    - q: risk neutral probability
    - R: risk free discount rate - used when pricing options (optional)
    - transform_func: 
    '''
    res_tree = np.zeros((n+1,n+1))
    
    Q = np.matrix( ( np.array([q, 1-q]) ) )
    
    # Run transformation to set the 'first' (i.e: t=n) vector according to the underlying type (e.g: Futures, Options).
    Sn = price_tree[:,n] 
    res_tree[:,n] = pricing_func(Sn, Sn, n, n, params)
    
    for t in xrange(n,0,-1):
        # Ct - value/cash flow distribution vector in time t (previous time)
        Ct = res_tree[:,t]
        # St - actual stock/underlying price distribution " t-1 (current time)
        St = price_tree[:,t-1]
        # Pt - helper matrix for E[Ct] calculation
        Pt = np.matrix(( np.concatenate((Ct[:t-n-1],np.zeros(n+1-t))), np.concatenate((Ct[1:],np.zeros(1))) ))
        res_tree[:,t-1] = pricing_func(St, Q*Pt, t-1, n, params) # This function runs backward, i.e: Calculates C[t-1] = f(C[t])

    return res_tree
        
    

class BPM(object):
    '''
    A representation of a Binomial Pricing Model, where the
    normal model parameters (u, d, q) are constructed by
    calibrating them to the Black-Scholes parameters (r,q)
    '''

    def __init__(self, s0, T, n, r, c, sigma=0.2):
        '''
        Constructs a new Binomial Pricing Model
        
        arguments (model parameters):
        - s0: Initial Price of the Security
        - T: Expiry time (in years)
        - n: number of periods
        - r: risk free interest rate
        - c: dividend yield
        - sigma: annualized volatility
        '''
        
        self.s0 = s0
        self.T = T
        self.n = n
        self.r = r
        self.c = c
        self.sigma = sigma
        
        self.__calc_black_scholes_params()
        
    def __calc_black_scholes_params(self):
        '''
        Adjusts the model parameters so that they can
        conform with the Black Scholes Model Parameters (r,sigma), I.E: 
        when n-->infinity, we will converge to
        the B&S formula.
        This is done by transforming B&S (r,sigma) to BPM (u, d, q)
        '''
        
        self.R = exp(self.r * (self.T / self.n)) # Risk free interest rate
        self.u = exp(self.sigma * sqrt(self.T / self.n))
        self.d = 1/self.u
        self.q = self.__calc_q()
        
    def __calc_q(self):
        '''
        Returns the risk free probability, in the form
        that diverges to the b&s formula when n-->infinity
        '''
            
        return ( exp( (self.r-self.c)*self.T/self.n ) - self.d ) / ( self.u - self.d )        
        

def run_tests():
    s0=100
    T=0.5
    n=10
    r=0.02
    c=0.01
    sigma=0.2
    
    # Futures
    bpm = BPM(s0, T, n, r, c, sigma)
    stock_lattice = build_binomial_price_tree(bpm.s0, bpm.u, bpm.d, bpm.n)
    price_valuation_lattice = build_binomial_value_tree(stock_lattice, bpm.q, bpm.n)
    print "Futures Lattice"
    print_tree(price_valuation_lattice)
    
    # European Call
    params = {
              'direction':1.0,
              'type':0,
              'strike':bpm.s0,
              'R':bpm.R
              }
    print ""
    print "European Call Options Lattice"
    price_valuation_lattice = build_binomial_value_tree(stock_lattice, bpm.q, bpm.n, option_price, params)
    print_tree(price_valuation_lattice)
    
    # American Put
    T=0.25
    n=3
    r=0.02
    c=0.01
    sigma=0.234
        
    bpm = BPM(s0, T, n, r, c, sigma)
    bpm.u = 1.07
    bpm.d = 0.93458
    bpm.q = 0.557009662
    bpm.R = 1.01
    stock_lattice = build_binomial_price_tree(bpm.s0, bpm.u, bpm.d, bpm.n)
    params = {
              'direction':-1.0,
              'type':1,
              'strike':bpm.s0,
              'R':bpm.R
              }
    print ""
    print "American Put Options Lattice"
    price_valuation_lattice = build_binomial_value_tree(stock_lattice, bpm.q, bpm.n, option_price, params)
    print_tree(price_valuation_lattice)
        
if __name__ == '__main__':
    run_tests()