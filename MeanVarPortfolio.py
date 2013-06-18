'''
Created on Mar 15, 2013

@author: GuyZ
'''

import numpy as np
import pylab as py

from math import sqrt


class _MeanVarPortfolio(object):
    '''
    Representation of a Mean-Var Portfolio. Used in CAPM.
    
    Summary:
        The Mean-Variance Portfolio Model leads to Markowitz Optimal Portfolio Selection,
        as well as the CAPM. Under this model (and assumptions!) we find two optimal portfolios and efficient frontiers:
        
        1. Mean-Variance Optimal Portfolio WITHOUT a risk free asset (see RiskyPortfolio->__compute_optimal_minvar_portfolio):
            - The optimal portfolio here is defined as the minimum variance portfolio. This is the
              vertex point/left-most point in the hyperbole efficient frontier (i.e: the parabola's minimum).
            - Two Fund Theorem: After calculating two optimal portfolios for ANY r1, r2, can calculate
              the optimal portfolio for any 'r' (much faster then doing this directly for every 'r').
            - Efficient Frontier - Can use the two fund theorem, which is a quadratic function to construct
              the efficient frontier. We only take the better solution out of the two ('half' of the parabola).
        
        2. Mean-Variance Optimal Portfolio WITH a risk free asset (see RiskFreePortfolio->__compute_optimal_sharpe_portfolio)
            - Also known as the Sharpe Optimal Portfolio and the 'Market Portfolio' - defined as the portfolio
              with the maximum achievable sharp ratio (excess_return/volatility)
            - A new concept - Excess return. With a risk-free asset we're always interested in the 
              return above the risk-free asset, and not the absolute return.
            - One Fund Theorem: Diversification between a single risky fund/portfolio x={x1,...,xn} and x0
              the risk free asset can yield the (sharpe) optimal portfolio and all efficient portfolios on the market line.
            - 'tau' determines the risk aversion (higher tau = we put more on the risk free asset), 
              but the optimal portfolio is independant of it, I.E: the Sharpe Ratio remains the same!
            - Efficient frontier: AKA the Market Capital Line, used in CAPM. 
              Unlike the portfolio above, the frontier is LINEAR - where the risk free asset is the intercept
              (return without risk), and the slope is the maximal achievable sharp-ratio, or in other words
              the maximum angle theta (slope).
              
        The two efficient frontiers meet in the Sharpe Optimal/Market Portfolio point (sigma_market, mean_market).
        They are tangent at this point, where in other points the market line is strictly greater.
        
        Important note: The maximum achievable Sharpe Ratio = the slope of the Capital Market Line (CML) = the Price of Risk.
        Used to compare projects and see if they are worth-while.
        
        AS A CONCLUSION - ALL EFFICIENT INVESTMENTS/PROJECTS WILL LIE ON THE CML. 
        
    '''


    def __init__(self, means, covars):
        '''
        Constructs a new portfolio. Should not be instantinated (only a base class)
        
        Properties:
        - means: a d-dimensional vector denoting the expected returns of the different assets
        - covars: covariance matrix between all assets
        '''
        
        self.d = len(means)
        self.means = np.mat(means) # in case it is an ndarray
        self.covars = np.mat(covars) # "
    
    def calc_return(self, x):
        '''
        Returns E(Rx(t)) =  --> the expected return (mean) of the portfolio.
        '''
        
        return (self.means*np.mat(x).T)
    
    def calc_variance(self, x):
        '''
        Returns the variance of the portfolio's return (square of the volatility) - sigmax^2 := x'*cov*x
        '''
        x1 = np.mat(x)
        return x1*self.covars*x1.T
    
    def calc_volatility(self, x):
        return sqrt(self.calc_variance(x))
        
        
class RiskyPortfolio(_MeanVarPortfolio):
    '''
    Representation of 'risky' portfolio (i.e: first type of portfolio - WITHOUT a risk free asset).
    '''
    
    def __init__(self, means, covars):
        _MeanVarPortfolio.__init__(self, means, covars)
        
        # Optimal portfolio function for this model - In a risky only portfolio, we define it as the minimum variance portfolio.
        self.compute_optimal_portfolio = self.__compute_optimal_minvar_portfolio        
        
    def efficient_volatility(self, r):
        '''
        Returns the optimal (efficient) volatility associated with this return (r).
        
        This is the inverse of the efficient frontier.
        '''
        
        optimal_portfolio = self.compute_optimal_portfolio_for_return(r)
        return self.calc_volatility(optimal_portfolio)
     
    def __compute_optimal_minvar_portfolio(self):
        '''
        Computes the optimal (lowest variance) portfolio 'x' for ANY return.
        
        This is the minimum of the parabola on efficient frontier.
        
        Solves the optimization problem - min(var(x)), s.t:
        (*): sum(x) = 1
        
        Optimization problem is solved using lagrangian.
        '''
        
        A = np.mat(np.zeros((self.d+1,self.d+1)))
        
        A[0:self.d,0:self.d] = 2.0*self.covars # minimize
        
        A[0:self.d,self.d] = -1.0 # constraint 'v'
        A[self.d,0:self.d] = 1.0 # constraint 'v'
        
        b = np.zeros(self.d+1)
        b[self.d] = 1.0
        
        return np.linalg.solve(A, b)[:-1] # returns A^(-1)*b        
         
    # TODO: create a single lagrangian solver for both (just for fun), or use a generic optimizer function
    # like LS.        
    def compute_optimal_portfolio_for_return(self, r):
        '''
        Computes the optimal (lowest variance) portfolio 'x' for a fixed target return 'r'.
        
        Solves the optimization problem - min(var(x)), s.t:
        (*): means*x'=r
        (**): sum(x) = 1
        
        Optimization problem is solved using lagrangian.
        '''
        
        A = np.mat(np.zeros((self.d+2,self.d+2)))
        
        A[0:self.d,0:self.d] = 2.0*self.covars # minimize
        
        A[0:self.d,self.d] = -1.0*self.means.T  # constraint 'u'
        A[0:self.d,self.d+1] = -1.0 # constraint 'v'
        
        A[self.d,0:self.d] = self.means  # constraint 'u'
        A[self.d+1,0:self.d] = 1.0 # constraint 'v'
        
        b = np.zeros(self.d+2)
        b[self.d] = r
        b[self.d+1] = 1.0
        
        return np.linalg.solve(A, b)[:-2]
        
class RiskFreePortfolio(_MeanVarPortfolio):
    '''
    Representation of 'risk-free' (i.e: second type of portfolio - WITH a risk free asset).
    
    Extra Properties:
    - rf: the risk free interest rate
    - tau: the risk aversion factor
    '''
    
    def __init__(self, means, covars, rf, tau):
        _MeanVarPortfolio.__init__(self, means, covars)

        # Optimal portfolio function for this model - In a risk-free portfolio, it is the Sharpe Portfolio.
        self.compute_optimal_portfolio = self.__compute_optimal_sharpe_portfolio        
        
        self.rf = rf
        self.tau = tau
        self.excess_mean = self.means - self.rf # excess returns
        self.market_beta = self.__calc_market_beta()
        
    def efficient_volatility(self, r):
        '''
        Returns the optimal (efficient) volatility associated with this return (r).
        
        This is the inverse of the efficient frontier.
        '''
        
        return (r-self.rf)/self.market_beta
    
    def efficient_return(self, sigma):
        '''
        Returns the optimal (efficient) return associated with this volatility (sigma).
        
        This is the efficient frontier.
        
        According to the one fund theorem - the frontier is linear function with intercept rf,
        and slope=maximum sharpe ratio
        '''
        
        return (self.rf + self.market_beta*sigma)
    
    def __compute_optimal_sharpe_portfolio(self):
        '''
        Computes the optimal(Sharpe) portfolio === the market portfolio.
        In others words - the portfolio with the lowest Sharpe Ratio = excess_return/volatility
        '''    
        
        positions = self.excess_mean*np.linalg.inv(self.covars)
        return (positions/np.sum(positions))
        
    def __calc_market_beta(self):
        '''
        Market beta - Capital Market Line Slope/Maximum Sharpe Ratio/Market risk.
        '''
        
        sharpe_portfolio = self.compute_optimal_portfolio()
        return (self.calc_return(sharpe_portfolio) - self.rf)/self.calc_volatility(sharpe_portfolio)

def test():
    cov_m = np.mat([[0.0010, 0.0013, -0.0006, -0.0007, 0.0001,  0.0001,  -0.0004, -0.0004],
                    [0.0013, 0.0073, -0.0013, -0.0006, -0.0022, -0.0010, 0.0014,  -0.0015],
                    [-0.0006, -0.0013, 0.0599, 0.0276, 0.0635, 0.0230, 0.0330, 0.0480],
                    [-0.0007, -0.0006, 0.0276, 0.0296, 0.0266, 0.0215, 0.0207, 0.0299],
                    [0.0001, -0.0022, 0.0635, 0.0266, 0.1025, 0.0427, 0.0399, 0.0660],
                    [0.0001, -0.0010, 0.0230, 0.0215, 0.0427, 0.0321, 0.0199, 0.0322],
                    [-0.0004, 0.0014, 0.0330, 0.0207, 0.0399, 0.0199, 0.0284, 0.0351],
                    [-0.0004, -0.0015, 0.0480, 0.0299, 0.0660, 0.0322, 0.0351, 0.0800]])
 
    mean_ret = np.array([0.0315, 0.0175, -0.0639, -0.0286, -0.0675, -0.0054, -0.0675, -0.0526])
     
    rf = 0.015
    tau = 0.01
    mv = RiskyPortfolio(mean_ret, cov_m)
    
    x = np.array([5.6762, -0.5214, 2.5467, -1.7475, -1.6448, 4.2951, -8.6474, 1.0431])
    assert round(mv.calc_return(x)*100.0, 2) == 67.36
    assert round(mv.calc_volatility(x)*100.0, 0) == 100.00
 
    m_p = mv.compute_optimal_portfolio()
    print "Volatility of min var portfolio: %f" % (mv.calc_volatility(m_p)*100.0)
 
    mv2 = RiskFreePortfolio(mean_ret, cov_m, rf, tau)    
    s_p = mv2.compute_optimal_portfolio()
    print "Mean return of sharpe portfolio: %f" % (mv2.calc_return(s_p)*100.0)
 
    print "Slope of CML: %f" % mv2.market_beta
    
def plot_efficient_frontiers_test():
    cov_m = np.mat([[0.0010, 0.0013, -0.0006, -0.0007, 0.0001,  0.0001,  -0.0004, -0.0004],
                    [0.0013, 0.0073, -0.0013, -0.0006, -0.0022, -0.0010, 0.0014,  -0.0015],
                    [-0.0006, -0.0013, 0.0599, 0.0276, 0.0635, 0.0230, 0.0330, 0.0480],
                    [-0.0007, -0.0006, 0.0276, 0.0296, 0.0266, 0.0215, 0.0207, 0.0299],
                    [0.0001, -0.0022, 0.0635, 0.0266, 0.1025, 0.0427, 0.0399, 0.0660],
                    [0.0001, -0.0010, 0.0230, 0.0215, 0.0427, 0.0321, 0.0199, 0.0322],
                    [-0.0004, 0.0014, 0.0330, 0.0207, 0.0399, 0.0199, 0.0284, 0.0351],
                    [-0.0004, -0.0015, 0.0480, 0.0299, 0.0660, 0.0322, 0.0351, 0.0800]])
 
    mean_ret = np.array([0.0315, 0.0175, -0.0639, -0.0286, -0.0675, -0.0054, -0.0675, -0.0526])
    
    rf = 0.015
    tau = 0.01
    mv = RiskyPortfolio(mean_ret, cov_m)
    mv2 = RiskFreePortfolio(mean_ret, cov_m, rf, tau)
         
    y = np.linspace(0,1.0,1000) # 100 linearly spaced numbers
    x = np.zeros(1000)
    x2 = np.zeros(1000)
    for i in xrange(len(y)):
        x[i] = mv.efficient_volatility(y[i])
        x2[i] = mv2.efficient_volatility(y[i])
    
    py.plot(np.round(x*100.0,2),np.round(y*100.0,2))
    py.plot(np.round(x2*100.0,2),np.round(y*100.0,2))
    
    # TODO: labels.
    # TODO: limit to x=100? (the frame).
    py.show()
    
#    mv2 = RiskFreePortfolio(mean_ret, cov_m, rf, tau)
        
        
#test()
plot_efficient_frontiers_test()