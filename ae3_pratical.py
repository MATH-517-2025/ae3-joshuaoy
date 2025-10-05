import numpy as np
import math
import matplotlib.pyplot as plt
n=1000    
np.random.seed(1)
#create the function m(x)
m=lambda x: np.sin((x/3+0.1)**(-1))

def create_data(alpha,beta,n,sigma):
    #draw n sample from a beta distribution 
    X = np.random.beta(alpha, beta,n)

    #draw n eplison error with mean zero and sigma=1
    epsilon = np.random.normal(0, sigma, n)
    #create Y as m(X)+ epsilon
    Y=m(X)+epsilon 
    return X,Y
#function to get blocks of size N sometimes approximately 
def get_blocks(data,N):
    number_of_block=len(data)//N
    return np.array_split(data, number_of_block)
#function to estimate theta_hat and sigma_hat
def estimate_unknown_quantities(data,N,):
    n=len(data,)
    X_blocks =get_blocks(data[0], N)
    Y_blocks = get_blocks(data[1], N)
    coeffs=[]
    theta_hat=0
    sigma_hat=0
    for i in range(len(X_blocks)):
        coeffs=np.polyfit(X_blocks[i], Y_blocks[i], 4)
        mhat_dd = 12*coeffs[0]*X_blocks[i]**2 + 6*coeffs[1]*X_blocks[i] + 2*coeffs[2]

        mhat=coeffs[0]*X_blocks[i]**4+coeffs[1]*X_blocks[i]**3+coeffs[2]*X_blocks[i]**2
        coeffs[3]*X_blocks[i]+coeffs[4]
        theta_hat+=np.sum(mhat_dd)
        residuals=Y_blocks[i]-mhat
        sigma_hat+=np.sum(residuals**2)
    theta_hat/=n
    sigma_hat/=(n-5*N)
    return theta_hat,sigma_hat

#function to estimate h_amise
def estimate_h_amise(n,  theta_hat,sigma_hat, supp_X=1):
    h_amise_hat = (35 * sigma_hat * supp_X / (n * theta_hat)) ** (1 / 5)
    return h_amise_hat

n=1000
alpha=4
beta=30
sigma=1
Ns=[5,10,20,50,100,200,500,1000]
results_h_amise_N=[]
#generate the h_values for deferent size of N
for N in Ns:
        data=create_data(alpha,beta,n,sigma)
        estimates=estimate_unknown_quantities(data,N)
        h_amise = estimate_h_amise(n,estimates[0],estimates[1])
        results_h_amise_N.append((N,h_amise))
print(results_h_amise_N)
Ns, h_values = zip(*results_h_amise_N)
plt.figure(figsize=(10, 6))
plt.plot(Ns, h_values, marker='o')
plt.xscale('log')
plt.title('Impact of N on h_AMISE')
plt.xlabel('Sample Size N')
plt.ylabel('h_AMISE')
plt.grid()
plt.savefig('./Impact_N_h_AMISE.png')  
plt.show()
plt.close()
