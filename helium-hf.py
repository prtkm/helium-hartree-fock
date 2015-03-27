
import numpy as np
from itertools import product
from scipy.linalg import eig


def Suv(z1, z2):
    ''' 
    Returns the overlap terms
    '''
    return (((z1 ** 3) * (z2 ** 3) / np.pi ** 2) ** 0.5) * 2 / (z1 + z2) ** 3 * 4 * np.pi 

def S_matrix(zetas, munus):
    '''
    Returns the overlap matrix
    '''
    S = np.zeros((2,2))
    
    for mu, nu in munus:
        S[mu - 1, nu - 1] =  Suv(zetas[mu], zetas[nu])

    return S   


def Tuv(z1, z2):
    '''
    Returns the kinetic energy integrals
    '''
    return 4 * z1 * z2 * np.sqrt(z1 ** 3 * z2 ** 3 ) / (z1 + z2) ** 3 


def Vuv(z1, z2):
    '''
    Returns the nuclear attraction integrals
    '''
    return - 8 * np.sqrt((z1 ** 3 * z2 ** 3)) / (z1 + z2) ** 2  


def Huv(z1, z2):
    '''
    Returns core hamiltonian elements
    '''
    return Tuv(z1, z2) + Vuv(z1, z2)

def H_matrix(zetas, munus):
    '''
    Returns the core hamiltonian matrix
    '''
    H = np.zeros((2,2))
    
    for mu, nu in munus:
        H[mu - 1, nu - 1] =  Huv(zetas[mu], zetas[nu])

    return H    

def I_two_electron(z):
    '''
    Calculates the two electron integrals
    Args: z = [z1, z2, z3, z4]
    Returns: (z1 z2 | z3 z4)
    '''
    A = np.prod(z) ** 1.5
    
    u = z[0] + z[1]
    v = z[2] + z[3]
    
    integral = 32 * A / u ** 2 * (1 / (u * v ** 2) - 1 / (u + v) ** 3 - 1 / u / (u + v) ** 2)
    return integral


def get_C21(z1, z2, k):

    '''
    Calculates C21
    '''

    S12 = Suv(z1, z2)
    C21 = (1 + k ** 2 + 2 * k * S12) ** -0.5
    return C21
    
def density_matrix(z1, z2, k):
    '''
    Returns the density matrix
    '''

    C21 = get_C21(z1, z2, k)

    P11 = 2 * C21 ** 2 * k ** 2
    P12 = 2 * k * C21 ** 2 
    P21 = P12
    P22 = 2 * C21 ** 2    
    P =  np.array([[P11, P12],
                   [P21, P22]])
    return P


def G_matrix(zetas, k, munus, lambdasigmas):

    '''
    Returns the G Matrix
    '''
    
    G = np.zeros((2,2))

    P = density_matrix(zetas[1], zetas[2], k)
                       
    for mu, nu in munus:    

        g = 0
        for l, s in lambdasigmas:

            int1 = I_two_electron((zetas[mu], zetas[nu], zetas[s], zetas[l]))
            int2 = I_two_electron((zetas[mu] , zetas[l], zetas[s], zetas[nu]))
  
            g+= P[l - 1, s - 1] * (int1 - 0.5 * int2)

        G[mu - 1, nu - 1] = g
    return G


def F_matrix(zetas, k, munus, lambdasigmas):
    '''
    Returns the Fock matrix
    '''
    return H_matrix(zetas, munus) + G_matrix(zetas, k, munus, lambdasigmas)


def secular_eqn(F, S):
    '''
    Returns the eigen values and eigen vectors of the secular eqn
    '''    
    ei, C = eig(F, S)
    return ei, C


def get_E0(P, H, F, orb_nos):

    '''
    Returns the hartree-fock energy
    '''
    
    E0 =0
    for mu in orb_nos:

        for nu in orb_nos:
            E0 += 0.5 * (P[mu -1, nu - 1] * (H[mu - 1, nu - 1] + F[mu - 1, nu - 1]))

    return E0

def calculate(z1, z2, k):
    '''
    Calculate HF energy, k, C11, C12
    '''
    
    orb_nos = [1,2]

    # Store zetas in a dictionary    
    zetas = {1:z1, 2:z2}

    # mu-nu combinations
    munus = list(product(orb_nos,repeat=2))
    
    # lambda-sigma combinations
    lambdasigmas =  list(product(orb_nos,repeat=2))
    
    # Calculate overlap integrals
    S = S_matrix(zetas, munus)

    # Calculate core hamiltonian
    H = H_matrix(zetas, munus)
    
    # Calculate density_matrix
    P = density_matrix(z1, z2, k)
    
    # Calculate Fock Matrix
    F = F_matrix(zetas, k, munus, lambdasigmas)

    # Solve secular eqn
    ei, C = secular_eqn(F, S)

    # get k
    k = C[0, 0] / C[1, 0]

    # Calculate HF energy
    E0 = get_E0(P, H, F, orb_nos)

    return E0, k, C[0,0], C[1,0]
     
def main(*args):
    '''
    Takes zeta1, zeta2, k, and max convergence steps as input and performs a 
    scf calculation on the helium atom.
    '''

    from argparse import ArgumentParser

    parser = ArgumentParser(description='Helium Hartree Fock')
    parser.add_argument('-z1', type=float, help="zeta 1", default=1.45)
    parser.add_argument('-z2', type=float, help="zeta2", default=2.91)
    parser.add_argument('-k0', type=float, help="k = C11 / C21", default=2.)      
    parser.add_argument('-n', default=20, type=int,
                        help='Max. number of scf steps')

    args = parser.parse_args()

    z1 = args.z1
    z2 = args.z2
    k0 = args.k0
    n = args.n
    
    k = k0
    C21_0 = get_C21(z1, z2, k)    
    C11_0 = k0 * C21_0
    
    print '-' * 20
    print 'Starting Simulation'
    print '-' * 20
    print '\nInitial Parameters:'
    print 'z1 = {0}, z2 = {1}, k = {2}\n'.format(z1, z2, k0)

    for i in range(n):

        print '-' * 20
        print 'Entering Iteration {0}'.format(i + 1)
        print '-' * 20
        
        print 'Using k = {0}\n'.format(k)

        E0, k, C11, C21 = calculate(z1, z2, k)
        
        print 'Iteration results:'
        print 'E0 = {E0}\nk = {k}\nC11 = {C11}\nC21 ={C21}\n'.format(**locals())
        print 'Convergence level:'
        print 'dC11 = {0:1.5f}'.format(np.abs(C11 - C11_0))
        print 'dC21 = {0:1.5f}\n'.format(np.abs(C21 - C21_0))

        if (np.abs(C11 - C11_0) < 1e-4) and (np.abs(C21 - C21_0) < 1e-4):
            print '\nReached required accuracy in {0} iterations. Stopping Simulation.'.format(i+1)
            print '-' * 20
            converged = True
            break

        else:
            C11_0 = C11
            C21_0 = C21
            
    return

if __name__ == '__main__':
    import sys
    main(*sys.argv)
