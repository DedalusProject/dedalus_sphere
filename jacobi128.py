import numpy             as np
import scipy.sparse      as sparse
import scipy.linalg      as linear
import scipy.special     as fun

dtype = np.float128

def sparse_symm_to_banded(matrix):
    """Convert sparse symmetric to upper-banded form."""
    diag = matrix.todia()
    B, I  = max(abs(diag.offsets)), diag.data.shape[1]
    banded = np.zeros((B+1, I), dtype=diag.dtype)
    for i, b in enumerate(diag.offsets):
        if b >= 0:
            banded[B-b] = diag.data[i]
    return banded

def grid_guess(Jacobi_matrix,symmetric=True):
    """Returns reasonable Gauss quadrature grid as the Jacobi matrix eigenvalues.
       For P(z) = [p_{0}(z),...,p_{N}(z)], J.P(z) = z P(z); or p_{N+1}(z) = 0.
    
       Parameters
       ----------
       Jacobi_matrix: square self-adjoint tri-diagonal matrix (arbitrary metric)
       symmetric: T/F
       A normalised metric implies a self-adjoint matrix is symmetric.
       
    """

    if symmetric:
        J_banded = sparse_symm_to_banded(Jacobi_matrix)
        return (np.sort(linear.eigvals_banded(J_banded).real)).astype(dtype)
    else:
        J_dense = Jacobi_matrix.todense()
        return (np.sort(linear.eigvals(J_dense).real)).astype(dtype)

def remainders(Jacobi_matrix,grid):
    """Given length-N three-term recursion, returns P_{N+1}(grid) and P'_{N+1}(grid)
        
       Parameters
       ----------
       Jacobi_matrix: square self-adjoint tri-diagonal matrix (arbitrary metric)
       grid: numpy array
       
    """
    
    J, z, N = sparse.dia_matrix(Jacobi_matrix).data, grid, len(grid)

    P = np.zeros((N,N),dtype=dtype)
    D = np.zeros((N,N),dtype=dtype)

    P[0] = np.ones(N,dtype=dtype)
    D[1] = P[0]/J[-1][1]
    
    if np.shape(J)[0] == 3:
        P[1]  = (z-J[1][0])*P[0]/J[-1][1]
        for n in range(2,N):
            P[n] = ((z-J[1][n-1])*P[n-1] - J[0][n-2]*P[n-2])/J[-1][n]
            D[n] = ((z-J[1][n-1])*D[n-1] - J[0][n-2]*D[n-2])/J[-1][n] + P[n-1]/J[-1][n]
        return ((z-J[1][N-1])*P[N-1] - J[0][N-2]*P[N-2]), ((z-J[1][N-1])*D[N-1] - J[0][N-2]*D[N-2]+P[N-1])
     
    P[1]  = z*P[0]/J[-1][1]
    for n in range(2,N):
        P[n] = (z*P[n-1] - J[0][n-2]*P[n-2])/J[-1][n]
        D[n] = (z*D[n-1] - J[0][n-2]*D[n-2])/J[-1][n] + P[n-1]/J[-1][n]

    return (z*P[N-1] - J[0][N-2]*P[N-2]), (z*D[N-1] - J[0][N-2]*D[N-2] + P[N-1])

def gauss_quadrature(Jacobi_matrix,mass=1,niter=3,guess=None,report_error=False):
    """Returns accurate grid and weights for general Gauss quadrature.
    
       Parameters
       ----------
       Jacobi_matrix: square self-adjoint tri-diagonal matrix (arbitrary metric)
       mass: float
       mass = integral_{-1}^{+1} w(z)dz
       niter: int
       Number of times to run the Newton iteration for P_{N+1}(grid) = 0
       guess: array
       Optional grid estimate
       report_error: T/F
       Show the error estimate in the grid values.
    
    """
    
    if guess == None:
        z = grid_guess(Jacobi_matrix)
    else:
        z = guess
    
    #Newton iteration
    for i in range(niter):
        P, D = remainders(Jacobi_matrix,z)
        if report_error: print(np.max(np.abs(P/D)))
        z = z - P/D

    w = 1/((1-z**2)*D**2)
    w = (mass/np.sum(w))*w
    
    return z, w

def three_term_recursion(Jacobi_matrix,grid,max_degree,init):
    """Returns weighted orthogonal polynomials on a grid from a three-term recursion.
       P_{-1}(z) = 0; P_{0}(z) = init(z)
       P_{n}(z)  = (A_{n}*z + B_{n})*P_{n-1}(z) + C_{n}*P_{n-2}(z)
        
       Parameters
       ----------
       Jacobi_matrix: square self-adjoint tri-diagonal matrix (arbitrary metric)
       grid: array
       max_degree: int
       Returns max_degree + 1 weighted polynomials
       init: array
       len(init) = len(grid)
    
    """
    
    if max_degree==0: return np.array([init])
    
    J, z, N = sparse.dia_matrix(Jacobi_matrix).data, grid, max_degree+1
    
    P     = np.zeros((N,len(grid)),dtype=dtype)
    P[0]  = init
    
    if np.shape(J)[0] == 3:
        P[1]  = (z-J[1][0])*P[0]/J[-1][1]
        for n in range(2,N):
            P[n] = ((z-J[1][n-1])*P[n-1] - J[0][n-2]*P[n-2])/J[-1][n]
        return P
    
    P[1]  = z*P[0]/J[-1][1]
    for n in range(2,N):
        P[n] = (z*P[n-1] - J[0][n-2]*P[n-2])/J[-1][n]
    return P

def normalise(functions,weights):
    """unit-normalise functions on a grid according to weighted L2 inner product. """
    for k in range(len(functions)):
        functions[k] = functions[k]/np.sqrt(np.sum(weights*functions[k]**2))

def quadrature(max_degree,a,b,**kw):
    """Returns accurate grid and weights for specific Gauss-Jacobi quadrature.
        
        Parameters
        ----------
        max_degree: int
        Integrates polynomials on (-1,+1) exactly up to dgree = 2*max_degree+1.
        a,b: int e
        Jacobi polynomial parameter for generating Jacobi matrix
        niter: int
        Number of times to run the Newton iteration for P_{N+1}(grid) = 0
        guess: array
        Optional grid estimate
        report_error: T/F
        Show the error estimate in the grid values.
        
    """
    
    mu = mass(a,b)
    J  = operator('J',max_degree,a,b)
    return gauss_quadrature(J,mass=mu,**kw)

def envelope(a,b,a0,b0,z):
    """Returns sqrt( ((1-z)**(a-a0)) * ((1+z)^(b-b0)) / mass(a,b) )
       Integral envelope(z)**2 (1-z)**a0 (1+z)**b0 dz = 1
        
       Parameters
       ----------
       a,b,a0,b0: float > -1
       z: grid array
       
    """

    mu = mass(a,b)
    return np.exp( ((a-a0)/2)*np.log(1-z) + ((b-b0)/2)*np.log(1+z) )/np.sqrt(mu)

def recursion(max_degree,a,b,grid,init):
    
    if max_degree==0: return np.array([init])
    
    J  = operator('J',max_degree,a,b)
    return three_term_recursion(J,grid,max_degree,init)

def push(op,data):
    """Pushforward"""
    return (op).dot(data)

def pull(op,data):
    """Pullback"""
    return (op.transpose()).dot(data)

def mass(a,b):
    if a=='inf' and b=='inf': return np.sqrt(np.pi)
    if a=='inf'             : return np.exp(fun.gammaln(b+1))
    if              b=='inf': return np.exp(fun.gammaln(a+1))
    return np.exp( (a+b+1)*np.log(2) + fun.gammaln(a+1) + fun.gammaln(b+1) - fun.gammaln(a+b+2) )

def operator(op,max_degree,a,b,format='csr',rescale=None):
    
    def diag(bands,locs):
        return sparse.dia_matrix((bands,locs),shape=(len(bands[0]),len(bands[0])))
        
    N = max_degree+1
    n = np.arange(0,N,dtype=dtype)
    na, nb, nab, nnab = n+a, n+b, n+a+b, 2*n+a+b
    
    # 0 = <a,b| 0
    if op == '0': out = diag([np.zeros(N,dtype=dtype)],[0])
    
    # <a,b| = <a,b| I
    if op == 'I': out = diag([np.ones(N,dtype=dtype)],[0])
    
    # (1-z) <a,b| = <a-1,b| A-
    if op == 'A-' and (a>0):
        if a+b==0:
            middle = na/(2*n+1)
            lower  = (nb+1)/(2*n+1)
            middle[0]  = 2*a
        else:
            middle = 2*na*nab/(nnab*(nnab+1))
            lower  = 2*(n+1)*(nb+1)/((nnab+1)*(nnab+2))
        out = diag([-np.sqrt(lower),np.sqrt(middle)],[-1,0])
        
    # <a,b| = <a+1,b| A+
    if op == 'A+':
        if a+b == 0 or a+b == -1:
            middle = (na+1)/(2*n+1)
            upper  = nb/(2*nab+1)
            middle[0], upper[0] = (1+a)*(1-(a+b)), 0
        else:
            middle = 2*(na+1)*(nab+1)/((nnab+1)*(nnab+2))
            upper  = 2*n*nb/(nnab*(nnab+1))
        out = diag([np.sqrt(middle),-np.sqrt(upper)],[0,+1])
    
    # (1+z) <a,b| = <a,b-1| B-
    if op == 'B-' and (b > 0):
        if a+b == 0:
            middle = nb/(2*n+1)
            lower  = (na+1)/(2*n+1)
            middle[0] = 2*b
        else:
            middle = 2*nb*nab/(nnab*(nnab+1))
            lower  = 2*(n+1)*(na+1)/((nnab+1)*(nnab+2))
        out = diag([np.sqrt(lower),np.sqrt(middle)],[-1,0])
        
    # <a,b| = <a,b+1| B+
    if op == 'B+':
        if a+b == 0 or a+b == -1:
            middle = (nb+1)/(2*n+1)
            upper  = na/(2*nab+1)
            middle[0], upper[0] = (1+b)*(1-(a+b)), 0
        else:
            middle = 2*(nb+1)*(nab+1)/((nnab+1)*(nnab+2))
            upper  = 2*n*na/(nnab*(nnab+1))
        out = diag([np.sqrt(middle),np.sqrt(upper)],[0,+1])

    # ( a - (1-z)*d/dz ) <a,b| = <a-1,b+1| C-
    if op == 'C-' and (a > 0):
        out = diag([np.sqrt(na*(nb+1))],[0])
        
    # ( b + (1+z)*d/dz ) <a,b| = <a+1,b-1| C+
    if op == 'C+' and (b > 0):
        out = diag([np.sqrt((na+1)*nb)],[0])
        
    # ( a(1+z) - b(1-z) - (1-z^2)*d/dz ) <a,b| = <a-1,b-1| D-
    if op == 'D-' and (a > 0) and (b > 0):
        out = diag([np.sqrt((n+1)*nab)],[-1])
        
    # d/dz <a,b| = <a+1,b+1| D+
    if op == 'D+':
        out = diag([np.sqrt(n*(nab+1))],[+1])
        
    # z <a,b| = <a,b| J
    if op == 'J':
        A, B = operator('A+',max_degree,a,b), operator('B+',max_degree,a,b)
        out = 0.5*( pull(B,B) - pull(A,A) )
    
    # <a,b|z=+1>
    if op == 'z=+1':
        n = np.arange(0,N,dtype=np.float64)
        out = np.sqrt(2*n+a+b+1)*np.sqrt(fun.binom(n+a,a))*np.sqrt(fun.binom(n+a+b,a))
        if a+b==-1:
            out[0] = np.sqrt(np.sin(np.pi*np.abs(a))/np.pi)

    # <a,b|z=-1>
    if op == 'z=-1':
        n = np.arange(0,N,dtype=np.float64)
        out = ((-1)**n)*np.sqrt(2*n+a+b+1)*np.sqrt(fun.binom(n+b,b))*np.sqrt(fun.binom(n+a+b,b))
        if a+b==-1:
            out[0] = np.sqrt(np.sin(np.pi*np.abs(b))/np.pi)

    if (op != 'z=+1') and (op != 'z=-1'):
        if format == 'dia': out = sparse.dia_matrix(out)
        if format == 'csr': out = sparse.csr_matrix(out)

    if rescale == None: return out

    return rescale*out

