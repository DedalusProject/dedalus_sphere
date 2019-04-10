
import numpy     as np
from . import disk128 as disk
from scipy.sparse import linalg as spla
from dedalus.tools.cache import CachedMethod

class Disk:
    def __init__(self,N_max,K_max=0,N_r=None,m_min=None,m_max=None):
        self.N_max, self.K_max  = N_max, K_max
        if N_r == None: N_r = N_max+1
        self.N_r = N_r

        if m_min == None: m_min = -N_max
        if m_max == None: m_max =  N_max
        self.m_min, self.m_max = m_min, m_max

        # grid and weights for the all transforms
        self.grid,self.weights = disk.quadrature(self.N_r-1,niter=3,report_error=False)
        self.radius = np.sqrt( (self.grid+1)/2 )

        self.pushQ, self.pullQ = {}, {}

        for m in range(max(self.m_min-self.K_max,0),self.m_max+self.K_max+1):
            Q = disk.polynomial(self.N_max+self.K_max-self.N_min(m),0,m,self.grid)
            self.pushQ[(m)] = (self.weights*Q).astype(np.float64)
            self.pullQ[(m)] = (Q.T).astype(np.float64)

        z0, weights0 = disk.quadrature(self.N_r-1,a=0.0)

        Q0           = disk.polynomial(self.N_r-1,0,0,z0)
        Q_projection = disk.polynomial(self.N_r-1,0,0,self.grid)

        self.dV = ((Q0.dot(weights0)).T).dot(self.weights*Q_projection)/4

        # downcast to double precision
        self.radius  = self.radius.astype(np.float64)
        self.weights = self.weights.astype(np.float64)
        self.dV = self.dV.astype(np.float64)

    @CachedMethod
    def op(self,op_name,N,k,m,dtype=np.float64):
        return disk.operator(op_name,N,k,m).astype(dtype)

    def zeros(self,m,deg_out,deg_in):
        return disk.zeros(self.N_max,m,deg_out,deg_in)

    @CachedMethod
    def N_min(self,m):
        return disk.N_min(m)

    def forward_component(self,m,deg,data):
        # grid --> coefficients
        N = self.N_max - self.N_min(m-self.K_max) + 1
        if m+deg >= 0: return (self.pushQ[(m+deg)][:N,:]).dot(data)
        else:
            shape = np.array(data.shape)
            shape[0] = N
            return np.zeros(shape)

    def backward_component(self,m,deg,data):
        # coefficients --> grid
        N = self.N_max - self.N_min(m-self.K_max) + 1
        if m+deg >= 0:
            return self.pullQ[(m+deg)][:,:N].dot(data)
        else:
            shape = np.array(data.shape)
            shape[0] = self.N_r
            return np.zeros(shape)

    @CachedMethod
    def tensor_index(self,m,rank):
        num = np.arange(2**rank)
        deg = (-1)**(1+num)
        for k in range(2,rank+1):
            deg += ((-1)**(1+num//2**(k-1))).astype(np.int64)

        if rank == 0: deg = [0]

        start_index = [0]
        end_index = []
        for k in range(2**rank):
            end_index.append(start_index[k]+self.N_max-self.N_min(m-self.K_max)+1)
            if k < 2**rank-1:
                start_index.append(end_index[k])

        return (start_index,end_index,deg)

    @CachedMethod
    def unitary(self,rank=1,adjoint=False):
        return disk.unitary(rank=rank,adjoint=adjoint)

    def forward(self,m,rank,data):

        if rank == 0:
            return self.forward_component(m,0,data[0])

        (start_index,end_index,deg) = self.tensor_index(m,rank)

        if m == 0 and rank == 1:
            unitary = np.sqrt(0.5)*np.array([[0,0],
                                             [1,1j]])
        elif m == 0 and rank == 2:
            unitary = 0.5*np.array([[ 0,  0,   0,  0],
                                    [1.,1.j,-1.j, 1.],
                                    [ 0,  0,   0,  0],
                                    [1.,1.j, 1.j,-1.]])
        elif m == 1 and rank == 2:
            unitary = 0.5*np.array([[ 0,  0,   0,  0],
                                    [1., 1.j,-1.j, 1.],
                                    [1.,-1.j, 1.j, 1.],
                                    [1., 1.j, 1.j,-1.]])
        else:
            unitary = self.unitary(rank=rank,adjoint=True)

        data = np.einsum("ij,j...->i...",unitary,data)

        shape = np.array(np.array(data).shape[1:])
        shape[0] = end_index[-1]

        data_c = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_c[start_index[i]:end_index[i]] = self.forward_component(m,deg[i],data[i])
        return data_c

    def backward(self,m,rank,data):

        if rank == 0:
            return self.backward_component(m,0,data)

        (start_index,end_index,deg) = self.tensor_index(m,rank)

        if m == 0 and rank == 1:
            unitary = np.sqrt(2)*np.array([[0,1],
                                           [0,-1j]])
        elif m == 0 and rank == 2:
            unitary = np.array([[0, 1. ,0, 1.],
                                [0,-1.j,0,-1.j],
                                [0, 1.j,0,-1.j],
                                [0, 1. ,0,-1.]])
        elif m == 1 and rank == 2:
            unitary = 0.5*np.array([[0, 1. , 2. , 1. ],
                                    [0,-1.j, 2.j,-1.j],
                                    [0, 1.j,   0,-1.j],
                                    [0, 1. ,   0,-1.]])
        else:
            unitary = self.unitary(rank=rank,adjoint=False)

        shape = np.array(np.array(data).shape)
        shape = np.concatenate(([2**rank],shape))
        shape[1] = self.N_max+1

        data_g = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_g[i] = self.backward_component(m,deg[i],data[start_index[i]:end_index[i]])
        return np.einsum("ij,j...->i...",unitary,data_g)

    def grad(self,m,rank_in,data_in,data_out):
        # data_in and data_out are in coefficient space

        (start_index_in,end_index_in,deg_in) = self.tensor_index(m,rank_in)
        rank_out = rank_in+1
        (start_index_out,end_index_out,deg_out) = self.tensor_index(m,rank_out)

        N = self.N_max-self.N_min(m-self.K_max)
        if m == 0 and rank_out == 1:
            normalization = 1.
        elif m == 0 and rank_out == 2:
            normalization = 1.
        else:
            normalization = 1.

        half = 2**(rank_out-1)
        for i in range(2**(rank_out)):

            if m+deg_in[i%half] >= 1 and i//half == 0:
              Dm = self.op('D-',N,0,m+deg_in[i%half])
              C  = self.op('E',N,0,m+deg_in[i%half]-1)
              np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                         spla.spsolve(C,Dm.dot(normalization*data_in[start_index_in[i%half]:end_index_in[i%half]])) )

            if m+deg_in[i%half] >= 0 and i//half == 1:
              Dp = self.op('D+',N,0,m+deg_in[i%half])
              C  = self.op('E',N,0,m+deg_in[i%half]+1)

              np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                         spla.spsolve(C,Dp.dot(normalization*data_in[start_index_in[i%half]:end_index_in[i%half]])) )


class TensorField_2D:

    def __init__(self,rank,D,domain):
        self.domain = domain
        self.D = D
        self.rank = rank

        self.m_min, self.m_max = D.m_min, D.m_max

        local_grid_shape = self.domain.distributor.layouts[-1].local_shape(scales=domain.dealias)
        grid_shape = np.append(2**rank,np.array(local_grid_shape))
        local_mr_shape = self.domain.distributor.layouts[1].local_shape(scales=domain.dealias)
        mr_shape = np.append(2**rank,np.array(local_mr_shape))

        self.grid_data = np.zeros(grid_shape,dtype=np.float64)
        self.mr_data = np.zeros(mr_shape,dtype=np.complex128)

        self.fields = domain.new_fields(2**rank)
        for field in self.fields:
            field.set_scales(domain.dealias)
            field.require_grid_space()
        self.coeff_data = []
        for m in range(self.m_min, self.m_max + 1):
            (start_index,end_index,deg) = self.D.tensor_index(m,rank)
            self.coeff_data.append(np.zeros(end_index[-1],dtype=np.complex128))

        self._layout = 'g'
        self.data = self.grid_data

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""

        self.require_layout(layout)
        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in specified layout."""

        self.layout = layout
        np.copyto(self.data, data)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        if self._layout == 'g':
            self.data = self.grid_data
        elif self._layout == 'c':
            self.data = self.coeff_data

    def require_layout(self, layout):

        if layout == 'g' and self._layout == 'c':
            self.require_grid_space()
        elif layout == 'c' and self._layout == 'g':
            self.require_coeff_space()

    def require_coeff_space(self):
        """Transform from grid space to coeff space"""

        rank = self.rank

        for i,field in enumerate(self.fields):
            field.require_grid_space()
            field.data = self.data[i]
            field.require_layout(self.domain.distributor.layouts[1])

        for m in range(self.m_min, self.m_max+1):
            m_local = m - self.m_min
            self.coeff_data[m_local] = self.D.forward(m,rank,
                                                      [self.fields[i].data[m_local] for i in range(2**rank)])

        self.data = self.coeff_data
        self.layout = 'c'

    def require_grid_space(self):
        """Transform from coeff space to grid space"""

        rank = self.rank

        for m in range(self.m_min, self.m_max + 1):
            m_local = m - self.m_min
            self.mr_data[:,m_local,:] = self.D.backward(m,rank,self.coeff_data[m_local])

        for i,field in enumerate(self.fields):
            field.layout = self.domain.distributor.layouts[1]
            field.data = self.mr_data[i]
            field.require_grid_space()
            self.grid_data[i] = field.data

        self.data = self.grid_data
        self._layout = 'g'


