# As of Python 3.7, "Dict keeps insertion order" is the ruling.
# So please use Python 3.7 or later version.
import numpy as np
import gudhi
import scipy.spatial.distance
import os

def mkdir_p(dir):
    """make a directory (dir) if it doesn't exist"""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def Partial_Column_Reduction(Mat, size_t):
    """
    This function can do a column reduction with respect to certain row indices.
    parameter
    _________
    Mat: a matrix (numpy array).
    size_t: 
    Output
    ------
    dh_star, P
    Column reduction.
    """
    Mat = Mat.astype('float64')
    n, m = Mat.shape
    if n == size_t:
        return Mat, np.identity(m)
    I = np.identity(m)
    Mat = np.concatenate((Mat, I), axis = 0)
    mathbbC = []
    
    mask = np.zeros(n+m, dtype=bool)
    mask[size_t: n] = True
    
    for j in range(m):
        if np.max(abs(Mat[size_t:n, j])) < 1e-5:
            continue # If a column is 0, just keep it.
        else:
            i = size_t
            while abs(Mat[i, j]) < 1e-5:
                i += 1 
                # find the highest row index in row_indices such that the entry is not zero.
            for j_prime in np.arange(j + 1, m):
                Mat[:, j_prime] -= Mat[i, j_prime]/Mat[i, j]*Mat[:, j]
    
    # scan the matrix from left to right and 
    # remember the column that are zero with respect to row_indices.
    for j in range(m):
        if np.max(abs(Mat[size_t:n, j])) < 1e-5:
            mathbbC.append(j)

    dh_star = Mat[:size_t, mathbbC]
    P = np.dot(Mat[-m:, mathbbC].T, Mat[-m:, mathbbC]) 
    return dh_star, P


def coboundary_constant_0(vertices, edges):
    """
    Conputes the coboundary matrix d_0 for a constant sheaf whose stalks are \mathbb{R}.
    Parameters
    ----------
    vertices: list. Indeed only the length of vertices is relevant.
    edges: dictionary. Keys are edges (which are tuples of points) and values are 
        tuples of the shape (index, filtration value)
    Outputs
    -------
    d_0: coboundary matrix
    """   
    # d_0 is the coboundary matrix
    d_0 = np.zeros((len(edges), len(vertices)))
    # We require that any edge must be of the form (n, m) where n <= m, 
    # otherwise the following code returns a wrong result.
    for edge, idx_and_t in edges.items():
        d_0[idx_and_t[0], edge[0]] = -1
        d_0[idx_and_t[0], edge[1]] = 1
    return d_0

def coboundary_constant_1(edges, faces):
    """
    Computes the coboundary matrix d_1 for a constant sheaf whose stalks are \mathbb{R}.
    Parameters
    ----------
    edges: dictionary. keys are edges (tuple) and values are tuples of indices and filtration values. 
    faces: dictionary. keys are 2-simplices (tuple) and values are tuples of indices and filtration values.
    Outputs
    -------
    d_1
    """
    d_1 = np.zeros((len(faces),len(edges)))
    for face, idx_and_t in faces.items():
        # construct faces of a 2-simplex. 
        face_face0 = (face[1], face[2])
        face_face1 = (face[0], face[2])
        face_face2 = (face[0], face[1])
        d_1[idx_and_t[0], edges[face_face0][0]] = 1
        d_1[idx_and_t[0], edges[face_face1][0]] = -1
        d_1[idx_and_t[0], edges[face_face2][0]] = 1
    return d_1


def coboundary_nonconstant_0(vertices, edges, charges, F):
    """
    Computes the coboundary matrix d_0 for the class of nonconstant sheaves defined in the paper.
    Parameters
    ----------
    vertices: list
    edges: dictionary
    charges: list of partial atomic charges. 
    F: a dictionary. A key is a simplex S and the value of it is F(S).
    Outputs
    -------
    d_0: d_0
    """   
    # d_0 is the coboundary matrix
    d_0 = np.zeros((len(edges), len(vertices)))
    for edge, idx_and_t in edges.items():
        # v_0 \leq [v_0, v_1], q_1/F([v_0, v_1])
        d_0[idx_and_t[0], edge[0]] = -charges[edge[1]]/F[edge]
        d_0[idx_and_t[0], edge[1]] = charges[edge[0]]/F[edge]
    return d_0

def coboundary_nonconstant_1(edges, faces, charges, F):
    """
    edges: dictionary. keys are edges (tuple) and values are tuples of indices and filtration values. 
    faces: dictionary. keys are faces (tuple) and values are tuples of indices and filtration values.
    charges: a list (or a 1d numpy array) of charges
    F: a dictionary. A key is a simplex S and the value of it is F(S).
    """
    d_1 = np.zeros((len(faces), len(edges)))
    for face, idx_and_t in faces.items():
        face_face0 = (face[1], face[2])
        face_face1 = (face[0], face[2])
        face_face2 = (face[0], face[1])
        # [v_0, v_1] \leq [v_0, v_1, v_2], F([v_0, v_1])q_2/F([v_0, v_1, v_2])
        d_1[idx_and_t[0], edges[face_face0][0]] = charges[face[0]]*F[face_face0]/F[face]
        d_1[idx_and_t[0], edges[face_face1][0]] = -charges[face[1]]*F[face_face1]/F[face]
        d_1[idx_and_t[0], edges[face_face2][0]] = charges[face[2]]*F[face_face2]/F[face]
    return d_1


class PSL():
    def __init__(self, pts, charges = None, filtration_type = 'alpha', constant = True, scale = False):
        """
        pts: a 2d numpy array. Each row is a point.
        charges: a 1d np array of atomic charges. Must be set to None if constant is True.     
        """
        self.filtration_type = filtration_type
        # If self.constant is True, implement constant sheaf of dim 1.
        self.constant = constant
        self.simplex_tree = None
        self.F = {} # Will be the dictionary that stores values of F
        self.pts = pts
        self.distance_matrix = scipy.spatial.distance.cdist(self.pts, self.pts)
        # scale_factor = np.prod(np.power(np.abs(charges), 1./len(charges)))
        if np.any(charges != None) and scale == True:
            #scale_factor = np.prod(np.power(np.abs(charges), 1./len(charges)))
            scale_factor = np.mean(charges)
            self.charges = charges/scale_factor*np.max(self.distance_matrix)
        elif np.any(charges != None) and scale == False:
            self.charges = charges

    def build_filtration(self):
        """
        """
        if self.filtration_type == 'rips':
            rips_complex = gudhi.RipsComplex(points=self.pts)
            self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)  
        elif self.filtration_type == 'alpha':   
            alpha_complex = gudhi.AlphaComplex(points=self.pts)
            self.simplex_tree = alpha_complex.create_simplex_tree()
       
        # Generate values of F
        if self.constant is False:
            for simplex, _ in self.simplex_tree.get_filtration():
                if len(simplex) == 1:
                   self.F[tuple(simplex)] = 1
                if len(simplex) == 2:
                   self.F[tuple(simplex)] = self.distance_matrix[simplex[0], simplex[1]]
                if len(simplex) == 3:
                   self.F[tuple(simplex)] = self.distance_matrix[simplex[0], simplex[1]]*self.distance_matrix[simplex[0], simplex[2]]*self.distance_matrix[simplex[1], simplex[2]]
   
    def build_simplicial_pair(self, t, p):
        if self.filtration_type == 'rips':
            f0, f1 = 2*t, 2*(t+p)
        elif self.filtration_type == 'alpha':   
            f0, f1 = t**2, (t+p)**2
       
        edge_idx = 0
        face_idx = 0
        self.C_0 = []
        self.C_1_t, self.C_1_tp, self.C_2_t, self.C_2_tp = {}, {}, {}, {}
        for simplex, filtration in self.simplex_tree.get_filtration():
            if len(simplex) == 1:
                self.C_0.append(simplex)
            if len(simplex) == 2:
                if filtration < f0:
                    self.C_1_t[tuple(simplex)] = (edge_idx, filtration)
                if filtration < f1:
                    self.C_1_tp[tuple(simplex)] = (edge_idx, filtration)
                edge_idx += 1
            if len(simplex) == 3:
                if filtration < f0:
                    self.C_2_t[tuple(simplex)] = (face_idx, filtration)
                if filtration < f1:
                    self.C_2_tp[tuple(simplex)] = (face_idx, filtration)
                face_idx += 1
               
    def psl_0(self):
        #if self.constant == False and len(self.charges) != self.pts.shape[0]:
        #    print('Error occurs. The number of partial charges is not equal to the number of points.')
        #    return None        
        if self.constant is True:
            d_0_tp = coboundary_constant_0(self.C_0, self.C_1_tp)
        else:
            d_0_tp = coboundary_nonconstant_0(self.C_0, self.C_1_tp, self.charges, self.F)
        psl_0 = np.dot(d_0_tp.T, d_0_tp)
        return psl_0
   
    def psl_1(self):
        #if self.constant == False and len(self.charges) != self.pts.shape[0]:
        #    print('Error occurs. The number of partial charges is not equal to the number of points.')
        #    return None
        if self.constant is True:
            d_0_t = coboundary_constant_0(self.C_0, self.C_1_t)
            d_1_tp = coboundary_constant_1(self.C_1_tp, self.C_2_tp)
            dh_star, P = Partial_Column_Reduction(d_1_tp.T, len(self.C_1_t))
        else:
            d_0_t = coboundary_nonconstant_0(self.C_0, self.C_1_t, self.charges, self.F)
            d_1_tp = coboundary_nonconstant_1(self.C_1_tp, self.C_2_tp, self.charges, self.F)
            dh_star, P = Partial_Column_Reduction(d_1_tp.T, len(self.C_1_t))            
        psl_1 = np.dot(d_0_t, d_0_t.T) + np.dot(np.dot(dh_star, np.linalg.inv(P)), dh_star.T)
        return psl_1


def matrix_reader(filename = 'test.txt'):
    text = open(filename, "r")
    array_list = []
    for line in text:
        array_list.append(list(map(float, line.split())))
    return np.array(array_list)  

def array_reader(filename = 'test.txt'):
    text = open(filename, "r")
    array_list = []
    for line in text:
        array_list.append(list(map(float, line.split())))
    return array_list

def pqr_parser(filename):
    import numpy as np
    f = open(filename, 'r')
    coordinates = []
    charges = []
    for line in f:
        if len(line.split()) == 10:
            coordinates.append(list(map(float, line.split()[-5:-2])))
            charges.append(float(line.split()[-2]))
    return np.array(coordinates), np.array(charges)



def array_writer(array, filename = 'test.txt'):
    """
    The array is a numpy array or list of lists. 
    We write it into a txt file.
    filename is the name of the txt file.
    """
    with open(filename, 'w') as f:
        for row in array:
            for i, entry in enumerate(row):
                f.write("{} ".format(entry))
            f.write("\n")
    f.close() 


