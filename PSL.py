# As of Python 3.7, "Dict keeps insertion order" is the ruling.
# please use Python 3.7 or later version.
import numpy as np
import gudhi
import scipy.spatial.distance
import os

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

def coboundary_constant_2(faces, tetras):
    """
    Computes the coboundary matrix d_2 for a constant sheaf whose stalks are \mathbb{R}.
    Parameters
    ----------
    Outputs
    -------
    d_2
    """
    d_2 = np.zeros((len(tetras),len(faces)))
    for tetra, idx_and_t in tetras.items():
        # construct faces of a 3-simplex. 
        tetra_face0 = (tetra[1], tetra[2], tetra[3])
        tetra_face1 = (tetra[0], tetra[2], tetra[3])
        tetra_face2 = (tetra[0], tetra[1], tetra[3])
        tetra_face3 = (tetra[0], tetra[1], tetra[2])
        d_2[idx_and_t[0], faces[tetra_face0][0]] = 1
        d_2[idx_and_t[0], faces[tetra_face1][0]] = -1
        d_2[idx_and_t[0], faces[tetra_face2][0]] = 1
        d_2[idx_and_t[0], faces[tetra_face3][0]] = -1
    return d_2

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

def coboundary_nonconstant_2(faces, tetras, charges, F):
    """
    Computes the coboundary matrix d_2 for a constant sheaf whose stalks are \mathbb{R}.
    Parameters
    ----------
    Outputs
    -------
    d_2
    """
    d_2 = np.zeros((len(tetras),len(faces)))
    for tetra, idx_and_t in tetras.items():
        # construct faces of a 3-simplex. 
        tetra_face0 = (tetra[1], tetra[2], tetra[3])
        tetra_face1 = (tetra[0], tetra[2], tetra[3])
        tetra_face2 = (tetra[0], tetra[1], tetra[3])
        tetra_face3 = (tetra[0], tetra[1], tetra[2])
        d_2[idx_and_t[0], faces[tetra_face0][0]] = charges[tetra[0]]*F[tetra_face0]/F[tetra]
        d_2[idx_and_t[0], faces[tetra_face1][0]] = -charges[tetra[1]]*F[tetra_face1]/F[tetra]
        d_2[idx_and_t[0], faces[tetra_face2][0]] = charges[tetra[2]]*F[tetra_face2]/F[tetra]
        d_2[idx_and_t[0], faces[tetra_face3][0]] = -charges[tetra[3]]*F[tetra_face3]/F[tetra]
    return d_2

class PSL():
    def __init__(self, pts, charges = None, filtration_type = 'alpha', radius_list = [], p = 0., constant = True, scale = False):
        """
        pts: a 2d numpy array. Each row is a point.
        charges: a 1d np array of atomic charges. Must be set to None if constant is True.     
        """
        self.filtration_type = filtration_type
        self.radius_list = radius_list
        self.p = p
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
            self.simplex_tree = rips_complex.create_simplex_tree(max_dimension=4)  
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
                if len(simplex) == 4:
                    self.F[tuple(simplex)] = 1
   
    def build_simplicial_pair(self):
        self.value_list = []
        if self.filtration_type == 'rips':
            for r in self.radius_list:
                self.value_list.append([2*r, 2*(r+self.p)])
        elif self.filtration_type == 'alpha':   
            for r in self.radius_list:
                self.value_list.append([r**2, (r+self.p)**2])
       
        # build dictionary of simplices that will be used for the calculation of coboundary matrices
        edge_idx = 0
        face_idx = 0
        tetra_idx = 0
        self.C_0 = []
        self.C_1, self.C_2, self.C_3 = {}, {}, {}
        self.fil_1, self.fil_2, self.fil_3 = [], [], [] # store filtration values, will be converted to numpy arrays.
        for simplex, filtration in self.simplex_tree.get_filtration():
            if filtration >= self.value_list[-1][-1]:
                break    
            if len(simplex) == 1:
                self.C_0.append(simplex)
            if len(simplex) == 2:
                self.fil_1.append(filtration)
                self.C_1[tuple(simplex)] = (edge_idx, filtration) 
                edge_idx += 1
            if len(simplex) == 3:
                self.fil_2.append(filtration)
                self.C_2[tuple(simplex)] = (face_idx, filtration)
                face_idx += 1
            if len(simplex) == 4:
                self.fil_3.append(filtration)
                self.C_3[tuple(simplex)] = (tetra_idx, filtration)
                tetra_idx += 1
        self.fil_1, self.fil_2, self.fil_3 = np.array(self.fil_1), np.array(self.fil_2), np.array(self.fil_3)

    def build_matrices(self):
        if self.constant is True:
            self.d_0 = coboundary_constant_0(self.C_0, self.C_1)
            self.d_1 = coboundary_constant_1(self.C_1, self.C_2)
            self.d_2 = coboundary_constant_2(self.C_2, self.C_3)
        else:
            self.d_0 = coboundary_nonconstant_0(self.C_0, self.C_1, self.charges, self.F)
            self.d_1 = coboundary_nonconstant_1(self.C_1, self.C_2, self.charges, self.F)
            self.d_2 = coboundary_nonconstant_2(self.C_2, self.C_3, self.charges, self.F)

    def psl_0(self): 
        res = [] 
        for _, v1 in self.value_list:     
            d_0_tp = self.d_0[:sum(self.fil_1<=v1)]
            res.append(np.dot(d_0_tp.T, d_0_tp)) 
        return res

    def psl_1(self):
        res = []
        for v0, v1 in self.value_list:
            d_0_t = self.d_0[:sum(self.fil_1<=v0)]
            d_1_tp = self.d_1[:sum(self.fil_2<=v1), :sum(self.fil_1<=v1)]
            if sum(self.fil_1<=v0) == sum(self.fil_1<=v1):
                res.append(np.dot(d_0_t, d_0_t.T) + np.dot(d_1_tp.T, d_1_tp)) 
            else:
                tmp = np.dot(d_1_tp.T, d_1_tp)
                tmp_idx = sum(self.fil_1<=v0)
                A, B, C, D = tmp[:tmp_idx, :tmp_idx], tmp[:tmp_idx, tmp_idx:], tmp[tmp_idx:, :tmp_idx],tmp[tmp_idx:, tmp_idx:]  
                res.append(np.dot(d_0_t, d_0_t.T) + A - B@np.linalg.pinv(D)@C)
        return res

    def psl_2(self):
        res = []
        for v0, v1 in self.value_list:
            d_1_t = self.d_1[:sum(self.fil_2<=v0), :sum(self.fil_1<=v0)]
            d_2_tp = self.d_2[:sum(self.fil_3<=v1), :sum(self.fil_2<=v1)]
            if sum(self.fil_2<=v0) == sum(self.fil_2<=v1):
                res.append(np.dot(d_1_t, d_1_t.T) + np.dot(d_2_tp.T, d_2_tp)) 
            else:
                tmp = np.dot(d_2_tp.T, d_2_tp)
                tmp_idx = sum(self.fil_2<=v0)
                A, B, C, D = tmp[:tmp_idx, :tmp_idx], tmp[:tmp_idx, tmp_idx:], tmp[tmp_idx:, :tmp_idx],tmp[tmp_idx:, tmp_idx:]  
                res.append(np.dot(d_1_t, d_1_t.T) + A - B@np.linalg.pinv(D)@C)
        return res