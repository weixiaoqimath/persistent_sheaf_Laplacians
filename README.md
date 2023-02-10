# Persistent sheaf Laplacians
A naive python implementation of persistent sheaf Laplacian. Currently it can only calculate PSL for a certain type of sheaves (see section 2.4 of https://arxiv.org/abs/2112.10906). When the parameter 'charges' is set to None and 'constant' to True, it calculates persistent Laplacian. The use of Schur complement is inspired by https://arxiv.org/abs/2012.02808. 

For a point cloud having more than 1000 points, it runs really slow. So this implementation is more suitable for pedagogical purpose.

If you have any questions, please email weixiaoq@msu.edu
