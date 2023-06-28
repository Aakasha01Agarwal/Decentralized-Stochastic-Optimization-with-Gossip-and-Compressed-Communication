import numpy as np
class Graph:

    # init function to declare class variables
    # adjacency matrix is given
    def __init__(self, n, topology):
        self.topology = topology
        self.n = n
        self.adj = self.create_adjacency_matrix()
        self.I = np.eye(self.n)

    def degree_matrix(self):

        adj = self.adj

        degree_matrix = np.zeros((self.n, self.n))

        for i in range(self.n):

            degree_matrix[i, i] = sum(adj[i])

        return degree_matrix

    def epsilon(self):
        #  0.9 can be anything, have to ask ma'am how to select this value !
        return 0.9*(1/np.max(self.degree_matrix()))

    def create_adjacency_matrix(self):

        if self.topology == "random":
            I = np.eye(self.n)
            connections = np.random.randint(0, 2,
                                            int(self.n * (self.n - 1) * 0.5))  # randomly connecting (n)(n-1)/2 nodes
            A = np.zeros((self.n, self.n))
            counter = 0

            # creating adjacency matrix out of the connections we have
            for i in range(0, self.n):
                for j in range(i + 1, self.n):
                    A[i, j] = connections[counter]
                    A[j, i] = connections[counter]
                    counter += 1
            return A

        if self.topology == "star":

            A = np.zeros((self.n, self.n))
            A[self.n - 1] = 1
            A[0:self.n - 1, self.n - 1] = 1
            A[self.n - 1, self.n - 1] = 0

            return A








