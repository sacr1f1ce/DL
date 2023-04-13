import torch


def backward_normalization(x):
    res =  (torch.eye(x.shape[0]) * (torch.linalg.norm(x) ** 2) - (x.reshape((-1, 1)) @ x.reshape((1, -1))))
    return res / (torch.linalg.norm(x) ** 3)


def backward_orthogonalization(Q, A):
    # Your code
    derivatives = {}
    for j in range(A.shape[1]):
        for k in range(j, A.shape[1]):
            if j == k:
                sum_term = sum([Q[:, i].reshape((-1, 1)) @ Q[:, i].reshape((1, -1)) / torch.linalg.norm(Q[:, i]) ** 2 for i in range(k)])
                derivatives[(k, j)] = torch.eye(Q.shape[0]) - sum_term
            else:
                result = torch.zeros((Q.shape[0], Q.shape[0]))
                for i in range(j, k):
                    result -= (1 / torch.linalg.norm(Q[:, i]) ** 2) * \
                        (Q[:, i].reshape((-1, 1)) @ A[:, k].reshape((1, -1)) + \
                         A[:, k] @ Q[:, i] * torch.eye(Q.shape[0]) - \
                            (2.0 * A[:, k] @ Q[:, i] / torch.linalg.norm(Q[:, i]) ** 2) \
                                      * (Q[:, i].reshape((-1, 1)) @ Q[:, i].reshape((1, -1)))) @ derivatives[(i, j)]
                derivatives[(k, j)] = result
    return derivatives


class QR():
    def forward(self, A):
        """
        Computes QR decomposition of matrix A

        Input: 
            A - n x m matrix
        Output:
            Q - n x m orthonormal matrix
            R - m x m upper triangular matrix
        """

        # Your code
        n, m = A.shape

        Q = torch.clone(A)

        for i in range(m):
            vec_i = torch.zeros((1, m)) #vector  to i-th column in matrix
            vec_i[0, i] = 1
            for j in range(i):
                pos_i, pos_j = torch.zeros(m), torch.zeros(m) #i-th and j-th column of matrix
                pos_j[j] = 1
                pos_i[i] = 1
                Q -= ((Q @ pos_i) @ (Q @ pos_j)) / ((Q @ pos_j) @ (Q @ pos_j)) * (Q @ pos_j)[..., None] @ vec_i
        
        self.Q_unnorm = Q
        Q = Q / torch.linalg.norm(Q, dim=0)

        R = Q.T @ A

        self.Q = Q
        self.A = A
        self.orthogonal_derivatives = backward_orthogonalization(self.Q_unnorm, self.A)  

        return Q, R
        

    def backward(self, grad_output):
        """
        Computes QR decomposition of matrix A

        Input: 
            grad_output - n x m matrix, derivative of the previous layer (derivative of loss dL/dQ  in our case)
        Output:
            grad_input - n x m derivative dL/dA
        """
        # Your code
        grad_input = torch.zeros_like(self.A)
        for j in range(self.A.shape[1]):
            for k in range(j, self.A.shape[1]):
                grad_input[:, j] += grad_output[:, j] @ backward_normalization(self.Q_unnorm[:, k]) @ self.orthogonal_derivatives[(k, j)]
         
        return grad_input
def modified_gram_schmidt(A):
    """
    Computes QR decomposition of matrix A
    
    Input: 
        A - n x m matrix
    Output:
        Q - n x m orthonormal matrix
        R - m x m upper triangular matrix
    """
    
    # Your code here
    n, m = A.shape

    Q = torch.clone(A)

    for i in range(m):
        vec_i = torch.zeros((1, m)) #vector  to i-th column in matrix
        vec_i[0, i] = 1
        for j in range(i):
            pos_i, pos_j = torch.zeros(m), torch.zeros(m) #i-th and j-th column of matrix
            pos_j[j] = 1
            pos_i[i] = 1
            Q -= ((Q @ pos_i) @ (Q @ pos_j)) / ((Q @ pos_j) @ (Q @ pos_j)) * (Q @ pos_j)[..., None] @ vec_i

    Q = Q / torch.linalg.norm(Q, dim=0)

    R = Q.T @ A

    return Q, R

    
def full_torch(A):
    Q, R = torch.linalg.qr(A) # torch version
    loss = Q.sum()
    loss.backward()
    return A.grad


def modified_gram_schmidt_autograd(A):
    Q, R = modified_gram_schmidt(A) # torch version
    loss = Q.sum()
    loss.backward()
    return A.grad


def custom(A):
    qr = QR()

    Q, R = qr.forward(A)
    loss1 = Q.sum()

    dL_dQ = torch.ones_like(Q)
    return qr.backward(dL_dQ)
    
