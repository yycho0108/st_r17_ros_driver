import sympy

def dh2T(alpha, a, d, q):
    """ Convert DH Parameters to Transformation Matrix """
    cq = cos(q)
    sq = sin(q)
    ca = cos(alpha)
    sa = sin(alpha)

    T = Matrix([
        [cq, -sq, 0, a],
        [sq*ca, cq*ca, -sa, -sa*d],
        [sq*sa, cq*sa, ca, ca*d],
        [0, 0, 0, 1]
        ])
    return T

# psi = matrix of jacobians
def _build():

    Ts = [dh2T(*dh) for dh in dhs]
    T_f = Matrix()
    T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> object

    x,y,z,r,p,y = sympy.Symbols('x,y,z,r,p,y') # target
    dx = [x,y,z,r,p,y]
    psi = F.jacobian(
    dP = pinv(psi).dot(dx)
    P += dP

