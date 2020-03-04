
import numpy as np


def _check_bounds(method):
    def method_wrapper(self, l, *args, **kwargs):
        if (np.array(l) < 0).any() or (np.array(l) > self._L).any():
            raise ValueError("Clothoid parameters should be computed within the bound [0, %f]. Received %f" % (self._L, l))
        return method(self, l, *args, **kwargs)
    return method_wrapper


def _input_to_np(method):
    def method_wrapper(self, l, *args, **kwargs):
        if type(l) == list:
            l = np.array(l)
        elif type(l) != np.ndarray:
            l = np.array([l])
        return method(self, l, *args, **kwargs)
    return method_wrapper


def _2input_to_np(method):
    def method_wrapper(self, l1, l2=0, *args, **kwargs):
        if type(l1) == list:
            l1 = np.array(l1)
        elif type(l1) != np.ndarray:
            l1 = np.array([l1])
        if type(l2) == list:
            l2 = np.array(l2)
        elif type(l2) != np.ndarray:
            l2 = np.array([l2])
        return method(self, l1, l2, *args, **kwargs)
    return method_wrapper


class Clothoidxy:
    def __init__(self, p0, theta0, c0, c1, L):
        self._p0 = p0
        self._theta0 = theta0
        self._c0 = c0
        self._c1 = c1
        self._L = L

    def to_np(self):
        np_array = np.zeros(len(self._p0)+4)
        np_array[0:2] = self._p0
        np_array[2] = self._theta0
        np_array[3] = self._c0
        np_array[4] = self._c1
        np_array[5] = self._L
        return np_array

    @classmethod
    def from_np(cls, np_array):
        if np.isnan(np_array).any():
            return None
        else:
            return cls(np_array[0:2].copy(), np_array[2], np_array[3], np_array[4], np_array[5])

    def get_L(self):
        return self._L

    @_check_bounds
    @_input_to_np
    def c(self, l):
        return self.unsafe_c(l)

    def unsafe_c(self, l):
        return self._c0 + self._c1*l

    @_check_bounds
    @_input_to_np
    def phi(self, l):
        return self.unsafe_phi(l)

    def unsafe_phi(self, l):
        return self._theta0 + self._c0 * l + self._c1 * l * l / 2

    @_2input_to_np
    @_check_bounds
    def p(self, longi, lat=0, n=None, eps=None):
        return self.unsafe_p(longi, lat, n, eps)

    def unsafe_p(self, longi, lat=0, n=None, eps=None):
        if eps is not None:
            n = np.max(self._unsafe_get_n(longi, eps))
        elif n is None:
            n = np.max(self._unsafe_get_n(longi, 1e-6))

        # print('Number of partitions ', n)
        n = max(min(np.max(n), 100), 1)
        h = longi/n
        partition = np.arange(n+1)*h
        phi_partition = self.unsafe_phi(partition)
        phi_even = phi_partition[2:-2:2]
        phi_odd = phi_partition[1:-1:2]

        sum_even_x = 2*np.sum(np.cos(phi_even), axis=0)
        sum_odd_x = 4*np.sum(np.cos(phi_odd), axis=0)
        sum_even_y = 2*np.sum(np.sin(phi_even), axis=0)
        sum_odd_y = 4*np.sum(np.sin(phi_odd), axis=0)

        p = np.zeros([longi.shape[0], 2])
        phi_longi = self.unsafe_phi(longi)
        p[:, 0] = h/3*(np.cos(self._theta0) + sum_even_x + sum_odd_x + np.cos(phi_longi))
        p[:, 1] = h/3*(np.sin(self._theta0) + sum_even_y + sum_odd_y + np.sin(phi_longi))

        return np.squeeze(p+self._p0+lat*np.array([-np.sin(phi_partition[-1]), np.sin(phi_partition[-1])]))

    @_check_bounds
    @_input_to_np
    def _get_n(self, l, eps):
        return self._unsafe_get_n(l, eps)

    def _unsafe_get_n(self, l, eps):
        l = np.abs(l)
        m = np.maximum(np.abs(self._c0), np.abs(self.unsafe_c(l)))
        m2 = m*m
        b = 3*self._c1*self._c1 + m2*m2 + 6*np.abs(self._c1)*m2

        n = 2*np.ceil((np.sqrt(np.sqrt(l*l*l*l*l*b/(180*eps))) + 1e-6)/2).astype(int)

        return n

    # def projection(self, p):
    #     def f(l):
    #         pl = self.p(l)
    #         # used to impose orthogonality of projection
    #         # phil = self.phi(l)
    #         # return ((p[0] - pl[0]) * np.cos(phil)+
    #         #         (p[1] - pl[1]) * np.sin(phil))
    #         return np.inner(pl-p, pl-p)
    #
    #     def df(l):
    #         pl = self.p(l)
    #         phil = self.phi(l)
    #         return self.c(l)*(-(p[0] - pl[0]) * np.sin(phil)+
    #                 (p[1] - pl[1]) * np.cos(phil))-1
    #
    #     d = np.linalg.norm(p-self._p0)
    #     # l_proj = scipy.optimize.newton(f, d, df, tol=1e-6, maxiter=10)
    #     l_proj = scipy.optimize.minimize_scalar(f, bounds=(0, 30), method='bounded')
    #     return l_proj.x

    def projection(self, p):
        diff = p - self._p0
        pop = np.arctan2(diff[1], diff[0])-self._theta0

        longitudinal_dist = np.maximum(np.minimum(np.linalg.norm(diff)*np.cos(pop), self._L), 0)
        if(longitudinal_dist == 0 or longitudinal_dist ==30):
            lateral_dist = 0
        else:
            lateral_dist = np.linalg.norm(diff)*np.sin(pop)

        return longitudinal_dist, lateral_dist