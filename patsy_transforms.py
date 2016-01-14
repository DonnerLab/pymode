import patsy
from pylab import *
class empty_transform(object):
    '''
    Transforms events into
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass


class Cc_t(empty_transform):
    def transform(self, x, levels=None):
        if levels is None:
            levels = unique(x)
        out = zeros((len(x), len(levels)))
        for i, level in enumerate(levels):
            idx = x==level
            out[:, i] = idx

        return out

class boxcar_t(empty_transform):
    def transform(self, x, pre=10, post=10, val='normalized'):
        if val == 'normalized':
            val = 1./(pre+post)
        idx = where(x)[0]
        for index in idx:
            x[idx-pre:idx+post] = val
        return x

class ramp_t(empty_transform):
    def transform(self, x, pre=10, post=10, ramp_type='upramp', start=0., end=1.):
        try:
            x = x.values
        except AttributeError:
            pass
        vals = start + arange(pre+post) * (end/(pre+post))
        if ramp_type == 'upramp':
            pass
        elif ramp_type == 'downramp':
            vals = vals[::-1]
        else:
            vals = ramp_type(arange(pre+post))
        if len(x.shape) == 1:
            idx = where(x[:])[0].ravel()
            for index in idx:
                x[index-pre:index+post] = vals
        else:
            for i in range(x.shape[1]):
                idx = where(x[:,i])[0].ravel()
                for index in idx:
                    x[index-pre:index+post, i] = vals
        return x

class spline_convolution(object):
    '''
    A stateful transform for patsy that convolves regressors with a spline basis
    function.

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, degree=2, df=5, length=100):
        try:
            x = x.values
        except AttributeError:
            pass
        knots = [0, 0] + linspace(0,1,df-1).tolist() + [1,1]
        basis = patsy.splines._eval_bspline_basis(linspace(0,1,length), knots, degree)
        basis /= basis.sum(0)
        if len(x.shape) > 1:
            return r_[[self.conv_base(t, basis, length) for t in x]]
        else:
            return self.conv_base(x, basis, length)

    def conv_base(self, x, basis, length):
        out = empty((len(x), basis.shape[1]))
        for i, base in enumerate(basis.T):
            out[:,i] = convolve(x, pad(base, [length, 0], mode='constant'), mode='same')
        return out

Cc = patsy.stateful_transform(Cc_t)
box = patsy.stateful_transform(boxcar_t)
ramp = patsy.stateful_transform(ramp_t)
bsconv = patsy.stateful_transform(spline_convolution)
