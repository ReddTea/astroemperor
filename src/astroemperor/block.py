# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# version 0.3
# date 14 nov 2022

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this
import numpy as np
from .utils import flatten, cps, GM_Estimator, get_support

class Parameter(object):
    def __init__(self, attributes_dict: dict):
        '''
        value, prior, limits, type, name, prargs, ptformargs,
        fixed, sigma, GM_parameter, posterior
        '''
        self.name = None

        for attr in attributes_dict:
            setattr(self, attr, attributes_dict[attr])

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self.name)


class Parameter_Block(object):
    def __init__(self, params, block_model=None, block_name=None, block_type=None, model_script='', display=True, math_display='', is_iterative=False, parameterisation=None, number=0, moav=0):
        self.list_ = np.array(params)
        self.ndim_ = len(self.list_)

        self.model = block_model
        self.name_ = block_name
        self.type_ = block_type
        self.display_on_data_ = display
        self.number_ = number
        self.bnumber_ = 0
        self.is_iterative_ = is_iterative
        self.slice = slice(None)
        self.additional_priors = []
        self.additional_priors_bool = False
        self.dynamics_bool = False
        self.model_script = model_script

        self.additional_parameters = []

        self.int1 = 1
        self.moav = moav
        self.bool1 = 1
        self.bool2 = 1

        self.C_ = []
        self.A_ = []

        self.gC_ = []
        self.gA_ = []

        empty_lists = ['extra_args', 'A_', 'C_', 'additional_priors']

        for attribute in empty_lists:
            setattr(self, attribute, [])

        self.parameterisation = parameterisation
        self.math_display_ = math_display

        self.b_fixed = self.get_attr('fixed')
        self.notfixed_bool_mask = [f is None for f in self.b_fixed]


    def get_attr(self, call):
        if type(call) == str:
            return [getattr(param, call) for param in self]
        elif type(call) == list:
            return [[getattr(param, c) for param in self] for c in call]


    def set_attr(self, call, val, silent=False):
        x = ''
        for i in range(len(self)):
            setattr(self[i], call, val[i])
            if not silent:
                x += '{0} {1} set to {2}\n'.format(self[i], call, val[i])
        return x


    def add_additional_priors(self, priors):
        for prior in priors:
            self.additional_priors.append(prior)
        self.additional_priors_bool = True


    def add_additional_parameters(self, param_dict):
        self.additional_parameters.append(Parameter(param_dict))


    def refresh_block(self):
        self.C_ = []
        self.A_ = []
        ndim = len(self)
        subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

        for i in range(ndim):
            if self[i].fixed is not None:
                self.A_.append(i)
                self[i].value = self[i].fixed
                self[i].prior = 'Fixed'
                self[i].limits = [np.nan, np.nan]
                ndim -= 1
            else:
                self.C_.append(i)

            if self[i].prior == 'Uniform':
                self[i].display_prior = '~𝓤 ({}, {})'.format(*np.round(self[i].limits, 3))
            elif self[i].prior == 'Normal':
                self[i].display_prior = '~𝓝 ({}, {})'.format(*np.round(self[i].prargs, 3))
            elif self[i].prior == 'GaussianMixture':
                gmparam = self[i].GM_parameter
                self[
                    i
                ].display_prior = f'𝛴{subscript_nums[gmparam.n_components]}~𝓝 ({list(np.round(gmparam.means, 3))}, {list(np.round(gmparam.sigmas, 3))})'
            elif self[i].prior == 'Fixed':
                self[i].display_prior = f'~𝛿 (x - {self[i].value})'
            else:
                self[i].display_prior = f'Method not built for {self[i].prior}'

        self.b_fixed = self.get_attr('fixed')
        self.notfixed_bool_mask = [f is None for f in self.b_fixed]

        self.ndim_ = ndim


    def calc_priors(self, theta):
        lp = 0.
        for i in range(len(self)):
            lp += getattr(my_stats, self[i].prior)(theta[self.slice][i], self[i].limits, self[i].prargs)

            #print(self[i].name, self[i].limits, theta[self.slice][i], lp)
            if lp == -np.inf:
                return lp

        for p in self.additional_parameters:
            if p.has_prior:
                if p.name == 'Amplitude':
                    x = theta[self.slice][1]**2 + theta[self.slice][2]**2
                elif p.name == 'Eccentricity':
                    x = theta[self.slice][3]**2 + theta[self.slice][4]**2
                elif p.name == 'Hill':
                    kplan = p.prargs[0]
                    if kplan > 1:
                        x = self.get_PAE(theta, kplan)
                else:
                    continue

                lp += getattr(my_stats, p.prior)(x, p.limits, p.prargs)
                if lp == -np.inf:
                    return lp

        return lp


    def get_PAE(self, theta, kplanets):
        ndim = len(self)
        periods = theta[slice(0, ndim*kplanets, ndim)]

        if self.parameterisation in [0, 2, 3]:
            amps = theta[slice(1, ndim*kplanets, ndim)]
            eccs = theta[slice(3, ndim*kplanets, ndim)]

        if self.parameterisation == 1:
            periods = np.exp(periods)
            amps = theta[slice(1, ndim*kplanets, ndim)] ** 2 + theta[slice(2, ndim*kplanets, ndim)] ** 2
            eccs = theta[slice(3, ndim*kplanets, ndim)] ** 2 + theta[slice(4, ndim*kplanets, ndim)] ** 2

        if self.parameterisation == 3:
            eccs = theta[slice(3, ndim*kplanets, ndim)] ** 2 + theta[slice(4, ndim*kplanets, ndim)] ** 2

        return np.array([periods, amps, eccs])


    def __repr__(self):
        return self.name_+'(%i)' % self.ndim_


    def __len__(self):
        return len(self.list_)


    def __getitem__(self, n):
        return self.list_[n]


    def __str__(self):
        return self.name_


    pass


class ReddModel(object):
    def __init__(self, data, bloques):
        self.bloques = bloques
        self.data = data

        self.x = data['BJD'].values
        self.y = data['RV'].values
        self.yerr = data['eRV'].values

        self.bloques_model = []
        self.bloques_error = []
        self.ndata = len(self.x)

        self.mod_fixed = flatten(self.get_attr_param('fixed'))
        self.A_ = []
        self.C_ = []

        self.notfixed_bool_mask = [f is None for f in self.mod_fixed]
        self.ndim__ = np.sum(self.get_attr_block('ndim_'))
        #self.additional_priors_bool = False
        #self.are_additional_priors = False

        self.model_script_no = 0


    def evaluate_model(self, theta):
        mod0 = np.zeros_like(self.y)
        err20 = self.yerr ** 2

        for a in self.A_:
            theta = np.insert(theta, a, self.mod_fixed[a])

        for b in self.bloques:
            if b.moav:
                b.model.fargs[4] = self.y - mod0

            mod, ferr = b.model(theta[b.slice])
            mod0 += mod
            err20 += ferr

        return mod0, err20  # returns model and errors ** 2


    def evaluate_logprior(self, theta):
        lp = 0.

        for a in self.A_:
            theta = np.insert(theta, a, self.mod_fixed[a])

        for b in self:
            lp += b.calc_priors(theta)
            if lp == -np.inf:
                return lp
        return lp


    def evaluate_ptform(self, theta, ptformargs):
        for a in self.A_:
            theta = np.insert(theta, a, self.mod_fixed[a])
        x = np.array(theta)
        for i in range(len(x)):
            a, b = ptformargs[i]
            x[i] =  a * (2. * x[i] - 1) + b
        return x


    def evaluate_loglikelihood(self, theta):
        model, err2 = self.evaluate_model(theta)
        return -0.5 * np.sum((self.y - model) ** 2 / err2 + np.log(err2)) + np.log(2*np.pi) * self.ndata


    def get_GMEstimates(self, chains):
        for b in self:
            for p in b:
                if p.cpointer is not None:
                    p.GM_parameter = GM_Estimator().estimate(chains[:, p.cpointer],
                                                            p.name, p.unit)
                else:
                    p.GM_parameter = p.value


    def refresh__(self):
        self.bloques_model = []
        self.bloques_ins = []
        self.kplan__ = 0
        self.nins__ = 0

        # update slices and sorts blocks

        nt = 0
        ntt = 0
        bn_ = 1
        for b in self:
            b.bnumber_ = bn_
            if b.type_ == 'Keplerian':
                self.bloques_model.append(b)
                self.kplan__ += 1
            if b.type_ == 'Instrumental':
                self.bloques_ins.append(b)
                self.nins__ += 1

            b.refresh_block()
            b.slice = slice(nt, nt+len(b))
            b.slice_true = slice(ntt, ntt+b.ndim_)
            nt += len(b)
            ntt += b.ndim_
            bn_ += 1
        # updates model's ndim
        self.ndim__ = np.sum(self.get_attr_block('ndim_'))
        for b in self.bloques_model:
            if b.dynamics_bool:
                for extra in b.additional_priors:
                    if extra[0] == 'Hill':
                        extra[3][0] = self.kplan__

        # updates model's A_ & C_
        for i in range(len(self.mod_fixed)):
            if self.mod_fixed[i] != None:
                self.A_.append(i)
            else:
                self.C_.append(i)

        # updates block's gA_ & gC_
        tdims = 0
        ntdims = 0
        for b in self:
            b.gC_ = self.C_[tdims:tdims+b.ndim_]

            b.gA_ = self.A_[ntdims:ntdims+(len(b)-b.ndim_)]

            b.cpointer = np.arange(tdims, tdims+b.ndim_)
            j = 0
            for p in b:
                if p.fixed == None:
                    p.cpointer = b.cpointer[j]
                    j += 1
                else:
                    p.cpointer = None

            tdims += b.ndim_
            ntdims += (len(b)-b.ndim_)


    def get_attr_param(self, call):
        return [b.get_attr(call) for b in self]


    def get_attr_block(self, call):
        return [getattr(b, call) for b in self]


    def display_math(self):
        return ' + '.join(self.get_attr_block('math_display_'))


    def write_model_(self, loc='', tail=''):
        saveloc = f'{loc}/temp/'
        self.data.to_csv(f'{saveloc}temp_data{tail}.csv')

        fname = f'{saveloc}/temp_model_{self.model_script_no}{tail}.py'
        with open(fname, 'w') as f:
            # data
            f.write('''
my_data = pd.read_csv('{}temp_data{}.csv')

'''.format(saveloc, tail))

            # model
            f.write('''
def my_model(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])

    model0 = np.zeros_like(my_data['BJD'])
    err20 = my_data['eRV'] ** 2

''')
            for b in self:
                if b.type_ == 'Keplerian':
                    f.write(open(get_support(f'models/{b.model_script}')).read().format(b.slice))
                elif b.type_ == 'Instrumental':
                    f.write(open(get_support(f'models/{b.model_script}')).read().format(b.ins_no,
                                                               b.slice,
                                                               b.moav))
                elif b.type_ == 'Acceleration':
                    f.write(open(get_support(f'models/{b.model_script}')).read().format(b.slice))
            f.write('''
    return model0, err20


''')
        self.model_script_no += 1
        return fname


    def __getitem__(self, n):
        return self.bloques[n]


    def __repr__(self):
        return str([b for b in self])


class my_stats():
    def Uniform(x, limits, args):
        if limits[0] <= x <= limits[1]:
            return 0.
        else:
            return -np.inf

    def Normal(x, limits, args):
        if limits[0] <= x <= limits[1]:
            mu, var = args[0], args[1]**2
            return - 0.5 * (np.log(2*np.pi*var) + (x - mu)**2/var)
        else:
            return -np.inf

    def Fixed(x, limits, args):
        return 0.

    def Jeffreys(x, limits, args):
        if limits[0] <= x <= limits[1]:
            return 0.
        else:
            return -np.inf

    def Hill(x, limits, args):
        kplanets = args[0]
        starmass = args[1]
        periods, amps, eccs = x

        gamma = np.sqrt(1 - eccs)
        sma, minmass = cps(periods, amps, eccs, starmass)
        orden = np.argsort(sma)
        sma = sma[orden]  # in AU
        minmass = minmass[orden]  # in Earth Masses

        periods, amps, eccs = periods[orden], amps[orden], eccs[orden]
        M = starmass * 1047.56 + np.sum(minmass)  # jupiter masses
        mu = minmass / M

        for k in range(kplanets-1):
            alpha = mu[k] + mu[k+1]
            delta = np.sqrt(sma[k+1] / sma[k])

            LHS = alpha**-3 * (mu[k] + (mu[k+1] / (delta**2))) * (mu[k] * gamma[k] + mu[k+1] * gamma[k+1] * delta)**2
            RHS = 1 + (3./alpha)**(4./3) * (mu[k] * mu[k+1])
            #LHS = delta
            #RHS = 2.4*alpha**(1./3)
            if LHS > RHS:
                pass
            else:
                return -np.inf

        return 0.

    def GaussianMixture(x, limits, args):
        #  = my_bgm.bgm_estimator[0]
        if limits[0] <= x <= limits[1]:
            return 0
        else:
            return -np.inf







#
