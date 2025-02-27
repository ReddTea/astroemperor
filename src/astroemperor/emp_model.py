# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# my coding convention
# **EVAL : evaluate the performance of this method
# **RED  : redo this
# **DEB  : debugging needed in this part
# **DEL  : DELETE AT SOME POINT
# **FIN  : Finish this

import numpy as np
from .qol_utils import get_support

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
    def __init__(self, params, attributes_dict: dict):
        self.list_ = np.array(params)

        self.int1 = 1
        self.int2 = 1
        self.bool1 = True
        self.bool2 = True


        empty_lists = ['extra_args', 'additional_priors',
                       'additional_parameters',
                       'dependencies',
                       'A_', 'C_', 'gC_', 'gA_']

        for attribute in empty_lists:
            setattr(self, attribute, [])

        for attr in attributes_dict:
            setattr(self, attr, attributes_dict[attr])

        self.refresh_block()


    def refresh_block(self):
        self.b_fixed = self.get_attr('fixed')
        self.not_fixed_bool = self.b_fixed == None

        ndim = len(self)

        self.C_ = np.arange(ndim)[self.not_fixed_bool]
        self.A_ = np.arange(ndim)[~self.not_fixed_bool]
        

        # Handle fixed values
        self.ndim_ = self._handle_fixed_params(ndim)



    def _handle_fixed_params(self, ndim):
        for fixed_param in self[self.A_]:
            fixed_param.value = fixed_param.fixed
            fixed_param.prior = 'Fixed'
            fixed_param.limits = [np.nan, np.nan]
            ndim -= 1       
        return ndim


    def _add_display_prior(self):
        subscript_nums = ['', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
        for param in self:
            if param.prior == 'Uniform':
                param.display_prior = '~𝓤 ({}, {})'.format(*np.round(param.limits, 3))
            elif param.prior == 'Normal':
                param.display_prior = '~𝓝 ({}, {})'.format(*np.round(param.prargs, 3))
            elif param.prior == 'GaussianMixture':
                gmparam = param.GM_parameter
                param.display_prior = f'𝛴{subscript_nums[gmparam.n_components]}~𝓝 ({list(np.round(gmparam.means, 3))}, {list(np.round(gmparam.sigmas, 3))})'
            elif param.prior == 'Beta':
                param.display_prior = '~𝛽 ({}, {})'.format(*np.round(param.prargs, 3))
            elif param.prior == 'Fixed':
                param.display_prior = f'~𝛿 (x - {param.value})'
            else:
                param.display_prior = f'Method not built for {param.prior}'


    def add_additional_priors(self, additional_priors):
        for prior in additional_priors:
            self.additional_priors.append(prior)
        self.additional_priors_bool = True


    def add_additional_parameters(self, addi_param):
        self.additional_parameters.append(addi_param)


    def write_model(self, f):
        write_args = [getattr(self, a) for a in self.write_args]
        f.write(open(get_support(f'models/{self.model_script}')).read().format(*write_args))


    def get_attr(self, call):
        try:
            if type(call) == str:
                return np.array([getattr(param, call) for param in self])
            else:
                return [[getattr(param, c) for param in self] for c in call]
        except Exception:
            print('Unidentified call type, try str, list or array')


    def set_attr(self, call, val, silent=False):
        x = ''
        for i in range(len(self)):
            setattr(self[i], call, val[i])
            if not silent:
                x += '{0} {1} set to {2}\n'.format(self[i], call, val[i])
        return x


    def __repr__(self):
        return f'{self.name_}({self.ndim_}): {list(self)}'


    def __len__(self):
        return len(self.list_)

    def __iter__(self):
        return iter(self.list_)
    

    def __getitem__(self, n):
        return self.list_[n]


    def __str__(self):
        return self.name_

import pandas as pd

class ReddModel(object):
    def __init__(self, bloques, data_RV=None, data_AM=None, data_PM=None):
        self.switch_RV = False
        self.switch_PM = False
        self.switch_AM = False

        self._init_blocks(bloques)

        self._init_data_RV(data_RV)

        self._init_data_AM(data_AM)

        if data_PM is not None:
            self._init_data_PM(data_PM)
        

        self.mod_fixed = np.array(self.get_attr_param('fixed', flat=True))
        self.model_script_no = 0
        self.ndim__ = np.sum(self.get_attr_block('ndim_'))
        self.dependencies = []

    def refresh__(self):
        '''Updates the model
        This method should sort what attributes the model has and
        what the model should do. It should also prepare 
        a write_out method.
        b.C_ index of free parameters respect to self
        b.gC_ index of free parameters respect to model
        b.cpointer corresponds to the true slice
        
        '''

        self.bloques_model = []
        self.bloques_data = []
        #self.bloques_error = []
        self.cornums = None
        self.kplan__ = 0
        self.nins__ = 0
        self.ndim__ = 0

        # Update Slices And Sorts Blocks

        blen = 0  # len(b)
        for i, b in enumerate(self):
            b.refresh_block()
            b.bnumber_ = int(i)

            if b.class_ == 'Data':
                self.bloques_data.append(b)
            elif b.class_ == 'Model':
                self.bloques_model.append(b)
            else:
                print(f'{b.class_ =}')

            # update nsai
            if b.type_ == 'StellarActivity':
                self.cornums = b.cornums

            b.slice = slice(blen, blen+len(b))
            b.slice_true = slice(self.ndim__, self.ndim__+b.ndim_)
            b.cpointer = np.arange(self.ndim__, self.ndim__+b.ndim_)

            blen += len(b)
            self.ndim__ += b.ndim_

        self.kplan__ = np.sum(np.array(self.get_attr_block('type_')) == 'Keplerian')
        for b in self:
            if b.type_ == 'Offset':
                self.nins__ = len(b)
        #self.nins__ = np.sum(np.array(self.get_attr_block('type_')) == 'Offset')


        # Handle additional Priors
        for b in self.bloques_model:
            if b.dynamics_bool:
                for extra in b.additional_priors:
                    if extra[0] == 'Hill':
                        extra[3][0] = self.kplan__


        # Update indexing
        self.C_ = np.arange(blen)
        self.A_ = np.arange(0)

        if np.any(self.mod_fixed):
            mask_fixed_bool = self.mod_fixed is None
            self.C_ = np.arange(blen)[mask_fixed_bool]
            self.A_ = np.arange(blen)[~mask_fixed_bool]


        j = 0
        for b in self:
            indices = np.arange(blen)[b.slice]
            b.gC_ = indices[b.C_]
            b.gA_ = indices[b.A_]

            for i, p in enumerate(b[b.C_]):
                p.cpointer = j
                p.C_ = b.C_[i]
                p.gC_ = b.gC_[i]
                p.A_ = None
                p.gA_ = None
                j += 1

            for i, p in enumerate(b[b.A_]):
                p.cpointer = None
                p.gC_ = None
                p.gA_ = b.gA_[i]

                p.C_ = None
                p.A_ = b.A_[i]


        # Add display priors
        for b in self:
            b._add_display_prior()
        # set dependencies?
        # set constants?
        self.model_constants = {'nan':'np.nan',
                                'gaussian_mixture_objects':'dict()',
                                'A_':self.A_,
                                'mod_fixed_':self.mod_fixed,
                                'cornums':self.cornums,
                                }

    
    def write_model(self, loc='.', tail=''):  # sourcery skip: remove-empty-nested-block, remove-redundant-if
        saveloc = f'{loc}/temp/'

        if self.switch_RV:
            self.data.to_csv(f'{saveloc}temp_data{tail}.csv')
        
        if self.switch_AM:
            self.AM_hg123.to_csv(f'{saveloc}temp_hg123{tail}.csv')
            self.AM_abs.to_csv(f'{saveloc}temp_abs{tail}.csv')
            self.AM_gost.to_csv(f'{saveloc}temp_gost{tail}.csv')

            c = 0
            for GDR in self.AM_cats:
                np.savetxt(f'{saveloc}GSV_{GDR}.txt', self.data_AM['Gaia_solution_vector'][c])

            self.data_AM['astro_gost'].to_csv(f'{saveloc}astro_gost{tail}.csv')
            
            self.data_AM['dead_gdr2'].to_csv(f'{saveloc}dead_gdr2{tail}.csv')
            self.data_AM['dead_gdr3'].to_csv(f'{saveloc}dead_gdr3{tail}.csv')

            # TODO: rel

        fname = f'{saveloc}temp_model_{self.model_script_no}{tail}.py'

        model_func_name = 'my_model'
        with open(fname, 'w') as f:
            if self.switch_RV:
                self._write_data_RV(f, saveloc, tail)

            if self.switch_AM:
               self._write_data_AM(f, saveloc, tail)

            if self.switch_PM:
                # TODO method here
                pass

            # Init Models
    
            if self.switch_RV:
                self.switch_RV_celerite = False
                for b in self:
                    if b.type_ == 'Celerite2':
                        self.switch_RV_celerite = True
                        self.rv_gp_slice = b.slice
                        model_func_name = 'my_model_support'

                self._write_model_RV(f, model_func_name)
                self._write_model_RV_GP(f)

            if self.switch_AM:
                self._write_model_AM(f)


        self.model_script_no += 1
        return fname


    def _write_data_RV(self, f, saveloc, tail):
        f.write(f'''
# BEGIN WRITE_DATA_RV FROM MODEL
my_data = pd.read_csv('{saveloc}temp_data{tail}.csv', index_col=0)

X_ = my_data['BJD'].values
Y_ = my_data['RV'].values
YERR_ = my_data['eRV'].values
ndat = len(X_)

''')        



        for n in range(self.nins__):
            f.write(f'''
mask{n+1} = (my_data['Flag'] == {n+1}).values
ndat{n+1} = np.sum(mask{n+1})
''')
        sum_cornums = np.sum(self.cornums) or 0
        if sum_cornums > 0:
            sai_count = 1
            for n in range(self.nins__):
                for _ in range(self.cornums[n]):
                    f.write(f'''
SAI{sai_count}_ = my_data.iloc[:, {3+sai_count}].values
''')
                    sai_count += 1

    def _write_data_AM(self, f, saveloc, tail):
        # TODO, add switches self.model.data_AM['AM_abs']
        f.write(f'''

from numpy.linalg import inv, slogdet
                

# BEGIN WRITE_DATA_AM FROM MODEL
data_hg123 = pd.read_csv('{saveloc}temp_hg123{tail}.csv', index_col=0)
data_abs = pd.read_csv('{saveloc}temp_abs{tail}.csv', index_col=0)
data_gost = pd.read_csv('{saveloc}temp_gost{tail}.csv', index_col=0)
AM_astro_gost = pd.read_csv('{saveloc}astro_gost{tail}.csv', index_col=0)
#data_relast = pd.read_csv('{saveloc}temp_relast{tail}.csv')

# Load Dead Points Data
data_dead_gdr2 = pd.read_csv('{saveloc}dead_gdr2{tail}.csv', comment='#', index_col=0)
data_dead_gdr3 = pd.read_csv('{saveloc}dead_gdr3{tail}.csv', comment='#', index_col=0)

# Define catalogs and reference index
AM_cats = {self.AM_cats}
AM_iref = -1  # Reference index pointing to GDR3 data

# Extract necessary columns from data_hg123
columns_to_extract = ['ref_epoch', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
AM_catalog_array = data_hg123[columns_to_extract].values

# Extract time and observation arrays
AM_tt = AM_catalog_array[:, 0]  # Reference epochs
AM_DT = AM_tt - AM_catalog_array[AM_iref, 0]  # Observations at the reference index (GDR3)
AM_obs = AM_catalog_array[AM_iref, 1:]  # Time differences relative to GDR3 epoch

# Extract observation times for astrometry data
AM_abs_T = data_abs['BJD'].values  # Hipparcos observation times
AM_gost_T = data_gost['BJD'].values  # GOST observation times
AM_gost_T_refed = AM_gost_T - AM_tt[AM_iref]  # Time differences relative to GDR3 epoch
AM_t_all = np.concatenate([AM_abs_T, AM_gost_T])  # Combine all observation times

# Parallax at reference epoch
AM_PLX = data_hg123['parallax'].values[AM_iref]


AM_flags = np.array([], dtype=float)
AM_flags = np.append(AM_flags, np.repeat('HIP', len(AM_abs_T)))
AM_flags = np.append(AM_flags, np.repeat('GOST', len(AM_gost_T)))


# Astrometry constants

AM_hipp_epoch = 2448348.75
AM_GDR1_ref_ep = 2457023.5
AM_GDR2_ref_ep = 2457206
AM_GDR3_ref_ep = 2457388.5

AM_GDR1_baseline = 2457281.5
AM_GDR2_baseline = 2457532
AM_GDR3_baseline = 2457902

gaia_offset = 1717.6256
gaia_scale_factor = 365.25 / 1461


# Convert dead intervals for GDR2/3
data_dead_gdr2['start'] = 2457023.75 + (data_dead_gdr2['start'] - 1717.6256)/(1461)*365.25
data_dead_gdr3['start'] = 2457023.75 + (data_dead_gdr3['start'] - 1717.6256)/(1461)*365.25

data_dead_gdr2['end']   = 2457023.75 + (data_dead_gdr2['end'] - 1717.6256)/(1461)*365.25
data_dead_gdr3['end']   = 2457023.75 + (data_dead_gdr3['end'] - 1717.6256)/(1461)*365.25

# Create validity masks for GDR2 and GDR3

am_valid1 = np.ones(len(AM_gost_T), dtype=bool)
am_valid2 = np.ones(len(AM_gost_T), dtype=bool)
am_valid3 = np.ones(len(AM_gost_T), dtype=bool)

for dead in data_dead_gdr2.values:
    am_valid2[np.logical_and(AM_gost_T >= dead[0], AM_gost_T <= dead[1])] = 0

for dead in data_dead_gdr3.values:
    am_valid3[np.logical_and(AM_gost_T >= dead[0], AM_gost_T <= dead[1])] = 0

mask_hipp = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]


mask_GDR2 = AM_gost_T < AM_GDR2_baseline
mask_GDR3 = AM_gost_T < AM_GDR3_baseline

mask_GDR2 &= am_valid2
mask_GDR3 &= am_valid3

# COEFS

AM_coeffs = dict()

for cat in AM_cats:
    if cat == 'GDR2':
        dgost = data_gost[mask_GDR2]
        ref_ep = AM_GDR2_ref_ep
    elif cat == 'GDR3':
        dgost = data_gost[mask_GDR3]
        ref_ep = AM_GDR3_ref_ep
    else:
        continue  # Skip unknown catalogs

    dgost_psi = dgost['psi'].values
    dgost_time = dgost['BJD'].values - ref_ep
    time_factor = dgost_time / 365.25

    sin_psi = np.sin(dgost_psi)
    cos_psi = np.cos(dgost_psi)

    AM_coeffs[cat] = dict([
        ('a1', sin_psi),
        ('a2', cos_psi),
        ('a3', dgost['parf'].values),
        ('a4', time_factor * sin_psi),
        ('a5', time_factor * cos_psi)
        ])


# GAIA SOL VECTOR

# Gaia Solution Vector

AM_GSV = dict()

for cat in AM_cats:
    coeffs = AM_coeffs[cat]
    XX_dr = np.column_stack([
        coeffs['a1'],
        coeffs['a2'],
        coeffs['a3'],
        coeffs['a4'],
        coeffs['a5']
    ])

    # Use pseudo-inverse for stability
    solution_vector = np.linalg.pinv(XX_dr)
    AM_GSV[cat] = solution_vector


# COV ASTRO

def construct_cov(catalog_data):
    keys = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    num_entries = len(catalog_data)
    cov_matrices = np.zeros((num_entries, 5, 5), dtype=float)

    for idx, row in catalog_data.iterrows():
        cov_matrix = np.zeros((5, 5), dtype=float)
        for i, key_i in enumerate(keys):
            error_key = f'{{key_i}}_error'
            cov_matrix[i, i] = row[error_key] ** 2
            for j in range(i + 1, 5):
                key_j = keys[j]
                cov_key = f'{{key_i}}_{{key_j}}_cov'
                cov_value = row[cov_key]
                cov_matrix[i, j] = cov_matrix[j, i] = cov_value
        cov_matrices[idx] = cov_matrix

    return cov_matrices

AM_COV = construct_cov(data_hg123)
AM_inv_COV = []
AM_log_det_COV = []


for cov_matrix in AM_COV:
    inv_cov_matrix = inv(cov_matrix)
    sign, log_det_cov = slogdet(cov_matrix)
    if sign <= 0:
        raise ValueError("Covariance matrix for "+str(cat)+ "is not positive definite.")
    AM_inv_COV.append(inv_cov_matrix)
    AM_log_det_COV.append(log_det_cov)

common_t = {self.common_t}

''')


    def _write_model_RV(self, f, model_func_name):
        f.write(f'''

def {model_func_name}(theta):
    for a in A_:
        theta = np.insert(theta, a, mod_fixed_[a])

    model0 = np.zeros(ndat)
    err20 = YERR_ ** 2
''')

        for b in self:
            if b.type_ == 'Keplerian':
                #f.write(open(get_support(f'models/{b.model_script}')).read().format(b.slice))
                b.write_model(f)

            # TODO replace with write_args in block formation
            

            elif b.type_ == 'Offset':
                for nin in range(self.nins__):
                    b.number_ = nin+1
                    b.write_model(f)

            elif b.type_ == 'Sinusoid':
                b.write_model(f)

            elif b.type_ == 'MagneticCycle':
                b.write_model(f)

            elif b.type_ == 'StellarActivity':
                f.write(f'''
    theta_sa = theta[{b.slice}]
''')
                sai_count = 1
                for n in range(self.nins__):
                    for _ in range(self.cornums[n]):
                        b.number_ = sai_count
                        b.write_model(f)
                        sai_count += 1

            elif b.type_ == 'StellarActivityPRO':
                f.write(f'''
    theta_sa = theta[{b.slice}]
''')
                                


            elif b.type_ == 'Acceleration':
                b.write_model(f)

            elif b.type_ == 'MOAV':
                f.write('''
    # add residuals for moav
    residuals = Y_ - model0

''')
                if b.is_global:
                    b.write_model(f)

                else:
                    for nin in range(self.nins__):
                        b.number_ = nin+1
                        b.write_model(f)

            elif b.type_ == 'Jitter':
                for nin in range(self.nins__):
                    b.number_ = nin+1
                    b.write_model(f)


        f.write('''
    return model0, err20


''')


    def _write_model_RV_GP(self, f):
        # TODO: we don't need to return gp.predict in my_model func for the mcmc run
        # TODO: we just need it for plots and models. kw?
        if not self.switch_RV_celerite:
            return
        f.write('''

''')
        kernel_file0 = open('temp_kernel00', 'r')
        kernel_string0 = kernel_file0.read()
        kernel_file0.close()

        kernel_file1 = open('temp_kernel01', 'r')
        kernel_string1 = kernel_file1.read()
        kernel_file1.close()

        if self.switch_plot:
            f.write(open(get_support('kernels/00plot.kernel')).read())
        else:
            f.write(open(get_support('kernels/00.kernel')).read().format(kernel_string0,
                                                                        kernel_string1,
                                                                        self.rv_gp_slice))


    def _write_model_AM(self, f):
        # self.model.data_AM['AM_abs']
        if self.data_AM['AM_abs']:
            for b in self:
                if b.type_ == 'AstrometryOffset':
                    # BARYCENTER, obs_lin_prop_PA
                    if True:
                        f.write('''
# Calculate Barycenter
pc2au = 206265
auyr2kms = 4.74047

def mas2deg(x):
    # Converts milliarcseconds to degrees 
    # using np.longdouble for high precision.
    return np.longdouble(x) / np.longdouble(3.6e6)

                            
def bl2xyz(b_rad, l_rad):
    # Converts spherical coordinates (latitude b_rad and 
    # longitude l_rad) to Cartesian coordinates (x, y, z)
    x = np.cos(b_rad) * np.cos(l_rad)
    y = np.cos(b_rad) * np.sin(l_rad)
    z = np.sin(b_rad)
    return np.array([x,y,z])

                            
def xyz2bl_vec(x,y,z):
    # Converts Cartesian coordinates back to spherical coordinates.
    b = np.arctan2(z,np.sqrt(x**2+y**2))
    ind = b>np.pi/2
    if(np.sum(ind)>0):
        b[ind] = b[ind]-np.pi
    l = np.arctan2(y,x)%(2*np.pi)
    return b,l


def obs_lin_prop_PA(obs):
    RA, DE, PLX, PMRA, PMDE, RV = obs
    ra = np.deg2rad(RA)
    de = np.deg2rad(DE)
    plx = PLX
    pmra = PMRA
    pmde = PMDE
    rv = RV

    cosde, sinde = np.cos(de), np.sin(de)
    cosra, sinra = np.cos(ra), np.sin(ra)

    d = 1 / plx  # kpc
    x, y, z = bl2xyz(de, ra) * d * 1e3  # pc
    vra = pmra * d
    vde = pmde * d
    
    vr = rv/auyr2kms  # au/yr
    vx_equ = vr*cosde*cosra - vde*sinde*cosra - vra*sinra  # note: vr is positive if the star is moving away from the Sun
    vy_equ = vr*cosde*sinra - vde*sinde*sinra + vra*cosra
    vz_equ = vr*sinde + vde*cosde

    # Compute positions at all times
    time_factor = AM_DT / (365.25 * pc2au)   
    x1 = x + vx_equ * time_factor
    y1 = y + vy_equ * time_factor
    z1 = z + vz_equ * time_factor
    
    # Convert positions back to observables
    de1_rad, ra1_rad = xyz2bl_vec(x1,y1,z1)  # rad
    d1 = np.sqrt(x1**2 + y1**2 + z1**2)*1e-3  # kpc

    ra1 = np.rad2deg(ra1_rad)  # degrees
    de1 = np.rad2deg(de1_rad)  # degrees

    # Prepare rotation matrices
    cosra1, sinra1 = np.cos(ra1_rad), np.sin(ra1_rad)
    cosde1, sinde1 = np.cos(de1_rad), np.sin(de1_rad)


    # Initial velocity vector
    vv = np.array([vx_equ, vy_equ, vz_equ])
    
    # Build rotation matrices
    zeros = np.zeros_like(cosra1)
    ones = np.ones_like(cosra1)

    # Rotation around the z-axis
    rotz = np.array([
        [cosra1, sinra1, zeros],
        [-sinra1, cosra1, zeros],
        [zeros, zeros, ones]
    ]).transpose(2, 0, 1)  # Shape: (n, 3, 3)

    # Rotation around the y-axis
    roty = np.array([
        [cosde1, zeros, sinde1],
        [zeros, ones, zeros],
        [-sinde1, zeros, cosde1]
    ]).transpose(2, 0, 1)  # Shape: (n, 3, 3)

    # Combine rotations for each time step
    rot = np.matmul(roty, rotz)  # Shape: (n, 3, 3)

    # Apply rotations to the velocity vector
    vequ = np.einsum('ijk,k->ij', rot, vv)

    pmra1 = vequ[:, 1] / d1  # mas/yr
    pmde1 = vequ[:, 2] / d1  # mas/yr
    rv1 = vequ[:, 0] * auyr2kms

    out0 = np.column_stack((ra1, de1, 1 / d1, pmra1, pmde1, rv1))
    return out0



def astrometry_bary(theta):
    theta0 = theta.copy()
    theta0[0] = mas2deg(theta0[0])/np.cos(np.deg2rad(AM_obs[1]))
    theta0[1] = mas2deg(theta0[1])

    theta0 = np.append(theta0, 0)                        
    
    obs = AM_obs - theta0
    
    return obs_lin_prop_PA(obs)


                            
''')
                        
                    # Thiele-Innes, calc_astro, calc_epoch
                    if True:
                        f.write(f'''
                                
# Calculate Epochs
def thiele_innes(omega, Omega, sinI, cosI):
    sinOM, cosOM = np.sin(Omega), np.cos(Omega)
    sinom, cosom = np.sin(omega), np.cos(omega)

    A = (cosom * cosOM
         - sinom * sinOM * cosI)
    B = (cosom * sinOM
         + sinom * cosOM * cosI)

    F = (-sinom * cosOM
         - cosom * sinOM * cosI)
    G = (-sinom * sinOM
         + cosom * cosOM * cosI)

    # these are negative in Catanzarite 2010    
    C = sinom * sinI
    H = cosom * sinI
    return A, B, F, G, C, H


def calc_astro(theta, plx):
    per, K, pha, ecc, omega, I, Omega = theta

    # Precompute constants
    sinI, cosI = np.sin(I), np.cos(I)
    sqrt1_e2 = np.sqrt(1 - ecc**2)

    # Mean anomaly
    #freq = two_pi / per
    freq = 2. * np.pi / per
    M = freq * (AM_t_all - common_t) + pha

    # Solve Kepler's equation
    E = kepler.solve(M, np.repeat(ecc, len(M)))
    f = (np.arctan(((1. + ecc) ** 0.5 / (1. - ecc) ** 0.5) * np.tan(E / 2.)) * 2.)

    # Thiele-Innes constants
    A, B, F, G, C, H = thiele_innes(omega, Omega, sinI, cosI)

    # Orbit in the orbital plane
    X = np.cos(E) - ecc
    Y = sqrt1_e2 * np.sin(E)

    # Velocity components in the orbital plane
    VX = -np.sin(f)
    VY = np.cos(f) + ecc

    # Auxiliary constants
    alpha0 = K/sinI/1e3/auyr2kms  # au/yr
    beta0 = per/365.25*(K/1e3/auyr2kms)*sqrt1_e2/(2*np.pi)/sinI  # au
    alpha = -alpha0 * plx
    beta = -beta0 * plx  # mas

    # Sky plane coordinates
    rasP = beta * (B*X + G*Y)
    decP = beta * (A*X + F*Y)
    plxP =-beta * (C*X + H*Y) * plx/206265e3  # parallax change

    # Proper motions
    pmrasP = alpha * (B*VX + G*VY)
    pmdecP = alpha * (A*VX + F*VY)
    rv = alpha0 * (C*VX + H*VY)  # km/s

    # Photocentric motion correction (if eta != 0)
    #sma, mp = cps(per, K, ecc)
    #mp /= sinI
    eta = 0  #calc_eta(mp)
    if eta != 0:
        xi = 1 / (eta + 1)
        rasP *= xi
        decP *= xi
        plxP *= xi
        pmrasP *= xi
        pmdecP *= xi

    # mas | mas/yr
    return np.array([rasP, decP, plxP, pmrasP, pmdecP, rv*auyr2kms])


def astrometry_epoch(theta):
    model = np.zeros((6, len(AM_t_all)))
    theta_am_off = theta[{b.slice}]
    plx0 = AM_PLX - theta_am_off[2]
''')
                        for bsub in self:
                            if bsub.type_ == 'Keplerian':
                                f.write(f'''
    # iterates for b in ReddModel
    #ms = algo
    #E = algo
    model += calc_astro(theta[{bsub.slice}], plx0)
''')

                        f.write('''
    return model


''')


                    # obs_lin_prop_simple, astrometry_kepler
                    if True:
                        f.write(f'''
def obs_lin_prop_simple(obs):
    RA, DEC, PLX, PMRA, PMDEC, RV = obs
    ra = np.deg2rad(RA)
    dec = np.deg2rad(DEC)
    plx = PLX
    pmra = PMRA
    pmdec = PMDEC
    rv = RV

    
    decs = dec + pmdec*AM_gost_T_refed/365.25/206265e3  # rad
    ras = ra + pmra*AM_gost_T_refed/365.25/np.cos(decs)/206265e3  # rad
    out0 = np.array([np.rad2deg(ras),
                     np.rad2deg(decs),
                     np.repeat(plx, len(AM_gost_T_refed)),
                     np.repeat(pmra, len(AM_gost_T_refed)),
                     np.repeat(pmdec, len(AM_gost_T_refed)),
                     np.repeat(rv, len(AM_gost_T_refed))]).T
                            
    return out0
                            

def astrometry_kepler(theta):
    theta_am_off = theta[{b.slice}]
    barycenter = astrometry_bary(theta_am_off)

    # TODO: add relAst = astrometry_rel here

    # Compute orbital motion
    epoch_all = astrometry_epoch(theta)

    # Process Gaia observations
    epoch_gost = epoch_all[:, -len(AM_gost_T):]

    # Get reference observation
    obs0 = barycenter[AM_iref, :]
    bary = obs_lin_prop_simple(obs0)

    ref_params = AM_catalog_array[AM_iref]
    ref_ra = ref_params[1]
    ref_dec = ref_params[2]

    # Update declination and right ascension
    dec = bary[:, 1] + mas2deg(epoch_gost[1, :])  # degrees
    dec_rad = np.deg2rad(dec)
    cos_dec = np.cos(dec_rad)

    dra = ((bary[:, 0] - ref_ra) * cos_dec * 3.6e6) + epoch_gost[0, :]  # mas
    ddec = (dec - ref_dec) * 3.6e6  # mas

    # Compute Gaia absolute astrometry signal
    psi = data_gost['psi'].values
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    parf = data_gost['parf'].values

    gabs_a = dra * sin_psi
    gabs_b = ddec * cos_psi
    gabs_c = (bary[:, 2] + epoch_gost[2, :]) * parf

    gabs = gabs_a + gabs_b + gabs_c  # mas

    # Compute residuals for each Gaia catalog
    cats = []
    for cat in AM_cats:
        current_mask = mask_GDR2 if cat == 'GDR2' else mask_GDR3
        current_idx = 0 if cat == 'GDR2' else 1
        yy = gabs[current_mask]

        # Get solution vector for current catalog
        solution_vector = AM_GSV[cat]

        # Compute parameters and residuals
        params = solution_vector @ yy
        ast = params  # Parameters from the fit
        res = AM_astro_gost.values[current_idx, :] - ast  # Residuals

        cats.append(res)

    return barycenter, epoch_all[:, :len(AM_abs_T)], cats

                                
''')
                        # calc_eta
                        if True:
                            f.write(f'''

def calc_eta(m1, mlow_m22=0, mup_m22=99, mrl_m22=None):
    # http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    #m2 = {self.starmass}
    eta = 0
    #if (m1>mlow_m22) and (m1<mup_m22) and (m2>mlow_m22) and (m2<mup_m22):
    #    f = mrl_m22
    #    dg = f(m2)-f(m1)
    #    eta = (1+m1/m2)/(10**(0.4*dg)-m1/m2)
    return eta


''')


                if b.type_ == 'AstrometryJitter':
                    f.write(f'''
def AstroDiff(obs1, obs2):
    # obs1, obs2: ra[deg], dec[deg], parallax [mas], pmra [mas/yr], pmdec [mas/yr], rv [km/s]

    dobs = obs2 - obs1
    dobs[:2] = dobs[:2] * 3.6e6  # deg2mas
    dobs[0] = dobs[0] * np.cos(np.deg2rad((obs1[1] + obs2[1])*0.5))

    return dobs
                            

def loglike_AM(theta):
    ll = 0.
    J_H, J_G = theta[{b.slice}]
    barycenter, epoch, cats = astrometry_kepler(theta)

    # LOGLIKELIHOOD HIP
    if True:
        dpmra = dpmde = dplx = 0
        dra = epoch[0]
        dde = epoch[1]

        # Compute astrometric differences
        dastro = AstroDiff(AM_catalog_array[0, 1:], barycenter[0, :])
        dra += dastro[0]
        dde += dastro[1]
        dplx += dastro[2]
        dpmra += dastro[3]
        dpmde += dastro[4]
        
        # Prepare data
        CPSI = data_abs['CPSI'].values
        SPSI = data_abs['SPSI'].values
        EPOCH = data_abs['EPOCH'].values
        PARF = data_abs['PARF'].values

        # Compute predicted astrometric observables
        dabs_new = (CPSI * (dra + dpmra * EPOCH) +
                    SPSI * (dde + dpmde * EPOCH) +
                    PARF * dplx)
    
        # Compute residuals and errors
        res = data_abs['RES'].values - dabs_new
        err2 = data_abs['SRES'].values**2 + J_H**2

        ll += -0.5 * (np.sum(res ** 2 / err2 + np.log(err2))) - 0.5 * np.log(2*np.pi) * len(res)

    # LOGLIKELIHOOD GOST
    if True:
        const_term = 0.5 * np.log(2 * np.pi)
        if 'GDR2' in AM_cats:
            x = cats[0]
            n_x = len(x)

            # Retrieve precomputed inverse covariance and log determinant
            inv_cov = AM_inv_COV[1]
            log_det_cov = AM_log_det_COV[1]

            # Compute Mahalanobis distance
            mahalanobis = x @ inv_cov @ x / J_G**2
            
            # Update log-likelihood
            ll -= 0.5 * (mahalanobis + n_x * np.log(J_G**2) + log_det_cov) + n_x * const_term


        if 'GDR3' in AM_cats:
            x = cats[1]
            n_x = len(x)

            # Retrieve precomputed inverse covariance and log determinant
            inv_cov = AM_inv_COV[2]
            log_det_cov = AM_log_det_COV[2]

            # Compute Mahalanobis distance
            mahalanobis = x @ inv_cov @ x / J_G**2
            
            # Update log-likelihood
            ll -= 0.5 * (mahalanobis + n_x * np.log(J_G**2) + log_det_cov) + n_x * const_term

    return ll

''')


    def display_math(self):
        return ' + '.join(self.get_attr_block('math_display_'))    


    def _init_blocks(self, bloques):
        self.bloques = bloques
        self.bloques_model = []
        self.bloques_data = []
        #self.bloques_error = []


    def _init_data_RV(self, data):
        # TODO handle through DataHandler object
        if data is None:
            return
        self.data = data

        self.x = self.data['BJD'].values
        self.y = self.data['RV'].values
        self.yerr = self.data['eRV'].values

        self.ndata = len(self.x)
        self.switch_plot = False

        self.switch_RV = True
        

    def _init_data_AM(self, data_AM):
        # TODO handle through DataHandler object
        if data_AM is None:
            return
        
        self.data_AM = data_AM
        
        # astro_array
        #to_get = ['ref_epoch','ra','dec','parallax','pmra','pmdec','radial_velocity']
        self.AM_hg123 = self.data_AM['df_hg123']#[to_get].values
        self.AM_abs = self.data_AM['df_abs']#[to_get].values
        self.AM_gost = self.data_AM['df_gost']#[to_get].values
        #self.AM_relast = self.data_AM['df_hg123']#[to_get].values

        self.AM_cats = self.data_AM['cats']
        self.common_t = self.data_AM['common_t']
        #self.AM_tt = self.AM_astro_array[:, 0]
        #iref = -1  # ref to GDR3
        #self.AM_DT = self.AM_tt - self.AM_astro_array[iref,0]
        #self.AM_obs = self.AM_astro_array[iref, 1:]


        #self.data_AM = {'astro_array':self.AM_astro_array,}



    def _init_data_PM(self, data_PM):
        # TODO handle through DataHandler object
        pass


    def get_dependencies(self):
        arr = self.get_attr_block('dependencies')
        flattened = [item for sublist in arr for item in sublist]
        flattened += self.dependencies

        return np.unique(flattened)


    def get_constants(self):
        for c in self.model_constants:
            if type(self.model_constants[c])==np.ndarray:
                self.model_constants[c] = list(self.model_constants[c])

        return self.model_constants


    def get_attr_param(self, call, flat=False):
        a = [b.get_attr(call) for b in self]
        if flat:
            a = [item for sublist in a for item in sublist]
        return a


    def get_attr_block(self, call):
        return [getattr(b, call) for b in self]


    def __getitem__(self, n):
        return self.bloques[n]

    def __iter__(self):
        return iter(self.bloques)

    def __repr__(self):
        return f"{self.get_attr_block('name_')}({self.ndim__})"