import os

import pandas as pd
import numpy as np
import tracemalloc
from termcolor import colored
from sklearn.mixture import GaussianMixture
from contextlib import contextmanager
from .globals import _OS_ROOT


def get_support(path):
    return os.path.join(_OS_ROOT, 'support', path)


class DataWrapper(object):
    def __init__(self, target_name, read_loc=''):
        self.target_name = target_name
        self.modes = {'RV':{'PATH':f'{read_loc}datafiles/{self.target_name}/RV/',
                            'KEY':'RV',
                            },
                      'PM':{'PATH':f'{read_loc}datafiles/{self.target_name}/PM/',
                            'KEY':'PM',
                            },
                      'AM':{'PATH':f'{read_loc}datafiles/{self.target_name}/AM/',
                            'KEY':'AM',
                            },
                      }

        self.RV_empty_lists = ['ndata', 'ncols', 'nsai', 'data', 'labels',
                               'RV_sets']
        
        self.PM_empty_lists = []

        self.AM_empty_lists = ['AM_labels','AM_sets',
                               'df_gost', 'df_hipgaia', 'df_hip',
                               ]

        self.activate()


    def activate(self):
        for m in self:
            p = m['PATH']
            m['use'] = False
            if os.path.exists(p):
                # macos fix
                filenames = [fn for fn in sorted(os.listdir(p)) if fn != '.DS_Store']

                if len(filenames) > 0:
                    m['use'] = True
                    m['filenames'] = filenames
                    for attribute in getattr(self, f"{m['KEY']}_empty_lists"):
                        m[attribute] = []

        for m in self:
            if m['use']:
                m['logger_msg'] = getattr(self, f"mk_{m['KEY']}")()
        

    def mk_RV(self):
        m = self['RV']
        str2prt = ''
        for file in m['filenames']:
            data = np.loadtxt('{0}{1}'.format(m['PATH'], file))
            ndat, ncol = data.shape

            m['ndata'].append(ndat)
            m['ncols'].append(ncol)
            m['labels'].append(file)

            names = ['BJD', 'RV', 'eRV']

            # identify and name SAI
            nsa = ncol - 3
            if nsa > 0:
                names.extend(f"Staract {len(m['ndata'])} {j}" for j in range(nsa))

            m['nsai'].append(nsa)

            # make dataframe
            df = pd.DataFrame(data, columns=names)
            
            if abs(df.mean()['RV']) > 1e-6:
                df['RV'] -= df.mean()['RV']

            for nam in names:
                if nam[:3] == 'Sta':
                    #if abs(df.mean()[nam]) > 1e-6:
                    df[nam] -= df.mean()[nam]
                    df[nam] = (df[nam] - df.min()[nam]) /(df.max()[nam]-df.min()[nam]) * (df.max()['RV'] - df.min()['RV']) + df.min()['RV']
                    #df[nam] = df[nam] / (df.max()[nam] - df.min()[nam])

            # create another column containing flags for the instrument
            df.insert(loc=3, column='Flag', value=np.ones(ndat, int) * len(m['ndata']))
            
            m['data'].append(df)
            str2prt += f'\nReading data from {file}'

        return str2prt


    def mk_AM(self):
        m = self['AM']
        str2prt = ''


        self.AM_set_data_constants()

        self.AM_load_files()

        self.AM_process_data()

        #self.astrometry_epochs()

        #self.astro_gost()

        return str2prt


    def AM_process_data(self):
        # which catalogs we are using
        m = self['AM']
        if m['AM_cata'] == True:
            # df_hg123
            self.astrometry_hg123()

        if m['AM_astro'] == True:
            self.astrometry_astro()

        if m['AM_hipp'] == True:
            # df_hipp
            self.astrometry_hipp()

        if m['AM_gost'] == True:
            self.astrometry_gost()

        #if identifier == 'hipgaia.astro':
        #    m['df_astro'] = pd.read_csv(ff, sep=r'\s+')


        if m['AM_relAst'] == True:
            print('Relative Astrometry not yet implemented')


    def AM_load_files(self):
        str2prt = ''
        m = self['AM']
        for file in m['filenames']:
            identifier = file.split('_')[-1]
            ff = m['PATH']+file

            # which catalogs we are using
            if identifier == 'hipgaia.hg123':
                m['AM_cata'] = True
                m['AM_astro'] = True

                m['df_hg123'] = pd.read_csv(ff, sep=r'\s+')

            # hipp iad
            elif identifier == 'hip2.abs':
                m['AM_hipp'] = True
                m['df_hipp'] = pd.read_csv(ff, sep=r'\s+')

            # gost iad
            elif identifier == 'gost.csv':
                m['AM_gost'] = True
                m['df_gost'] = pd.read_csv(ff)
                m['df_gost'].columns = m['df_gost'].columns.astype(str).str.strip()

            elif identifier == 'hipgaia.astro':
                pass

            elif identifier == 'Image.rel1':
                m['AM_relAst'] = True
                m['df_rel'] = pd.read_csv(ff)
                
            else:
                print(f'File format not identified for {identifier}')
            # gaia astro
            '''
                m['AM_astro'] = True
                m['df_astro'] = pd.read_csv(ff, sep=r'\s+')
            '''

            str2prt += f'\nReading data from {file}'
        return str2prt


    def mk_PM(self):
        pass


    def get_data__(self, sortby='BJD'):
        m = self['RV']
        df = pd.concat(m['data']).sort_values(sortby)
        m['common_t'] = df['BJD'].min()
        self['AM']['common_t'] = m['common_t']  # also pass to AM data
        df['BJD'] -= m['common_t']
        return df.fillna(0)


    def AM_set_data_constants(self):
        m = self['AM']
        m['AM_cata'] = False
        m['AM_hipp'] = False
        m['AM_gost'] = False
        m['AM_relAst'] = False

        m['cats'] = ['GDR2', 'GDR3']  # GYX GDR2?

        m['hipp_epoch'] = self.time_all_2jd(1991.25,
                                            fmt='decimalyear')  # 2448348.75

        m['AM_GDR1_ref_ep'] = 2457023.5  # self.time_all_2jd(2015, fmt='decimalyear')  # 2457023.5

        m['AM_GDR2_ref_ep'] = 2457206  # self.time_all_2jd(2015.5, fmt='decimalyear')  # 2457206 hardcode?
        m['gdr2_baseline'] = 2457532

        m['AM_GDR3_ref_ep'] = 2457388.5  # self.time_all_2jd(2016, fmt='decimalyear')  # 2457388.5
        m['gdr3_baseline'] = 2457902

        m['dead_gdr2'] = pd.read_csv(get_support('deadtime/astrometric_gaps_gaiadr2_08252020.csv'), comment='#')
        m['dead_gdr3'] = pd.read_csv(get_support('deadtime/astrometric_gaps_gaiaedr3_12232020.csv'), comment='#')


    def astrometry_hg123(self):
        m = self['AM']
        # delete DR1 data?
        #m['df_hg123'] = m['df_hg123'].drop(1)
        #m['df_hg123'] = m['df_hg123'].reset_index(drop=True)
        data_hg123 = self['AM']['df_hg123']
        columns_to_extract = ['ref_epoch', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        AM_catalogs_ = data_hg123[columns_to_extract]

        m['AM_iref_'] = -1
        m['AM_catalogs_'] = AM_catalogs_


    def astrometry_astro(self):
        m = self['AM']
        df = self['AM']['df_hg123'][['ra','dec','parallax','pmra','pmdec']][-2:]

        df['ra'] = (df['ra']-df['ra'].iloc[-1])*np.cos(df['dec'].iloc[-2]*np.pi/180)*3.6e6
        df['dec'] = (df['dec']-df['dec'].iloc[-1])*3.6e6
        df.rename(columns={'ra': 'dra', 'dec':'ddec'}, inplace=True)
        
        self['AM']['df_astro'] = df

        if 'GDR2' not in m['cats']:
            self['AM']['astro_gost'] = self['AM']['astro_gost'].drop(1)        
        


    def astrometry_hipp(self):
        # assert df_iad_gost is correct
        self['AM']['data_iad_gost'] = self['AM']['df_hipp']
        pass


    def astrometry_gost(self):
        self.astrometry_gost_readable_()

        self.astrometry_gost_filter()

        # GYX: check dead time of DR2 and DR3
        self.astrometry_validate_epochs()
        #self.astrometry_gost_epochs_legacy()

        self.astrometry_gost_transform()

        # GDRx to use
        for gdr in self['AM']['cats']:
            number = gdr[-1]
            self.astrometry_gost_coef(number)
        
        self.astrometry_gost_gaia_sol_vec()

        self.astrometry_construct_covariance()




    def astrometry_gost_readable_(self):
        m = self['AM']
        if isinstance(m['df_gost'], pd.DataFrame):
            A = 'ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]'
            B = 'scanAngle[rad]'
            C = 'parallaxFactorAlongScan'
            D = 'parallaxFactorAcrossScan'
            column_mapping = {A: 'BJD',
                              B: 'psi',
                              C: 'parf',
                              D: 'parx',
                              'ra[rad]':'RA',
                              'dec[rad]':'DEC',
                              }
            
            m['df_gost'] = m['df_gost'].rename(columns=column_mapping)

    def astrometry_gost_filter(self):
        m = self['AM']
        a = m['df_gost']
        filtered = a[a['BJD'] < m['gdr3_baseline']]
        m['df_gost'] = filtered[['BJD', 'psi', 'parf', 'parx']]

    def astrometry_validate_epochs(self):
        m = self['AM']
        N_GOST = len(m['df_gost'])
        gaia_offset = 1717.6256
        DAY_PER_YEAR = 365.25
        gaia_scale_factor = DAY_PER_YEAR / 1461
        # Convert dead intervals for GDR2/3
        m['dead_gdr2']['start'] = 2457023.75 + (m['dead_gdr2']['start'] - gaia_offset)*gaia_scale_factor
        m['dead_gdr3']['start'] = 2457023.75 + (m['dead_gdr3']['start'] - gaia_offset)*gaia_scale_factor

        m['dead_gdr2']['end']   = 2457023.75 + (m['dead_gdr2']['end'] - gaia_offset)*gaia_scale_factor
        m['dead_gdr3']['end']   = 2457023.75 + (m['dead_gdr3']['end'] - gaia_offset)*gaia_scale_factor

        # Create validity masks for GDR2 and GDR3

        am_valid1 = np.ones(N_GOST, dtype=bool)
        am_valid2 = np.ones(N_GOST, dtype=bool)
        am_valid3 = np.ones(N_GOST, dtype=bool)

        time_iad_gost = m['df_gost'].BJD.values

        for dead in m['dead_gdr2'].values:
            am_valid2[np.logical_and(time_iad_gost >= dead[0], time_iad_gost <= dead[1])] = 0

        for dead in m['dead_gdr3'].values:
            am_valid3[np.logical_and(time_iad_gost >= dead[0], time_iad_gost <= dead[1])] = 0

        mask_GDR2 = time_iad_gost < m['gdr2_baseline']
        mask_GDR3 = time_iad_gost < m['gdr3_baseline']

        mask_GDR2 &= am_valid2
        mask_GDR3 &= am_valid3

        m['mask_GDR2'] = mask_GDR2
        m['mask_GDR3'] = mask_GDR3
        
        m['a1'], m['a2'], m['a3'], m['a4'], m['a5'] = [], [], [], [], []

    def astrometry_gost_transform(self):
        m = self['AM']
        m['df_gost']['CPSI'] = np.cos(m['df_gost']['psi'].values)
        m['df_gost']['SPSI'] = np.sin(m['df_gost']['psi'].values)

    def astrometry_gost_epochs_legacy(self):
        m = self['AM']
        m0 = m['df_gost']
        if isinstance(m0, pd.DataFrame):
            t = m0.BJD.values

            m['gdr1_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2015-09-16 16:00:00')]
            m['gdr2_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2016-05-23 11:35:00')]
            m['gdr3_epoch'] = [self.time_all_2jd('2014-07-25 10:30:00'),
                            self.time_all_2jd('2017-05-28 08:44:00')]

            m['mask_hipp'] = (m['hipp_epoch'] <= t) & (t <= m['gdr1_epoch'][0])
            m['mask_gdr1'] = (m['gdr1_epoch'][0] <= t) & (t<= m['gdr1_epoch'][1])
            m['mask_gdr2'] = (m['gdr2_epoch'][0] <= t) & (t<= m['gdr2_epoch'][1])
            m['mask_gdr3'] = (m['gdr3_epoch'][0] <= t) & (t<= m['gdr3_epoch'][1])

            m['a1'], m['a2'], m['a3'], m['a4'], m['a5'] = [], [], [], [], []
    
    def astrometry_gost_coef(self, gdr):
        m = self['AM']
        DAY_PER_YEAR = 365.25
        m['AM_coeffs'] = dict()
        for cat in m['cats']:
            mask = m[f'mask_{cat}']
            dgost = m['df_gost'][mask]
            ref_ep = m[f'AM_{cat}_ref_ep']
            
            dgost_time = dgost['BJD'].values - ref_ep
            time_factor = dgost_time / DAY_PER_YEAR

            sin_psi = dgost['SPSI'].values
            cos_psi = dgost['CPSI'].values

            m['AM_coeffs'][cat] = dict([
                        ('a1', sin_psi),
                        ('a2', cos_psi),
                        ('a3', dgost['parf'].values),
                        ('a4', time_factor * sin_psi),
                        ('a5', time_factor * cos_psi)
                        ])

    def astrometry_gost_gaia_sol_vec(self):
        m = self['AM']
        m['AM_GSV'] = dict()

        for cat in m['cats']:
            coeffs = m['AM_coeffs'][cat]
            XX_dr = np.column_stack([
                    coeffs['a1'],
                    coeffs['a2'],
                    coeffs['a3'],
                    coeffs['a4'],
                    coeffs['a5']
            ])

            # Use pseudo-inverse for stability
            solution_vector = np.linalg.pinv(XX_dr)
            #solution_vector = np.linalg.inv(XX_dr.T@XX_dr).astype(float)@XX_dr.T
            m['AM_GSV'][cat] = solution_vector


    def astrometry_construct_covariance(self):
        from numpy.linalg import inv, slogdet
        m = self['AM']
        AM_COV = self._construct_cov(m['df_hg123'])
        AM_inv_COV = []
        AM_log_det_COV = []

        for cov_matrix in AM_COV:
            inv_cov_matrix = inv(cov_matrix)
            sign, log_det_cov = slogdet(cov_matrix)
            if sign <= 0:
                raise ValueError("Covariance matrix for "+str(cat)+ "is not positive definite.")
            AM_inv_COV.append(inv_cov_matrix)
            AM_log_det_COV.append(log_det_cov)
        m['AM_inv_COV'] = np.array(AM_inv_COV)
        m['AM_log_det_COV'] = AM_log_det_COV


    def _construct_cov(self, catalog_data):
        keys = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        num_entries = len(catalog_data)
        cov_matrices = np.zeros((num_entries, 5, 5), dtype=float)

        for idx, row in catalog_data.iterrows():
            cov_matrix = np.zeros((5, 5), dtype=float)
            for i, key_i in enumerate(keys):
                error_key = f'{key_i}_error'
                cov_matrix[i, i] = row[error_key] ** 2
                for j in range(i + 1, 5):
                    key_j = keys[j]
                    cov_key = f'{key_i}_{key_j}_cov'
                    cov_value = row[cov_key]
                    cov_matrix[i, j] = cov_matrix[j, i] = cov_value
            cov_matrices[idx] = cov_matrix

        return cov_matrices        



    def time_all_2jd(self, time_str, fmt='iso'):
        from astropy.time import Time as AstroTime
        t = AstroTime(time_str, format=fmt)
        return t.to_value('jd')


    def __repr__(self):
        x = ''
        for p in self:
            x += f'{p}\n'
        return x


    def __getitem__(self, string):
        return self.modes[string]


    def __iter__(self):
        for key in self.modes:
            yield self.modes[key]


class reddlog(object):
    def __init__(self):
        self.log = ''
        try:
            self.terminal_width = os.get_terminal_size().columns
        except Exception:
            self.terminal_width = pd.get_option('display.width')
        self.baddies_list = ["\x1b[", '0m', '1m', '4m', '7m', '31m', '32m']

        self.start_message()


    def saveto(self, location, name=''):
        np.savetxt(f'{location}/log{name}.dat', np.array([self.log]), fmt='%100s')


    def start_message(self):
        self('   ', center=True, save=False, c='green', attrs=['bold', 'reverse'])
        self('~~ Simulation Successfully Initialized ~~', center=True, save=False, c='green', attrs=['bold', 'reverse'])
        self('   ', center=True, save=False, c='green', attrs=['bold', 'reverse'])
    

    def end_run_message(self):
        self._centered_mag_message('~~ End of the Run ~~')


    def next_run_message(self):
        self._centered_mag_message('~~ Proceeding with the next run ! ~~')


    def _centered_mag_message(self, arg0):
        self.line()
        self('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
        self(arg0, center=True, c='magenta', attrs=['bold', 'reverse'])
        self('   ', center=True, c='magenta', attrs=['bold', 'reverse'])
        self.line()


    def _add_block_message(self, block_type, block_name):
        msg = '{} {}, {}'.format(colored(block_type, 'green', attrs=['bold']),
                                 colored('block added', 'green'),
                                 colored(block_name, 'green'))
        msg = f'                              {msg}'
        self(msg, center=True)
        self.line()


    def _add_subtitle(self, msg, double=False, postprocess=False, center=True):
        if double:
            self.line()
        self.line()
        if postprocess:
            self(f'~~ {msg} ~~', center=center, c='yellow', attrs=['bold', 'reverse'])
        else:
            self(f'{msg}', center=center)
        self.line()
        if double:
            self.line()


    def check(self, msg, cond):
        box = colored('✘', attrs=['reverse'], color='red')
        if cond:
            box = colored('✔', attrs=['reverse'], color='green')
        self(f'{msg}: {box}', c='blue', save_extra_n=True)


    def _apply_condition_message(self, c):
        msg = '\nCondition applied: Parameter {} attribute {} set to {}'.format(
                    colored(c[0], attrs=['underline']),
                    colored(c[1], attrs=['underline']),
                    colored(c[2], attrs=['underline']))
        self(msg)


    def message_width(self, msg, val, width):
        self(f"{msg:<{width}}" + colored(val, attrs=['bold']), c='blue', save_extra_n=True)


    def line(self, *args):
        self('\n', *args)

    def dline(self):
        self.line()
        self.line()
    
    def help(self):
        print('Colors: grey, red, green, yellow, blue, magenta, cyan, white')
        print('On_Colors: on_<color>')
        print('Attrs: bold, dark, underline, blink, reverse, concealed')


    def __call__(self, msg, center=False, save=True, c=None,
                 oc=None, attrs=None, save_extra_n=False):
        if attrs is None:
            attrs = []
        if 'reversed' in attrs:
            self.line()
        if save:
            msg0 = msg
            if save_extra_n:
                msg0 += '\n'
            for b in self.baddies_list:
                msg0 = msg0.replace(b, '')

            self.log += msg0
        if center:
            msg = msg.center(self.terminal_width)
        if c:
            msg = colored(msg, c, oc, attrs)

        print(msg)


class Debugger(object):
    def __init__(self):
        self.debugging = False
        self.initialized = False

        
    def debug_script(self, f, msg):
        if self.debugging:
            f.write(f'''
{msg}
''')

    def debug_snapshot(self):
        if self.debugging:
            self._check_trace()

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            print("----[ Top 30 ]----")
            for stat in top_stats[:30]:
                print(stat)
            print("----[ Top 30 ]----")


    def _check_trace(self):
        if not self.initialized:
            tracemalloc.start()
            self.initialized = True


    def __call__(self, msg):
        if self.debugging:
            print(msg)


class GaussianMixture_addon(GaussianMixture):
    def __init__(self, name, unit, **kw):
        # this one is per parameter
        self.name = name
        self.unit = unit
        if self.unit is None:
            self.unit = ''
        self.means = None
        self.covariances = None
        self.sigmas = None
        self.weights = None
        self.mixture_mean = None
        self.mixture_variance = None
        self.mixture_sigma = None
        super().__init__(**kw)

    def update(self):
        self.means = self.means_.flatten()
        self.covariances = self.covariances_.flatten()
        self.sigmas = np.sqrt(self.covariances)
        self.weights = self.weights_
        self.mixture_mean = self.Mixture_Mean()
        self.mixture_variance = self.Mixture_Variance()
        self.mixture_sigma = np.sqrt(self.mixture_variance)

    def Mixture_Variance(self):
        s1 = np.sum(self.weights*self.covariances)
        s2 = np.sum(self.weights*self.means**2)
        s3 = np.sum(self.weights*self.means)**2
        return s1 + s2 - s3

    def Mixture_Mean(self):
        return np.sum(self.means*self.weights)


    def __repr__(self):
        return f'{self.means}; {self.sigmas}'


class GM_Estimator:
    def __init__(self):
        # This one is per param
        self.gm_estimator = None
        self.max_n_components = 5
        self.BIC0_ = np.inf
        self.BIC_Tolerance = 0
        self.name__ = 'GM_Estimator'


    def estimate(self, chain_p, name, unit):
        comp_bic = self.BIC0_
        gm0 = None
        chain_p = chain_p[:, np.newaxis]

        for n in range(self.max_n_components):
            gm = GaussianMixture_addon(name, unit, **{'n_components':n+1}).fit(chain_p)
            gm.update()
            mu = gm.means_
            sel_bic = gm.bic(mu)
            if sel_bic - comp_bic < self.BIC_Tolerance:
                comp_bic = sel_bic
                gm0 = gm
            else:
                break                


        self.gm_estimator = gm0
        return self.gm_estimator



@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True, suppress_stdin=True):
    # Save the original file descriptors
    saved_stdout_fd = os.dup(1)  # FD for stdout
    saved_stderr_fd = os.dup(2)  # FD for stderr
    saved_stdin_fd = os.dup(0)   # FD for stdin

    # Open the null device
    devnull_fd = os.open(os.devnull, os.O_RDWR)

    try:
        if suppress_stdout:
            os.dup2(devnull_fd, 1)  # Replace stdout with devnull
        if suppress_stderr:
            os.dup2(devnull_fd, 2)  # Replace stderr with devnull
        if suppress_stdin:
            os.dup2(devnull_fd, 0)  # Replace stdin with devnull
        yield
    finally:
        if suppress_stdout:
            os.dup2(saved_stdout_fd, 1)  # Restore stdout
            os.close(saved_stdout_fd)
        if suppress_stderr:
            os.dup2(saved_stderr_fd, 2)  # Restore stderr
            os.close(saved_stderr_fd)
        if suppress_stdin:
            os.dup2(saved_stdin_fd, 0)   # Restore stdin
            os.close(saved_stdin_fd)
        os.close(devnull_fd)


def create_directory_structure(name, base_path='', k=0, first=True):
    # Determine unique run path
    aux = 1
    while os.path.exists(f'{base_path}datalogs/{name}/run_{aux}'):
        aux += 1
    
    # Adjust for 'first' condition
    if not first:
        aux -= 1
    
    dr0 = f'{base_path}datalogs/{name}/run_{aux}'

    # Final directory with 'k' parameter
    dr = f'{dr0}/k{k}'

    # Define all subdirectories to create
    subdirectories = ['plots/betas',
                        'plots/histograms',
                        'plots/GMEstimates',
                        'plots/arviz/cornerplots',
                        'plots/arviz/traces',
                        'plots/arviz/normed_posteriors',
                        'plots/arviz/density_intervals',
                        'plots/models/uncertainpy',
                        'plots/posteriors/scatter',
                        'plots/posteriors/hexbin',
                        'plots/posteriors/gaussian',
                        'plots/posteriors/chains',
                        'samples/posteriors',
                        'samples/likelihoods',
                        'samples/chains',
                        'temp/models',
                        'restore/backends',
                        'tables',
                        ]

    # Create the base directory and all specified subdirectories
    os.makedirs(dr, exist_ok=True)
    for subdirectory in subdirectories:
        os.makedirs(os.path.join(dr, subdirectory), exist_ok=True)

    return dr


def adjust_table_tex(to_tab0, rounder=3):
    for j in range(len(to_tab0)):
        if type(to_tab0[j][0]) == str:
            to_tab0[j] = adjust_str_length(to_tab0[j])

        elif type(to_tab0[j][0]) in [np.float64, list, np.ndarray]:
            to_tab0[j] = np.round(to_tab0[j], rounder)

    return to_tab0


def adjust_str_length(my_list):
    max_length = max(len(item) for item in my_list)
    return [item.ljust(max_length) for item in my_list]


def mk_table(headers, values):
    # sets the val length to the length of the biggest name
    par_box_names = values[0]
    max_length = max(len(item) for item in par_box_names)
    par_box_names = [item.ljust(max_length) for item in par_box_names]


def fold_dataframe(df, per=None):
    if per is None:
        per = 2 * np.pi
    df['BJD'] = df['BJD'] % per
    return df.sort_values('BJD')


def sec_to_clock(seconds):
    # Calculate hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)

    # Format the result as hh:mm:ss
    return f'{hours:02d}:{minutes:02d}:{remaining_seconds:02d}'




#