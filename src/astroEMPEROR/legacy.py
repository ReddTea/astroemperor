# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def model_line(theta, x):
    m, n, j = theta
    return m * x + n

# MAIN BODY; SIMULATION

    def plot2(self):
        # FULL
        fig = pl.figure(figsize=(10, 10))
        ax = pl.gca()
        c = ['b', 'g', 'r', 'c', 'm', 'y']
        c = flatten([c,c,c,c,c])
        for i in range(self.nins__):
            x = self.datax__[self.dataflag__==i]
            y = self.datay__[self.dataflag__==i]
            yerr = self.datayerr__[self.dataflag__==i]
            dib = ax.errorbar(x, y, yerr=yerr, fmt='%so' % c[i], alpha=0.9)



        minx, maxx = minmax(self.datax__)
        xmod = np.linspace(minx, maxx, 1000)

        DATA = [xmod, np.zeros_like(xmod), np.zeros_like(xmod), np.zeros_like(xmod)]

        #self.plot_model = Model(DATA, self.blocks__)
        ymod, err = Model(DATA, self.blocks__).evaluate_plot(self.ajuste, xmod)
        #y_mod, err = _model_sinusoid([2.8181, 15.1515, 0.3926], x_mod, np.zeros_like(x_mod), np.zeros_like(x_mod), np.zeros_like(x_mod))
        pl.plot(xmod, ymod, 'r-')
        pl.show()


        # PHASED

        # phased real data


        p_ = self.ajuste[0]
        fig = pl.figure(figsize=(10, 10))
        ax = pl.gca()
        for i in range(self.nins__):
            x = self.datax__[self.dataflag__==i]
            y = self.datay__[self.dataflag__==i]
            yerr = self.datayerr__[self.dataflag__==i]
            xf, yf, yerrf = fold(x, y, per=p_, yerr=yerr)

            dib = ax.errorbar(xf, yf, yerr=yerrf, fmt='%so' % c[i], alpha=0.9)

        # phased model
        minx, maxx = minmax(xf)
        x1f = np.linspace(minx, minx+p_, 1000)
        y1f, err = Model(DATA, self.blocks__).evaluate_plot(self.ajuste, x1f)

        pl.plot(x1f, y1f, 'r-')
        pl.show()
        pass


    def plot3(self):
        c = ['b', 'g', 'r', 'c', 'm', 'y']
        c = flatten([c,c,c,c,c])
        alp, ms = 0.3, 2
        #pl.tight_layout()


        samples_h = self.sampler.flatchain[0]
        posts_h = self.posteriors[0]


        #fig, ax = pl.subplots(5, 1, sharey=True, figsize=(16,16), constrained_layout=True)
        i, j = 0, 0
        axsize = 5
        if self.ndim__ < 5:
            axsize = self.ndim__
        for b in self.blocks__:
            for p in b:
                if j % 5 == 0:
                    fig, ax = pl.subplots(axsize, 1, sharey=True, figsize=(8,6), constrained_layout=True)
                ax[j].tick_params(direction='out', length=6, width=2, labelsize=14)
                ax[j].plot(samples_h[:, i], posts_h, '%so' % c[i], alpha=alp, markersize=ms)
                ax[j].set_xlabel(p.name, fontsize=22)
                l1 = ax[j].axvline(p.value)
                i += 1
                j += 1
                if j % 5 == 0 or i==self.ndim__:
                    pl.ylabel('Posterior Prob', fontsize=22)
                    pl.title('Posteriors', fontsize=18)
                    pl.show()
                    j = 0

        pass
