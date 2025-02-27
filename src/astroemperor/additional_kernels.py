from celerite2.terms import TermSum, SHOTerm

class GonzRotationTerm(TermSum):

    @staticmethod
    def get_test_parameters():
        return dict(period=3.45, tau=0.5)

    def __init__(self, *, period, tau):
        self.period = float(period)
        self.tau = float(tau)

        # One term with a period of period
        rho1 = period

        # Another term at half the period
        rho2 = period * 0.5

        super().__init__(
            SHOTerm(S0=1, rho=rho1, tau=tau), SHOTerm(S0=1, rho=rho2, tau=tau)
        )
