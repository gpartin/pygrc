from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
import typing as tp
import matplotlib.pyplot as plt


class Fit:
    """
    Class to make fitting easier
    relies on iminuit for fitting
    """

    def __init__(
        self,
        data_x: tp.Sequence,
        data_y: tp.Sequence,
        *args: tp.Union[float, tp.Sequence[float]]
    ):
        """
        Args:
            data_x (tuple): input data on x axis
            data_y (tuple): input data on y axis
            *args (float): Takes initial parameter values


        """
        self.args = args
        print(type(data_x))
        if "numpy" not in str(type(data_x)):
            self.data_x = data_x.to_numpy()
        else:
            self.data_x = data_x
        if "numpy" not in str(type(data_y)):
            self.data_y = data_y.to_numpy()
        else:
            self.data_y = data_y

    def fit_lsq(
        self,
        function: tp.Callable,
        limit: tp.Union[float, tp.Sequence[float]],
        err: tp.Union[float, tp.Sequence[float]],
        bools: tp.Union[bool, tp.Sequence[bool]] = "NAN",
    ):
        """
        function to fit a given curve with least_square
        Args:
            function: the definition of function
            limit (float): limits for the parameters
            err (float): error step
            bools: to say if any parameters are fixed
        Returns:
            iminuit object

        Example:

        ```python
        pygrc.Fit(x, y, 1,1).fit_lsq(function,[(-1,10),(-2,10)],.1)
        ```
        here we have two variables and two parameters (initial values of parameters is 1.)
        where [..] are the limits for the two parameter and .1 is the error.


        """
        least_square = LeastSquares(self.data_x, self.data_y, err, function)
        m = Minuit(least_square, *self.args)
        m.limits = limit
        if bools != "NAN":
            m.fixed = bools
        print(m.limits)
        m.migrad()
        m.hesse()
        m.minos()
        return m

    def get_profile(self, m: Minuit, par: str):
        """
        function to get profile curve for a given parameter
        Args:
            m: iminuit object
            par (str): name of the parameter
        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300

        if par not in m.pos2var:
            raise ValueError("Please Enter a valid variable")
        fig, ax = plt.subplots()
        data = m.draw_mnprofile(par, band=True, text=False)
        ax.plot(data[0], data[1])
        ax.set_xlabel(par)
        ax.set_ylabel("Least Square")
        ax.set_title("Log Likelihood for parameter " + par)
        fig.savefig(par + "_profile.pdf")

    def draw_contours(self, m: Minuit, var: tp.Union[str, tp.Sequence[str]]):
        """
        function to get profile curve for a given parameter
        Args:
            m: iminuit object
            par (list): name of the parameters. At least 2 required.
        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        if len(var) <= 1:
            raise ValueError("Takes two or more paramters as input")

        _var1list = []
        for _var in var:
            if _var not in m.pos2var:
                print(_var)
                raise ValueError("Please Enter a valid variable")
        for _var1 in var:
            _var1list.append(_var1)
            for _var2 in var:
                if _var1 == _var2 or _var2 in _var1list:
                    continue
                fig, ax = plt.subplots()
                ax = m.draw_mncontour(_var1, _var2, cl=[0.68, 0.95])
                fig.savefig(_var1 + "_" + _var2 + "_profile.pdf")

    def fit(self, m: Minuit, function: tp.Callable, confidence: bool = True):
        """
        function to get profile curve for a given parameter
        Args:
            m: iminuit object
            function: the definition of function
            confidence (bool) : if confidence is needed set true else false
        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        fit_x = np.linspace(self.data_x.min(), self.data_x.max(), 100)
        fig, ax = plt.subplots()
        pars = []
        for par in range(len(m.parameters)):
            pars.append(m.values[par])
        ax.plot(self.data_x, self.data_y, marker="+", linestyle="none", label="Data")
        ax.plot(
            fit_x, function(fit_x, *pars), marker="+", linestyle="none", label="Fit"
        )
        if confidence:
            # boostrapping method was inspired from https://iminuit.readthedocs.io/en/stable/notebooks/error_bands.html

            par_boostrap = np.random.default_rng().multivariate_normal(
                m.values, m.covariance, size=1000
            )
            y_boostrap = [function(fit_x, *p) for p in par_b]
            yerr_boostrap_1 = np.std(y_boostrap, axis=0)
            yerr_boostrap_2 = 2 * np.std(y_boostrap, axis=0)
            plt.fill_between(
                fit_x,
                function(fit_x, *pars) - yerr_boostrap_1,
                function(fit_x, *pars) + yerr_boostrap_1,
                facecolor="C1",
                alpha=0.5,
                label="68% Confidence Region",
            )
            plt.fill_between(
                fit_x,
                function(fit_x, *pars) - yerr_boostrap_2,
                function(fit_x, *pars) + yerr_boostrap_2,
                facecolor="C1",
                alpha=0.5,
                label="95% Confidence Region",
            )

        plt.legend()
        fig.savefig("fit_curve.pdf")
