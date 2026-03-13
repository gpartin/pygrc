import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from pygrc.reader import Reader
import numpy as np
from iminuit import Minuit
import typing as tp


class Plot:
    """Class for Generating Matplotlib Plots"""

    def __init__(self):
        """ """
        self.units_dict: dict = {
            "Rad": "kpc",
            "Vobs": "km/s",
            "errV": "km/s",
            "Vgas": "km/s",
            "Vdisk": "km/s",
            "Vbul": "km/s",
            "SBdisk": "L/pc2",
            "SBbul": "L/pc2",
        }

    def plot(
        self, data: pd.DataFrame, column_x: str, column_y: str
    ):  # , save_name: str =""):
        """
        function to plot data on x axis and y axis.
        Args:
            data  :  pd.DataFrame
            column_x: name of dataframe's column for x axis
            column_y: name of dataframe's column for y axis

        """
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        if data[column_x].min() < 0:
            min_x = data[column_x].min() * 0.4
        else:
            min_x = 0

        if data[column_y].min() < 0:
            min_y = data[column_y].min() * 0.4
        else:
            min_y = 0
        max_x = data[column_x].max()
        max_y = data[column_y].max()

        if min_x == max_x or min_y == max_y:
            return 0
        fig, ax = plt.subplots()
        ax.errorbar(data[column_x], data[column_y], linestyle="none", marker="o")
        ax.set_xlim(min_x, max_x * 1.1)
        ax.set_ylim(min_y, max_y * 1.1)
        x_unit = self.units_dict[column_x]
        y_unit = self.units_dict[column_y]
        ax.set_xlabel(column_x + " " + x_unit)
        ax.set_ylabel(column_y + " " + y_unit)
        ax.set_title(
            column_y
            + " "
            + self.units_dict[column_y]
            + " vs "
            + column_x
            + " "
            + self.units_dict[column_x]
        )
        fig.savefig(column_x + "_" + column_y + ".pdf")
        plt.show()

    def overlap(
        self,
        data: pd.DataFrame,
        column_x: str,
        column_y: list,
        x_label: str = "",
        y_label: str = "",
        save_name: str = "",
    ):
        """
        function to plot two differeny y data on the same x axis.
        Args:
            data  :  pd.DataFrame
            column_x (str): name of dataframe's column for x axis
            column_y (list): name of dataframe's column for y axis
            y_label (str): label for the y axis
        """
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        if type(column_x) != str:
            raise ValueError("Please Enter a single column as string for X axis")
        if type(column_y) != list:
            raise ValueError("Please Enter a List for X axis")
        fig, ax = plt.subplots()
        for column_y in column_y:
            if data[column_x].min() < 0:
                min_x = data[column_x].min() * 0.4
            else:
                min_x = 0
            max_x = data[column_x].max()
            ax.errorbar(
                data[column_x],
                data[column_y],
                linestyle="none",
                marker="o",
                label=column_y,
            )
            ax.set_xlim(min_x, max_x * 1.1)
            # ax.set_xlabel(column_x + " " + self.units_dict[column_x])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(save_name)
            plt.legend()
        fig.savefig(column_x + "_" + "vars" + "_" + save_name + ".pdf")

    def plot_all(
        self,
        data: pd.DataFrame,
    ):
        """
        function to plot all the possible plots from data.
        Args:
            data  :  pd.DataFrame
        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300

        for column_x in data.columns:
            for column_y in data.columns:
                if column_x == column_y:
                    continue
                else:
                    if data[column_x].min() < 0:
                        min_x = data[column_x].min() * 0.4
                    else:
                        min_x = 0

                    if data[column_y].min() < 0:
                        min_y = data[column_y].min() * 0.4
                    else:
                        min_y = 0
                    max_x = data[column_x].max()
                    max_y = data[column_y].max()

                    if min_x == max_x or min_y == max_y:
                        continue
                    fig, ax = plt.subplots()
                    ax.errorbar(
                        data[column_x], data[column_y], linestyle="none", marker="o"
                    )
                    ax.set_xlim(min_x, max_x * 1.1)
                    ax.set_ylim(min_y, max_y * 1.1)
                    ax.set_xlabel(column_x + " " + self.units_dict[column_x])
                    ax.set_ylabel(column_y + " " + self.units_dict[column_y])
                    ax.set_title(
                        column_y
                        + " "
                        + self.units_dict[column_y]
                        + " vs "
                        + column_x
                        + " "
                        + self.units_dict[column_x]
                    )
                    fig.savefig(column_x + "_" + column_y + ".pdf")
                    plt.show()

    def corr_map(self, data: pd.DataFrame):
        """
        function to plot correlation map.
        Args:
            data  :  pd.DataFrame
        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        m_corr = Reader.correlation(data)
        fig, ax = plt.subplots()
        sb.heatmap(m_corr, vmax=0.3, center=0, square=True, linewidths=0.5)
        fig.savefig("correlation.pdf")
        plt.show()

    def rotation(self, data: pd.DataFrame):
        """
        function for plotting galaxy rotation curve
        Args:
            data  :  pd.DataFrame

        """

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        self.plot(data, "Rad", ["Vobs"])

    def plot_grc(
        self,
        data: pd.DataFrame,
        m: Minuit,
        function: tp.Callable,
        name: str,
        title: str,
        ax,
    ):
        """
        function to plot two differeny y data on the same x axis.
        Args:
            data  (pd.DataFrame) : data
            m : Minuit  Object
            function: the definition of function without parameters values
            label: string name of the galaxy
        """
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        handles, labels = plt.gca().get_legend_handles_labels()

        x = np.linspace(data["Rad"].min(), data["Rad"].max(), 200)

        if "Data" not in labels:
            ax.plot(
                data["Rad"], data["Vobs"], marker="o", linestyle="none", label="Data"
            )
        ax.plot(x, function(x, *m.values), linestyle="--", label=name)
        plt.legend()
        ax.set_xlabel("Distance (kpc)")
        ax.set_ylabel("Velocity (Km/s)")
        ax.set_title(title)
        plt.savefig(title + "rot_fit.pdf")
