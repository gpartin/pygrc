import os
import pandas as pd
import numpy as np
import pathlib


class Reader:
    """Reader Class"""

    def read(
        filepath: os.PathLike,
        columns: list = [
            "Rad",
            "Vobs",
            "errV",
            "Vgas",
            "Vdisk",
            "Vbul",
            "SBdisk",
            "SBbul",
        ],
        units: list = [
            "kpc",
            "km/s",
            "km/s",
            "km/s",
            "km/s",
            "km/s",
            "L/pc^2",
            "L/pc^2",
        ],
    ):
        """
        Reading a file

        Args:

        filepath : path of the file.

        columns  : list of columns.

        units    : dict for the units.

        """
        if pathlib.PurePosixPath(filepath).suffix == ".dat":
            rawdata = np.loadtxt(filepath)
            data = pd.DataFrame(rawdata, columns=columns)
            return data
        elif pathlib.PurePosixPath(filepath).suffix == ".csv":
            data = pd.read_csv(filepath)
            return data
        else:
            raise ValueError("Format not supported")
            return 0

    def drop_columns(data: pd.DataFrame, columns: list):
        """
        Drop columns from the DataFrame

        Args:

        data  :  pandas dataFrame

        columns: list of columns to be droped
        """
        if len(columns) != 0:
            data = data.drop(columns=columns, axis=1)
        return data

    def read_corr(data: pd.DataFrame, row: int, column: int):
        """
        read a correlation matrix

        Args:

        data  :  pandas dataFrame.

        row   :  number of row element.

        column   :  number of column element.

        """
        return data.iloc[row, column]

    def correlation(data: pd.DataFrame):
        """
        provides correlation matrix for the DataFrame

        Args:

        data  :  pandas dataFrame
        """
        data = data.dropna(axis=1, how="all")
        m_corr = data.corr(method="pearson")
        return m_corr

    def len(data: pd.DataFrame):
        """
        number of rows in the dataframe

        Args:

        data  :  pandas dataFrame
        """
        return len(data)
