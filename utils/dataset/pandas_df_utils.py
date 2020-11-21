# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from functools import wraps
import numpy as np
import pandas as pd
from utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_LABEL_COL,
)

logger = logging.getLogger(__name__)

def has_columns(df, columns):
    """
    Check if DataFrame has necessary columns
    :param df(pd.DataFrame):  DataFrame
    :param columns: (list(str)) columns to check for
    :return: (bool) True if DataFrame has specified columns
    """
    df_columns = df.columns
    result = True
    for column in columns:
        if column not in df_columns:
            logger.error("Missing column: {} in DataFrame".format(column))
            result = False

    return result


def has_same_base_dtype(df1, df2, columns=None):
    """
    Check if specified columns have the same base dtypes across both DataFrames
    :param df1: (pd.DataFrame) first DataFrame
    :param df2: (pd.DataFrame) second DataFrame
    :param columns: (list(str)) columns to check, None checks all columns
    :return: (bool) True if DataFrames columns have the same base dtypes
    """

    if columns is None:
        if any(set(df1.columns).symmetric_difference(set(df2.columns))):
            logger.error(
                "Cannot test all columns because they are not all shared across DataFrames"
            )
            return False
        columns = df1.columns

    if not (
            has_columns(df=df1, columns=columns) and has_columns(df=df2, columns=columns)
    ):
        return False

    result = True
    for column in columns:
        if df1[column].dtype.type.__base__ != df2[column].dtype.type.__base__:
            logger.error("Columns {} do not have the same base datatype".format(column))
            result = False

    return result


def check_column_dtypes(func):
    """Checks columns of DataFrame inputs
    This includes the checks on:
    1. whether the input columns exist in the input DataFrames
    2. whether the data types of col_user as well as col_item are matched in the two input DataFrames.
    Args:
        func (function): function that will be wrapped
    """

    @wraps(func)
    def check_column_dtypes_wrapper(
            rating_true,
            rating_pred,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
            *args,
            **kwargs
    ):
        """Check columns of DataFrame inputs
        Args:
            rating_true (pd.DataFrame): True data
            rating_pred (pd.DataFrame): Predicted data
            col_user (str): column name for user
            col_item (str): column name for item
            col_rating (str): column name for rating
            col_prediction (str): column name for prediction
        """

        if not has_columns(rating_true, [col_user, col_item, col_rating]):
            raise ValueError("Missing columns in true rating DataFrame")
        if not has_columns(rating_pred, [col_user, col_item, col_prediction]):
            raise ValueError("Missing columns in predicted rating DataFrame")
        if not has_same_base_dtype(
                rating_true, rating_pred, columns=[col_user, col_item]
        ):
            raise ValueError("Columns in provided DataFrames are not the same datatype")

        return func(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction=col_prediction,
            *args,
            **kwargs
        )

    return check_column_dtypes_wrapper