"""
Utilities for defining the tree-based discontinuity search mechanism.
"""
import numpy as np
import pandas as pd

from llr import compute_llr


class Node:
    """class for individual nodes in the RDDTree"""

    def __init__(self, split_col=None, split_val=None, llr=None, group=None):
        """Attributes:
            split_col (str): the data column the split was defined on
            split_value (float): the value of split_col that was split on:
            llr (float): the llr value of the split
            group (np.array)): a binary array of group assignment data points
            left (Node): the left node
            right (Node): the right node
            data (df.DataFrame): the data, only present for terminals
        """
        self.split_col = split_col
        self.split_val = split_val
        self.llr = llr
        self.group = group
        self.right = None
        self.left = None
        self.data = None

    def __str__(self):
        if (self.left is None) and (self.right is None):
            output = "Terminal data shape: {}".format(self.data.shape)
        else:
            output = "Split col: {}, val: {}, llr: {}".format(self.split_col,
                                                              self.split_val,
                                                              self.llr)

        return output


class RDDTree:
    def __init__(self, df, max_depth, min_size, threshold=None):
        self.df = df
        self.max_depth = max_depth
        self.min_size = min_size
        # TODO initialize threshold if one isn't provivded
        self.threshold = threshold

        self.root = None

    def _create_split(self, df, col, value):
        """
        Creates a split of the given DataFrame on the given value and column.

        Args:
            df (pd.DataFrame): the data to split
            col (str): the column to split on
            value (float): the value within the column to split on

        Returns:
            np.array: a binary array representing group assignments (the Gs)
        """
        assert col in df

        Gs = (df[col] > value).astype(int)
        return Gs.values

    def _get_split(self, df, search="exhaustive"):
        """
        Returns a candidate split of the given DataFrame, by searching over
        all data points over all columns.

        TODO implement random search case

        Args:
            df (pd.DataFrame): the data to search over, assumes presence of Ts
                               and Ps column
            search (str): the type of search to perform, currently only
                          "exhaustive" is supported

        Returns:
            Node with split_col, split_val, llr, and group populated
        """

        # TODO implement random search
        if search == "random":
            raise NotImplementedError

        if search == "exhaustive":
            exclude_cols = ['Ts', 'Ps', 'Gs']
            sel_df = df.drop(exclude_cols, axis=1, errors='ignore')
            best_col, best_val = None, None
            best_llr, best_group = -np.inf, None

            for col in sel_df.columns:
                print("candidate column: {}".format(col))
                # TODO hope to reduce the number of candidate values
                candidate_vals = sel_df[col].round(decimals=3)
                candidate_vals = sorted(candidate_vals.unique())
                for val in candidate_vals:
                    Gs = self._create_split(sel_df, col, val)
                    Ps = df['Ps']
                    Ts = df['Ts']
                    assert Ps.shape[0] == Gs.shape[0]

                    llr = compute_llr(Ps, Ts, Gs)
                    if llr > self.threshold and llr > best_llr:
                        print("Split col: {}. val: {}, llr: {}".format(col, val, llr))
                        best_col, best_val = col, val
                        best_llr, best_group = llr, Gs

        return Node(split_col=best_col,
                    split_val=best_val,
                    llr=best_llr,
                    group=best_group)

    def _is_terminal(self, df):
        """
        Checks whether a df is "pure," namely whether all the treatment
        assignments are the same.

        Args:
            df (pd.DataFrame)

        Returns:
            True if the given dataframe is "pure", False otherwise
        """
        return (df['Ts'].values[0] == df['Ts']).all()

    def _process_terminal(self, node, df):
        """
        Populates the given node with proper terminal information

        Args:
            node (dict): dict representation of a node
            df (pd.DataFrame): the selected data for this node

        Returns:
            None
        """
        node.left = None
        node.right = None
        node.data = df

    def _split_node(self, node, df, depth):
        """
        Splits a node, populating both 'left' and 'right' if possible.

        Args:
            node (dict): the dictionary representation of a node
            df (pd.DataFrame): the selected dataframe to split over
            depth (int): the current depth of the algorithm
        Returns
            None
        """
        print("split at level {}".format(depth))
        # check if if no good split was found or node is pure
        if (node.group is None) or self._is_terminal(df):
            self._process_terminal(node, df)
            return

        groups = node.group
        left_group = df[groups == 0].reset_index(drop=True)
        right_group = df[groups == 1].reset_index(drop=True)

        assert (right_group.shape[0] + left_group.shape[0]) == df.shape[0]

        left_node = Node()
        right_node = Node()

        if depth >= self.max_depth:
            self._process_terminal(left_node, left_group)
            self._process_terminal(right_node, right_group)
            node.left = left_node
            node.right = right_node
            return

        if left_group.shape[0] > self.min_size:
            left_node = self._get_split(left_group)
            node.left = left_node
            self._split_node(left_node, left_group, depth+1)
        else:
            self._process_terminal(left_node, left_group)
            node.left = left_node

        if right_group.shape[0] > self.min_size:
            right_node = self._get_split(right_group)
            node.right = right_node
            self._split_node(right_node, right_group, depth+1)
        else:
            self._process_terminal(right_node, right_group)
            node.right = right_node

    def build_tree(self):
        """
        Builds a tree with nodes as dicts.
        """
        self.root = self._get_split(self.df)
        self._split_node(self.root, self.df, 0)
