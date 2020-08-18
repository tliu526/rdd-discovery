import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sim import gen_fuzzy_rdd
from tree import RDDTree

class RDDTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 1000
        df = gen_fuzzy_rdd(n, 0.8, 0.8, 0.2, seed=0)
        
        lr = LogisticRegression(random_state=0)
        X = df['x'].values.reshape(-1, 1)
        lr.fit(X, df['t'])
        all_Ps = lr.predict_proba(X)[:, 1]
        all_Ts = df['t']

        df['Ts'] = all_Ts
        df['Ps'] = all_Ps

        cls.oneD_df = df


    def test_create_split(self):
        df = pd.DataFrame.from_dict(dict(col=[-1,-2,1,3]))
        tree = RDDTree(df, 1,2,1)
        gs = tree._create_split(df, 'col', 0.5)

        self.assertEqual(gs.tolist(), [0,0,1,1])
    

    def test_get_split(self):

        # 1D case
        d = {
            "Ps": [0.1,0.2,0.3,0.9,0.95,1],
            "X": [1,2,3,5,6,7],
            "Ts": [0,0,0,1,1,1]
        }    

        df = pd.DataFrame.from_dict(d)
        tree = RDDTree(df, 1,2,-np.inf)
        node = tree._get_split(df)

        self.assertEqual(node['col'] , 'X')
        self.assertListEqual(node['group'].tolist(), [0,0,0,1,1,1])
        

    def test_is_terminal(self):
        true_df = pd.DataFrame.from_dict(dict(Ts=[1,1,1,1]))
        false_df = pd.DataFrame.from_dict(dict(Ts=[1,1,0,1]))
        tree = RDDTree(None, 1,2,1)
        
        self.assertTrue(tree._is_terminal(true_df))
        self.assertFalse(tree._is_terminal(false_df))


    def check_is_terminal(self, node, df):
        self.assertEqual(node['left'], None)
        self.assertEqual(node['right'], None)
        self.assertTrue(node['data'].equals(df))


    def test_process_terminal(self):
        tree = RDDTree(None, 1,2,1)
        df = pd.DataFrame()
        node = {'group': None}
        tree._process_terminal(node, df)
        self.check_is_terminal(node, df)


    def test_split_node_terminal_cases(self):
        tree = RDDTree(None, 2,2,2)

        empty_df = pd.DataFrame()
        empty_node = {'group': None}

        tree._split_node(empty_node, empty_df, 1)
        self.check_is_terminal(empty_node, empty_df)

        pure_df = pd.DataFrame.from_dict(dict(Ts=[1,1,1,1]))
        tree._split_node(empty_node, pure_df, 1)
        self.check_is_terminal(empty_node, pure_df)


    def test_split_node_1D_cases(self):
        tree = RDDTree(None, 2,2,2)

        root_dict = {
            "Ps": [0.1,0.2,0.3,0.9,0.95,1],
            "X": [1,2,3,5,6,7],
            "Ts": [0,0,0,1,1,1]
        }   
        
 
        left_dict = {
            "Ps": [0.1,0.2,0.3],
            "X": [1,2,3,],
            "Ts": [0,0,0]
        }

        right_dict = {
            "Ps": [0.9,0.95,1],
            "X": [5,6,7],
            "Ts": [1,1,1]
        }   

        root_df = pd.DataFrame.from_dict(root_dict)
        left_df = pd.DataFrame.from_dict(left_dict)
        right_df = pd.DataFrame.from_dict(right_dict)

        # terminal group
        node = {'group': np.array([0,0,0,1,1,1])}
        tree._split_node(node, root_df, 1)

        left_node = node['left']
        right_node = node['right']

        self.check_is_terminal(left_node, left_df)
        self.check_is_terminal(right_node, right_df)





if __name__ == '__main__':
    unittest.main()


