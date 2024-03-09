import datetime
import numpy as np
import pandas as pd
import os
import sys
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import unittest

sys.path.append('c:\\Users\\Emile\\Documents\\lifecycle_investing')

import lc_investing.birthday_rule
import lc_investing.constant_percent_stock
import lc_investing.lifecycle_strategy
from lc_investing.backtest import Backtest

class TestComputation(unittest.TestCase):
    def setUp(self):
        self.data_folder = 'c:/Users/Emile/Documents/lifecycle_investing/lc_investing/data/'

    def test_birthday_rule(self):
        """birthday rule"""
        s1 = lc_investing.birthday_rule.Simulation(data_folder=self.data_folder)
        s1.retirement_savings_before_period.head(2)
        s1.retirement_savings_before_period.Final.describe()
        self.assertEqual(1, 1, 'should be equal')
    
    def test_constant_percent(self):
        """constant percent stock"""
        # replicate the constant percent stock backtest
        lc_investing.constant_percent_stock = reload(lc_investing.constant_percent_stock)
        s2 = lc_investing.constant_percent_stock.Simulation(data_folder=self.data_folder)
        s2.retirement_savings_before_period.Final.describe()

    def test_lifecycle_strategy(self):
        """lifecycle strategy"""
        # replicate the lifecycle strategy backtest
        s3 = lc_investing.lifecycle_strategy.Simulation(data_folder=self.data_folder)
        s3.calc_retirement_savings_before_period()
        s3.retirement_savings_before_period.FINAL.describe()

    def test_backtest_1(self):
        """"""
        b = Backtest()
        self.assertEqual(b.get_data(), 1, 'should be equal')

if __name__ == "__main__":
    unittest.main(module="tests_1")