# This module exists only to allow Sphinx to generate docs
# for the result objects returned by some functions in stats.

"""
Result classes
--------------

.. currentmodule:: scipy.stats._result_classes

.. autosummary::
   :toctree: generated/

   RelativeRiskResult
   BinomTestResult
   TukeyHSDResult
   FitResult

"""

__all__ = ['BinomTestResult', 'RelativeRiskResult', 'TukeyHSDResult',
           'FitResult']


from ._binomtest import BinomTestResult
from ._relative_risk import RelativeRiskResult
from ._hypotests import TukeyHSDResult
from ._fit import FitResult
