# sklearn offers many useful transformations, however, own task tranformations for custom cleanup operations and combining specific attributes could be created.
# To do so, the transformer needs to work seamlessly with sklearn functionalities (such as pipelines), and since sklearn relies on duck typing (not inheritence),
# all that is required is to create a class and implement three methods: fit() [returning self], transform(), and fit_transform().

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# BaseEstimator provides two methods (get_params() and set_params() for automatic hyperparameter tuning)
# TransformerMixin provides fit_transform()

housing = pd.read_csv("C:/Users/dushy/datasets/housing/housing.csv")

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs constructors
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombineAttributesAdder(add_bedrooms_per_room = False)
housing_per_attribs = attr_adder.transform(housing.values)

print(housing_per_attribs.describe())