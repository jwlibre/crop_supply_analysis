import matplotlib.pyplot as plt
import pandas as pd
#import Tkinter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pdb

# maize dataset
data_path = '/home/user/projects/food_security/maize_data.csv'

types = [('metsite', 'U50'), ('year', 'i'), ('N', 'f'), ('DE', 'f'), ('H', 'f'),
        ('S', 'U6'), ('CU', 'U50'), ('pawc', 'f'), ('ylds', 'f'),
         ('Junemaxt', 'f'), ('Junemint', 'f'), ('Juneradn', 'f'),
          ('Junerain', 'f'), ('Junevpd', 'f'), ('Julymaxt', 'f'),
           ('Julymint', 'f'), ('Julyradn', 'f'), ('Julyrain', 'f'),
            ('Julyvpd', 'f'), ('Augmaxt', 'f'), ('Augmint', 'f'),
             ('Augradn', 'f'), ('Augrain', 'f'), ('Augvpd', 'f'),
              ('JJAmaxt', 'f'), ('JJAmint', 'f'), ('JJAradn', 'f'),
               ('JJArain', 'f'), ('JJAvpd', 'f')]

# maize = np.genfromtxt(data_path, dtype=types, delimiter=',', names=True)
# turns out it's easier to manipulate in pandas first!
maize = pd.read_csv(data_path, header=0)

# find all the unique years, this will determine our training / test sets
years = list(set(maize['year']))

# "TASK: Using only simulations for 1980-2010 for training,
# predict the yields for a field for 2011-2015 using only weather information."

# DATA PREPROCESSING

# I will use 1980-2005 as training, 2006-2010 as CV, and 2011-2015 as test
training_start = min(years) # 1980
training_end = 2005
cv_start = 2006
cv_end = 2010
test_start = 2011
test_end = max(years)

weather_headers = ['Junemaxt', 'Junemint', 'Juneradn', 'Junerain', 'Junevpd',
                   'Julymaxt', 'Julymint', 'Julyradn', 'Julyrain', 'Julyvpd',
                   'Augmaxt', 'Augmint', 'Augradn', 'Augrain', 'Augvpd']

# EXTENSION TASK ########
# Use feature selection to decide which weather variables to use

################

# Mean normalisation
# Feature extraction


maize_training = maize.loc[maize['year'] <= training_end]
maize_cv = maize.loc[(maize['year'] >= cv_start) & (maize['year'] <= cv_end)]
maize_test = maize.loc[maize['year'] >= test_start]

maize_training = maize_training[weather_headers + ['ylds']]
maize_test = maize_test[weather_headers + ['ylds']]

# TRAINING
X_train = np.array(maize_training[weather_headers])
y_train = np.array(maize_training['ylds'])

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# CROSS-VALIDATION
X_cv = np.array(maize_cv[weather_headers])
hyp_cv = regressor.predict(X_cv)

y_cv = np.array(maize_cv['ylds'])

# cross-validation: performance evaluation
msqerr_cv = mean_squared_error(y_cv, hyp_cv)
r2_cv = r2_score(y_cv, hyp_cv)

pdb.set_trace()

# Establish whether the algorithm is high-bias or high-variance
# does LinearRegression output J values?
# Tweak learning parameter?




# TEST
# run the trained model on test data
X_test = maize_test[weather_headers]
hyp = regressor.predict(X_test)

# actual values
y_test = maize_test['ylds']

msqerr = mean_squared_error(y_test, hyp)
r2 = r2_score(y_test, hyp)

# basic plots
# fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot(maize['H'],maize['Junerain'])
pdb.set_trace()
