import matplotlib.pyplot as plt
import Tkinter
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pdb

# maize dataset
data_path = '/home/user/projects/food_security/maize_data.csv'

# "metsite","year","N","DE","H","S","CU","pawc","ylds","Junemaxt","Junemint","Juneradn","Junerain","Junevpd","Julymaxt","Julymint","Julyradn","Julyrain","Julyvpd","Augmaxt","Augmint","Augradn","Augrain","Augvpd","JJAmaxt","JJAmint","JJAradn","JJArain","JJAvpd"
# "newton_long",1980,200,4.5,0.6,"24-apr","Pioneer_3394",150,1.5139,27.6833333333333,14.4033333333333,29.7795,3.83533333333333,1.57352776853867,32.3677419354839,18.5741935483871,28.5303870967742,2.13064516129032,2.07005261495749,30.5870967741935,18.0709677419355,24.5088064516129,4.87516129032258,1.76209615884348,30.2402173913043,17.0445652173913,27.5826086956522,3.61130434782609,1.80437440265207

types = [('metsite', 'U50'), ('year', 'i'), ('N', 'f'), ('DE', 'f'), ('H', 'f'),
        ('S', 'U6'), ('CU', 'U50'), ('pawc', 'f'), ('ylds', 'f'),
         ('Junemaxt', 'f'), ('Junemint', 'f'), ('Juneradn', 'f'),
          ('Junerain', 'f'), ('Junevpd', 'f'), ('Julymaxt', 'f'),
           ('Julymint', 'f'), ('Julyradn', 'f'), ('Julyrain', 'f'),
            ('Julyvpd', 'f'), ('Augmaxt', 'f'), ('Augmint', 'f'),
             ('Augradn', 'f'), ('Augrain', 'f'), ('Augvpd', 'f'),
              ('JJAmaxt', 'f'), ('JJAmint', 'f'), ('JJAradn', 'f'),
               ('JJArain', 'f'), ('JJAvpd', 'f')]

maize = np.genfromtxt(data_path, dtype=types, delimiter=',', names=True)

# basic plots
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(maize['H'],maize['Junerain'])
