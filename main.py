# print("hi")

# print("x has been set to" +str(206))

# name = input("what is your name? ")
# print("the name you entered is " + name)

# import math

#
# print(type(math.pi))

# numbers = [10, 2]
# print(sum(numbers) / len(numbers))


# mylist = (True, 6 < 5, 1 == 3, None is None)
# # for elements in mylist:
# #     print(elements)
# print(str(mylist[0]))

# import numpy as np
# import pandas as pd
# a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# print(type(np.mean(a)))
# import pandas as pd

# import pandas as pd
# url = "YtT9NroBEemddAqBQMk_og_f682394f0a2542cea617efa59089ab2b_Cartwheeldata.csv"
# df = pd.read_csv(url)
# print(type(df))
# print(df.loc[2,"Gender"])
# print(df.iloc[3, :])


# import pandas as pa
# url = "YtPruboBEemdqA7UJJ_tgg_63e179e3722f4ef783f58ff6e395feb7_nhanes_2015_2016.csv"
# da = pa.read_csv(url)
# print(da.loc[:, "BMXARML"])
# print(da)
# print(type(da))
# print(da.iloc[3, :])

# name = input("whats your name?")
# if name == "amirhosein":
#     print(" the name you entered is a good guy")
# elif name == "fatemeh":
#     print(" the name you entered is a NERD")
# elif name == "mahla":
#     print(" the name you entered is a is an amazing artist")
# elif name == "navid":
#     print(" the name you entered is a is a aligator on a wheelchair" )


# from pandas import DataFrame
# from pandas.io.parsers import TextFileReader

# url = "YtPruboBEemdqA7UJJ_tgg_63e179e3722f4ef783f58ff6e395feb7_nhanes_2015_2016.csv"
#
# import pandas as pa
#
# df = pa.read_csv(url)

# result = []
#
# for x in range(3):
#     for y in range(5):
#         if x * y > 30:
#             result.append((x, y))
#
# print(result)

# for x in range(7):
#     print(df)

##################################################
# import numpy as np

# a = np.array([7, 8, 9])

# print(type(a))
# print(a.shape)
# print(a[0], a[2])

# b = np.array([[3, 4], [6, 3], [7, 9]])
# print(type(b))
# print(b.shape)
# print(b[0,1], b[1,1], b[2,0])

### 2x7 zero array
# d = np.zeros((2, 7))
# print(d)

### 2x6 ones array
# e = np.ones((2, 6))
# print(e)

### 3x2 constant array
# f = np.full((3,2), 8)
# print(f)

### 3x3 random array
# g = np.random.random((3, 3))
# print(g)

### Create 3x4 array
# z = np.array([1,2,3])
# h = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# i = h[0:2, 1:3]
# i[0, 0] = 206
# print(i)
# print(h)
# i[0, 0] = 1738
# i = h[2:, 3:]

# print(i)

### Integer
# j = np.array([1, 2])
# print(j.dtype)


### Float
# k = np.array([1.0, 2.0])
# print(k.dtype)

### Force Data Type
# l = np.array([1.0, 2.0], dtype=np.int64)
# print(l.dtype)

# x = np.array([[1], [2]], dtype=np.float64)
# y = np.array([[3], [4]], dtype=np.float64)
# print(x + y)
# print(x)

# x = np.array([[1,2],[3,4]], dtype=np.float64)
# y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
# print(x + y)
# print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
# print(x - y)
# print(np.subtract(x, y))


# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
# print(np.sqrt(x))

# x = np.array([[1, 2],[3, 4]])

### Compute sum of all elements; prints "10"
# print(np.sum(x))

### Compute sum of each column; prints "[4 6]"
# print(np.sum(x, axis=0))
### Compute sum of each row; prints "[3 7]"
# print(np.sum(x, axis=1))
### Compute mean of all elements; prints "2.5"
# print(np.mean(x))
### Compute mean of each column; prints "[2 3]"
# print(np.mean(x, axis=0))
### Compute mean of each row; prints "[1.5 3.5]"
# print(np.mean(x, axis=1))

from scipy import stats
import numpy as np

# print(stats.norm.rvs(size = 10))

# from pylab import *

# crate some test data
dx = .5
X = np.arange(-2, 2, dx)
Y = np.exp(-X ** 2)
# print(Y)
#
# # normalize the data to proper PDF
Y /= (dx * Y).sum()
# print(Y)
#
# # compute the CDF
CY = np.cumsum(Y * dx)
# print(CY)

############################################################
# # plot both
# import matplotlib
# matplotlib.use('Agg')  # Specify the 'Agg' backend
# import matplotlib.pyplot as plt

# Rest of your plotting code
# plt.plot(X, Y)
# plt.plot(X, CY, 'r--')

# Save the plot as an image file (e.g., PNG)
# plt.savefig('my_plot.png')

# Don't call plt.show() since the 'Agg' backend is non-interactive

#################################################################

# # compute the normal CDF of certain values
# print(stats.norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6])))

# # descriptive statistic
# np.random.seed(282629734)

# #generate 1000 student's T contunes random variable
# x = stats.t.rvs(10, size=1000)
# print(x)

# Do some descriptive statistics
# print(x.min())
# print(x.max())
# print(x.mean())
# print(x.var())

#######################################3

# Compute the x and y coordinates for points on a sine curve

import matplotlib

matplotlib.use('TkAgg')  # Or 'Agg' or another compatible backend
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x)
# y = np.linspace(3, 9, len(x))

# std_dev = 0.2  ## You should specify your desired standard deviation

## Create a shaded region to represent standard deviation

# plt.fill_between(x, y - std_dev, y + std_dev, alpha=0.5, label='Standard Deviation')

## plot the points using matplotlop

# tip: 'x' This is the array of x-coordinates that you want to plot on the horizontal axis (usually the independent variable).
# 'y' This is the array of y-coordinates that you want to plot on the vertical axis (usually the dependent variable).

# plt.plot(x, y)
# plt.show()

### Compute the x and y coordinates for points on sine and cosine curves
# def test( v):
#     return v ** 2 * np.tan(v)

# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# z = np.arange(-100, 100, 0.1)
# m = np.arange(-100, 100, 0.1)
# y = test(z)

## plot the points using matplotlip
# plt.subplot(2, 1, 1)
# plt.plot(z, y, color="blue")
# plt.title('test')
# plt.subplot(2, 1, 2)
# plt.plot(x, y_sin, color="red")
# plt.plot(x, y_cos, color="yellow")
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('sine and cosine')
# plt.legend(['sine', 'cosine'])
# ax = plt.gca()
# leg = ax.get_legend()
# leg.legendHandles[0].set_color('red')
# leg.legendHandles[1].set_color('yellow')
# plt.show()

### Subplots

## Compute the x and y coordinates for points on sine and cosine curves
# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.tan(x)
# y_cos = np.cos(x)

## Set up a subplot grid that has height 2 and width 1,
## and set the first such subplot as active.

# plt.subplot(2, 1, 1)

# The first argument 2 indicates that you want to create a grid of 2 rows of subplots.
# The second argument 1 indicates that you want to create a grid of 1 column of subplots.
# The third argument 1 indicates that you want to select the first subplot in this grid.

## Make the first plot
# plt.plot(x, y_sin)
# plt.title('sin')
# Make the second...
# plt.subplot(2, 1, 2)
# plt.plot(x, y_cos)
# plt.title('cosine')

# plt.show()

##########################################
#########        SEABORN      ############
##########################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

url = "YtT9NroBEemddAqBQMk_og_f682394f0a2542cea617efa59089ab2b_Cartwheeldata.csv"
df = pd.read_csv(url)
ff = np.array(df['Age'])
# print(ff)

# create scatterplots
# sns.lmplot(x="Wingspan", y="CWDistance", data=df)

# plt.show()

# Scatterplot arguments
# sns.lmplot(x='Wingspan', y='CWDistance', data=df,
#            fit_reg=False,  ### No regression line ###
#            hue='Gender')  ### Color by evolution stage ###

# plt.show()

## Construct Cartwheel distance plot
# sns.swarmplot(x="Gender", y="CWDistance", data=df)

# plt.show()

#######   BOXPLOTS   #######
# sns.boxplot(data=df.loc[:, ["Age","Height"]])
# plt.show()

## Male Boxplot
# sns.boxplot(data=df.loc[df['Gender'] == 'F', ["Age", "Height", "Wingspan", "CWDistance", "Score"]])
# plt.show()

######   HISTOGRAM   #######
## distribution plot (a.k.a Histogram)
# sns.distplot(df.CWDistance)
# plt.show()
# sns.histplot(df.Glasses)
# plt.xticks(rotation=-45)
# plt.show()
# sns.histplot(df["CWDistance"], kde=False)
# plt.show()

######   COUNT PLOT   ######
## Count Plot (a.k.a. Bar Plot)
# sns.countplot(x='Gender', data=df)
# plt.xticks(rotation=-45)
# plt.show()

###################################################################

## We first need to import the packages that we will be using
# import seaborn as sns
# import matplotlib.pyplot as plt

## Load in the data set
tips_data = sns.load_dataset("tips")
## Print out the first few rows of the data
# tips_data.head(df)
## Print out the summary statistics for the quantitative variables
# x = tips_data.describe()
# print(x)
## Plot a histogram of the total bill
# sns.distplot(tips_data["total_bill"], kde = False).set_title("Histogram of tota bill")
# plt.show()
## Plot a histogram of tips only
# sns.distplot(tips_data["tip"], kde = True).set_title("histogram of total tip")
# plt.show()

########     Creating a Boxplot      ########

## Create a boxplot of the total bill amounts
## sns.boxplot(tips_data["total_bill"]).set_title("box plot of the total bill")
# plt.show()
## Create a boxplot of the tips amounts
# sns.boxplot(tips_data["tip"]).set_title("box plot of the tip")
# plt.show()

######      Creating Histograms and Boxplots Plotted by Groups      ######

## Create a boxplot and histogram of the tips grouped by smoking status
# sns.boxplot(x = tips_data["tip"], y = tips_data["smoker"])
# plt.show()
## Create a boxplot and histogram of the tips grouped by time of day

# sns.boxplot(x = tips_data["tip"], y = tips_data["smoker"])

# g = sns.FacetGrid(tips_data, row = "time" )
# g = g.map(plt.hist, "tip")
# plt.show()

# Create a boxplot and histogram of the tips grouped by the day
# sns.boxplot(x = tips_data["tip"], y = tips_data["day"])
# g = sns.FacetGrid(tips_data, row = "day")
# g = g.map(plt.hist, "tip")
# plt.show()

#########      Univariate data analyses       #########
### First of All

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

da = pd.read_csv("../YtPruboBEemdqA7UJJ_tgg_63e179e3722f4ef783f58ff6e395feb7_nhanes_2015_2016.csv")

# a = da.DMDEDUC2.value_counts()
# print(a)

# print(da.DMDEDUC2.value_counts().sum())
# print(a.sum())
# print(da.shape)

# Another way to obtain this result is to locate all the null
# (missing) values in the data set using the [isnull]
# (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.isnull.html)
# Pandas function, and count the number of such locations.

# print(pd.isnull(da.DMDEDUC2).sum)

# In some cases it is useful to replace integer codes
# with a text label that reflects the code's meaning.
# Below we create a new variable called 'DMDEDUC2x' that is recoded with text labels,
# then we generate its frequency distribution.

# da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "<9", 2:"9-11", 3:"HS/GED",
#                                        4: "Some college/AA", 5:"College",
#                                        7:"refused", 9:"don't know"})
# print(da.DMDEDUC2x.value_counts())

### We will also want to have a relabeled version of the gender variable,
### so we will construct that now as well.

# da["RIAGENDRx"] = da.RIAGENDR.replace({1:"Male", 2:"Female"})
# print(da.RIAGENDRx.value_counts())

### For many purposes it is more relevant to consider the proportion
### of the sample with each of the possible category values,
### rather than the number of people in each category. We can do this as follows:

# x = da.DMDEDUC2x.value_counts()
# print(x / x.sum())

### In some cases we will want to treat the missing response category as another category of observed response,
### rather than ignoring it when creating summaries. Below we create a new category
### called "Missing", and assign all missing values to it usig fillna.
### Then we recalculate the frequency distribution. We see that 4.6% of the responses are missing.

# da["DMDEDUC2x"] = da.DMDEDUC2x.fillna("Missing")
# x = da.DMDEDUC2x.value_counts()
# print(x / x.sum())
# print(da.DMDEDUC2x.value_counts())

########     Numerical summaries      ########
###  use of deropna method for generating a summries except of missed data

# print(da.BMXWT.dropna().describe())

### It's also possible to calculate individual summary statistics
### from one column of a data set. This can be done using Pandas methods,
### or with numpy functions:

# x = da.BMXWT.dropna()
# print(x.mean())
# print(np.mean(x))
#
# print(x.median())
# print(np.percentile(x, 50)) # 50th percentile, same as the median7
#
# print(np.percentile(x, 75))
# print(x.quantile(0.75)) # Pandas method for quantiles, equivalent to 75th percentaile

### "&" means "and"

# print(np.mean((da.BPXSY1 >= 120) & (da.BPXSY2 <=139)))

### "|" means "or"

# a = (da.BPXSY1 >= 120) & (da.BPXSY2 <= 139)
# b = (da.BPXDI1 >= 80) & (da.BPXDI2 <= 89)
# print(np.mean(a | b))

# print(np.mean(da.BPXSY1 - da.BPXSY2))
# print(np.mean(da.BPXDI1 - da.BPXDI2))
#########################################
# print(da.BPXSY1[0], da.BPXSY2[0])
# print(np.mean(da.BPXSY1[0] - da.BPXSY2[0]))
# print(da.BPXSY1[1], da.BPXSY2[1])
# print(np.mean(da.BPXSY1))
# print(np.mean(da.BPXSY2))
# print(np.mean(da.BPXSY1) - np.mean(da.BPXSY2))
#########################################

######    Graphical summaries     ######

# sns.distplot(da.BMXWT.dropna())
# plt.show()

# sns.distplot(da.BPXSY1.dropna(), kde = False)
# plt.show()

#this branch was created by amirhoosein