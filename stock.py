import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from datetime import date
from nsepy import get_history
infy = get_history(symbol='INFY',
                   start=date(2015,1,1),
                   end=date(2016,1,1))

tcs = get_history(symbol='TCS',
                   start=date(2015,1,1),
                   end=date(2016,1,1))

nifty_it = get_history(symbol="NIFTY IT",
                            start=date(2015,1,1),
                            end=date(2016,1,1),
                            index=True)
#to View column names
infy.columns.tolist()

#Keeping only OCHLV  
infy=infy[['Open','Close', 'High','Low','Volume']]
nifty_it=nifty_it[['Open','Close', 'High','Low','Volume']]
tcs=tcs[['Open','Close', 'High','Low','Volume']]

infy.index=pd.to_datetime(infy.index)
tcs.index=pd.to_datetime(tcs.index)
nifty_it.index=pd.to_datetime(nifty_it.index)

#Calculating Moving Averages
def moving_average(df):
    df['4week']=df['Close'].rolling(window=28).mean()
    df['16week']=df['Close'].rolling(window=112).mean()
    df['32week']=df['Close'].rolling(window=224).mean()
    df['54week']=df['Close'].rolling(window=378).mean()

moving_average(infy)
moving_average(tcs)
moving_average(nifty_it)



def plotMovingAverage(df):
    plt.title("Moving average")
    plt.plot(df['Close'],label="Original")
    plt.plot(df['4week'], label="4 Week Rolling mean trend")
    plt.plot(df['16week'], label="16 week Rolling mean trend")
    plt.plot(df['32week'], label="32 week Rolling mean trend")
    plt.plot(df['54week'], label="54 week Rolling mean trend")
    plt.legend(loc='best')
    plt.show()


plotMovingAverage(infy)
plotMovingAverage(tcs)
plotMovingAverage(nifty_it)


def change_dir(n):
    if n>0.1 :
        return 0
    if n<-0.1:
        return 1
    
    
def change(n):
    if n>0.1 or n<-0.1:
        return 1
    else:
        return 0

def change_dir_price(n):
    if n>0.02 :
        return 0
    if n<-0.02:
        return 1
    
    
def change_price(n):
    if n>0.02 or n<-0.02:
        return 1
    else:
        return 0
    
def volume_shock(df):
    df['vol_change'] = df['Volume'].pct_change(periods=1)
    df['vol_shock']=df['vol_change'].map(change)
    df['vol_shock_dir']=df['vol_change'].map(change_dir)

volume_shock(infy)
volume_shock(tcs)
volume_shock(nifty_it)


def price_shock(df):
    df['price_change'] = df['Close'].pct_change(periods=1)
    df['price_shock']=df['price_change'].map(change_price)
    df['price_shock_dir']=df['price_change'].map(change_dir_price)

price_shock(infy)
price_shock(tcs)
price_shock(nifty_it)

##Price Shock


def price_shock_without_vol(df):
    df['price_vol_shock']= np.where((df['price_shock'] == 1) & (df['vol_shock'] ==0)
                     , 1, 0)
price_shock_without_vol(infy)
price_shock_without_vol(tcs)
price_shock_without_vol(nifty_it)



#Part 2
#Visualizations
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral11, colorblind, Inferno, BuGn, brewer
from bokeh.models import HoverTool, value, LabelSet, Legend, ColumnDataSource,LinearColorMapper,BasicTicker, PrintfTickFormatter, ColorBar

def timeseries_plot(df):
    TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
    p = figure(title="Closing Prices 2015-2016", x_axis_type='datetime',y_axis_type="linear", plot_height = 400,
               tools = TOOLS, plot_width = 800)
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Closing Price'
    p.line(df.index,df['Close'],line_color="blue", line_width = 3)
    
    p.select_one(HoverTool).tooltips = [
            ('Date', '@x'),
            ('Closing Price', '@y'),
            ]
    
    new=np.where((df['price_shock'] == 1) & (df['vol_shock'] ==0),df['Close'],None)
    p.circle(df.index, new, legend="Price shock without volume shock", fill_color="orange", size=8)

    output_file("line_chart.html", title="Line Chart")
    show(p)



timeseries_plot(infy)
timeseries_plot(tcs)
timeseries_plot(nifty_it)

from statsmodels.graphics.tsaplots import plot_pacf

#partial autocorrelation plot
def pacf(df):
    plot_pacf(df['Close'])
    plt.show()
    
    
pacf(infy)
pacf(tcs)
pacf(nifty_it)


#Part 3

def test_stationarity(timeseries):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value1 in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value1
    print(dfoutput)



infy['4week'].dropna(inplace=True)
test_stationarity(infy['4week'])


tcs['Close'].dropna(inplace=True)
test_stationarity(tcs['Close'])

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def linear_regression(df,df1):
    lm=LinearRegression()
    X=df.iloc[:,0:4].copy()
    X=X.loc[:, X.columns != 'Close'].copy()
    y=df.loc[:,'Close'].copy()
    

    X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2018)

    lm.fit(X_train,y_train)

    y_pred=lm.predict(X_test)
    print("Training split MAPE:")
    print(mean_absolute_percentage_error(y_test,y_pred))

    df1=df1[['Open','Close', 'High','Low','Volume']]

    X_test1=df1.iloc[:,0:4].copy()
    X_test1=X_test1.loc[:, X.columns != 'Close'].copy()
    y_test1=df1.loc[:,'Close'].copy()
    
    new_pred=lm.predict(X_test1)
    print("Test MAPE:")
    print(mean_absolute_percentage_error(y_test1,new_pred))

infy_test = get_history(symbol='INFY',
                   start=date(2018,1,1),
                   end=date(2018,8,28))

tcs_test = get_history(symbol='TCS',
                   start=date(2018,1,1),
                   end=date(2018,8,28))


linear_regression(infy,infy_test)
linear_regression(tcs,tcs_test)


#Lasso Linear Regression
from sklearn import linear_model
def lasso_regression(df,df1):
    #reg = linear_model.LassoLars(alpha=0.01)
    X=df.iloc[:,0:4].copy()
    X=X.loc[:, X.columns != 'Close'].copy()
    y=df.loc[:,'Close'].copy()
    

    X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2018)

    llm = linear_model.LassoLars(alpha=0.1)
    llm.fit(X_train, y_train)
    predict_y = llm.predict(X_test)
    print("Training split MAPE:")
    print(mean_absolute_percentage_error(y_test,predict_y))

    df1=df1[['Open','Close', 'High','Low','Volume']]

    X_test1=df1.iloc[:,0:4].copy()
    X_test1=X_test1.loc[:, X.columns != 'Close'].copy()
    y_test1=df1.loc[:,'Close'].copy()
    
    new_pred=llm.predict(X_test1)
    print("Test MAPE:")
    print(mean_absolute_percentage_error(y_test1,new_pred))
    
print("INFY:")    
lasso_regression(infy,infy_test)
print("TCS")
lasso_regression(tcs,tcs_test)
