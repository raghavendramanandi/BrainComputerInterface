from pandas import read_csv
from pandas import get_dummies
from pandas.compat import to_str
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#pip install pandas
#pip install sklearn -> pip install scipy
#pip install matplotlib

def getNormalizedColumn(df, columnName):
    df[columnName] = df[columnName].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    return df[columnName]



df = read_csv('/Users/rmramesh/A/Volume-Forecast/venv/resources/WA_Sales_Products_2012-14.csv',
              names=['Retailer_country', 'Order_method_type', 'Retailer_type', 'Product_line', 'Product_type',
                     'Product', 'Year', 'Quarter', 'Revenue', 'Quantity', 'Gross_margin'])
revenue = 'Revenue'

df = df.dropna()

df = df[(df.T !=0).any()]

#Select only US data
df = df[(df.Retailer_country) == 'United States']

#Select relavent columns
df = df[['Order_method_type', 'Retailer_type', 'Product_type', 'Quarter', 'Revenue', 'Quantity', 'Gross_margin']]

#Normalize data
revenue_min = df.Revenue.min()
revenue_max = df.Revenue.max()

quantity_min = df.Quantity.min()
quantity_max = df.Quantity.max()

Gross_margin_min = df.Gross_margin.min()
Gross_margin_max = df.Gross_margin.max()

df.Revenue = df.Revenue.apply(lambda x: ((x - revenue_min) / (revenue_max - revenue_min)))
df.Quantity = df.Quantity.apply(lambda x: ((x - quantity_min) / (quantity_max - quantity_min)))
df.Gross_margin = df.Gross_margin.apply(lambda x: ((x - Gross_margin_min) / (Gross_margin_max - Gross_margin_min)))

#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(df['Revenue'])

#verify data

result_df = df.groupby(['Order_method_type', 'Retailer_type', 'Product_type', 'Quarter']).sum()
result_df = result_df.reset_index()


#print(result_df.count())
#print(result_df.groupby(['Order_method_type', 'Retailer_type', 'Product_type', 'Quarter']).count())

#Convert categorical data to Numeric information

cols_to_transform = ['Order_method_type', 'Retailer_type', 'Product_type', 'Quarter']
df_with_dummies = get_dummies(result_df, columns=cols_to_transform)

df = df_with_dummies
#print(df_with_dummies.count())
#print(df_with_dummies.sample(10))
#print(df_with_dummies.info())

#Independent variable
y = df[['Quantity']]

#print(list(df))
x= df[['Revenue', 'Gross_margin', 'Order_method_type_E-mail', 'Order_method_type_Fax', 'Order_method_type_Mail', 'Order_method_type_Sales visit', 'Order_method_type_Special', 'Order_method_type_Telephone', 'Order_method_type_Web', 'Retailer_type_Department Store', 'Retailer_type_Direct Marketing', 'Retailer_type_Equipment Rental Store', 'Retailer_type_Eyewear Store', 'Retailer_type_Golf Shop', 'Retailer_type_Outdoors Shop', 'Retailer_type_Sports Store', 'Retailer_type_Warehouse Store', 'Product_type_Binoculars', 'Product_type_Climbing Accessories', 'Product_type_Cooking Gear', 'Product_type_Eyewear', 'Product_type_First Aid', 'Product_type_Golf Accessories', 'Product_type_Insect Repellents', 'Product_type_Irons', 'Product_type_Knives', 'Product_type_Lanterns', 'Product_type_Navigation', 'Product_type_Packs', 'Product_type_Putters', 'Product_type_Rope', 'Product_type_Safety', 'Product_type_Sleeping Bags', 'Product_type_Sunscreen', 'Product_type_Tents', 'Product_type_Tools', 'Product_type_Watches', 'Product_type_Woods', 'Quarter_Q1 2012', 'Quarter_Q1 2013', 'Quarter_Q1 2014', 'Quarter_Q2 2012', 'Quarter_Q2 2013', 'Quarter_Q2 2014', 'Quarter_Q3 2012', 'Quarter_Q3 2013', 'Quarter_Q3 2014', 'Quarter_Q4 2012', 'Quarter_Q4 2013']]

#print(x.sample(10))

#Split data into test and learn data
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.30)


#Fit the data in model
model = LinearRegression()
model.fit(x_train, y_train)

predictions=model.predict(x_test)


#print(predictions.sample(10))


print(model.coef_)
print("============================")
print(model.intercept_)
print("============================")
print(model.get_params())
print("============================")
print("rmse: ", mean_squared_error(y_test, predictions))

# Plot outputs

#plt.plot(predictions, color='blue', linewidth=1)
#plt.plot(y_test, color='blue')
#plt.show()


count =0
for val in predictions:
    print(val)

print("==============================================================")
print(y_test)