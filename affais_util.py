import processRB as prb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def regression_plots_one_independent_variable(dataframe):
    dummy_df = dataframe.select_dtypes(exclude=['object'])
    dummy_df = dummy_df.drop(["ID"], axis=1)
    common_array_of_fields = np.array(dummy_df.columns).tolist()
    i = 0
    colors = ["", "teal", "lime", "blue", "coral", "c", "tomato", "darksalmon"]
    plt.figure(figsize=(25, 25)).suptitle('Scatter Plots for PT survey (attributes Vs target) with regression lines')
    plt.subplots_adjust(hspace=0.5)
    seaborn_regression_data_per_field = []
    for field in common_array_of_fields:
        field_min, field_max, field_range = control_x_range(field, dummy_df)
        copy = np.array(dummy_df.columns).tolist()
        copy.remove("affairs")
        i = i + 1
        if field == "affairs":
            i = 0
            continue
        else:
            field_regression_data = []
            field_regression_data.append(field)
            slope_intercept_for_field = []
            copy.remove(field)
            XY_term_PT = dummy_df.drop(copy, axis=1)
            plt.subplot(3, 3, i)
            # sns.regplot(field, "affairs", data=XY_term_PT, scatter_kws={"color": colors[i], "alpha": 0.05, "s": 200})
            s = sns.regplot(field, "affairs", data=XY_term_PT,
                            scatter_kws={"color": colors[i], "alpha": 0.05, "s": 200})
            xd = s.get_lines()[0].get_xdata()
            yd = s.get_lines()[0].get_ydata()
            slope1 = (yd[1] - yd[0]) / (xd[1] - xd[0])
            slope2 = (yd[60] - yd[59]) / (xd[60] - xd[59])
            slope_extreme = (yd[-1] - yd[0]) / (xd[-1] - xd[0])
            intercept1 = yd[0] - slope1 * xd[0]
            intercept2 = yd[0] - slope2 * xd[0]
            intercept_extreme = yd[0] - slope_extreme * xd[0]
            print("xdata for: ", field)
            print(s.get_lines()[0].get_xdata())
            print("ydata for: ", field)
            print("slope for field '", field, "'", " is : ", slope1)
            print("verify slope for field '", field, "'", " is : ", slope2)
            print("Extreme slope for field '", field, "'", " is : ", slope_extreme)
            print("intercept for field '", field, "'", " is : ", intercept1)
            print("verify intercept for field '", field, "'", " is : ", intercept1)
            print("Extreme intercept for field '", field, "'", " is : ", intercept_extreme)
            slope_intercept_for_field.append(slope_extreme)
            slope_intercept_for_field.append(intercept_extreme)
            field_regression_data.append(slope_intercept_for_field)
            seaborn_regression_data_per_field.append(field_regression_data)
            print(s.get_lines()[0].get_ydata())
            plt.xlim(field_min - 0.1 * field_range, field_max + 0.1 * field_range)
            plt.ylim(-5, 15)
    plt.show()
    return seaborn_regression_data_per_field


def control_x_range(field, dataframe):
    field_min = dataframe[field].min()
    field_max = dataframe[field].max()
    field_range = field_max - field_min
    return field_min, field_max, field_range


def describe_all(df):
    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))


def draw_correlation_heatmap(dataframe):
    plt.figure(figsize=(10, 10)).suptitle('Correlation matrix heatmap')
    plt.subplot(2, 1, 1)
    matrix = np.triu(dataframe.corr())
    sns.heatmap(dataframe.corr(), annot=True, mask=matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()


def perform_one_variable_regression_linear(dataframe, X_train, X_test, y_train, y_test):
    coefficients_one_variable = []
    intercepts_one_variable = []
    R2_one_variable = []
    for field in dataframe.columns:
        lr = LinearRegression()
        X_train_one_var = X_train[field].values.reshape(-1, 1)
        X_test_one_var = X_test[field].values.reshape(-1, 1)
        lr.fit(X_train_one_var, y_train)
        ep_predict_one = lr.predict(X_test_one_var)
        coefficients_one_variable.append(lr.coef_[0][0])
        intercepts_one_variable.append(lr.intercept_[0])
        R2_one_variable.append(r2_score(y_test, ep_predict_one))
        print("One variable study for predictor: ", field)
        print("==========================================")
        print("Coefficients: ", lr.coef_[0])
        print("Intercept: ", lr.intercept_)
        print("MSE: ", mean_squared_error(y_test, ep_predict_one))
        print("Root MSE: ", np.sqrt(mean_squared_error(y_test, ep_predict_one)))
        print("MEA: ", mean_absolute_error(y_test, ep_predict_one))
        print("R2 score: ", r2_score(y_test, ep_predict_one))
        print("VIF: ", 1 / (1 - r2_score(y_test, ep_predict_one)))
        print()
        print("PLOTS")
        print("::::::::::::::::")
        plt.scatter(X_test[field], y_test, color='black')
        x = np.array(X_test[field])
        y = x * lr.coef_[0] + lr.intercept_
        plt.plot(x, y, color='blue', linewidth=3)
        plt.show()