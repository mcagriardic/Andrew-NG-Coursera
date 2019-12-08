import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


EPSILON = 10e-10 # to silence the RuntimeWarning: divide by zero encountered errors


def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


def calculate_model_performance(actual, predicted):
    TP = 0; TN = 0; FP = 0; FN = 0

    for act, pred in zip(actual, predicted):
        if act == 1:
            if act == pred:    TP += 1
            else:              FN += 1
        else:
            if act == pred:    TN += 1
            else:              FP += 1

    def specificity():
        return TN / (TN + FP + EPSILON) * 100

    def sensitivity_recall():
        return TP / (TP + FN + EPSILON) * 100

    def accuracy():
        return (TP + TN) / (TP + TN + FP + FN + EPSILON) * 100

    def prevalence():
        return (TP + FN) / (TP + TN + FP + FN + EPSILON) * 100

    def precision():
        return TP / (TP + FP + EPSILON) * 100
    
    def false_positive():
        return FP / (FP + TN + EPSILON) * 100
    
    def F1():
        return (2 * (
                (precision() * sensitivity_recall())
                / (precision() + sensitivity_recall() + EPSILON)
            ))

    model_metrics = {
        "specificty": specificity(),
        "sensitivity/recall": sensitivity_recall(),
        "accuracy": accuracy(),
        "prevalence": prevalence(),
        "precision": precision(),
        "F1": F1(),
        "false_positive_rate": false_positive()
    }
    return model_metrics


def calculate_TP_FP(actual, func_to_predict, threshold_spacing):

    true_positive_rate = []
    false_positive_rate = []
    for threshold in np.linspace(0, 1, threshold_spacing):
        model_metrics = calculate_model_performance(actual, func_to_predict(threshold))
        true_positive_rate.append(model_metrics["sensitivity/recall"])
        false_positive_rate.append(model_metrics["false_positive_rate"])

    return true_positive_rate, false_positive_rate


# **************************************** PLOTTING *******************************************************************
def plot_data(
    ax,
    data1,
    data2,
    param_dict
):

    ax.scatter(data1, data2, **param_dict)


def get_decision_boundary(
    X,
    thetas,
    is_polynomial=False,
    PolynomialFeatures_instance=None
):
    thetas = thetas.reshape(-1, 1)

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    if is_polynomial:
        h = sigmoid(PolynomialFeatures_instance.fit_transform(np.c_[xx1.flatten(), xx2.flatten()]) @ (thetas))

    else:
        constant = np.ones((xx1.flatten().shape[0], 1))
        h = sigmoid(np.c_[constant, xx1.flatten(), xx2.flatten()] @ (thetas))

    h = h.reshape(xx1.shape)

    return xx1, xx2, h


def plot_lambdas_grid(
    data,
    X,
    y,
    X_poly,
    PolynomialFeatures_instance,
    grid_rows,
    grid_columns,
    LAMBDA_start,
    LAMBDA_end,
    cost_func,
    gradient
):

    n = X_poly.shape[1]              # n: no of features

    fig, ax = plt.subplots(grid_rows, grid_columns, figsize=(15, 5 * grid_rows))
    ax = ax.reshape((grid_rows, grid_columns))

    LAMBDA_vals = np.linspace(LAMBDA_start, LAMBDA_end, grid_rows * grid_columns)

    negatives = data[data[:, -1] == 0]
    positives = data[data[:, -1] == 1]

    thetas = np.zeros(n).reshape(-1, 1)

    i = 0
    for row in range(grid_rows):
        for column in range(grid_columns):

            res = minimize(
                    fun=cost_func,
                    x0=thetas,
                    args=(LAMBDA_vals[i], X_poly, y),
                    method=None,
                    jac=gradient,
                    options={'maxiter':3000}
            )

            ax[row, column].set_xlabel("Microchip test 1")
            ax[row, column].set_ylabel("Microchip test 2")
            plot_data(ax[row, column], negatives[:, 0], negatives[:, 1], param_dict={"c": "black", "marker": "x", "label": "not admitted"})
            plot_data(ax[row, column], positives[:, 0], positives[:, 1], param_dict={"c": "y", "marker": "d", "label": "admitted"})

            xx1, xx2, h = get_decision_boundary(
                            X=X,
                            thetas=res.x,
                            is_polynomial=True,
                            PolynomialFeatures_instance=PolynomialFeatures_instance
                        )

            ax[row, column].contour(xx1, xx2, h, [0.5], linewidths=2, colors="blue")
            ax[row, column].set_title('Lambda = %s' %LAMBDA_vals[i])
            i += 1

def plot_ROC(
    actual,
    func_to_predict,
    threshold_spacing=50,
):
    true_positive_rate, false_positive_rate = calculate_TP_FP(actual, func_to_predict, threshold_spacing)
            
    #Here we multiply with -1 because recall is in decreasing order and therefore,
    #np.trapz returns a negative value. However, taking the integral of an equation
    #should return us the area under a curve which cannot be negative.
    AUC = -1 * np.trapz(y=true_positive_rate, x=false_positive_rate) 
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(false_positive_rate, true_positive_rate)
    ax.plot([0, 100], [0, 100])
    ax.set_xlabel("False Positive Rate", fontsize=15, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=15, labelpad=10)
    ax.tick_params(labelsize=14)

    # AUC is divided by 100 here because in calculate_model_performance function,
    # these metrics are multiplied by 100
    ax.set_title("ROC Curve - AUC %.3f" %(AUC/100), fontsize=20) 
    ax.legend(["Logistic Regression", "Random guess"], loc='lower right', prop={'size': 14});
# *********************************************************************************************************************
