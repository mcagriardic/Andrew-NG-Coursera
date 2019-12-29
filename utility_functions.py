import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


EPSILON = 10e-10 # to silence the RuntimeWarning: divide by zero encountered errors


def calculate_model_performance(actual, predicted):
    # http://www.academicos.ccadet.unam.mx/jorge.marquez/cursos/Instrumentacion/FalsePositive_TrueNegative_etc.pdf
    # https://www.lexjansen.com/nesug/nesug10/hl/hl07.pdf
    # https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c
    # https://towardsdatascience.com/data-science-performance-metrics-for-everyone-4d68f4859eef

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
        "specificity": specificity(),
        "accuracy": accuracy(),
        "prevalence": prevalence(),
        "precision": precision(),
        "sensitivity/recall": sensitivity_recall(),
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


def one_hot_encode(y):
    """Convert an iterable of indices to one-hot encoded labels."""
    y = y.flatten() # Sometimes not flattened vector is passed e.g (118,1) in these cases
    # the function ends up creating a tensor e.g. (118, 2, 1). flatten removes this issue
    nb_classes = len(np.unique(y)) # get the number of unique classes
    standardised_labels = dict(zip(np.unique(y), np.arange(nb_classes))) # get the class labels as a dictionary
    # which then is standardised. E.g imagine class labels are (4,7,9) if a vector of y containing 4,7 and 9 is
    # directly passed then np.eye(nb_classes)[4] or 7,9 throws an out of index error.
    # standardised labels fixes this issue by returning a dictionary;
    # standardised_labels = {4:0, 7:1, 9:2}. The values of the dictionary are mapped to keys in y array.
    # standardised_labels also removes the error that is raised if the labels are floats. E.g. 1.0; element
    # cannot be called by a float index e.g y[1.0] - throws an index error.
    targets = np.vectorize(standardised_labels.get)(y) # map the dictionary values to array.
    return np.eye(nb_classes)[targets]


class FractionError(Exception):
    pass


def split_data_as(X, y, **kwargs):
    args_passed = list(kwargs.keys())
    split_ratios = kwargs.values()
    if not np.isclose(sum(split_ratios), 1):
        raise FractionError("Passed fractions add up to %.3f! The fractions should add up to 1!" %sum(split_ratios))
    print("Splitting the dataset as %s..." %(', '.join(args_passed[:-1]) + ' and ' + args_passed[-1]))
    
    dataset = np.c_[X,y]
    shuffled = np.arange(len(dataset))
    np.random.shuffle(shuffled)
    dataset_shuffled = dataset[shuffled]
    
    if len(args_passed) == 3:
        arg_1, arg_2, arg_3 = np.split(
            dataset_shuffled,
            [int(kwargs[args_passed[0]] * len(dataset_shuffled)),
             int((kwargs[args_passed[0]] + kwargs[args_passed[1]]) * len(dataset_shuffled))]
        )
        return arg_1, arg_2, arg_3

    elif len(args_passed) == 2:       
        arg_1, arg_2 = np.split(
            dataset_shuffled,
            [int(kwargs[args_passed[0]] * len(dataset_shuffled))]
        )
        return arg_1, arg_2


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
