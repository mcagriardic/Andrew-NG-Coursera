import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product


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
    nb_classes = len(np.unique(y)) # get the number of unique classes
    standardised_labels = dict(zip(np.unique(y), np.arange(nb_classes))) # get the class labels as a dictionary
    targets = np.vectorize(standardised_labels.get)(y) # map the dictionary values to array.
    return np.eye(nb_classes)[targets]


class FractionError(Exception):
    pass


def shuffle(X, y):
    dataset = np.c_[X,y]
    shuffled = np.arange(len(dataset))
    np.random.shuffle(shuffled)
    return dataset[shuffled]


def split_data_as(X, y, **kwargs):
    args_passed = list(kwargs.keys())
    split_ratios = kwargs.values()
    if not np.isclose(sum(split_ratios), 1):
        raise FractionError("Passed fractions add up to %.3f! The fractions should add up to 1!" %sum(split_ratios))
    print("Splitting the dataset as %s..." %(', '.join(args_passed[:-1]) + ' and ' + args_passed[-1]))

    dataset_shuffled = shuffle(X, y)

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


def metrics_averaged_and_sorted(model_metrics, metrics, sort_by=None):

    # by default sort by the first metric passed
    if not sort_by:
        sort_by = metrics[0]

    # if single metric is passed to metrics e.g. "accuracy"
    if not isinstance(metrics, list):
        index_sort = 1
        metrics = [metrics]
    else:
        index_sort = metrics.index(sort_by)

    results_average_dict = {}
    for metric in metrics:
        for model in model_metrics:
            retrieve_metric = {}
            for fold in model_metrics[model]:
                retrieve_metric[model + "_" + fold] = model_metrics[model][fold][metric]
            
            vals = np.fromiter(retrieve_metric.values(), dtype=float)
            try:
                results_average_dict[model] = results_average_dict[model], (metric, np.average(vals))
            except KeyError:
                results_average_dict[model] = metric, np.average(vals)
                
    return sorted(results_average_dict.items(), key=lambda x: x[1][index_sort], reverse=True)


def param_grid(param_grid_dict):
    keys = param_grid_dict.keys()
    for element in product(*param_grid_dict.values()):
        yield dict(zip(keys, element))


def grid_search_stratified(X, y, clf, metrics, sort_by=None, n_fold=3, verbose=True, **kwargs):
    # X and y are in the shape of (no_of_training_examples, no_of_features)
    dataset_shuffled = shuffle(X, y)
    classes, counts = np.unique(y, return_counts=True)
    classes = np.int_(classes)
    counts_perc = counts / y.shape[0]

    classes_holder_dict = {}
    for class_ in classes:
        classes_holder_dict[class_] = dataset_shuffled[dataset_shuffled[:, -1] == class_]

    fold_size = y.shape[0] / n_fold

    train_dataset = {}
    array_training = np.empty_like(classes, dtype=object)
    test_dataset = {}
    array_test = np.empty_like(classes, dtype=object)
    for fold in range(n_fold):
        for class_ in classes:
            grab = np.int(fold_size * counts_perc[class_])
            array_test[class_] = classes_holder_dict[class_][fold * grab:(fold + 1) * grab ,:]
            array_training[class_] = np.delete(
                classes_holder_dict[class_],
                (range(fold * grab, (fold + 1) * grab)),
                axis=0
            )
            
        train_dataset[fold] = np.concatenate(array_training)
        test_dataset[fold] = np.concatenate(array_test)

    models = {}
    results_dict_all_models = {}

    grid_len= param_grid(kwargs['param_grid_dict'])
    n_to_run = len(list(grid_len)) * n_fold

    grid = param_grid(kwargs['param_grid_dict'])
    count = 1
    for index_model, params in enumerate(grid):
        models["model_" + str(index_model + 1)] = clf(**params)

        results_dict = {}
        for index_fold in range(n_fold):
            if verbose:
                print("\n*********{}/{}*********".format(count ,n_to_run))
                print(
                    "Running model {0} fold {1}".format(
                    str(index_model + 1),
                    str(index_fold + 1)
                    )
                )

            x_train = train_dataset[fold][:, :-1].T
            y_train = one_hot_encode(train_dataset[fold][:, -1]).T

            x_test = test_dataset[fold][:, :-1].T
            y_test = one_hot_encode(test_dataset[fold][:, -1]).T

            models["model_" + str(index_model + 1)].fit(x_train, y_train)

            results_dict["fold_" + str(index_fold + 1)] = \
            calculate_model_performance(
                np.argmax(y_test, axis=0),
                models["model_" + str(index_model + 1)].predict(x_test)
            )
            count += 1

        results_dict_all_models["model_" + str(index_model + 1)] = results_dict
        results_average_dict = metrics_averaged_and_sorted(results_dict_all_models,
                                                           metrics,
                                                           sort_by)

    return results_dict_all_models, results_average_dict, models


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
