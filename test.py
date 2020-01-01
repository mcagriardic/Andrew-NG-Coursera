from utility_functions import (calculate_model_performance,
                               plot_ROC,
                               one_hot_encode,
                               split_data_as,
                               grid_search_stratified) 

x = np.array(
    [
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [11,12,13],
        [14,15,16],
        [17,18,19],
        [20,21,22],
        [23,24,25],
        [26,27,28],
        [29,30,31],
        [32,33,34],
        [35,36,37]
    ]
)

y = np.array(
    [
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
        [2],
        [2],
        [2],
        [2]
    ]
)

train_dataset, test_dataset = grid_search_stratified(
    x,
    y,
    metric='accuracy',
    clf=NeuralNetwork,
    n_fold=2,
    param_grid_dict={
        'batch_size': [8, 16],
        'input_layer': [(2, 'relu'), (2, 'tanh')],
        'hidden_layer': [
            [(6,'relu'), (4,'softmax')],
            [(4,'sigmoid'),(4,'softmax')]
        ],
        'output_layer': [3],
        'alpha': [1, 2, 4],
        'verbose': [False],
        'epoch': [1000]
    }
)

def display_array(dict_):
    
    for ary in dict_.values():
        print(ary)
        print("\n")


display_array(train_dataset)
display_array(test_dataset)