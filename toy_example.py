import torch
from src.search_space import create_search_space, SearchInstance



# juste pour l'exemple
from itertools import product
from random import randint


if __name__ == '__main__':
    var_1_range = range(10)
    var_2_range = range(20)
    var_3_range = range(10)
    encodings = product(var_1_range, var_2_range, var_3_range)
    encodings = torch.Tensor(list(encodings))

    # We're running this without pretraining
    # encoding_to_net is useful for PyTorch nn.Module objects
    
    search_space = create_search_space(name='Exemple',
                                       save_filename='test_search_space.dill',
                                       encodings=encodings,
                                       encoding_to_net=None,
                                       device='cpu')

    # search_space.preprocess() for the normal setup
    search_space.preprocess_no_pretraining()

    # Objective function (toy example): sum of the variables
    # We used the actual sum for the high fidelity evaluation
    # And the sum + perturbation for the low fidelity one
    # In a normal NAS scenario, example a vision model which needs 300 training epochs
    # We'd use e.g. 300 epochs for the hi_fi_eval, 15 epochs for lo_fi_eval
    
    hi_fi_eval = lambda encodings_lst: [sum(encoding) for encoding in encodings_lst]
    hi_fi_cost = 200
    
    lo_fi_eval = lambda encodings_lst: [sum(encoding) + randint(-2,3) for encoding in encodings_lst]
    lo_fi_cost = 12

    search_instance = SearchInstance(name='Exemple',
                                     save_filename='test_search_inst.dill',
                                     search_space_filename='test_search_space.dill',
                                     hi_fi_eval = hi_fi_eval,
                                     hi_fi_cost = hi_fi_cost,
                                     lo_fi_eval = lo_fi_eval,
                                     lo_fi_cost = lo_fi_cost,
                                     device='cpu')

    search_instance.run_search(eval_budget=int(1e6))

    # This will save at each iteration.
    # To resume, load it using dill as follows, and execute run_search

    with open('test_search_inst.dill', 'rb') as f:
        s = dill.load(f)
