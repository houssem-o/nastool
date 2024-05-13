# NAS-Tool

Paper link: https://hal.science/hal-04076075

We include a toy example in `toy_example.py`

What needs to be defined:
  - encodings: vector representations of architectures in the search space
  - objective function: encoding as input -> quality/fitness
  - (Optional) For PyTorch `nn.Module` objects, an `encoding_to_net` (encoding -> `nn.Module`) is useful to automatically generate the pretraining metrics (uses PyTorch profiling tools)

# Steps:
## Search space definition:
1. Include the search space encodings, and optionally the `encoding_to_net` function (otherwise pass None, or custom data in a list of tensors):
```
ss = create_search_space(name = "search-space",
                         save_filename = "ss. dill",
                         encodings = encodings,
                         encoding_to_net = encoding_to_net)
```
2. Launch the preprocessing step, which generates pretraining data (if `encoding_to_net` is specified), and pretrains the ensemble on this data.
```
ss.preprocess(sample_input = torch.rand(16, 3, 224, 224),
              threads =16)
```

## Search instance definition (multiple search instances might use one search space):
```
s = SearchInstance(name = "search-inst",
                   save_filename = "search.dill",
                   search_space_filename = "ss.dill",
                   hi_fi_eval = hi_fi_eval,
                   hi_fi_cost = 240,
                   lo_fi_eval = lo_fi_eval,
                   lo_fi_cost = 12)
```

## Run search. Progress is saved automatically.
```
s.run_search(eval_budget=n)
```
To resume, load the SearchInstance object (using dill) and run the search.
```
 with open('search.dill', 'rb') as f:
    s = dill.load(f)
```

Helper functions are available for image classification, based on Timm and the `datasets` package. Check the paper for details.
