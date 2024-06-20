% utils for hardware constraints
high_flops :- flops(V), flops_th(Th), V > Th.
high_numb_params :- nparams(V), nparams_th(Th), V > Th.

% rules for hardware constraints
problem(latency) :- high_flops.
problem(model_size) :- high_numb_params.

% rules for hardware constraints
t(0.4)::action(dec_neurons, latency).
t(0.5)::action(dec_layers, latency).
t(0.4)::action(dec_neurons, model_size).
t(0.5)::action(dec_layers, model_size).
