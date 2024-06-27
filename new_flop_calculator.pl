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

%problem rules
0.45::action(new_fc_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_conv_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.4::action(dec_neurons,latency):- problem(latency).
1.0::action(dec_layers,latency):- problem(latency).
0.4::action(dec_neurons,model_size):- problem(model_size).
0.5::action(dec_layers,model_size):- problem(model_size).