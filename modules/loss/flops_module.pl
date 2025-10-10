% utils for hardware constraints
high_flops :- flops(V), flops_th(Th), V > Th.
high_numb_params :- nparams(V), nparams_th(Th), V > Th.

% rules for hardware constraints
problem(latency) :- high_flops.
problem(model_size) :- high_numb_params.

% rules for hardware constraints
t(0.70)::action(new_fc_layer, underfitting).
t(0.70)::action(new_conv_layer, underfitting).
t(0.70)::action(inc_neurons, underfitting).
t(0.70)::action(dec_neurons, latency).
t(0.70)::action(dec_layers, latency).
t(0.70)::action(dec_neurons, model_size).
t(0.70)::action(dec_layers, model_size).
t(0.70)::action(dec_fc,model_size).

%problem rules
0.70::action(new_fc_layer, underfitting):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.70::action(new_conv_layer, underfitting):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.70::action(inc_neurons, underfitting):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.70::action(dec_neurons, latency):- problem(latency).
0.70::action(dec_layers, latency):- problem(latency).
0.70::action(dec_neurons, model_size):- problem(model_size).
0.70::action(dec_layers, model_size):- problem(model_size).
0.70::action(dec_fc,model_size):- problem(model_size).