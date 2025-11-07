% utils for hardware constraints
high_latency :- hw_latency(L), max_latency(Max_L), L > Max_L.

% rules for hardware constraints
problem(out_range):- high_latency.

% rules for power constraints
t(0.70)::action(dec_layers, out_range).
t(0.70)::action(dec_fc, out_range).
t(0.40)::action(dec_neurons, out_range).

%problem rules
0.70::action(dec_layers, out_range):- problem(out_range).
0.70::action(dec_fc, out_range):- problem(out_range).
0.70::action(dec_neurons, out_range):- problem(out_range).

%expanding old rules
0.70::action(new_fc_layer, underfitting):- problem(underfitting), \+problem(out_range).
0.70::action(new_conv_layer, underfitting):- problem(underfitting), \+problem(out_range).
0.40::action(inc_neurons, underfitting):- problem(underfitting), \+problem(out_range).
