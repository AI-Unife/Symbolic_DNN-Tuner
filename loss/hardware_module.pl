% utils for hardware constraints
high_latency :- hw_latency(L), max_latency(Max_L), L > (Max_L / 2).
high_cost :- total_cost(C), C > 0.6.

% rules for hardware constraints
problem(out_range):- high_latency.
problem(out_range):- high_cost.

% rules for power constraints
t(0.5)::action(dec_layers, out_range).
t(0.5)::action(dec_fc, out_range).
t(0.5)::action(dec_neurons, out_range).

%problem rules
0.5::action(dec_layers, out_range):- problem(out_range).
0.5::action(dec_fc, out_range):- problem(out_range).
0.5::action(dec_neurons, out_range):- problem(out_range).

%expanding old rules
0.5::action(new_fc_layer):- \+problem(out_range).
0.45::action(new_conv_layer):- \+problem(out_range).