% utils for power constraints
low_power :- flops_w(F), low_p(P), F > P.
high_power :- flops_w(F), high_p(P), F < P.
diff_flops :- flops(F), last_flops(P), P \= F.

% rules for power constraints
problem(out_range) :- \+high_power.
problem(out_range) :- \+low_power.
problem(config_problem) :- problem(out_range), diff_flops.
problem(config_tuning) :- \+high_power, \+diff_flops.

% rules for power constraints
t(0.4)::action(new_config, config_problem).
t(0.5)::action(dec_layers, config_tuning).
t(0.5)::action(dec_fc, config_tuning).

%problem rules
0.4::action(new_config,config_problem):- problem(config_problem).
0.5::action(dec_layers,config_tuning):- problem(config_tuning).
0.5::action(dec_fc,config_tuning):- problem(config_tuning).

%expanding old rules
0.5::action(new_fc_layer):- \+problem(config_tuning).
0.45::action(new_conv_layer):- \+problem(config_tuning).