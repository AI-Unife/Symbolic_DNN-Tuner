0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
0.0::action(new_fc_layer,underfitting):- problem(underfitting).
0.0::action(inc_batch_size,floating_loss):- problem(floating_loss).
1.0::action(decr_lr,floating_loss):- problem(floating_loss).