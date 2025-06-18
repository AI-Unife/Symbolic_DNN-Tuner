0.99::action(decr_lr,inc_loss):- problem(inc_loss).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.99::action(decr_lr,high_lr):- problem(high_lr).
0.99::action(inc_lr,low_lr):- problem(low_lr).
0.40::action(reg_l2,overfitting):- problem(overfitting).
0.40::action(inc_dropout,overfitting):- problem(overfitting).
0.60::action(data_augmentation,overfitting):- problem(overfitting).
0.30::action(dec_neurons,overfitting):- problem(overfitting).
0.30::action(dec_fc, overfitting):- problem(overfitting).
0.30::action(dec_layers,overfitting):- problem(overfitting).
0.20::action(decr_lr,underfitting):- problem(underfitting).
0.50::action(inc_neurons,underfitting):- problem(underfitting).
0.30::action(new_fc_layer, underfitting):- problem(underfitting).
0.30::action(new_conv_layer, underfitting):- problem(underfitting).

