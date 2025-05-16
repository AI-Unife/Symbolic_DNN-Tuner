t(0.99)::action(decr_lr, inc_loss).
t(0.99)::action(decr_lr, high_lr).
t(0.99)::action(inc_lr, low_lr).
t(0.4)::action(reg_l2, overfitting).
t(0.4)::action(inc_dropout, overfitting).
t(0.6)::action(data_augmentation, overfitting).
t(0.3)::action(decr_lr, underfitting).
t(0.4)::action(inc_neurons, underfitting).
t(0.6)::action(new_fc_layer, underfitting).
t(0.99)::action(new_conv_layer, underfitting).
t(0.85)::action(inc_batch_size, floating_loss).
t(0.15)::action(decr_lr, floating_loss).

