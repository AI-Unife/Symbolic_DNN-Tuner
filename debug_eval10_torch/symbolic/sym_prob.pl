0.5::action(reg_l2,overfitting):-problem(overfitting).
0.5::action(inc_dropout,overfitting) :- problem(overfitting).
0.7::action(dec_conv_layers,overfitting) :- problem(overfitting).
0.7::action(dec_conv_block,overfitting) :- problem(overfitting).
0.7::action(dec_fc_layers,overfitting) :- problem(overfitting).
0.4::action(dec_neurons,overfitting) :- problem(overfitting).
0.5::action(remove_residual,overfitting):-problem(overfitting).
0.99999::action(data_augmentation,underfitting) :- problem(underfitting).
0.5::action(remove_reg_l2,underfitting):-problem(underfitting).
0.5::action(dec_dropout,underfitting) :- problem(underfitting).
0.5::action(decr_lr,underfitting) :- problem(underfitting).
1e-05::action(inc_neurons,underfitting):-\+problem(out_range),problem(underfitting).
0.4::action(new_fc_layers,underfitting) :- \+problem(out_range),problem(underfitting).
1e-05::action(inc_conv_layers,underfitting):-problem(underfitting).
0.4::action(new_conv_block,underfitting) :- problem(underfitting).
0.6::action(decr_lr,inc_loss) :- problem(inc_loss).
0.99::action(decr_lr,high_lr) :- problem(high_lr).
0.8::action(inc_batch_size,floating_loss) :- problem(floating_loss).
0.3::action(decr_lr,floating_loss) :- problem(floating_loss).
0.99999::action(add_residual,need_skip) :- problem(need_skip).
1e-05::action(dec_layers,out_range):-problem(out_range).
1e-05::action(dec_fc,out_range):-problem(out_range).
0.4::action(dec_neurons,out_range) :- problem(out_range).
0.7::action(new_conv_layers,underfitting):-problem(underfitting),\+problem(out_range).