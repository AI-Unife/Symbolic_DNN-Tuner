l([2.31160569190979, 2.2722229957580566]).
sl([2.31160569190979, 2.295852613449097]).
a([0.10999999940395355, 0.12999999523162842]).
sa([0.10999999940395355, 0.11799999773502351]).
vl([2.295408248901367, 2.298469305038452]).
va([0.05999999865889549, 0.05999999865889549]).
int_loss(2.2969387769699097).
int_slope(2.2969387769699097).
lacc(0.15).
hloss(1.2).
flops(58069354).
flops_th(1200).
nparams(1063388.0).
nparams_th(23851784).

0.99::eve.
action(reg_l2,overfitting):- eve, problem(overfitting).
action(decr_lr,inc_loss):- eve, problem(inc_loss).
action(decr_lr,high_lr):- eve, problem(high_lr).
action(inc_lr,low_lr):- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.0::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_conv_layer):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.4::action(dec_neurons,latency):- problem(latency).
0.0::action(dec_layers,latency):- problem(latency).
0.4::action(dec_neurons,model_size):- problem(model_size).
0.5::action(dec_layers,model_size):- problem(model_size).

% DIAGNOSIS SECTION ----------------------------------------------------------------------------------------------------
:- use_module(library(lists)).

% UTILITY
abs2(X,Y) :- Y is abs(X).
isclose(X,Y,W) :- D is X - Y, abs2(D,D1), D1 =< W.

add_to_UpList([_],0).
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H =< H1, U is U1+1.
add_to_UpList([H|[H1|T]], U) :- add_to_UpList([H1|T], U1), H > H1, U is U1+0.

add_to_DownList([_],0).
add_to_DownList([H|[H1|T]], U) :- add_to_DownList([H1|T], U1), H > H1, U is U1+1.
add_to_DownList([H|[H1|T]], U) :- add_to_DownList([H1|T], U1), H =< H1, U is U1+0.

area_sub(R) :- int_loss(A), int_slope(B), Rt is A - B, abs2(Rt,R).
threshold_up(Th) :- int_slope(A), Th is A/4.
threshold_down(Th) :- int_slope(A), Th is A*(3/4).

% ANALYSIS
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA),
                Res is LTA - ScoreA, abs2(Res,Res1), Res1 > 0.2.
gap_tr_te_loss :- l(L), vl(VL), last(L,LTL), last(VL,ScoreL),
                Res is LTL - ScoreL, abs2(Res,Res1), Res1 > 0.2.
low_acc :- va(A), lacc(Tha), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > Tha.
high_loss :- vl(L), hloss(Thl), last(L,LTL), \+isclose(LTL,0,Thl).
growing_loss_trend :- l(L),add_to_UpList(L,Usl), length(L,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- a(A),add_to_UpList(A,Usa), add_to_DownList(A,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- l(L),add_to_UpList(L,Usl), add_to_DownList(L,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.
to_low_lr :- area_sub(As), threshold_up(Th), As < Th.
to_high_lr :- area_sub(As), threshold_down(Th), As > Th.


% POSSIBLE PROBLEMS
problem(overfitting) :- gap_tr_te_acc; gap_tr_te_loss.
problem(underfitting) :- low_acc; high_loss.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.
problem(low_lr) :- to_low_lr.
problem(high_lr) :- to_high_lr.

% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).

% rules utils in 'new_flop_calculator'
high_flops :- flops(V), flops_th(Th), V > Th.
high_numb_params :- nparams(V), nparams_th(Th), V > Th.
problem(latency) :- high_flops.
problem(model_size) :- high_numb_params.

