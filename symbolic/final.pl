l([1.5882375240325928, 1.1512850522994995, 0.948762059211731, 0.8362171649932861, 0.7518475651741028]).
sl([1.5882375240325928, 1.4134565353393553, 1.2275787448883055, 1.0710341129302978, 0.9433594938278198]).
a([0.4173400104045868, 0.5926399827003479, 0.6668599843978882, 0.7085599899291992, 0.7376599907875061]).
sa([0.4173400104045868, 0.4874599993228912, 0.5592199933528901, 0.6189559919834138, 0.6664375915050507]).
vl([1.2091732025146484, 0.9136554002761841, 0.8261662125587463, 0.7781446576118469, 0.7075143456459045]).
va([0.5667999982833862, 0.6775000095367432, 0.7106999754905701, 0.7335000038146973, 0.7555000185966492]).
int_loss(3.476310044527054).
int_slope(3.833375096321106).
lacc(0.15).
hloss(1.2).
hw_latency(0.009216748).
max_latency(0.01).
new_acc(0.7555000185966492).

0.99::eve.
action(reg_l2,overfitting):- eve, problem(overfitting).
action(decr_lr,inc_loss):- eve, problem(inc_loss).
action(decr_lr,high_lr):- eve, problem(high_lr).
action(inc_lr,low_lr):- eve, problem(low_lr).
0.4::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
1.0::action(inc_neurons,underfitting):- problem(underfitting).
0.60::action(new_fc_layer):- \+problem(out_range), problem(underfitting).
0.45::action(new_conv_layer):- \+problem(out_range), problem(underfitting).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.5::action(dec_layers, out_range):- problem(out_range).
0.5::action(dec_fc, out_range):- problem(out_range).
0.5::action(dec_neurons, out_range):- problem(out_range).


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

% rules utils in 'hardware_module'
high_latency :- hw_latency(L), max_latency(Max_L), L > Max_L.
problem(out_range):- high_latency.

