l([2.2038835697174073, 2.043703546524048]).
sl([2.2038835697174073, 2.1398115604400636]).
a([0.194, 0.263]).
sa([0.194, 0.22160000000000002]).
vl([2.058048336029053, 1.9956424407958984]).
va([0.291, 0.277]).
int_loss(2.0268453884124757).
int_slope(2.0268453884124757).
lacc(0.2).
hloss(2.302585092994046).
grad_global_norm(1.0).
vanish_th(1e-08).
exploding_th(100.0).
hw_latency(9.7215e-11).
max_latency(0.008).
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
0.7::action(inc_neurons,underfitting) :- \+problem(out_range),problem(underfitting).
0.4::action(new_fc_layers,underfitting) :- \+problem(out_range),problem(underfitting).
1e-05::action(inc_conv_layers,underfitting) :- problem(underfitting).
0.4::action(new_conv_block,underfitting) :- problem(underfitting).
0.6::action(decr_lr,inc_loss) :- problem(inc_loss).
0.99::action(decr_lr,high_lr) :- problem(high_lr).
0.8::action(inc_batch_size,floating_loss) :- problem(floating_loss).
0.3::action(decr_lr,floating_loss) :- problem(floating_loss).
0.99999::action(add_residual,need_skip) :- problem(need_skip).
0.7::action(dec_layers,out_range) :- problem(out_range).
1e-05::action(dec_fc,out_range):-problem(out_range).
0.4::action(dec_neurons,out_range) :- problem(out_range).
0.7::action(new_conv_layers,underfitting):-problem(underfitting),\+problem(out_range).

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

% Utility: safe denom
safe_den(0.0, 1.0e-9).
safe_den(A, A) :- A =\= 0.0.

% Relative improvements on training loss
rel_impr_loss_early(R) :-
    l(L), nth0(0, L, L0),
    length(L, N), K is min(5, N-1),            % prime ~5 epoche (adatta se vuoi)
    nth0(K, L, Lk),
    safe_den(L0, D), R is (L0 - Lk)/D.

rel_impr_loss_total(R) :-
    l(L), nth0(0, L, L0),
    last(L, Lend),
    safe_den(L0, D), R is (L0 - Lend)/D.

% Instabilità (oscillazioni) su loss/acc già definite altrove, ma le raccogliamo qui
instability_signal :- up_down_loss ; up_down_acc.

% Slow start: la loss scende pochissimo all inizio (tipico di reti che beneficiano di skip)
slow_start :- rel_impr_loss_early(R), R < 0.05.                  % <5% nelle prime ~5 epoche

% Plateau precoce: miglioramento totale modesto
early_plateau :- rel_impr_loss_total(R), R < 0.20.               % <20% sull’intero run

% ANALYSIS
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA),
                Res is LTA - ScoreA, abs2(Res,Res1), Res1 > 0.2.
gap_tr_te_loss :- l(L), vl(VL), last(L,LTL), last(VL,ScoreL),
                Res is LTL - ScoreL, abs2(Res,Res1), Res1 > 0.2.
low_acc :- a(A), lacc(Tha), last(A,LTA),
                Res is LTA - 1.0, abs2(Res,Res1), Res1 > Tha.
high_loss :- l(L), hloss(Thl), last(L,LTL), \+isclose(LTL,0,Thl).
growing_loss_trend :- l(L),add_to_UpList(L,Usl), length(L,Length_u), G is (Usl*100)/Length_u, G > 50.
up_down_acc :- a(A),add_to_UpList(A,Usa), add_to_DownList(A,Dsa), isclose(Usa,Dsa,150), Usa > 0, Dsa > 0.
up_down_loss :- l(L),add_to_UpList(L,Usl), add_to_DownList(L,Dsl), isclose(Usl,Dsl,150), Usl > 0, Dsl > 0.
to_low_lr :- area_sub(As), threshold_up(Th), As < Th.
to_high_lr :- area_sub(As), threshold_down(Th), As > Th.

loss_increase([],_,_Th,_M):-false.
loss_increase([H|_],C,Th,_M):-
    C > Th.
loss_increase([H|T],C,Th,M):-
    H >= M,
    C1 is C + 1,
    loss_increase(T,C1,Th,M).
loss_increase([H|T],_C,Th,M):-
    H < M,
    loss_increase(T,0,Th,H).

min_list([],_M):-!,false.
min_list([H|T],M):-
    H >= M,
    min_list(T,M).
min_list([H|T],M):-
    H < M,
    min_list(T,H).

is_overfitting:- vl(L_val), loss_increase(L_val,0,5,100).

vanish_gradient :- grad_global_norm(G), vanish_th(Th), G < Th.
exploding_gradient :- grad_global_norm(G), exploding_th(Th), G > Th.


% POSSIBLE PROBLEMS
problem(overfitting) :- is_overfitting, \+ problem(underfitting).
problem(underfitting) :- low_acc.
problem(inc_loss) :- growing_loss_trend.
problem(floating_loss) :- up_down_loss.
problem(low_lr) :- to_low_lr.
problem(high_lr) :- to_high_lr.
problem(gradient) :- vanish_gradient; exploding_gradient.
% PROBLEMI SPECIFICI CHE SUGGERISCONO SKIP
problem(need_skip) :- vanish_gradient; exploding_gradient ; slow_start ; early_plateau ; instability_signal.


% QUERY ----------------------------------------------------------------------------------------------------------------
query(action(_,_)).

% rules utils in 'hardware_module'
high_latency :- hw_latency(L), max_latency(Max_L), L > Max_L.
problem(out_range):- high_latency.
