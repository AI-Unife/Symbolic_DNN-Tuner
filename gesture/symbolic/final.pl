l([21.130870819091797, 14.032437324523926, 9.146753311157227, 6.024165630340576, 4.0388336181640625, 3.055676221847534, 2.672644853591919, 2.31317400932312, 2.0904500484466553, 1.9315619468688965, 1.8266184329986572, 1.6581448316574097, 1.4670218229293823, 1.4493169784545898, 1.1831296682357788, 1.013831615447998, 0.853383481502533, 0.6893221139907837, 0.63726407289505, 0.5717300772666931, 0.5614028573036194, 0.5148025155067444, 0.48265865445137024]).
sl([21.130870819091797, 18.29149742126465, 14.633599777221681, 11.189826118469238, 8.329429118347168, 6.219927959747314, 4.801014717285156, 3.8058784341003418, 3.119707079838867, 2.6444490266508787, 2.31731678918999, 2.053648006176958, 1.8189975328779278, 1.6711253111085926, 1.475927053959467, 1.2910888785548795, 1.116006719733941, 0.945332877436678, 0.8221053556200268, 0.7219552442786933, 0.6577342894886637, 0.600561579895896, 0.5534004097180857]).
a([0.2646239697933197, 0.475394606590271, 0.5598886013031006, 0.6016713380813599, 0.6657381653785706, 0.6982358694076538, 0.670380711555481, 0.7502321004867554, 0.7558031678199768, 0.7845867872238159, 0.7697307467460632, 0.7920148372650146, 0.8133704662322998, 0.7910863757133484, 0.834726095199585, 0.8783658146858215, 0.893221914768219, 0.9136490225791931, 0.8978644609451294, 0.9229340553283691, 0.9117920398712158, 0.9201485514640808, 0.9312906265258789]).
sa([0.2646239697933197, 0.34893222451210026, 0.43331477522850037, 0.5006574003696442, 0.5666897063732148, 0.6193081715869904, 0.6397371875743866, 0.683935152739334, 0.7126823587715911, 0.741444130152481, 0.752758776789914, 0.7684612009799543, 0.7864249070808925, 0.7882894945338749, 0.8068641348001588, 0.8354648067544239, 0.858567649959942, 0.8806001990076424, 0.8875059037826372, 0.90167716440093, 0.9057231145890444, 0.9114932893390589, 0.919412224213787]).
vl([463.72113037109375, 395.1612243652344, 193.0277557373047, 76.4725341796875, 22.970111846923828, 18.542449951171875, 10.625117301940918, 5.095067977905273, 7.503688335418701, 8.257682800292969, 10.4631986618042, 7.357207775115967, 5.3425211906433105, 5.346693515777588, 5.903203010559082, 5.7220964431762695, 7.229875087738037, 9.526532173156738, 9.664865493774414, 9.820127487182617, 10.308761596679688, 10.516067504882812, 10.37429141998291]).
va([0.125, 0.12121212482452393, 0.15530303120613098, 0.18560606241226196, 0.21590909361839294, 0.25, 0.35227271914482117, 0.4280303120613098, 0.3333333432674408, 0.36742424964904785, 0.35227271914482117, 0.3295454680919647, 0.4015151560306549, 0.47727271914482117, 0.38257575035095215, 0.3901515007019043, 0.21590909361839294, 0.1401515156030655, 0.13636364042758942, 0.1401515156030655, 0.13636364042758942, 0.13636364042758942, 0.13257575035095215]).
int_loss(1071.9044933319092).
int_slope(5215.049639701844).
lacc(0.10625000000000001).
hloss(0.4125).
new_acc(0.13257575035095215).

0.99::eve.
action(reg_l2,overfitting):- eve, problem(overfitting).
action(decr_lr,inc_loss):- eve, problem(inc_loss).
action(decr_lr,high_lr):- eve, problem(high_lr).
action(inc_lr,low_lr):- eve, problem(low_lr).
0.5::action(inc_dropout,overfitting):- problem(overfitting).
0.333333333333333::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.490066225165563::action(inc_neurons,underfitting):- problem(underfitting).
0.45::action(new_fc_layer):- problem(underfitting), \+problem(out_range).
0.45::action(new_conv_layer):- problem(underfitting), \+problem(out_range).
0.504273504273504::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
:- problem(out_range).

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

