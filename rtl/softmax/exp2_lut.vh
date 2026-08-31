// ============================================================================
//  exp2_lut.vh -- exp 유닛 소수부 PWL LUT   (AUTO-GENERATED, do not hand-edit)
// ----------------------------------------------------------------------------
//  g(f) = 2^-f ,  f in [0,1)  ->  g in (0.5, 1],  UQ1.17 로 저장
//  128 세그먼트 균일분할, 세그먼트 내 위치 13b
//     g = exp_base_rom[seg] + ((exp_delta_rom[seg]*frac + (1<<12)) >>> 13)
//  ROM 폭은 실제 값 범위에 맞춰 최소화 : base 18b(unsigned), delta 11b(signed)
//  최대 근사오차 = 1.019e-05  (= 1.34 LSB of 2^-17)
//  * 세그먼트 수/EF 를 바꾸면 exp2_unit.v 의 SEGB 와 함께 재생성해야 한다.
// ============================================================================
localparam integer EXP_BW = 18;   // base ROM 폭 (unsigned)
localparam integer EXP_DW = 11;   // delta ROM 폭 (signed)
reg        [EXP_BW-1:0] exp_base_rom  [0:127];
reg signed [EXP_DW-1:0] exp_delta_rom [0:127];
initial begin
    exp_base_rom[  0] = 18'd131072; exp_delta_rom[  0] = -11'sd708;  // f=0.000000  2^-f=1.00000000
    exp_base_rom[  1] = 18'd130364; exp_delta_rom[  1] = -11'sd704;  // f=0.007812  2^-f=0.99459942
    exp_base_rom[  2] = 18'd129660; exp_delta_rom[  2] = -11'sd700;  // f=0.015625  2^-f=0.98922801
    exp_base_rom[  3] = 18'd128960; exp_delta_rom[  3] = -11'sd697;  // f=0.023438  2^-f=0.98388561
    exp_base_rom[  4] = 18'd128263; exp_delta_rom[  4] = -11'sd692;  // f=0.031250  2^-f=0.97857206
    exp_base_rom[  5] = 18'd127571; exp_delta_rom[  5] = -11'sd689;  // f=0.039062  2^-f=0.97328721
    exp_base_rom[  6] = 18'd126882; exp_delta_rom[  6] = -11'sd685;  // f=0.046875  2^-f=0.96803090
    exp_base_rom[  7] = 18'd126197; exp_delta_rom[  7] = -11'sd682;  // f=0.054688  2^-f=0.96280297
    exp_base_rom[  8] = 18'd125515; exp_delta_rom[  8] = -11'sd678;  // f=0.062500  2^-f=0.95760328
    exp_base_rom[  9] = 18'd124837; exp_delta_rom[  9] = -11'sd674;  // f=0.070312  2^-f=0.95243167
    exp_base_rom[ 10] = 18'd124163; exp_delta_rom[ 10] = -11'sd671;  // f=0.078125  2^-f=0.94728799
    exp_base_rom[ 11] = 18'd123492; exp_delta_rom[ 11] = -11'sd667;  // f=0.085938  2^-f=0.94217209
    exp_base_rom[ 12] = 18'd122825; exp_delta_rom[ 12] = -11'sd663;  // f=0.093750  2^-f=0.93708382
    exp_base_rom[ 13] = 18'd122162; exp_delta_rom[ 13] = -11'sd660;  // f=0.101562  2^-f=0.93202302
    exp_base_rom[ 14] = 18'd121502; exp_delta_rom[ 14] = -11'sd656;  // f=0.109375  2^-f=0.92698956
    exp_base_rom[ 15] = 18'd120846; exp_delta_rom[ 15] = -11'sd652;  // f=0.117188  2^-f=0.92198328
    exp_base_rom[ 16] = 18'd120194; exp_delta_rom[ 16] = -11'sd650;  // f=0.125000  2^-f=0.91700404
    exp_base_rom[ 17] = 18'd119544; exp_delta_rom[ 17] = -11'sd645;  // f=0.132812  2^-f=0.91205169
    exp_base_rom[ 18] = 18'd118899; exp_delta_rom[ 18] = -11'sd642;  // f=0.140625  2^-f=0.90712609
    exp_base_rom[ 19] = 18'd118257; exp_delta_rom[ 19] = -11'sd639;  // f=0.148438  2^-f=0.90222708
    exp_base_rom[ 20] = 18'd117618; exp_delta_rom[ 20] = -11'sd635;  // f=0.156250  2^-f=0.89735454
    exp_base_rom[ 21] = 18'd116983; exp_delta_rom[ 21] = -11'sd632;  // f=0.164062  2^-f=0.89250831
    exp_base_rom[ 22] = 18'd116351; exp_delta_rom[ 22] = -11'sd628;  // f=0.171875  2^-f=0.88768825
    exp_base_rom[ 23] = 18'd115723; exp_delta_rom[ 23] = -11'sd625;  // f=0.179688  2^-f=0.88289422
    exp_base_rom[ 24] = 18'd115098; exp_delta_rom[ 24] = -11'sd622;  // f=0.187500  2^-f=0.87812608
    exp_base_rom[ 25] = 18'd114476; exp_delta_rom[ 25] = -11'sd618;  // f=0.195312  2^-f=0.87338369
    exp_base_rom[ 26] = 18'd113858; exp_delta_rom[ 26] = -11'sd615;  // f=0.203125  2^-f=0.86866692
    exp_base_rom[ 27] = 18'd113243; exp_delta_rom[ 27] = -11'sd612;  // f=0.210938  2^-f=0.86397562
    exp_base_rom[ 28] = 18'd112631; exp_delta_rom[ 28] = -11'sd608;  // f=0.218750  2^-f=0.85930965
    exp_base_rom[ 29] = 18'd112023; exp_delta_rom[ 29] = -11'sd605;  // f=0.226562  2^-f=0.85466888
    exp_base_rom[ 30] = 18'd111418; exp_delta_rom[ 30] = -11'sd602;  // f=0.234375  2^-f=0.85005318
    exp_base_rom[ 31] = 18'd110816; exp_delta_rom[ 31] = -11'sd598;  // f=0.242188  2^-f=0.84546240
    exp_base_rom[ 32] = 18'd110218; exp_delta_rom[ 32] = -11'sd595;  // f=0.250000  2^-f=0.84089642
    exp_base_rom[ 33] = 18'd109623; exp_delta_rom[ 33] = -11'sd592;  // f=0.257812  2^-f=0.83635509
    exp_base_rom[ 34] = 18'd109031; exp_delta_rom[ 34] = -11'sd589;  // f=0.265625  2^-f=0.83183829
    exp_base_rom[ 35] = 18'd108442; exp_delta_rom[ 35] = -11'sd586;  // f=0.273438  2^-f=0.82734588
    exp_base_rom[ 36] = 18'd107856; exp_delta_rom[ 36] = -11'sd582;  // f=0.281250  2^-f=0.82287774
    exp_base_rom[ 37] = 18'd107274; exp_delta_rom[ 37] = -11'sd580;  // f=0.289062  2^-f=0.81843372
    exp_base_rom[ 38] = 18'd106694; exp_delta_rom[ 38] = -11'sd576;  // f=0.296875  2^-f=0.81401371
    exp_base_rom[ 39] = 18'd106118; exp_delta_rom[ 39] = -11'sd573;  // f=0.304688  2^-f=0.80961757
    exp_base_rom[ 40] = 18'd105545; exp_delta_rom[ 40] = -11'sd570;  // f=0.312500  2^-f=0.80524517
    exp_base_rom[ 41] = 18'd104975; exp_delta_rom[ 41] = -11'sd567;  // f=0.320312  2^-f=0.80089638
    exp_base_rom[ 42] = 18'd104408; exp_delta_rom[ 42] = -11'sd564;  // f=0.328125  2^-f=0.79657108
    exp_base_rom[ 43] = 18'd103844; exp_delta_rom[ 43] = -11'sd561;  // f=0.335938  2^-f=0.79226913
    exp_base_rom[ 44] = 18'd103283; exp_delta_rom[ 44] = -11'sd557;  // f=0.343750  2^-f=0.78799042
    exp_base_rom[ 45] = 18'd102726; exp_delta_rom[ 45] = -11'sd555;  // f=0.351562  2^-f=0.78373482
    exp_base_rom[ 46] = 18'd102171; exp_delta_rom[ 46] = -11'sd552;  // f=0.359375  2^-f=0.77950220
    exp_base_rom[ 47] = 18'd101619; exp_delta_rom[ 47] = -11'sd549;  // f=0.367188  2^-f=0.77529244
    exp_base_rom[ 48] = 18'd101070; exp_delta_rom[ 48] = -11'sd546;  // f=0.375000  2^-f=0.77110541
    exp_base_rom[ 49] = 18'd100524; exp_delta_rom[ 49] = -11'sd542;  // f=0.382812  2^-f=0.76694100
    exp_base_rom[ 50] = 18'd99982; exp_delta_rom[ 50] = -11'sd540;  // f=0.390625  2^-f=0.76279908
    exp_base_rom[ 51] = 18'd99442; exp_delta_rom[ 51] = -11'sd537;  // f=0.398438  2^-f=0.75867952
    exp_base_rom[ 52] = 18'd98905; exp_delta_rom[ 52] = -11'sd535;  // f=0.406250  2^-f=0.75458221
    exp_base_rom[ 53] = 18'd98370; exp_delta_rom[ 53] = -11'sd531;  // f=0.414062  2^-f=0.75050703
    exp_base_rom[ 54] = 18'd97839; exp_delta_rom[ 54] = -11'sd528;  // f=0.421875  2^-f=0.74645386
    exp_base_rom[ 55] = 18'd97311; exp_delta_rom[ 55] = -11'sd526;  // f=0.429688  2^-f=0.74242258
    exp_base_rom[ 56] = 18'd96785; exp_delta_rom[ 56] = -11'sd522;  // f=0.437500  2^-f=0.73841307
    exp_base_rom[ 57] = 18'd96263; exp_delta_rom[ 57] = -11'sd520;  // f=0.445312  2^-f=0.73442522
    exp_base_rom[ 58] = 18'd95743; exp_delta_rom[ 58] = -11'sd517;  // f=0.453125  2^-f=0.73045890
    exp_base_rom[ 59] = 18'd95226; exp_delta_rom[ 59] = -11'sd515;  // f=0.460938  2^-f=0.72651400
    exp_base_rom[ 60] = 18'd94711; exp_delta_rom[ 60] = -11'sd511;  // f=0.468750  2^-f=0.72259040
    exp_base_rom[ 61] = 18'd94200; exp_delta_rom[ 61] = -11'sd509;  // f=0.476562  2^-f=0.71868800
    exp_base_rom[ 62] = 18'd93691; exp_delta_rom[ 62] = -11'sd506;  // f=0.484375  2^-f=0.71480667
    exp_base_rom[ 63] = 18'd93185; exp_delta_rom[ 63] = -11'sd503;  // f=0.492188  2^-f=0.71094630
    exp_base_rom[ 64] = 18'd92682; exp_delta_rom[ 64] = -11'sd501;  // f=0.500000  2^-f=0.70710678
    exp_base_rom[ 65] = 18'd92181; exp_delta_rom[ 65] = -11'sd497;  // f=0.507812  2^-f=0.70328800
    exp_base_rom[ 66] = 18'd91684; exp_delta_rom[ 66] = -11'sd496;  // f=0.515625  2^-f=0.69948984
    exp_base_rom[ 67] = 18'd91188; exp_delta_rom[ 67] = -11'sd492;  // f=0.523438  2^-f=0.69571219
    exp_base_rom[ 68] = 18'd90696; exp_delta_rom[ 68] = -11'sd490;  // f=0.531250  2^-f=0.69195494
    exp_base_rom[ 69] = 18'd90206; exp_delta_rom[ 69] = -11'sd487;  // f=0.539062  2^-f=0.68821799
    exp_base_rom[ 70] = 18'd89719; exp_delta_rom[ 70] = -11'sd485;  // f=0.546875  2^-f=0.68450121
    exp_base_rom[ 71] = 18'd89234; exp_delta_rom[ 71] = -11'sd482;  // f=0.554688  2^-f=0.68080451
    exp_base_rom[ 72] = 18'd88752; exp_delta_rom[ 72] = -11'sd479;  // f=0.562500  2^-f=0.67712777
    exp_base_rom[ 73] = 18'd88273; exp_delta_rom[ 73] = -11'sd477;  // f=0.570312  2^-f=0.67347089
    exp_base_rom[ 74] = 18'd87796; exp_delta_rom[ 74] = -11'sd474;  // f=0.578125  2^-f=0.66983376
    exp_base_rom[ 75] = 18'd87322; exp_delta_rom[ 75] = -11'sd471;  // f=0.585938  2^-f=0.66621627
    exp_base_rom[ 76] = 18'd86851; exp_delta_rom[ 76] = -11'sd469;  // f=0.593750  2^-f=0.66261832
    exp_base_rom[ 77] = 18'd86382; exp_delta_rom[ 77] = -11'sd467;  // f=0.601562  2^-f=0.65903980
    exp_base_rom[ 78] = 18'd85915; exp_delta_rom[ 78] = -11'sd464;  // f=0.609375  2^-f=0.65548061
    exp_base_rom[ 79] = 18'd85451; exp_delta_rom[ 79] = -11'sd461;  // f=0.617188  2^-f=0.65194063
    exp_base_rom[ 80] = 18'd84990; exp_delta_rom[ 80] = -11'sd459;  // f=0.625000  2^-f=0.64841978
    exp_base_rom[ 81] = 18'd84531; exp_delta_rom[ 81] = -11'sd457;  // f=0.632812  2^-f=0.64491794
    exp_base_rom[ 82] = 18'd84074; exp_delta_rom[ 82] = -11'sd454;  // f=0.640625  2^-f=0.64143501
    exp_base_rom[ 83] = 18'd83620; exp_delta_rom[ 83] = -11'sd451;  // f=0.648438  2^-f=0.63797089
    exp_base_rom[ 84] = 18'd83169; exp_delta_rom[ 84] = -11'sd450;  // f=0.656250  2^-f=0.63452548
    exp_base_rom[ 85] = 18'd82719; exp_delta_rom[ 85] = -11'sd446;  // f=0.664062  2^-f=0.63109868
    exp_base_rom[ 86] = 18'd82273; exp_delta_rom[ 86] = -11'sd445;  // f=0.671875  2^-f=0.62769038
    exp_base_rom[ 87] = 18'd81828; exp_delta_rom[ 87] = -11'sd442;  // f=0.679688  2^-f=0.62430049
    exp_base_rom[ 88] = 18'd81386; exp_delta_rom[ 88] = -11'sd439;  // f=0.687500  2^-f=0.62092891
    exp_base_rom[ 89] = 18'd80947; exp_delta_rom[ 89] = -11'sd437;  // f=0.695312  2^-f=0.61757553
    exp_base_rom[ 90] = 18'd80510; exp_delta_rom[ 90] = -11'sd435;  // f=0.703125  2^-f=0.61424027
    exp_base_rom[ 91] = 18'd80075; exp_delta_rom[ 91] = -11'sd433;  // f=0.710938  2^-f=0.61092302
    exp_base_rom[ 92] = 18'd79642; exp_delta_rom[ 92] = -11'sd430;  // f=0.718750  2^-f=0.60762368
    exp_base_rom[ 93] = 18'd79212; exp_delta_rom[ 93] = -11'sd427;  // f=0.726562  2^-f=0.60434216
    exp_base_rom[ 94] = 18'd78785; exp_delta_rom[ 94] = -11'sd426;  // f=0.734375  2^-f=0.60107837
    exp_base_rom[ 95] = 18'd78359; exp_delta_rom[ 95] = -11'sd423;  // f=0.742188  2^-f=0.59783220
    exp_base_rom[ 96] = 18'd77936; exp_delta_rom[ 96] = -11'sd421;  // f=0.750000  2^-f=0.59460356
    exp_base_rom[ 97] = 18'd77515; exp_delta_rom[ 97] = -11'sd419;  // f=0.757812  2^-f=0.59139236
    exp_base_rom[ 98] = 18'd77096; exp_delta_rom[ 98] = -11'sd416;  // f=0.765625  2^-f=0.58819850
    exp_base_rom[ 99] = 18'd76680; exp_delta_rom[ 99] = -11'sd414;  // f=0.773438  2^-f=0.58502188
    exp_base_rom[100] = 18'd76266; exp_delta_rom[100] = -11'sd412;  // f=0.781250  2^-f=0.58186243
    exp_base_rom[101] = 18'd75854; exp_delta_rom[101] = -11'sd410;  // f=0.789062  2^-f=0.57872004
    exp_base_rom[102] = 18'd75444; exp_delta_rom[102] = -11'sd407;  // f=0.796875  2^-f=0.57559461
    exp_base_rom[103] = 18'd75037; exp_delta_rom[103] = -11'sd405;  // f=0.804688  2^-f=0.57248607
    exp_base_rom[104] = 18'd74632; exp_delta_rom[104] = -11'sd403;  // f=0.812500  2^-f=0.56939432
    exp_base_rom[105] = 18'd74229; exp_delta_rom[105] = -11'sd401;  // f=0.820312  2^-f=0.56631926
    exp_base_rom[106] = 18'd73828; exp_delta_rom[106] = -11'sd399;  // f=0.828125  2^-f=0.56326081
    exp_base_rom[107] = 18'd73429; exp_delta_rom[107] = -11'sd397;  // f=0.835938  2^-f=0.56021888
    exp_base_rom[108] = 18'd73032; exp_delta_rom[108] = -11'sd394;  // f=0.843750  2^-f=0.55719337
    exp_base_rom[109] = 18'd72638; exp_delta_rom[109] = -11'sd392;  // f=0.851562  2^-f=0.55418421
    exp_base_rom[110] = 18'd72246; exp_delta_rom[110] = -11'sd390;  // f=0.859375  2^-f=0.55119129
    exp_base_rom[111] = 18'd71856; exp_delta_rom[111] = -11'sd388;  // f=0.867188  2^-f=0.54821454
    exp_base_rom[112] = 18'd71468; exp_delta_rom[112] = -11'sd386;  // f=0.875000  2^-f=0.54525387
    exp_base_rom[113] = 18'd71082; exp_delta_rom[113] = -11'sd384;  // f=0.882812  2^-f=0.54230918
    exp_base_rom[114] = 18'd70698; exp_delta_rom[114] = -11'sd382;  // f=0.890625  2^-f=0.53938040
    exp_base_rom[115] = 18'd70316; exp_delta_rom[115] = -11'sd380;  // f=0.898438  2^-f=0.53646743
    exp_base_rom[116] = 18'd69936; exp_delta_rom[116] = -11'sd378;  // f=0.906250  2^-f=0.53357020
    exp_base_rom[117] = 18'd69558; exp_delta_rom[117] = -11'sd375;  // f=0.914062  2^-f=0.53068861
    exp_base_rom[118] = 18'd69183; exp_delta_rom[118] = -11'sd374;  // f=0.921875  2^-f=0.52782259
    exp_base_rom[119] = 18'd68809; exp_delta_rom[119] = -11'sd371;  // f=0.929688  2^-f=0.52497204
    exp_base_rom[120] = 18'd68438; exp_delta_rom[120] = -11'sd370;  // f=0.937500  2^-f=0.52213689
    exp_base_rom[121] = 18'd68068; exp_delta_rom[121] = -11'sd368;  // f=0.945312  2^-f=0.51931705
    exp_base_rom[122] = 18'd67700; exp_delta_rom[122] = -11'sd365;  // f=0.953125  2^-f=0.51651244
    exp_base_rom[123] = 18'd67335; exp_delta_rom[123] = -11'sd364;  // f=0.960938  2^-f=0.51372297
    exp_base_rom[124] = 18'd66971; exp_delta_rom[124] = -11'sd362;  // f=0.968750  2^-f=0.51094857
    exp_base_rom[125] = 18'd66609; exp_delta_rom[125] = -11'sd359;  // f=0.976562  2^-f=0.50818916
    exp_base_rom[126] = 18'd66250; exp_delta_rom[126] = -11'sd358;  // f=0.984375  2^-f=0.50544464
    exp_base_rom[127] = 18'd65892; exp_delta_rom[127] = -11'sd356;  // f=0.992188  2^-f=0.50271495
end
