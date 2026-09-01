// ==========================================================================
//  Rsqrt_Lut.vh -- 1/sqrt 유닛 PWL LUT      (AUTO-GENERATED, do not hand-edit)
// --------------------------------------------------------------------------
//  r(m, par) = (m * 2^par)^-0.5 ,  m = 1+f in [1,2),  UQ1.17
//     par=0 : 1/sqrt(m)  in (0.7071, 1.0]      <- var 의 지수가 짝수
//     par=1 : 1/sqrt(2m) in (0.5,    0.7071]   <- 홀수 (2를 mantissa 로 흡수)
//  뱅크당 128 세그먼트 균일분할, 세그먼트 내 위치 12b, 인덱스 = {par, seg[6:0]}
//     r = rsq_base_rom[idx] + ((rsq_delta_rom[idx]*frac + (1<<11)) >>> 12)
//  ROM 폭은 실제 값 범위에 맞춰 최소화 : base 18b(unsigned), delta 10b(signed)
//  최대 근사오차 = 1.2107e-05  (상대오차 <= 1.5065e-05)
//  * NSEG/FRB/RF 를 바꾸면 Rsqrt_Unit.v 의 SEGB/FRB 와 함께 재생성해야 한다.
// ==========================================================================
localparam integer RSQ_BW = 18;   // base ROM 폭 (unsigned)
localparam integer RSQ_DW = 10;   // delta ROM 폭 (signed)
reg        [RSQ_BW-1:0] rsq_base_rom  [0:255];
reg signed [RSQ_DW-1:0] rsq_delta_rom [0:255];
initial begin
    rsq_base_rom[  0] = 18'd131072; rsq_delta_rom[  0] = -10'sd509;  // par=0 m=1.000000  r=1.00000000
    rsq_base_rom[  1] = 18'd130563; rsq_delta_rom[  1] = -10'sd503;  // par=0 m=1.007812  r=0.99611649
    rsq_base_rom[  2] = 18'd130060; rsq_delta_rom[  2] = -10'sd498;  // par=0 m=1.015625  r=0.99227788
    rsq_base_rom[  3] = 18'd129562; rsq_delta_rom[  3] = -10'sd491;  // par=0 m=1.023438  r=0.98848330
    rsq_base_rom[  4] = 18'd129071; rsq_delta_rom[  4] = -10'sd486;  // par=0 m=1.031250  r=0.98473193
    rsq_base_rom[  5] = 18'd128585; rsq_delta_rom[  5] = -10'sd481;  // par=0 m=1.039062  r=0.98102294
    rsq_base_rom[  6] = 18'd128104; rsq_delta_rom[  6] = -10'sd475;  // par=0 m=1.046875  r=0.97735555
    rsq_base_rom[  7] = 18'd127629; rsq_delta_rom[  7] = -10'sd470;  // par=0 m=1.054688  r=0.97372899
    rsq_base_rom[  8] = 18'd127159; rsq_delta_rom[  8] = -10'sd465;  // par=0 m=1.062500  r=0.97014250
    rsq_base_rom[  9] = 18'd126694; rsq_delta_rom[  9] = -10'sd460;  // par=0 m=1.070312  r=0.96659535
    rsq_base_rom[ 10] = 18'd126234; rsq_delta_rom[ 10] = -10'sd455;  // par=0 m=1.078125  r=0.96308682
    rsq_base_rom[ 11] = 18'd125779; rsq_delta_rom[ 11] = -10'sd450;  // par=0 m=1.085938  r=0.95961623
    rsq_base_rom[ 12] = 18'd125329; rsq_delta_rom[ 12] = -10'sd445;  // par=0 m=1.093750  r=0.95618289
    rsq_base_rom[ 13] = 18'd124884; rsq_delta_rom[ 13] = -10'sd441;  // par=0 m=1.101562  r=0.95278613
    rsq_base_rom[ 14] = 18'd124443; rsq_delta_rom[ 14] = -10'sd436;  // par=0 m=1.109375  r=0.94942533
    rsq_base_rom[ 15] = 18'd124007; rsq_delta_rom[ 15] = -10'sd431;  // par=0 m=1.117188  r=0.94609983
    rsq_base_rom[ 16] = 18'd123576; rsq_delta_rom[ 16] = -10'sd427;  // par=0 m=1.125000  r=0.94280904
    rsq_base_rom[ 17] = 18'd123149; rsq_delta_rom[ 17] = -10'sd422;  // par=0 m=1.132812  r=0.93955235
    rsq_base_rom[ 18] = 18'd122727; rsq_delta_rom[ 18] = -10'sd419;  // par=0 m=1.140625  r=0.93632918
    rsq_base_rom[ 19] = 18'd122308; rsq_delta_rom[ 19] = -10'sd414;  // par=0 m=1.148438  r=0.93313895
    rsq_base_rom[ 20] = 18'd121894; rsq_delta_rom[ 20] = -10'sd409;  // par=0 m=1.156250  r=0.92998111
    rsq_base_rom[ 21] = 18'd121485; rsq_delta_rom[ 21] = -10'sd406;  // par=0 m=1.164062  r=0.92685511
    rsq_base_rom[ 22] = 18'd121079; rsq_delta_rom[ 22] = -10'sd401;  // par=0 m=1.171875  r=0.92376043
    rsq_base_rom[ 23] = 18'd120678; rsq_delta_rom[ 23] = -10'sd398;  // par=0 m=1.179688  r=0.92069654
    rsq_base_rom[ 24] = 18'd120280; rsq_delta_rom[ 24] = -10'sd394;  // par=0 m=1.187500  r=0.91766294
    rsq_base_rom[ 25] = 18'd119886; rsq_delta_rom[ 25] = -10'sd390;  // par=0 m=1.195312  r=0.91465912
    rsq_base_rom[ 26] = 18'd119496; rsq_delta_rom[ 26] = -10'sd386;  // par=0 m=1.203125  r=0.91168461
    rsq_base_rom[ 27] = 18'd119110; rsq_delta_rom[ 27] = -10'sd382;  // par=0 m=1.210938  r=0.90873893
    rsq_base_rom[ 28] = 18'd118728; rsq_delta_rom[ 28] = -10'sd379;  // par=0 m=1.218750  r=0.90582163
    rsq_base_rom[ 29] = 18'd118349; rsq_delta_rom[ 29] = -10'sd375;  // par=0 m=1.226562  r=0.90293224
    rsq_base_rom[ 30] = 18'd117974; rsq_delta_rom[ 30] = -10'sd372;  // par=0 m=1.234375  r=0.90007032
    rsq_base_rom[ 31] = 18'd117602; rsq_delta_rom[ 31] = -10'sd368;  // par=0 m=1.242188  r=0.89723545
    rsq_base_rom[ 32] = 18'd117234; rsq_delta_rom[ 32] = -10'sd364;  // par=0 m=1.250000  r=0.89442719
    rsq_base_rom[ 33] = 18'd116870; rsq_delta_rom[ 33] = -10'sd362;  // par=0 m=1.257812  r=0.89164514
    rsq_base_rom[ 34] = 18'd116508; rsq_delta_rom[ 34] = -10'sd357;  // par=0 m=1.265625  r=0.88888889
    rsq_base_rom[ 35] = 18'd116151; rsq_delta_rom[ 35] = -10'sd355;  // par=0 m=1.273438  r=0.88615804
    rsq_base_rom[ 36] = 18'd115796; rsq_delta_rom[ 36] = -10'sd352;  // par=0 m=1.281250  r=0.88345221
    rsq_base_rom[ 37] = 18'd115444; rsq_delta_rom[ 37] = -10'sd348;  // par=0 m=1.289062  r=0.88077101
    rsq_base_rom[ 38] = 18'd115096; rsq_delta_rom[ 38] = -10'sd345;  // par=0 m=1.296875  r=0.87811408
    rsq_base_rom[ 39] = 18'd114751; rsq_delta_rom[ 39] = -10'sd342;  // par=0 m=1.304688  r=0.87548105
    rsq_base_rom[ 40] = 18'd114409; rsq_delta_rom[ 40] = -10'sd339;  // par=0 m=1.312500  r=0.87287156
    rsq_base_rom[ 41] = 18'd114070; rsq_delta_rom[ 41] = -10'sd336;  // par=0 m=1.320312  r=0.87028527
    rsq_base_rom[ 42] = 18'd113734; rsq_delta_rom[ 42] = -10'sd333;  // par=0 m=1.328125  r=0.86772183
    rsq_base_rom[ 43] = 18'd113401; rsq_delta_rom[ 43] = -10'sd330;  // par=0 m=1.335938  r=0.86518091
    rsq_base_rom[ 44] = 18'd113071; rsq_delta_rom[ 44] = -10'sd327;  // par=0 m=1.343750  r=0.86266219
    rsq_base_rom[ 45] = 18'd112744; rsq_delta_rom[ 45] = -10'sd325;  // par=0 m=1.351562  r=0.86016533
    rsq_base_rom[ 46] = 18'd112419; rsq_delta_rom[ 46] = -10'sd322;  // par=0 m=1.359375  r=0.85769003
    rsq_base_rom[ 47] = 18'd112097; rsq_delta_rom[ 47] = -10'sd318;  // par=0 m=1.367188  r=0.85523597
    rsq_base_rom[ 48] = 18'd111779; rsq_delta_rom[ 48] = -10'sd317;  // par=0 m=1.375000  r=0.85280287
    rsq_base_rom[ 49] = 18'd111462; rsq_delta_rom[ 49] = -10'sd313;  // par=0 m=1.382812  r=0.85039041
    rsq_base_rom[ 50] = 18'd111149; rsq_delta_rom[ 50] = -10'sd311;  // par=0 m=1.390625  r=0.84799830
    rsq_base_rom[ 51] = 18'd110838; rsq_delta_rom[ 51] = -10'sd308;  // par=0 m=1.398438  r=0.84562628
    rsq_base_rom[ 52] = 18'd110530; rsq_delta_rom[ 52] = -10'sd306;  // par=0 m=1.406250  r=0.84327404
    rsq_base_rom[ 53] = 18'd110224; rsq_delta_rom[ 53] = -10'sd303;  // par=0 m=1.414062  r=0.84094133
    rsq_base_rom[ 54] = 18'd109921; rsq_delta_rom[ 54] = -10'sd301;  // par=0 m=1.421875  r=0.83862787
    rsq_base_rom[ 55] = 18'd109620; rsq_delta_rom[ 55] = -10'sd298;  // par=0 m=1.429688  r=0.83633340
    rsq_base_rom[ 56] = 18'd109322; rsq_delta_rom[ 56] = -10'sd296;  // par=0 m=1.437500  r=0.83405766
    rsq_base_rom[ 57] = 18'd109026; rsq_delta_rom[ 57] = -10'sd294;  // par=0 m=1.445312  r=0.83180039
    rsq_base_rom[ 58] = 18'd108732; rsq_delta_rom[ 58] = -10'sd291;  // par=0 m=1.453125  r=0.82956136
    rsq_base_rom[ 59] = 18'd108441; rsq_delta_rom[ 59] = -10'sd289;  // par=0 m=1.460938  r=0.82734030
    rsq_base_rom[ 60] = 18'd108152; rsq_delta_rom[ 60] = -10'sd286;  // par=0 m=1.468750  r=0.82513700
    rsq_base_rom[ 61] = 18'd107866; rsq_delta_rom[ 61] = -10'sd284;  // par=0 m=1.476562  r=0.82295120
    rsq_base_rom[ 62] = 18'd107582; rsq_delta_rom[ 62] = -10'sd282;  // par=0 m=1.484375  r=0.82078268
    rsq_base_rom[ 63] = 18'd107300; rsq_delta_rom[ 63] = -10'sd280;  // par=0 m=1.492188  r=0.81863122
    rsq_base_rom[ 64] = 18'd107020; rsq_delta_rom[ 64] = -10'sd278;  // par=0 m=1.500000  r=0.81649658
    rsq_base_rom[ 65] = 18'd106742; rsq_delta_rom[ 65] = -10'sd275;  // par=0 m=1.507812  r=0.81437856
    rsq_base_rom[ 66] = 18'd106467; rsq_delta_rom[ 66] = -10'sd274;  // par=0 m=1.515625  r=0.81227693
    rsq_base_rom[ 67] = 18'd106193; rsq_delta_rom[ 67] = -10'sd271;  // par=0 m=1.523438  r=0.81019149
    rsq_base_rom[ 68] = 18'd105922; rsq_delta_rom[ 68] = -10'sd269;  // par=0 m=1.531250  r=0.80812204
    rsq_base_rom[ 69] = 18'd105653; rsq_delta_rom[ 69] = -10'sd267;  // par=0 m=1.539062  r=0.80606835
    rsq_base_rom[ 70] = 18'd105386; rsq_delta_rom[ 70] = -10'sd265;  // par=0 m=1.546875  r=0.80403025
    rsq_base_rom[ 71] = 18'd105121; rsq_delta_rom[ 71] = -10'sd263;  // par=0 m=1.554688  r=0.80200753
    rsq_base_rom[ 72] = 18'd104858; rsq_delta_rom[ 72] = -10'sd262;  // par=0 m=1.562500  r=0.80000000
    rsq_base_rom[ 73] = 18'd104596; rsq_delta_rom[ 73] = -10'sd259;  // par=0 m=1.570312  r=0.79800747
    rsq_base_rom[ 74] = 18'd104337; rsq_delta_rom[ 74] = -10'sd257;  // par=0 m=1.578125  r=0.79602975
    rsq_base_rom[ 75] = 18'd104080; rsq_delta_rom[ 75] = -10'sd256;  // par=0 m=1.585938  r=0.79406667
    rsq_base_rom[ 76] = 18'd103824; rsq_delta_rom[ 76] = -10'sd253;  // par=0 m=1.593750  r=0.79211803
    rsq_base_rom[ 77] = 18'd103571; rsq_delta_rom[ 77] = -10'sd252;  // par=0 m=1.601562  r=0.79018368
    rsq_base_rom[ 78] = 18'd103319; rsq_delta_rom[ 78] = -10'sd250;  // par=0 m=1.609375  r=0.78826342
    rsq_base_rom[ 79] = 18'd103069; rsq_delta_rom[ 79] = -10'sd248;  // par=0 m=1.617188  r=0.78635710
    rsq_base_rom[ 80] = 18'd102821; rsq_delta_rom[ 80] = -10'sd246;  // par=0 m=1.625000  r=0.78446454
    rsq_base_rom[ 81] = 18'd102575; rsq_delta_rom[ 81] = -10'sd244;  // par=0 m=1.632812  r=0.78258558
    rsq_base_rom[ 82] = 18'd102331; rsq_delta_rom[ 82] = -10'sd243;  // par=0 m=1.640625  r=0.78072006
    rsq_base_rom[ 83] = 18'd102088; rsq_delta_rom[ 83] = -10'sd241;  // par=0 m=1.648438  r=0.77886781
    rsq_base_rom[ 84] = 18'd101847; rsq_delta_rom[ 84] = -10'sd240;  // par=0 m=1.656250  r=0.77702869
    rsq_base_rom[ 85] = 18'd101607; rsq_delta_rom[ 85] = -10'sd237;  // par=0 m=1.664062  r=0.77520253
    rsq_base_rom[ 86] = 18'd101370; rsq_delta_rom[ 86] = -10'sd236;  // par=0 m=1.671875  r=0.77338919
    rsq_base_rom[ 87] = 18'd101134; rsq_delta_rom[ 87] = -10'sd235;  // par=0 m=1.679688  r=0.77158852
    rsq_base_rom[ 88] = 18'd100899; rsq_delta_rom[ 88] = -10'sd232;  // par=0 m=1.687500  r=0.76980036
    rsq_base_rom[ 89] = 18'd100667; rsq_delta_rom[ 89] = -10'sd232;  // par=0 m=1.695312  r=0.76802458
    rsq_base_rom[ 90] = 18'd100435; rsq_delta_rom[ 90] = -10'sd229;  // par=0 m=1.703125  r=0.76626103
    rsq_base_rom[ 91] = 18'd100206; rsq_delta_rom[ 91] = -10'sd228;  // par=0 m=1.710938  r=0.76450957
    rsq_base_rom[ 92] = 18'd99978; rsq_delta_rom[ 92] = -10'sd227;  // par=0 m=1.718750  r=0.76277007
    rsq_base_rom[ 93] = 18'd99751; rsq_delta_rom[ 93] = -10'sd225;  // par=0 m=1.726562  r=0.76104239
    rsq_base_rom[ 94] = 18'd99526; rsq_delta_rom[ 94] = -10'sd223;  // par=0 m=1.734375  r=0.75932640
    rsq_base_rom[ 95] = 18'd99303; rsq_delta_rom[ 95] = -10'sd222;  // par=0 m=1.742188  r=0.75762196
    rsq_base_rom[ 96] = 18'd99081; rsq_delta_rom[ 96] = -10'sd220;  // par=0 m=1.750000  r=0.75592895
    rsq_base_rom[ 97] = 18'd98861; rsq_delta_rom[ 97] = -10'sd219;  // par=0 m=1.757812  r=0.75424723
    rsq_base_rom[ 98] = 18'd98642; rsq_delta_rom[ 98] = -10'sd218;  // par=0 m=1.765625  r=0.75257669
    rsq_base_rom[ 99] = 18'd98424; rsq_delta_rom[ 99] = -10'sd216;  // par=0 m=1.773438  r=0.75091721
    rsq_base_rom[100] = 18'd98208; rsq_delta_rom[100] = -10'sd215;  // par=0 m=1.781250  r=0.74926865
    rsq_base_rom[101] = 18'd97993; rsq_delta_rom[101] = -10'sd213;  // par=0 m=1.789062  r=0.74763090
    rsq_base_rom[102] = 18'd97780; rsq_delta_rom[102] = -10'sd212;  // par=0 m=1.796875  r=0.74600385
    rsq_base_rom[103] = 18'd97568; rsq_delta_rom[103] = -10'sd210;  // par=0 m=1.804688  r=0.74438737
    rsq_base_rom[104] = 18'd97358; rsq_delta_rom[104] = -10'sd209;  // par=0 m=1.812500  r=0.74278135
    rsq_base_rom[105] = 18'd97149; rsq_delta_rom[105] = -10'sd208;  // par=0 m=1.820312  r=0.74118569
    rsq_base_rom[106] = 18'd96941; rsq_delta_rom[106] = -10'sd207;  // par=0 m=1.828125  r=0.73960026
    rsq_base_rom[107] = 18'd96734; rsq_delta_rom[107] = -10'sd205;  // par=0 m=1.835938  r=0.73802497
    rsq_base_rom[108] = 18'd96529; rsq_delta_rom[108] = -10'sd204;  // par=0 m=1.843750  r=0.73645969
    rsq_base_rom[109] = 18'd96325; rsq_delta_rom[109] = -10'sd202;  // par=0 m=1.851562  r=0.73490434
    rsq_base_rom[110] = 18'd96123; rsq_delta_rom[110] = -10'sd202;  // par=0 m=1.859375  r=0.73335880
    rsq_base_rom[111] = 18'd95921; rsq_delta_rom[111] = -10'sd200;  // par=0 m=1.867188  r=0.73182297
    rsq_base_rom[112] = 18'd95721; rsq_delta_rom[112] = -10'sd198;  // par=0 m=1.875000  r=0.73029674
    rsq_base_rom[113] = 18'd95523; rsq_delta_rom[113] = -10'sd198;  // par=0 m=1.882812  r=0.72878003
    rsq_base_rom[114] = 18'd95325; rsq_delta_rom[114] = -10'sd196;  // par=0 m=1.890625  r=0.72727273
    rsq_base_rom[115] = 18'd95129; rsq_delta_rom[115] = -10'sd195;  // par=0 m=1.898438  r=0.72577474
    rsq_base_rom[116] = 18'd94934; rsq_delta_rom[116] = -10'sd194;  // par=0 m=1.906250  r=0.72428597
    rsq_base_rom[117] = 18'd94740; rsq_delta_rom[117] = -10'sd193;  // par=0 m=1.914062  r=0.72280632
    rsq_base_rom[118] = 18'd94547; rsq_delta_rom[118] = -10'sd192;  // par=0 m=1.921875  r=0.72133571
    rsq_base_rom[119] = 18'd94355; rsq_delta_rom[119] = -10'sd190;  // par=0 m=1.929688  r=0.71987403
    rsq_base_rom[120] = 18'd94165; rsq_delta_rom[120] = -10'sd189;  // par=0 m=1.937500  r=0.71842121
    rsq_base_rom[121] = 18'd93976; rsq_delta_rom[121] = -10'sd189;  // par=0 m=1.945312  r=0.71697714
    rsq_base_rom[122] = 18'd93787; rsq_delta_rom[122] = -10'sd187;  // par=0 m=1.953125  r=0.71554175
    rsq_base_rom[123] = 18'd93600; rsq_delta_rom[123] = -10'sd185;  // par=0 m=1.960938  r=0.71411495
    rsq_base_rom[124] = 18'd93415; rsq_delta_rom[124] = -10'sd185;  // par=0 m=1.968750  r=0.71269665
    rsq_base_rom[125] = 18'd93230; rsq_delta_rom[125] = -10'sd184;  // par=0 m=1.976562  r=0.71128676
    rsq_base_rom[126] = 18'd93046; rsq_delta_rom[126] = -10'sd183;  // par=0 m=1.984375  r=0.70988521
    rsq_base_rom[127] = 18'd92863; rsq_delta_rom[127] = -10'sd181;  // par=0 m=1.992188  r=0.70849191
    rsq_base_rom[128] = 18'd92682; rsq_delta_rom[128] = -10'sd360;  // par=1 m=1.000000  r=0.70710678
    rsq_base_rom[129] = 18'd92322; rsq_delta_rom[129] = -10'sd356;  // par=1 m=1.007812  r=0.70436073
    rsq_base_rom[130] = 18'd91966; rsq_delta_rom[130] = -10'sd351;  // par=1 m=1.015625  r=0.70164642
    rsq_base_rom[131] = 18'd91615; rsq_delta_rom[131] = -10'sd348;  // par=1 m=1.023438  r=0.69896325
    rsq_base_rom[132] = 18'd91267; rsq_delta_rom[132] = -10'sd344;  // par=1 m=1.031250  r=0.69631062
    rsq_base_rom[133] = 18'd90923; rsq_delta_rom[133] = -10'sd340;  // par=1 m=1.039062  r=0.69368798
    rsq_base_rom[134] = 18'd90583; rsq_delta_rom[134] = -10'sd336;  // par=1 m=1.046875  r=0.69109474
    rsq_base_rom[135] = 18'd90247; rsq_delta_rom[135] = -10'sd332;  // par=1 m=1.054688  r=0.68853037
    rsq_base_rom[136] = 18'd89915; rsq_delta_rom[136] = -10'sd329;  // par=1 m=1.062500  r=0.68599434
    rsq_base_rom[137] = 18'd89586; rsq_delta_rom[137] = -10'sd325;  // par=1 m=1.070312  r=0.68348613
    rsq_base_rom[138] = 18'd89261; rsq_delta_rom[138] = -10'sd322;  // par=1 m=1.078125  r=0.68100522
    rsq_base_rom[139] = 18'd88939; rsq_delta_rom[139] = -10'sd318;  // par=1 m=1.085938  r=0.67855114
    rsq_base_rom[140] = 18'd88621; rsq_delta_rom[140] = -10'sd315;  // par=1 m=1.093750  r=0.67612340
    rsq_base_rom[141] = 18'd88306; rsq_delta_rom[141] = -10'sd311;  // par=1 m=1.101562  r=0.67372154
    rsq_base_rom[142] = 18'd87995; rsq_delta_rom[142] = -10'sd309;  // par=1 m=1.109375  r=0.67134509
    rsq_base_rom[143] = 18'd87686; rsq_delta_rom[143] = -10'sd305;  // par=1 m=1.117188  r=0.66899361
    rsq_base_rom[144] = 18'd87381; rsq_delta_rom[144] = -10'sd302;  // par=1 m=1.125000  r=0.66666667
    rsq_base_rom[145] = 18'd87079; rsq_delta_rom[145] = -10'sd298;  // par=1 m=1.132812  r=0.66436384
    rsq_base_rom[146] = 18'd86781; rsq_delta_rom[146] = -10'sd296;  // par=1 m=1.140625  r=0.66208471
    rsq_base_rom[147] = 18'd86485; rsq_delta_rom[147] = -10'sd293;  // par=1 m=1.148438  r=0.65982888
    rsq_base_rom[148] = 18'd86192; rsq_delta_rom[148] = -10'sd289;  // par=1 m=1.156250  r=0.65759595
    rsq_base_rom[149] = 18'd85903; rsq_delta_rom[149] = -10'sd287;  // par=1 m=1.164062  r=0.65538554
    rsq_base_rom[150] = 18'd85616; rsq_delta_rom[150] = -10'sd284;  // par=1 m=1.171875  r=0.65319726
    rsq_base_rom[151] = 18'd85332; rsq_delta_rom[151] = -10'sd281;  // par=1 m=1.179688  r=0.65103077
    rsq_base_rom[152] = 18'd85051; rsq_delta_rom[152] = -10'sd279;  // par=1 m=1.187500  r=0.64888568
    rsq_base_rom[153] = 18'd84772; rsq_delta_rom[153] = -10'sd275;  // par=1 m=1.195312  r=0.64676167
    rsq_base_rom[154] = 18'd84497; rsq_delta_rom[154] = -10'sd273;  // par=1 m=1.203125  r=0.64465837
    rsq_base_rom[155] = 18'd84224; rsq_delta_rom[155] = -10'sd271;  // par=1 m=1.210938  r=0.64257546
    rsq_base_rom[156] = 18'd83953; rsq_delta_rom[156] = -10'sd268;  // par=1 m=1.218750  r=0.64051262
    rsq_base_rom[157] = 18'd83685; rsq_delta_rom[157] = -10'sd265;  // par=1 m=1.226562  r=0.63846951
    rsq_base_rom[158] = 18'd83420; rsq_delta_rom[158] = -10'sd263;  // par=1 m=1.234375  r=0.63644583
    rsq_base_rom[159] = 18'd83157; rsq_delta_rom[159] = -10'sd260;  // par=1 m=1.242188  r=0.63444127
    rsq_base_rom[160] = 18'd82897; rsq_delta_rom[160] = -10'sd258;  // par=1 m=1.250000  r=0.63245553
    rsq_base_rom[161] = 18'd82639; rsq_delta_rom[161] = -10'sd255;  // par=1 m=1.257812  r=0.63048832
    rsq_base_rom[162] = 18'd82384; rsq_delta_rom[162] = -10'sd253;  // par=1 m=1.265625  r=0.62853936
    rsq_base_rom[163] = 18'd82131; rsq_delta_rom[163] = -10'sd251;  // par=1 m=1.273438  r=0.62660836
    rsq_base_rom[164] = 18'd81880; rsq_delta_rom[164] = -10'sd248;  // par=1 m=1.281250  r=0.62469505
    rsq_base_rom[165] = 18'd81632; rsq_delta_rom[165] = -10'sd247;  // par=1 m=1.289062  r=0.62279916
    rsq_base_rom[166] = 18'd81385; rsq_delta_rom[166] = -10'sd244;  // par=1 m=1.296875  r=0.62092042
    rsq_base_rom[167] = 18'd81141; rsq_delta_rom[167] = -10'sd242;  // par=1 m=1.304688  r=0.61905859
    rsq_base_rom[168] = 18'd80899; rsq_delta_rom[168] = -10'sd239;  // par=1 m=1.312500  r=0.61721340
    rsq_base_rom[169] = 18'd80660; rsq_delta_rom[169] = -10'sd238;  // par=1 m=1.320312  r=0.61538462
    rsq_base_rom[170] = 18'd80422; rsq_delta_rom[170] = -10'sd235;  // par=1 m=1.328125  r=0.61357199
    rsq_base_rom[171] = 18'd80187; rsq_delta_rom[171] = -10'sd234;  // par=1 m=1.335938  r=0.61177529
    rsq_base_rom[172] = 18'd79953; rsq_delta_rom[172] = -10'sd231;  // par=1 m=1.343750  r=0.60999428
    rsq_base_rom[173] = 18'd79722; rsq_delta_rom[173] = -10'sd230;  // par=1 m=1.351562  r=0.60822874
    rsq_base_rom[174] = 18'd79492; rsq_delta_rom[174] = -10'sd227;  // par=1 m=1.359375  r=0.60647843
    rsq_base_rom[175] = 18'd79265; rsq_delta_rom[175] = -10'sd226;  // par=1 m=1.367188  r=0.60474316
    rsq_base_rom[176] = 18'd79039; rsq_delta_rom[176] = -10'sd223;  // par=1 m=1.375000  r=0.60302269
    rsq_base_rom[177] = 18'd78816; rsq_delta_rom[177] = -10'sd222;  // par=1 m=1.382812  r=0.60131682
    rsq_base_rom[178] = 18'd78594; rsq_delta_rom[178] = -10'sd220;  // par=1 m=1.390625  r=0.59962535
    rsq_base_rom[179] = 18'd78374; rsq_delta_rom[179] = -10'sd218;  // par=1 m=1.398438  r=0.59794807
    rsq_base_rom[180] = 18'd78156; rsq_delta_rom[180] = -10'sd216;  // par=1 m=1.406250  r=0.59628479
    rsq_base_rom[181] = 18'd77940; rsq_delta_rom[181] = -10'sd214;  // par=1 m=1.414062  r=0.59463532
    rsq_base_rom[182] = 18'd77726; rsq_delta_rom[182] = -10'sd213;  // par=1 m=1.421875  r=0.59299945
    rsq_base_rom[183] = 18'd77513; rsq_delta_rom[183] = -10'sd211;  // par=1 m=1.429688  r=0.59137702
    rsq_base_rom[184] = 18'd77302; rsq_delta_rom[184] = -10'sd209;  // par=1 m=1.437500  r=0.58976782
    rsq_base_rom[185] = 18'd77093; rsq_delta_rom[185] = -10'sd208;  // par=1 m=1.445312  r=0.58817170
    rsq_base_rom[186] = 18'd76885; rsq_delta_rom[186] = -10'sd206;  // par=1 m=1.453125  r=0.58658846
    rsq_base_rom[187] = 18'd76679; rsq_delta_rom[187] = -10'sd204;  // par=1 m=1.460938  r=0.58501794
    rsq_base_rom[188] = 18'd76475; rsq_delta_rom[188] = -10'sd202;  // par=1 m=1.468750  r=0.58345997
    rsq_base_rom[189] = 18'd76273; rsq_delta_rom[189] = -10'sd201;  // par=1 m=1.476562  r=0.58191437
    rsq_base_rom[190] = 18'd76072; rsq_delta_rom[190] = -10'sd200;  // par=1 m=1.484375  r=0.58038100
    rsq_base_rom[191] = 18'd75872; rsq_delta_rom[191] = -10'sd198;  // par=1 m=1.492188  r=0.57885968
    rsq_base_rom[192] = 18'd75674; rsq_delta_rom[192] = -10'sd196;  // par=1 m=1.500000  r=0.57735027
    rsq_base_rom[193] = 18'd75478; rsq_delta_rom[193] = -10'sd195;  // par=1 m=1.507812  r=0.57585260
    rsq_base_rom[194] = 18'd75283; rsq_delta_rom[194] = -10'sd193;  // par=1 m=1.515625  r=0.57436653
    rsq_base_rom[195] = 18'd75090; rsq_delta_rom[195] = -10'sd192;  // par=1 m=1.523438  r=0.57289190
    rsq_base_rom[196] = 18'd74898; rsq_delta_rom[196] = -10'sd190;  // par=1 m=1.531250  r=0.57142857
    rsq_base_rom[197] = 18'd74708; rsq_delta_rom[197] = -10'sd189;  // par=1 m=1.539062  r=0.56997640
    rsq_base_rom[198] = 18'd74519; rsq_delta_rom[198] = -10'sd187;  // par=1 m=1.546875  r=0.56853524
    rsq_base_rom[199] = 18'd74332; rsq_delta_rom[199] = -10'sd186;  // par=1 m=1.554688  r=0.56710496
    rsq_base_rom[200] = 18'd74146; rsq_delta_rom[200] = -10'sd185;  // par=1 m=1.562500  r=0.56568542
    rsq_base_rom[201] = 18'd73961; rsq_delta_rom[201] = -10'sd183;  // par=1 m=1.570312  r=0.56427649
    rsq_base_rom[202] = 18'd73778; rsq_delta_rom[202] = -10'sd182;  // par=1 m=1.578125  r=0.56287804
    rsq_base_rom[203] = 18'd73596; rsq_delta_rom[203] = -10'sd181;  // par=1 m=1.585938  r=0.56148993
    rsq_base_rom[204] = 18'd73415; rsq_delta_rom[204] = -10'sd179;  // par=1 m=1.593750  r=0.56011203
    rsq_base_rom[205] = 18'd73236; rsq_delta_rom[205] = -10'sd178;  // par=1 m=1.601562  r=0.55874424
    rsq_base_rom[206] = 18'd73058; rsq_delta_rom[206] = -10'sd177;  // par=1 m=1.609375  r=0.55738641
    rsq_base_rom[207] = 18'd72881; rsq_delta_rom[207] = -10'sd175;  // par=1 m=1.617188  r=0.55603844
    rsq_base_rom[208] = 18'd72706; rsq_delta_rom[208] = -10'sd174;  // par=1 m=1.625000  r=0.55470020
    rsq_base_rom[209] = 18'd72532; rsq_delta_rom[209] = -10'sd173;  // par=1 m=1.632812  r=0.55337157
    rsq_base_rom[210] = 18'd72359; rsq_delta_rom[210] = -10'sd172;  // par=1 m=1.640625  r=0.55205245
    rsq_base_rom[211] = 18'd72187; rsq_delta_rom[211] = -10'sd171;  // par=1 m=1.648438  r=0.55074271
    rsq_base_rom[212] = 18'd72016; rsq_delta_rom[212] = -10'sd169;  // par=1 m=1.656250  r=0.54944226
    rsq_base_rom[213] = 18'd71847; rsq_delta_rom[213] = -10'sd168;  // par=1 m=1.664062  r=0.54815097
    rsq_base_rom[214] = 18'd71679; rsq_delta_rom[214] = -10'sd167;  // par=1 m=1.671875  r=0.54686874
    rsq_base_rom[215] = 18'd71512; rsq_delta_rom[215] = -10'sd165;  // par=1 m=1.679688  r=0.54559547
    rsq_base_rom[216] = 18'd71347; rsq_delta_rom[216] = -10'sd165;  // par=1 m=1.687500  r=0.54433105
    rsq_base_rom[217] = 18'd71182; rsq_delta_rom[217] = -10'sd163;  // par=1 m=1.695312  r=0.54307539
    rsq_base_rom[218] = 18'd71019; rsq_delta_rom[218] = -10'sd163;  // par=1 m=1.703125  r=0.54182837
    rsq_base_rom[219] = 18'd70856; rsq_delta_rom[219] = -10'sd161;  // par=1 m=1.710938  r=0.54058990
    rsq_base_rom[220] = 18'd70695; rsq_delta_rom[220] = -10'sd160;  // par=1 m=1.718750  r=0.53935989
    rsq_base_rom[221] = 18'd70535; rsq_delta_rom[221] = -10'sd159;  // par=1 m=1.726562  r=0.53813824
    rsq_base_rom[222] = 18'd70376; rsq_delta_rom[222] = -10'sd158;  // par=1 m=1.734375  r=0.53692484
    rsq_base_rom[223] = 18'd70218; rsq_delta_rom[223] = -10'sd157;  // par=1 m=1.742188  r=0.53571962
    rsq_base_rom[224] = 18'd70061; rsq_delta_rom[224] = -10'sd156;  // par=1 m=1.750000  r=0.53452248
    rsq_base_rom[225] = 18'd69905; rsq_delta_rom[225] = -10'sd155;  // par=1 m=1.757812  r=0.53333333
    rsq_base_rom[226] = 18'd69750; rsq_delta_rom[226] = -10'sd154;  // par=1 m=1.765625  r=0.53215208
    rsq_base_rom[227] = 18'd69596; rsq_delta_rom[227] = -10'sd152;  // par=1 m=1.773438  r=0.53097865
    rsq_base_rom[228] = 18'd69444; rsq_delta_rom[228] = -10'sd152;  // par=1 m=1.781250  r=0.52981294
    rsq_base_rom[229] = 18'd69292; rsq_delta_rom[229] = -10'sd151;  // par=1 m=1.789062  r=0.52865488
    rsq_base_rom[230] = 18'd69141; rsq_delta_rom[230] = -10'sd150;  // par=1 m=1.796875  r=0.52750438
    rsq_base_rom[231] = 18'd68991; rsq_delta_rom[231] = -10'sd149;  // par=1 m=1.804688  r=0.52636136
    rsq_base_rom[232] = 18'd68842; rsq_delta_rom[232] = -10'sd148;  // par=1 m=1.812500  r=0.52522573
    rsq_base_rom[233] = 18'd68694; rsq_delta_rom[233] = -10'sd146;  // par=1 m=1.820312  r=0.52409743
    rsq_base_rom[234] = 18'd68548; rsq_delta_rom[234] = -10'sd146;  // par=1 m=1.828125  r=0.52297636
    rsq_base_rom[235] = 18'd68402; rsq_delta_rom[235] = -10'sd146;  // par=1 m=1.835938  r=0.52186246
    rsq_base_rom[236] = 18'd68256; rsq_delta_rom[236] = -10'sd144;  // par=1 m=1.843750  r=0.52075564
    rsq_base_rom[237] = 18'd68112; rsq_delta_rom[237] = -10'sd143;  // par=1 m=1.851562  r=0.51965584
    rsq_base_rom[238] = 18'd67969; rsq_delta_rom[238] = -10'sd142;  // par=1 m=1.859375  r=0.51856298
    rsq_base_rom[239] = 18'd67827; rsq_delta_rom[239] = -10'sd142;  // par=1 m=1.867188  r=0.51747698
    rsq_base_rom[240] = 18'd67685; rsq_delta_rom[240] = -10'sd140;  // par=1 m=1.875000  r=0.51639778
    rsq_base_rom[241] = 18'd67545; rsq_delta_rom[241] = -10'sd140;  // par=1 m=1.882812  r=0.51532530
    rsq_base_rom[242] = 18'd67405; rsq_delta_rom[242] = -10'sd139;  // par=1 m=1.890625  r=0.51425948
    rsq_base_rom[243] = 18'd67266; rsq_delta_rom[243] = -10'sd138;  // par=1 m=1.898438  r=0.51320024
    rsq_base_rom[244] = 18'd67128; rsq_delta_rom[244] = -10'sd137;  // par=1 m=1.906250  r=0.51214752
    rsq_base_rom[245] = 18'd66991; rsq_delta_rom[245] = -10'sd136;  // par=1 m=1.914062  r=0.51110125
    rsq_base_rom[246] = 18'd66855; rsq_delta_rom[246] = -10'sd136;  // par=1 m=1.921875  r=0.51006137
    rsq_base_rom[247] = 18'd66719; rsq_delta_rom[247] = -10'sd134;  // par=1 m=1.929688  r=0.50902781
    rsq_base_rom[248] = 18'd66585; rsq_delta_rom[248] = -10'sd134;  // par=1 m=1.937500  r=0.50800051
    rsq_base_rom[249] = 18'd66451; rsq_delta_rom[249] = -10'sd133;  // par=1 m=1.945312  r=0.50697940
    rsq_base_rom[250] = 18'd66318; rsq_delta_rom[250] = -10'sd132;  // par=1 m=1.953125  r=0.50596443
    rsq_base_rom[251] = 18'd66186; rsq_delta_rom[251] = -10'sd132;  // par=1 m=1.960938  r=0.50495552
    rsq_base_rom[252] = 18'd66054; rsq_delta_rom[252] = -10'sd131;  // par=1 m=1.968750  r=0.50395263
    rsq_base_rom[253] = 18'd65923; rsq_delta_rom[253] = -10'sd129;  // par=1 m=1.976562  r=0.50295569
    rsq_base_rom[254] = 18'd65794; rsq_delta_rom[254] = -10'sd130;  // par=1 m=1.984375  r=0.50196464
    rsq_base_rom[255] = 18'd65664; rsq_delta_rom[255] = -10'sd128;  // par=1 m=1.992188  r=0.50097943
end
