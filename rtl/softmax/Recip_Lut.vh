// ============================================================================
//  Recip_Lut.vh -- 역수 유닛 PWL LUT        (AUTO-GENERATED, do not hand-edit)
// ----------------------------------------------------------------------------
//  r(f) = 1/(1+f) ,  f in [0,1)  ->  m = 1+f in [1,2),  r in (0.5,1],  UQ1.17
//  128 세그먼트 균일분할, 세그먼트 내 위치 16b  (= (SW-1)-SEGB, SW=24)
//     r = rcp_base_rom[seg] + ((rcp_delta_rom[seg]*frac + (1<<15)) >>> 16)
//  ROM 폭은 실제 값 범위에 맞춰 최소화 : base 18b(unsigned), delta 11b(signed)
//  최대 근사오차 = 2.063e-05  (상대오차 <= 4.126e-05)
//  * 세그먼트 수/RF/EF 를 바꾸면 Recip_Unit.v 의 SEGB 와 함께 재생성해야 한다.
// ============================================================================
localparam integer RCP_BW = 18;   // base ROM 폭 (unsigned)
localparam integer RCP_DW = 11;   // delta ROM 폭 (signed)
reg        [RCP_BW-1:0] rcp_base_rom  [0:127];
reg signed [RCP_DW-1:0] rcp_delta_rom [0:127];
initial begin
    rcp_base_rom[  0] = 18'd131072; rcp_delta_rom[  0] = -11'sd1016;  // m=1.000000  1/m=1.00000000
    rcp_base_rom[  1] = 18'd130056; rcp_delta_rom[  1] = -11'sd1000;  // m=1.007812  1/m=0.99224806
    rcp_base_rom[  2] = 18'd129056; rcp_delta_rom[  2] = -11'sd986;  // m=1.015625  1/m=0.98461538
    rcp_base_rom[  3] = 18'd128070; rcp_delta_rom[  3] = -11'sd970;  // m=1.023438  1/m=0.97709924
    rcp_base_rom[  4] = 18'd127100; rcp_delta_rom[  4] = -11'sd956;  // m=1.031250  1/m=0.96969697
    rcp_base_rom[  5] = 18'd126144; rcp_delta_rom[  5] = -11'sd941;  // m=1.039062  1/m=0.96240602
    rcp_base_rom[  6] = 18'd125203; rcp_delta_rom[  6] = -11'sd927;  // m=1.046875  1/m=0.95522388
    rcp_base_rom[  7] = 18'd124276; rcp_delta_rom[  7] = -11'sd914;  // m=1.054688  1/m=0.94814815
    rcp_base_rom[  8] = 18'd123362; rcp_delta_rom[  8] = -11'sd901;  // m=1.062500  1/m=0.94117647
    rcp_base_rom[  9] = 18'd122461; rcp_delta_rom[  9] = -11'sd887;  // m=1.070312  1/m=0.93430657
    rcp_base_rom[ 10] = 18'd121574; rcp_delta_rom[ 10] = -11'sd875;  // m=1.078125  1/m=0.92753623
    rcp_base_rom[ 11] = 18'd120699; rcp_delta_rom[ 11] = -11'sd862;  // m=1.085938  1/m=0.92086331
    rcp_base_rom[ 12] = 18'd119837; rcp_delta_rom[ 12] = -11'sd850;  // m=1.093750  1/m=0.91428571
    rcp_base_rom[ 13] = 18'd118987; rcp_delta_rom[ 13] = -11'sd838;  // m=1.101562  1/m=0.90780142
    rcp_base_rom[ 14] = 18'd118149; rcp_delta_rom[ 14] = -11'sd826;  // m=1.109375  1/m=0.90140845
    rcp_base_rom[ 15] = 18'd117323; rcp_delta_rom[ 15] = -11'sd815;  // m=1.117188  1/m=0.89510490
    rcp_base_rom[ 16] = 18'd116508; rcp_delta_rom[ 16] = -11'sd803;  // m=1.125000  1/m=0.88888889
    rcp_base_rom[ 17] = 18'd115705; rcp_delta_rom[ 17] = -11'sd793;  // m=1.132812  1/m=0.88275862
    rcp_base_rom[ 18] = 18'd114912; rcp_delta_rom[ 18] = -11'sd781;  // m=1.140625  1/m=0.87671233
    rcp_base_rom[ 19] = 18'd114131; rcp_delta_rom[ 19] = -11'sd771;  // m=1.148438  1/m=0.87074830
    rcp_base_rom[ 20] = 18'd113360; rcp_delta_rom[ 20] = -11'sd761;  // m=1.156250  1/m=0.86486486
    rcp_base_rom[ 21] = 18'd112599; rcp_delta_rom[ 21] = -11'sd751;  // m=1.164062  1/m=0.85906040
    rcp_base_rom[ 22] = 18'd111848; rcp_delta_rom[ 22] = -11'sd741;  // m=1.171875  1/m=0.85333333
    rcp_base_rom[ 23] = 18'd111107; rcp_delta_rom[ 23] = -11'sd731;  // m=1.179688  1/m=0.84768212
    rcp_base_rom[ 24] = 18'd110376; rcp_delta_rom[ 24] = -11'sd721;  // m=1.187500  1/m=0.84210526
    rcp_base_rom[ 25] = 18'd109655; rcp_delta_rom[ 25] = -11'sd712;  // m=1.195312  1/m=0.83660131
    rcp_base_rom[ 26] = 18'd108943; rcp_delta_rom[ 26] = -11'sd703;  // m=1.203125  1/m=0.83116883
    rcp_base_rom[ 27] = 18'd108240; rcp_delta_rom[ 27] = -11'sd694;  // m=1.210938  1/m=0.82580645
    rcp_base_rom[ 28] = 18'd107546; rcp_delta_rom[ 28] = -11'sd685;  // m=1.218750  1/m=0.82051282
    rcp_base_rom[ 29] = 18'd106861; rcp_delta_rom[ 29] = -11'sd676;  // m=1.226562  1/m=0.81528662
    rcp_base_rom[ 30] = 18'd106185; rcp_delta_rom[ 30] = -11'sd668;  // m=1.234375  1/m=0.81012658
    rcp_base_rom[ 31] = 18'd105517; rcp_delta_rom[ 31] = -11'sd659;  // m=1.242188  1/m=0.80503145
    rcp_base_rom[ 32] = 18'd104858; rcp_delta_rom[ 32] = -11'sd652;  // m=1.250000  1/m=0.80000000
    rcp_base_rom[ 33] = 18'd104206; rcp_delta_rom[ 33] = -11'sd643;  // m=1.257812  1/m=0.79503106
    rcp_base_rom[ 34] = 18'd103563; rcp_delta_rom[ 34] = -11'sd635;  // m=1.265625  1/m=0.79012346
    rcp_base_rom[ 35] = 18'd102928; rcp_delta_rom[ 35] = -11'sd628;  // m=1.273438  1/m=0.78527607
    rcp_base_rom[ 36] = 18'd102300; rcp_delta_rom[ 36] = -11'sd620;  // m=1.281250  1/m=0.78048780
    rcp_base_rom[ 37] = 18'd101680; rcp_delta_rom[ 37] = -11'sd612;  // m=1.289062  1/m=0.77575758
    rcp_base_rom[ 38] = 18'd101068; rcp_delta_rom[ 38] = -11'sd606;  // m=1.296875  1/m=0.77108434
    rcp_base_rom[ 39] = 18'd100462; rcp_delta_rom[ 39] = -11'sd598;  // m=1.304688  1/m=0.76646707
    rcp_base_rom[ 40] = 18'd99864; rcp_delta_rom[ 40] = -11'sd591;  // m=1.312500  1/m=0.76190476
    rcp_base_rom[ 41] = 18'd99273; rcp_delta_rom[ 41] = -11'sd583;  // m=1.320312  1/m=0.75739645
    rcp_base_rom[ 42] = 18'd98690; rcp_delta_rom[ 42] = -11'sd578;  // m=1.328125  1/m=0.75294118
    rcp_base_rom[ 43] = 18'd98112; rcp_delta_rom[ 43] = -11'sd570;  // m=1.335938  1/m=0.74853801
    rcp_base_rom[ 44] = 18'd97542; rcp_delta_rom[ 44] = -11'sd564;  // m=1.343750  1/m=0.74418605
    rcp_base_rom[ 45] = 18'd96978; rcp_delta_rom[ 45] = -11'sd557;  // m=1.351562  1/m=0.73988439
    rcp_base_rom[ 46] = 18'd96421; rcp_delta_rom[ 46] = -11'sd551;  // m=1.359375  1/m=0.73563218
    rcp_base_rom[ 47] = 18'd95870; rcp_delta_rom[ 47] = -11'sd545;  // m=1.367188  1/m=0.73142857
    rcp_base_rom[ 48] = 18'd95325; rcp_delta_rom[ 48] = -11'sd538;  // m=1.375000  1/m=0.72727273
    rcp_base_rom[ 49] = 18'd94787; rcp_delta_rom[ 49] = -11'sd533;  // m=1.382812  1/m=0.72316384
    rcp_base_rom[ 50] = 18'd94254; rcp_delta_rom[ 50] = -11'sd527;  // m=1.390625  1/m=0.71910112
    rcp_base_rom[ 51] = 18'd93727; rcp_delta_rom[ 51] = -11'sd520;  // m=1.398438  1/m=0.71508380
    rcp_base_rom[ 52] = 18'd93207; rcp_delta_rom[ 52] = -11'sd515;  // m=1.406250  1/m=0.71111111
    rcp_base_rom[ 53] = 18'd92692; rcp_delta_rom[ 53] = -11'sd509;  // m=1.414062  1/m=0.70718232
    rcp_base_rom[ 54] = 18'd92183; rcp_delta_rom[ 54] = -11'sd504;  // m=1.421875  1/m=0.70329670
    rcp_base_rom[ 55] = 18'd91679; rcp_delta_rom[ 55] = -11'sd498;  // m=1.429688  1/m=0.69945355
    rcp_base_rom[ 56] = 18'd91181; rcp_delta_rom[ 56] = -11'sd493;  // m=1.437500  1/m=0.69565217
    rcp_base_rom[ 57] = 18'd90688; rcp_delta_rom[ 57] = -11'sd488;  // m=1.445312  1/m=0.69189189
    rcp_base_rom[ 58] = 18'd90200; rcp_delta_rom[ 58] = -11'sd482;  // m=1.453125  1/m=0.68817204
    rcp_base_rom[ 59] = 18'd89718; rcp_delta_rom[ 59] = -11'sd477;  // m=1.460938  1/m=0.68449198
    rcp_base_rom[ 60] = 18'd89241; rcp_delta_rom[ 60] = -11'sd473;  // m=1.468750  1/m=0.68085106
    rcp_base_rom[ 61] = 18'd88768; rcp_delta_rom[ 61] = -11'sd467;  // m=1.476562  1/m=0.67724868
    rcp_base_rom[ 62] = 18'd88301; rcp_delta_rom[ 62] = -11'sd462;  // m=1.484375  1/m=0.67368421
    rcp_base_rom[ 63] = 18'd87839; rcp_delta_rom[ 63] = -11'sd458;  // m=1.492188  1/m=0.67015707
    rcp_base_rom[ 64] = 18'd87381; rcp_delta_rom[ 64] = -11'sd452;  // m=1.500000  1/m=0.66666667
    rcp_base_rom[ 65] = 18'd86929; rcp_delta_rom[ 65] = -11'sd449;  // m=1.507812  1/m=0.66321244
    rcp_base_rom[ 66] = 18'd86480; rcp_delta_rom[ 66] = -11'sd443;  // m=1.515625  1/m=0.65979381
    rcp_base_rom[ 67] = 18'd86037; rcp_delta_rom[ 67] = -11'sd439;  // m=1.523438  1/m=0.65641026
    rcp_base_rom[ 68] = 18'd85598; rcp_delta_rom[ 68] = -11'sd434;  // m=1.531250  1/m=0.65306122
    rcp_base_rom[ 69] = 18'd85164; rcp_delta_rom[ 69] = -11'sd431;  // m=1.539062  1/m=0.64974619
    rcp_base_rom[ 70] = 18'd84733; rcp_delta_rom[ 70] = -11'sd425;  // m=1.546875  1/m=0.64646465
    rcp_base_rom[ 71] = 18'd84308; rcp_delta_rom[ 71] = -11'sd422;  // m=1.554688  1/m=0.64321608
    rcp_base_rom[ 72] = 18'd83886; rcp_delta_rom[ 72] = -11'sd417;  // m=1.562500  1/m=0.64000000
    rcp_base_rom[ 73] = 18'd83469; rcp_delta_rom[ 73] = -11'sd413;  // m=1.570312  1/m=0.63681592
    rcp_base_rom[ 74] = 18'd83056; rcp_delta_rom[ 74] = -11'sd410;  // m=1.578125  1/m=0.63366337
    rcp_base_rom[ 75] = 18'd82646; rcp_delta_rom[ 75] = -11'sd405;  // m=1.585938  1/m=0.63054187
    rcp_base_rom[ 76] = 18'd82241; rcp_delta_rom[ 76] = -11'sd401;  // m=1.593750  1/m=0.62745098
    rcp_base_rom[ 77] = 18'd81840; rcp_delta_rom[ 77] = -11'sd397;  // m=1.601562  1/m=0.62439024
    rcp_base_rom[ 78] = 18'd81443; rcp_delta_rom[ 78] = -11'sd394;  // m=1.609375  1/m=0.62135922
    rcp_base_rom[ 79] = 18'd81049; rcp_delta_rom[ 79] = -11'sd389;  // m=1.617188  1/m=0.61835749
    rcp_base_rom[ 80] = 18'd80660; rcp_delta_rom[ 80] = -11'sd386;  // m=1.625000  1/m=0.61538462
    rcp_base_rom[ 81] = 18'd80274; rcp_delta_rom[ 81] = -11'sd382;  // m=1.632812  1/m=0.61244019
    rcp_base_rom[ 82] = 18'd79892; rcp_delta_rom[ 82] = -11'sd379;  // m=1.640625  1/m=0.60952381
    rcp_base_rom[ 83] = 18'd79513; rcp_delta_rom[ 83] = -11'sd375;  // m=1.648438  1/m=0.60663507
    rcp_base_rom[ 84] = 18'd79138; rcp_delta_rom[ 84] = -11'sd372;  // m=1.656250  1/m=0.60377358
    rcp_base_rom[ 85] = 18'd78766; rcp_delta_rom[ 85] = -11'sd368;  // m=1.664062  1/m=0.60093897
    rcp_base_rom[ 86] = 18'd78398; rcp_delta_rom[ 86] = -11'sd364;  // m=1.671875  1/m=0.59813084
    rcp_base_rom[ 87] = 18'd78034; rcp_delta_rom[ 87] = -11'sd362;  // m=1.679688  1/m=0.59534884
    rcp_base_rom[ 88] = 18'd77672; rcp_delta_rom[ 88] = -11'sd358;  // m=1.687500  1/m=0.59259259
    rcp_base_rom[ 89] = 18'd77314; rcp_delta_rom[ 89] = -11'sd354;  // m=1.695312  1/m=0.58986175
    rcp_base_rom[ 90] = 18'd76960; rcp_delta_rom[ 90] = -11'sd352;  // m=1.703125  1/m=0.58715596
    rcp_base_rom[ 91] = 18'd76608; rcp_delta_rom[ 91] = -11'sd348;  // m=1.710938  1/m=0.58447489
    rcp_base_rom[ 92] = 18'd76260; rcp_delta_rom[ 92] = -11'sd345;  // m=1.718750  1/m=0.58181818
    rcp_base_rom[ 93] = 18'd75915; rcp_delta_rom[ 93] = -11'sd342;  // m=1.726562  1/m=0.57918552
    rcp_base_rom[ 94] = 18'd75573; rcp_delta_rom[ 94] = -11'sd339;  // m=1.734375  1/m=0.57657658
    rcp_base_rom[ 95] = 18'd75234; rcp_delta_rom[ 95] = -11'sd336;  // m=1.742188  1/m=0.57399103
    rcp_base_rom[ 96] = 18'd74898; rcp_delta_rom[ 96] = -11'sd333;  // m=1.750000  1/m=0.57142857
    rcp_base_rom[ 97] = 18'd74565; rcp_delta_rom[ 97] = -11'sd330;  // m=1.757812  1/m=0.56888889
    rcp_base_rom[ 98] = 18'd74235; rcp_delta_rom[ 98] = -11'sd327;  // m=1.765625  1/m=0.56637168
    rcp_base_rom[ 99] = 18'd73908; rcp_delta_rom[ 99] = -11'sd324;  // m=1.773438  1/m=0.56387665
    rcp_base_rom[100] = 18'd73584; rcp_delta_rom[100] = -11'sd321;  // m=1.781250  1/m=0.56140351
    rcp_base_rom[101] = 18'd73263; rcp_delta_rom[101] = -11'sd319;  // m=1.789062  1/m=0.55895197
    rcp_base_rom[102] = 18'd72944; rcp_delta_rom[102] = -11'sd315;  // m=1.796875  1/m=0.55652174
    rcp_base_rom[103] = 18'd72629; rcp_delta_rom[103] = -11'sd313;  // m=1.804688  1/m=0.55411255
    rcp_base_rom[104] = 18'd72316; rcp_delta_rom[104] = -11'sd311;  // m=1.812500  1/m=0.55172414
    rcp_base_rom[105] = 18'd72005; rcp_delta_rom[105] = -11'sd307;  // m=1.820312  1/m=0.54935622
    rcp_base_rom[106] = 18'd71698; rcp_delta_rom[106] = -11'sd306;  // m=1.828125  1/m=0.54700855
    rcp_base_rom[107] = 18'd71392; rcp_delta_rom[107] = -11'sd302;  // m=1.835938  1/m=0.54468085
    rcp_base_rom[108] = 18'd71090; rcp_delta_rom[108] = -11'sd300;  // m=1.843750  1/m=0.54237288
    rcp_base_rom[109] = 18'd70790; rcp_delta_rom[109] = -11'sd297;  // m=1.851562  1/m=0.54008439
    rcp_base_rom[110] = 18'd70493; rcp_delta_rom[110] = -11'sd295;  // m=1.859375  1/m=0.53781513
    rcp_base_rom[111] = 18'd70198; rcp_delta_rom[111] = -11'sd293;  // m=1.867188  1/m=0.53556485
    rcp_base_rom[112] = 18'd69905; rcp_delta_rom[112] = -11'sd290;  // m=1.875000  1/m=0.53333333
    rcp_base_rom[113] = 18'd69615; rcp_delta_rom[113] = -11'sd288;  // m=1.882812  1/m=0.53112033
    rcp_base_rom[114] = 18'd69327; rcp_delta_rom[114] = -11'sd285;  // m=1.890625  1/m=0.52892562
    rcp_base_rom[115] = 18'd69042; rcp_delta_rom[115] = -11'sd283;  // m=1.898438  1/m=0.52674897
    rcp_base_rom[116] = 18'd68759; rcp_delta_rom[116] = -11'sd281;  // m=1.906250  1/m=0.52459016
    rcp_base_rom[117] = 18'd68478; rcp_delta_rom[117] = -11'sd278;  // m=1.914062  1/m=0.52244898
    rcp_base_rom[118] = 18'd68200; rcp_delta_rom[118] = -11'sd276;  // m=1.921875  1/m=0.52032520
    rcp_base_rom[119] = 18'd67924; rcp_delta_rom[119] = -11'sd274;  // m=1.929688  1/m=0.51821862
    rcp_base_rom[120] = 18'd67650; rcp_delta_rom[120] = -11'sd272;  // m=1.937500  1/m=0.51612903
    rcp_base_rom[121] = 18'd67378; rcp_delta_rom[121] = -11'sd269;  // m=1.945312  1/m=0.51405622
    rcp_base_rom[122] = 18'd67109; rcp_delta_rom[122] = -11'sd268;  // m=1.953125  1/m=0.51200000
    rcp_base_rom[123] = 18'd66841; rcp_delta_rom[123] = -11'sd265;  // m=1.960938  1/m=0.50996016
    rcp_base_rom[124] = 18'd66576; rcp_delta_rom[124] = -11'sd263;  // m=1.968750  1/m=0.50793651
    rcp_base_rom[125] = 18'd66313; rcp_delta_rom[125] = -11'sd261;  // m=1.976562  1/m=0.50592885
    rcp_base_rom[126] = 18'd66052; rcp_delta_rom[126] = -11'sd259;  // m=1.984375  1/m=0.50393701
    rcp_base_rom[127] = 18'd65793; rcp_delta_rom[127] = -11'sd257;  // m=1.992188  1/m=0.50196078
end
