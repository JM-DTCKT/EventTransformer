// -----------------------------------------------------------------------------
// Fp32_Add : IEEE-754 단정밀도 덧셈, round-to-nearest-even
//
// LayerNorm 의 `mean` 과 `mean(centered²)` 를 위한 블록입니다 (hw_flow.md §2.6).
//
// ## 왜 bf16 이 아니라 fp32 로 누산하는가
//
// bf16 은 가수가 8비트뿐이라 128개를 bf16 으로 이어 더하면 오차가 누적돼 **분산이
// 망가집니다**. 그래서 누산만 fp32 로 하고 **마지막에 한 번만** bf16 으로 내립니다
// (`quant_lib/hw_quant.py` 와 같은 구조).
//
// ## 순서가 정의되어 있어야 합니다
//
// 골든(`nonlinear_script/quantize.py`)은 k = 0,1,…,E-1 **순차** 누산입니다.
// `torch.sum` 은 pairwise 라 순서가 다르고, 실제로 이 순서를 못박자 정확도가
// 98.52% → 98.46% 로 움직였습니다. 부동소수 덧셈은 결합법칙이 성립하지 않으므로
// **RTL 과 골든이 같은 순서**여야 비트 단위로 맞습니다.
//
// 이미지 32장은 레인 병렬이라 루프는 k 축만 돕니다 → E 사이클/패스.
//
// ## residual 도 이 블록입니다
//
// `x = bf16(x + h)` 는 bf16→fp32 확장(가수 0 채움, 공짜) → fp32 덧셈 → bf16 반올림.
// 골든이 정확히 그렇게 정의돼 있습니다.
//
// ## 구현
//
//   ① 언팩 / 크기 비교 / 지수 정렬 (guard·round·sticky 3비트)
//   ② 가감산 → 정규화(LZC) → RNE 반올림 → 팩
//
// denormal 은 0 으로 플러시합니다 (`Requant/Int32_To_Bf16.v` 등 기존 bf16 블록과
// 같은 정책). Inf/NaN 은 그대로 통과시킵니다 — 이 데이터패스엔 오지 않지만 시뮬에서
// X 가 번지지 않게 하기 위한 것입니다.
//
// 파이프라인 2단, 처리율 1/사이클.
// -----------------------------------------------------------------------------
module Fp32_Add (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         out_valid,
    output reg  [31:0] y
);
    // ---------------- 언팩 ----------------
    wire        sa = a[31],      sb = b[31];
    wire [7:0]  ea = a[30:23],   eb = b[30:23];
    wire [22:0] ma = a[22:0],    mb = b[22:0];

    wire a_zero = (ea == 8'd0);                 // denormal 포함 → 0
    wire b_zero = (eb == 8'd0);
    wire a_spec = (ea == 8'hFF);                // Inf / NaN
    wire b_spec = (eb == 8'hFF);

    // 크기 비교 (지수+가수를 한 덩어리로)
    wire a_big = ({ea, ma} >= {eb, mb});

    wire        s_big = a_big ? sa : sb,   s_sml = a_big ? sb : sa;
    wire [7:0]  e_big = a_big ? ea : eb,   e_sml = a_big ? eb : ea;
    wire [22:0] m_big = a_big ? ma : mb,   m_sml = a_big ? mb : ma;
    wire        z_sml = a_big ? b_zero : a_zero;  // 작은 수가 0인지 나타내는 bit

    // 24비트 유효숫자를 3비트(Guard,Round,Sticky) 자리만큼 올려 27비트로, 
    wire [26:0] sig_big = {1'b1, m_big, 3'b000};
    wire [26:0] sig_sml = z_sml ? 27'd0 : {1'b1, m_sml, 3'b000};

    wire [7:0]  d = e_big - e_sml;

    // ---------------- 정렬 (sticky 보존) ----------------
    reg  [26:0] sml_sh;
    reg         sticky;
    integer     i;
    always @* begin
        if (d >= 8'd27) begin
            sml_sh = 27'd0;
            sticky = |sig_sml;
        end else begin
            sml_sh = sig_sml >> d;
            sticky = |(sig_sml & ((27'd1 << d) - 27'd1)); // 밀려난 하위 d bit 중 하나라도 1이 있는지 나타내는 bit
        end
    end
    wire [26:0] sml_al = sml_sh | {26'd0, sticky};

    wire same_sign = (s_big == s_sml);
    wire [27:0] sum = same_sign ? ({1'b0, sig_big} + {1'b0, sml_al})
                                : ({1'b0, sig_big} - {1'b0, sml_al});

    // ---------------- stage 1 ----------------
    reg         v1, s1, spec1, bigzero1;
    reg [31:0]  spec_val1;
    reg [7:0]   e1;
    reg [27:0]  sum1;

    always @(posedge clk) begin
        if (rst) begin
            v1 <= 1'b0; spec1 <= 1'b0;
        end else begin
            v1   <= in_valid;
            s1   <= s_big;
            e1   <= e_big;
            sum1 <= sum;
            // 특수값: Inf/NaN 이 하나라도 있으면 그걸 그대로.  둘 다 0 이면 +0.
            spec1     <= a_spec | b_spec | (a_zero & b_zero);
            spec_val1 <= a_spec ? a : (b_spec ? b : 32'd0);
            bigzero1  <= a_zero & b_zero;
        end
    end

    // ---------------- 정규화 : leading-zero count ----------------
    // 선행 1 을 **bit 26** 으로 올리는 시프트량입니다 (mant 가 norm[26:3] 이므로).
    // 자리올림(bit 27)은 아래에서 따로 처리하므로 여기서는 26..0 만 훑습니다.
    reg [4:0] lz;
    reg       lz_found;
    always @* begin
        lz = 5'd0; lz_found = 1'b0;
        for (i = 26; i >= 0; i = i - 1)
            if (sum1[i] && !lz_found) begin
                lz = 5'd26 - i[4:0];
                lz_found = 1'b1;
            end
    end

    reg  [27:0] norm;
    reg  [8:0]  e_norm;
    always @* begin
        if (sum1[27]) begin                           // 자리올림 → 오른쪽 1
            norm   = {1'b0, sum1[27:1]} | {27'd0, sum1[0]};   // sticky 유지
            e_norm = {1'b0, e1} + 9'd1;
        end else begin
            norm   = sum1 << lz;                      // 선행 1 을 bit26 으로
            e_norm = {1'b0, e1} - {4'd0, lz};
        end
    end

    // ---------------- RNE 반올림 ----------------
    wire [23:0] mant  = norm[26:3];                   // 1.xxx (24비트)
    wire        g     = norm[2], r = norm[1], st = norm[0];
    wire        up    = g & (r | st | mant[0]);
    wire [24:0] mant_r = {1'b0, mant} + {24'd0, up};
    wire        carry  = mant_r[24];
    wire [23:0] mant_f = carry ? mant_r[24:1] : mant_r[23:0];
    wire [8:0]  e_fin  = e_norm + {8'd0, carry};

    wire zero_res = (sum1 == 28'd0) || (e_fin[8] == 1'b1) || (e_fin == 9'd0);
    wire ovf      = (e_fin >= 9'd255) && !e_fin[8];

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0; y <= 32'd0;
        end else begin
            out_valid <= v1;
            if (v1) begin
                if (spec1)        y <= bigzero1 ? 32'd0 : spec_val1;
                else if (ovf)     y <= {s1, 8'hFF, 23'd0};        // Inf
                else if (zero_res) y <= 32'd0;                     // 언더플로 → +0
                else              y <= {s1, e_fin[7:0], mant_f[22:0]};
            end
        end
    end
endmodule
