// ============================================================================
//  gelu_pwl.v  --  GELU activation unit (Residual + PWL-LUT method)
// ----------------------------------------------------------------------------
//  GELU:  G(x) = x * Phi(x),   Phi = standard-normal CDF
//
//  Residual formulation (a = |x|, a >= 0):
//     R(a) = a - G(a) = a*(1 - Phi(a)) = (a/2)*erfc(a/sqrt2)
//     x >= 0 : G(x) =  x - R(a)
//     x <  0 : G(x) =    - R(a)          ( G(-a) = -R(a) )
//
//  R(a) is a small, bounded bump (0 <= R <= 0.17, ->0 for a>~4) so it is cheap
//  to approximate with a uniform piecewise-linear LUT.
//
//  Fixed-point:
//     x, y  : signed Q5.10  (16-bit, res = 2^-10)
//     R     : internal Q?.14 (base14/delta14 stored in gelu_lut.vh)
//     a in [0,4)  -> seg = a[11:6] (64 seg, width 2^-4), frac = a[5:0]
//     R14  = base14[seg] + (delta14[seg]*frac >>> 6)      (linear interp)
//     a >= 4 -> R = 0  (natural saturation: G=x for x>=0, G=0 for x<0)
//
//  Pipeline latency = 3 clocks (input reg -> lookup/interp -> combine).
// ============================================================================
`timescale 1ns/1ps

module gelu_pwl #(
    parameter W  = 16,   // data width
    parameter QF = 11,   // fractional bits of x/y  (Q(W-1-QF).QF)
    parameter FR = 14    // fractional bits of internal R
)(
    input                     clk,
    input                     rst_n,
    input                     in_valid,
    input      signed [W-1:0] x,        // Q(W-1-QF).QF
    output reg                out_valid,
    output reg signed [W-1:0] y         // Q(W-1-QF).QF  ~= GELU(x)
);
    localparam integer K        = 4;               // segment width = 2^-K = 0.0625
    localparam integer A_INT    = 4;               // A_MAX = 4.0
    localparam integer NSEG     = A_INT << K;      // = 64  (A_MAX/width), QF 무관
    localparam integer IDXBITS  = $clog2(NSEG);    // = 6   (segment index bits)
    localparam integer FRACBITS = QF - K;          // in-segment frac bits (QF-4)
    localparam integer A_MAX_FIX= (A_INT << QF);   // a>=4 -> R(a)=0
    localparam integer RSH      = FR - QF;          // Q(FR) -> Q(QF)

    // ---- LUT (base14_rom[64], delta14_rom[64]) ----
    `include "gelu_lut.vh"

    // ======================= Stage 1 : decode + LUT lookup ==================
    // magnitude a = |x|  (17-bit to hold |-32768| = 32768)
    wire signed [W:0]  x_ext = {x[W-1], x};
    wire        [W:0]  a     = x[W-1] ? (~x_ext + 1'b1) : x_ext;   // |x|, unsigned
    wire               ge4   = (a >= A_MAX_FIX);                   // a >= 4.0

    wire [IDXBITS-1:0]  seg  = a[FRACBITS+IDXBITS-1 : FRACBITS];   // a[11:6]
    wire [FRACBITS-1:0] frac = a[FRACBITS-1 : 0];                  // a[5:0]

    reg                 vld_s1, sign_s1, ge4_s1;
    reg  signed [W-1:0] x_s1;
    reg  signed [W-1:0] base_s1, delta_s1;
    reg  [FRACBITS-1:0] frac_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_s1 <= 1'b0; sign_s1 <= 1'b0; ge4_s1 <= 1'b0;
            x_s1 <= 0; base_s1 <= 0; delta_s1 <= 0; frac_s1 <= 0;
        end else begin
            vld_s1   <= in_valid;
            x_s1     <= x;
            sign_s1  <= x[W-1];
            ge4_s1   <= ge4;
            frac_s1  <= frac;
            base_s1  <= base14_rom [seg];
            delta_s1 <= delta14_rom[seg];
        end
    end

    // ======================= Stage 2 : linear interpolation =================
    // interp = delta*frac >>> FRACBITS  (frac 은 세그먼트를 2^FRACBITS 등분한 위치)
    // r14 = base + interp   (R in Q(FR)=Q14) ;  r_out = round(r14) -> Q(QF)
    wire signed [W+FRACBITS:0] prod   = delta_s1 * $signed({1'b0, frac_s1});
    wire signed [W:0]          interp = prod >>> FRACBITS;                 // R in Q(FR)
    wire signed [W:0]          r14    = ge4_s1 ? {(W+1){1'b0}} : (base_s1 + interp);
    wire signed [W:0]          r_out  = (r14 + (1 << (RSH-1))) >>> RSH;    // Q(FR)->Q(QF), round

    reg                 vld_s2, sign_s2;
    reg  signed [W-1:0] x_s2;
    reg  signed [W-1:0] r_out_s2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_s2 <= 1'b0; sign_s2 <= 1'b0; x_s2 <= 0; r_out_s2 <= 0;
        end else begin
            vld_s2   <= vld_s1;
            sign_s2  <= sign_s1;
            x_s2     <= x_s1;
            r_out_s2 <= r_out[W-1:0];
        end
    end

    // ======================= Stage 3 : combine ==============================
    //   x >= 0 : y = x - R ;   x < 0 : y = -R   (모두 Q(QF))
    wire signed [W-1:0] y_next = sign_s2 ? (-r_out_s2) : (x_s2 - r_out_s2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0; y <= 0;
        end else begin
            out_valid <= vld_s2;
            y         <= y_next;
        end
    end

endmodule
