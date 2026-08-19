/* ---------------------------------------------------------------------------
 * main_evt.c — ZCU102 베어메탈: EvT(DVS128 제스처) 추론
 *
 * `fpga_nl/sw/main_nl.c` 와 골격은 같지만 흐름이 하나 늘었습니다 —
 * **타임스텝마다 입력을 새로 넣어야** 합니다. X/PIN 은 A_Mem 에 한 타임스텝분만
 * 들어가기 때문입니다 (20벌이면 24k 워드로 A_Mem 을 넘습니다).
 *
 *   1회      W / PB / PG / Step 을 DMA
 *   샘플마다  latinit → Z, LATV / bkv → BKV, N_TIME 쓰고 start
 *            T 번 { tok_req 대기 → X, PIN DMA → TOK_N → TOK_ACK }
 *            done 대기 → RES_CLASS
 *
 * DDR 이미지는 `board/load_ddr.tcl` 이 미리 올려 둡니다 (아래 주소 상수와
 * 한 벌이어야 합니다).
 * ------------------------------------------------------------------------- */
#include <stdio.h>
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xtime_l.h"

/* 베이스 주소 매크로 이름은 Vitis 버전/BD 이름에 따라 갈립니다 — `fpga_nl` 에서
   실제로 둘 다 본 적이 있어 셋 다 받아 둡니다. 주소는 BD 의 0xA001_0000 입니다. */
#if defined(XPAR_EVT_ACCEL_0_BASEADDR)
  #define ACC XPAR_EVT_ACCEL_0_BASEADDR
#elif defined(XPAR_EVT_ACCEL_0_S_AXI_BASEADDR)
  #define ACC XPAR_EVT_ACCEL_0_S_AXI_BASEADDR
#else
  #define ACC 0xA0010000U
#endif
#define DMA_ID XPAR_AXIDMA_0_DEVICE_ID

/* ---- 가속기 레지스터 (rtl/Evt_Accel.v 의 맵과 한 벌) ---- */
#define R_CTRL       0x000
#define R_STAT       0x004
#define R_NBODY      0x008
#define R_NTAIL      0x00C
#define R_NTIME      0x010
#define R_LSEL       0x014
#define R_LBASE      0x018
#define R_VER        0x01C
#define R_CYC        0x020
#define R_EPS        0x024
#define R_DBASE      0x028
#define R_DLEN       0x02C
#define R_WLOAD      0x030
#define R_TOKN       0x034
#define R_TACK       0x038
#define R_CLASS      0x03C
#define R_LOGIT      0x400

#define ST_DONE(s)   ((s) & 1u)
#define ST_TOKREQ(s) (((s) >> 14) & 1u)
#define ST_T(s)      (((s) >> 17) & 0x3Fu)

#define WR(o, v)  Xil_Out32(ACC + (o), (u32)(v))
#define RD(o)     Xil_In32(ACC + (o))

/* ---- 적재 목적지 ---- */
#define SEL_W 0u
#define SEL_A 1u
#define SEL_PB 2u
#define SEL_PG 3u
#define SEL_S 4u
#define SEL_POS 5u   /* pos enc 표 → PL BRAM (`Pos_Gather`) */

/* ---- A_Mem 영역 (data/schedule.json 과 한 벌) ---- */
#define R_X_BASE     0
#define R_PIDX_BASE  576     /* pos_idx — `Pos_Gather` 입력 */
#define R_PIN_BASE   580
#define R_LATV_BASE  2756
#define R_Z_BASE     3140
#define R_BKV_BASE   6728

/* ---- 실행 파라미터 (data/schedule.json) ---- */
#define N_BODY 118
#define N_TAIL 5
#define POS_ROWS 441         /* 21 x 21 */

/* ---- DDR 배치 (board/load_ddr.tcl 과 한 벌) ---- */
#define DDR_W        0x10000000u    /* wmem.bin      14000 x 32B */
#define DDR_PB       0x10100000u    /* pbmem.bin       869 x 32B */
#define DDR_PG       0x10110000u    /* pgmem.bin       208 x 32B */
#define DDR_STEP     0x10120000u    /* stepmem.bin     123 x 32B */
#define DDR_LATINIT  0x10130000u    /* latinit.bin     384 x 64B */
#define DDR_BKV      0x10140000u    /* bkv.bin          24 x 64B */
#define DDR_POSTBL   0x10150000u    /* posmem.bin      441 x 64B  → PL BRAM */
#define DDR_X        0x20000000u    /* amem_x.int16.bin        */
#define DDR_PIDX     0x30000000u    /* amem_pidx.int16.bin — pos_idx 만 */
#define DDR_INDEX    0x40000000u    /* board_index.int32.bin   */
#define DDR_SAMPLES  0x40100000u    /* board_samples.int32.bin */

#define W_WORDS   14000
#define PB_WORDS  869
#define PG_WORDS  208
#define S_WORDS   123
#define LAT_WORDS 384
#define BKV_WORDS 24

static XAxiDma dma;

/* 워드 하나가 몇 바이트인가 — A_Mem 만 512b 이고 나머지는 256b */
/* A_Mem(1) 과 POS 표(5) 만 512b — 나머지는 256b (로더와 한 벌) */
static u32 wbytes(u32 sel) { return (sel == SEL_A || sel == SEL_POS) ? 64u : 32u; }

/* DDR → 온칩. 로더를 arm 한 뒤 스트림을 흘립니다. */
static int load(u32 sel, u32 base, UINTPTR addr, u32 nword)
{
    u32 nb = nword * wbytes(sel);
    WR(R_LSEL, sel);
    WR(R_LBASE, base);                      /* 쓰는 순간 arm */
    /* 캐시는 켜 둡니다 (끄면 DDR 읽기가 느려집니다). DMA 가 읽기 전에
       그 구간만 내려씁니다 — `fpga_nl` 에서 검증된 방식입니다. */
    Xil_DCacheFlushRange(addr, nb);
    if (XAxiDma_SimpleTransfer(&dma, addr, nb, XAXIDMA_DMA_TO_DEVICE)
        != XST_SUCCESS) {
        xil_printf("  [DMA] sel=%u transfer failed\r\n", (unsigned)sel);
        return -1;
    }
    while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) { }
    /* 로더가 마지막 워드를 쓰는 데 몇 클럭 걸립니다 (시뮬에서도 4클럭 필요) */
    { volatile int i; for (i = 0; i < 64; i++) { } }
    u32 got = RD(R_WLOAD);
    if (got != nword) {
        xil_printf("  [DMA] sel=%u : %u words loaded (expected %u)\r\n",
                   (unsigned)sel, (unsigned)got, (unsigned)nword);
        return -1;
    }
    return 0;
}

int main(void)
{
    xil_printf("\r\n===== EvT DVS128_10 accelerator =====\r\n");
    u32 ver = RD(R_VER);
    xil_printf("VERSION = %08x  (expect 45565401)\r\n", (unsigned)ver);
    if (ver != 0x45565401u) { xil_printf("** check the bitstream\r\n"); return 1; }

    XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_ID);
    if (!cfg || XAxiDma_CfgInitialize(&dma, cfg) != XST_SUCCESS) {
        xil_printf("** DMA init failed\r\n"); return 1;
    }
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);

    WR(R_NBODY, N_BODY);
    WR(R_NTAIL, N_TAIL);
    WR(R_EPS,   0x3727c5acu);            /* 1e-5, 골든과 같은 비트 */

    xil_printf("-- loading constants\r\n");
    /* pos enc 표는 **샘플과 무관**해서 여기서 한 번만 올립니다. 전에는 호스트가
       타임스텝마다 펴서 보냈고 그 이미지가 96.7 MB 였습니다 — 이제 27.6 KB 표가
       PL BRAM 에 있고 오는 것은 `pos_idx`(타임스텝당 최대 246 B) 뿐입니다. */
    if (load(SEL_W,   0, (UINTPTR)DDR_W,      W_WORDS)  ||
        load(SEL_PB,  0, (UINTPTR)DDR_PB,     PB_WORDS) ||
        load(SEL_PG,  0, (UINTPTR)DDR_PG,     PG_WORDS) ||
        load(SEL_S,   0, (UINTPTR)DDR_STEP,   S_WORDS)  ||
        load(SEL_POS, 0, (UINTPTR)DDR_POSTBL, POS_ROWS)) return 1;
    xil_printf("   W %u / PB %u / PG %u / Step %u / POS %u words\r\n",
               W_WORDS, PB_WORDS, PG_WORDS, S_WORDS, POS_ROWS);

    /* 인덱스/샘플 표는 XSCT 가 JTAG 로 직접 써 넣은 것이라 **캐시에 없습니다**.
       무효화하지 않으면 옛 캐시 라인을 읽습니다. */
    Xil_DCacheInvalidateRange((UINTPTR)DDR_INDEX,   6 * 4 * 8192);
    Xil_DCacheInvalidateRange((UINTPTR)DDR_SAMPLES, 4 * 4 * 512);
    const volatile int *idx = (const volatile int *)(UINTPTR)DDR_INDEX;
    const volatile int *sam = (const volatile int *)(UINTPTR)DDR_SAMPLES;

    int n_sample = 0;
    while (sam[n_sample * 4 + 1] > 0 && n_sample < 264) n_sample++;
    xil_printf("-- %d samples\r\n", n_sample);

    int correct = 0;
    XTime t0, t1;
    XTime_GetTime(&t0);

    for (int s = 0; s < n_sample; s++) {
        int ts0 = sam[s * 4 + 0];
        int T   = sam[s * 4 + 1];
        int lab = sam[s * 4 + 2];

        /* latent 는 샘플마다 초기값에서 다시 시작합니다 */
        if (load(SEL_A, R_Z_BASE,    (UINTPTR)DDR_LATINIT, LAT_WORDS) ||
            load(SEL_A, R_LATV_BASE, (UINTPTR)DDR_LATINIT, LAT_WORDS) ||
            load(SEL_A, R_BKV_BASE,  (UINTPTR)DDR_BKV,     BKV_WORDS)) return 1;

        WR(R_NTIME, T);
        WR(R_CTRL, 1);                       /* start */

        for (int t = 0; t < T; t++) {
            u32 st;
            do { st = RD(R_STAT); } while (!ST_TOKREQ(st));

            const volatile int *e = idx + (ts0 + t) * 6;
            u32 x_off = (u32)e[0], x_n = (u32)e[1];
            u32 p_off = (u32)e[2], p_n = (u32)e[3];   /* pos_idx : 타일당 1워드 */
            u32 ntok  = (u32)e[4];

            if (load(SEL_A, R_X_BASE,    (UINTPTR)DDR_X    + (UINTPTR)x_off * 64u, x_n) ||
                load(SEL_A, R_PIDX_BASE, (UINTPTR)DDR_PIDX + (UINTPTR)p_off * 64u, p_n))
                return 1;

            WR(R_TOKN, ntok);
            WR(R_TACK, 1);
        }

        u32 st;
        do { st = RD(R_STAT); } while (!ST_DONE(st));
        int cls = (int)(RD(R_CLASS) & 0xF);
        if (cls == lab) correct++;

        /* 샘플이 적을 때는 **하나씩** 찍습니다. 어느 샘플이 틀리는지 알아야
           논리 버그(항상 같은 샘플)와 타이밍(돌릴 때마다 다름)을 가릅니다. */
        if (n_sample <= 32)
            xil_printf("   s%-3d T=%-2d label %d  pred %d  %s   %u cyc\r\n",
                       s, T, lab, cls, (cls == lab) ? "ok" : "MISS",
                       (unsigned)RD(R_CYC));
        else if ((s + 1) % 20 == 0 || s + 1 == n_sample)
            xil_printf("   %3d/%3d  correct %3d  (%d.%02d %%)\r\n",
                       s + 1, n_sample, correct,
                       correct * 100 / (s + 1), (correct * 10000 / (s + 1)) % 100);
    }

    XTime_GetTime(&t1);
    u64 us = ((u64)(t1 - t0)) * 1000000ull / COUNTS_PER_SECOND;

    /* 마지막 샘플의 로짓 — 오답이 간발의 차인지(타이밍/양자화) 아니면
       완전히 다른지(논리) 구분하는 데 씁니다. */
    if (n_sample <= 32) {
        xil_printf("\r\nlast sample logits:\r\n");
        for (int c = 0; c < 10; c++)
            xil_printf("   [%d] %d\r\n", c, (int)RD(R_LOGIT + 4 * c));
    }

    xil_printf("\r\n===== result =====\r\n");
    xil_printf("accuracy : %d/%d = %d.%02d %%\r\n", correct, n_sample,
               correct * 100 / n_sample, (correct * 10000 / n_sample) % 100);
    xil_printf("golden   : 97.34 %% (257/264)\r\n");
    xil_printf("time     : %u ms  (%u us per sample)\r\n",
               (unsigned)(us / 1000), (unsigned)(us / (n_sample ? n_sample : 1)));
    xil_printf("last run : %u cycles\r\n", (unsigned)RD(R_CYC));
    return 0;
}
