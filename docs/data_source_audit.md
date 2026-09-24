# Data-source audit

Audit date: 23 September 2026. This document records the sources selected before validation modelling. Any substitution must be recorded in the protocol amendment log.

## Connectivity smoke test

All configured identifiers were queried successfully for January 2025 on 23 September 2026:

| Series | Observations returned |
|---|---:|
| EUR/USD | 22 |
| AAA 2-year yield | 22 |
| AAA 10-year yield | 22 |
| All-ratings 10-year yield | 22 |
| New CISS | 23 |
| EURO STOXX 50 | 21 |

This verifies identifiers and basic parsing only. Full-period coverage, gaps, units, and staleness remain to be audited after the frozen raw download.

The first full pull found that `^STOXX50E` begins on 30 March 2007. The common sample was therefore amended to start on that date before any feature construction or market-outcome analysis. The original protocol is preserved, and the reason is recorded in `protocol/amendments.md`. A long-lived EUR-denominated ETF (`EXW1`) was considered, but the selected provider did not supply its missing 2005–2007 history; mixing ETF and index returns was rejected.

The ECB EUR/USD response also contains 25 dated rows without observation values. They do not remove any Friday model week because nearby valid observations exist. The downloader records the count in `manifest.json` and excludes these unusable rows from the normalized raw Parquet file.

| Series | Source | Coverage needed | Reuse decision | Project decision |
|---|---|---:|---|---|
| EURO STOXX 50 | Yahoo Finance, `^STOXX50E` | 2007-03–2025-12 daily | Unofficial third-party feed; raw values will not be redistributed | Prototype input behind a replaceable adapter |
| EUR/USD | ECB EXR, `D.USD.EUR.SP00.A` | 2005-09–2025-12 daily | Official ECB series; attribution required | Core input |
| AAA 2-year yield | ECB YC, `B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y` | 2005-09–2025-12 business daily | Official ECB curve; attribution required | Core input |
| AAA 10-year yield | ECB YC, `B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y` | 2005-09–2025-12 business daily | Official ECB curve; attribution required | Core input |
| All-ratings 10-year yield | ECB YC, `B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y` | 2005-09–2025-12 business daily | Official ECB curve; attribution required | Core fragmentation proxy |
| New CISS | ECB CISS, `D.U2.Z0Z.4F.EC.SS_CIN.IDX` | 2005-09–2025-12 daily | Official ECB index; not independent of market inputs | External validation only |

## Sources deliberately excluded from version 1

- VSTOXX and additional STOXX sector indices: full history and redistribution are licensed.
- ICE BofA euro credit spreads: third-party rights and shrinking free historical coverage make the series unsuitable for the frozen core sample.
- ECB deposit-facility rate: it is a step function and would create artificial persistence at weekly frequency.
- Gold and Brent: useful robustness variables, but not necessary to answer the primary question.
- ECB announcement dates: reserved for a later descriptive extension; no causal event-study claim belongs in the core project.

## Acceptance checks before feature construction

1. Each core series must cover at least 95% of the intended common weekly sample.
2. Dates must be unique and increasing within each series.
3. Units must match the catalogue before any basis-point conversion.
4. Missing market observations must remain missing until the weekly sampling rule is applied.
5. A weekly observation older than five calendar days receives a stale flag and is excluded.
6. The raw-download manifest must record the URL or symbol, retrieval time, row count, and SHA-256 hash.

## Primary documentation

- ECB API examples: <https://data.ecb.europa.eu/help/api/data-examples>
- ECB yield-curve catalogue: <https://data.ecb.europa.eu/data/datasets/YC>
- ECB New CISS series: <https://data.ecb.europa.eu/data/datasets/CISS/CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX>
- ECB statistics reuse policy: <https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html>
- iShares EXW1 fund information: <https://www.ishares.com/ch/individual/en/literature/fact-sheet/exw1-ishares-core-euro-stoxx-50-ucits-etf-%28de%29-fund-fact-sheet-en-ch.pdf>
