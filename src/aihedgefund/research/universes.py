"""Shared research universes (deterministic tickers, no live fetch).

`BROAD_LIQUID_CANDIDATE_UNIVERSE` is the S&P-500-near ticker list used by
the Phase-2 momentum-breadth probe. It is today's membership → mild
survivorship bias; suitable for relative diagnostics only, not as a
tradeable alpha claim.

`SMALL_CAP_CANDIDATE_UNIVERSE` is a liquid US small/mid snapshot outside
that broad list (Russell-2000-segment style). Survivorship bias is stronger
for small caps; relative diagnosis only.
"""

from __future__ import annotations

from typing import Final

SURVIVORSHIP_BIAS_NOTE: Final[str] = (
    "BROAD_LIQUID_CANDIDATE_UNIVERSE is current index membership "
    "(survivorship bias). Results are a relative 50-vs-broad diagnosis, "
    "not a tradeable alpha proof."
)

# fmt: off
BROAD_LIQUID_CANDIDATE_UNIVERSE: Final[tuple[str, ...]] = (
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE",
    "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL",
    "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE",
    "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK",
    "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY",
    "BDX", "BEN", "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR",
    "BLK", "BMY", "BNY", "BR", "BRK-B", "BRO", "BSX", "BX",
    "BXP", "C", "CAH", "CARR", "CASY", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL", "CLX",
    "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF",
    "COHR", "COIN", "COO", "COP", "COR", "COST", "CPAY", "CPRT",
    "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL",
    "DASH", "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX",
    "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW",
    "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
    "EBAY", "ECHO", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EME", "EMR", "EOG", "EQIX", "EQR", "EQT", "ERIE",
    "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS",
    "FDX", "FDXF", "FE", "FFIV", "FICO", "FIS", "FISV", "FITB",
    "FIX", "FLEX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS",
    "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN",
    "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD",
    "HIG", "HII", "HLT", "HON", "HONA", "HOOD", "HPE", "HPQ",
    "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU",
    "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB",
    "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LII", "LIN", "LITE", "LLY", "LMT", "LNT", "LOW",
    "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA",
    "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST", "MO",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH", "MRVL", "MS",
    "MSCI", "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ",
    "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW",
    "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS",
    "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG",
    "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "PPG",
    "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL",
    "Q", "QCOM", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNDK", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE",
    "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF",
    "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL",
    "TER", "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL",
    "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN",
    "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER",
    "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VEEV", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN",
    "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
    "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN", "XEL",
    "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS",
)
# fmt: on

SMALL_CAP_SURVIVORSHIP_BIAS_NOTE: Final[str] = (
    "SMALL_CAP_CANDIDATE_UNIVERSE is a static snapshot of liquid US small/mid "
    "names outside the S&P-near broad list (Russell-2000-segment style). Current "
    "membership plus small-cap delistings intensify survivorship bias vs large-cap; "
    "results are a relative universe diagnosis only, not a tradeable alpha proof."
)

# fmt: off
SMALL_CAP_CANDIDATE_UNIVERSE: Final[tuple[str, ...]] = (
    "AAOI", "AAT", "ABCB", "ABG", "ABR", "ACA", "ACAD", "ACHC",
    "ACIW", "ACLS", "ADC", "ADUS", "AEO", "AFG", "AGIO", "AGM",
    "AGX", "AHH", "AIN", "AIR", "AIT", "AKR", "AL", "ALGT",
    "ALKS", "ALKT", "ALSN", "ALX", "AMBA", "AMED", "AMKR", "AMN",
    "AMPH", "AMR", "AN", "ANDE", "ANF", "ANIP", "AORT", "AOSL",
    "APAM", "APG", "APLS", "APOG", "AR", "ARCB", "AROC", "ARWR",
    "ASB", "ASGN", "ASH", "ASO", "ASTE", "ASTH", "ASTS", "AT",
    "ATGE", "ATI", "ATKR", "ATNI", "ATR", "ATRC", "AUB", "AVA",
    "AVAV", "AVNT", "AVT", "AWI", "AX", "AXSM", "AXTA", "AZZ",
    "BANC", "BANF", "BANR", "BCPC", "BDC", "BE", "BEAM", "BELFB",
    "BFH", "BGC", "BGS", "BHE", "BHF", "BHLB", "BJ", "BKE",
    "BKU", "BL", "BLBD", "BLKB", "BMI", "BOH", "BOOT", "BOX",
    "BPMC", "BPOP", "BRBR", "BRC", "BRKL", "BRY", "BTU", "BURL",
    "BUSE", "BV", "BXC", "BY", "BYD", "CABO", "CAC", "CADE",
    "CAL", "CALM", "CALX", "CARG", "CARS", "CASH", "CATY", "CAVA",
    "CBRL", "CBU", "CCOI", "CCS", "CDE", "CDP", "CDRE", "CENTA",
    "CENX", "CERT", "CEVA", "CFR", "CGNX", "CHCO", "CHE", "CHEF",
    "CHRD", "CHWY", "CIVI", "CLF", "CLH", "CLS", "CLSK", "CMC",
    "CNO", "CNX", "CNXC", "COKE", "COLB", "COLL", "COLM", "CORT",
    "COUR", "CPRX", "CRC", "CRGY", "CRI", "CRK", "CRNX", "CROX",
    "CRSP", "CRUS", "CSGS", "CSWC", "CTKB", "CTRE", "CTS", "CUBI",
    "CVBF", "CVCO", "CVI", "CVLT", "CW", "CWAN", "CWK", "CWT",
    "CYTK", "DAN", "DAR", "DBX", "DCOM", "DDS", "DEI", "DFH",
    "DGII", "DHT", "DIOD", "DK", "DKS", "DNLI", "DOCN", "DOCS",
    "DOLE", "DORM", "DRH", "DT", "DV", "DXC", "DY", "EAT",
    "EBC", "ECVT", "EEFT", "EGBN", "EHC", "ELAN", "ELF", "ELME",
    "ELS", "EMBC", "ENOV", "ENS", "ENSG", "ENVA", "ENVX", "EPAC",
    "EPC", "EPRT", "EQH", "ESAB",
)
# fmt: on
