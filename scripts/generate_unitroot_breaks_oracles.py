"""Provenance and cross-checks for the break/bubble critical-value tables.

QA-only; not imported by the runtime. Run once with network access:

    python scripts/generate_unitroot_breaks_oracles.py tables --cache /tmp/breaks-cache \
        --write src/openecon/econometrics/unitroot/break_tables.py
    python scripts/generate_unitroot_breaks_oracles.py psy --replications 5000
    python scripts/generate_unitroot_breaks_oracles.py size --datasets 400 --boot 199

``tables`` downloads every primary source listed below, transcribes the tables
programmatically, cross-checks independent transcriptions entry by entry and
(optionally) writes the runtime module. It refuses to write when two
transcriptions disagree.

Sources (retrieved 2026-10-10)
------------------------------
* Bai--Perron supF(k), supF(l+1|l), UDmax/WDmax: Pierre Perron's GAUSS code
  ``BRCODE.SRC`` (procedures getcv1/getcv2/getdmax, October 2004 version,
  https://blogs.bu.edu/perron/files/2020/05/m-Break.zip) and the CSV tables of
  the CRAN package mbreaks 1.0.1 (Nguyen, Yamamoto and Perron;
  https://github.com/cran/mbreaks/tree/master/R/{supF,supF_next,Dmax}). Both
  reproduce Bai and Perron (1998, Econometrica 66, Table I) and Bai and Perron
  (2003, Econometrics Journal 6, Tables 1-2).
* Perron--Vogelsang (1992) model 0 (mean shift, non-trending data) asymptotic
  distribution of min t_alpha (IO/AO): percentiles 1..99 of function f1 in
  Tim Vogelsang's ``PVAL/FF1.C`` (6/17/93), distributed with the Perron--Ng
  RATS procedures (https://blogs.bu.edu/perron/files/2020/05/unit-root.zip).
* Lee--Strazicich: one break, Lee and Strazicich (2013), Economics Bulletin
  33(4), Table 1, p. 2488 (https://www.accessecon.com/Pubs/EB/2013/Volume33/EB-13-V33-I4-P234.pdf),
  typed below and cross-checked against Aptech tspdlib ``src/lm_1br.src``;
  two breaks, Lee and Strazicich (2003), REStat 85(4), Table 2 as transcribed
  in tspdlib ``src/lm_2br.src`` (https://github.com/aptech/tspdlib); the
  journal text itself was not retrievable.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import urllib.request
import zipfile

PERRON_ZIP = "https://blogs.bu.edu/perron/files/2020/05/m-Break.zip"
UNITROOT_ZIP = "https://blogs.bu.edu/perron/files/2020/05/unit-root.zip"
MBREAKS = "https://raw.githubusercontent.com/cran/mbreaks/master/R/{kind}/cv_{index}.csv"
TSPDLIB = "https://raw.githubusercontent.com/aptech/tspdlib/master/src/{name}"
EPSILONS = (0.05, 0.10, 0.15, 0.20, 0.25)
LEVELS = ("10%", "5%", "2.5%", "1%")

# Lee and Strazicich (2013), Table 1, p. 2488, typed from the journal PDF.
LS_2013_A = (-4.239, -3.566, -3.211)
LS_2013_C = {0.1: (-5.11, -4.50, -4.21), 0.2: (-5.07, -4.47, -4.20),
             0.3: (-5.15, -4.45, -4.18), 0.4: (-5.05, -4.50, -4.18),
             0.5: (-5.11, -4.51, -4.17)}


def fetch(url: str, cache: str) -> bytes:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, re.sub(r"[^A-Za-z0-9._-]", "_", url))
    if not os.path.exists(path):
        request = urllib.request.Request(url, headers={"User-Agent": "openecon-qa"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        with open(path, "wb") as handle:
            handle.write(data)
    with open(path, "rb") as handle:
        return handle.read()


def gauss_tables(source: str, proc: str) -> dict[tuple[float, int], list[list[float]]]:
    """Tables cv={...} of a GAUSS proc keyed by (eps1, signif)."""
    start = source.index(f"proc {proc}(")
    end = source.index("endp;", start)
    body = source[start:end]
    tables, eps = {}, None
    for token in re.finditer(r"eps1\s*==\s*(\.\d+)|signif\s*==\s*(\d)|cv\s*=\s*\{(.*?)\}",
                             body, re.S):
        if token.group(1):
            eps = float(token.group(1))
        elif token.group(2):
            signif = int(token.group(2))
        else:
            rows = [[float(v) for v in row.split()] for row in token.group(3).split(",")]
            tables[(eps, signif)] = rows
    return tables


def mbreaks_tables(kind: str, cache: str) -> dict[tuple[float, int], list[list[float]]]:
    tables = {}
    for index, eps in enumerate(EPSILONS, start=1):
        text = fetch(MBREAKS.format(kind=kind, index=index), cache).decode()
        rows = [[float(v) for v in line.split(",")] for line in text.split() if line.strip()]
        for signif in range(1, 5):
            tables[(eps, signif)] = rows[(signif - 1) * 10:signif * 10]
    return tables


def compare(name: str, a: dict, b: dict) -> list[str]:
    problems = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            problems.append(f"{name} {key}: missing in one transcription")
            continue
        for q, (ra, rb) in enumerate(zip(a[key], b[key], strict=True), start=1):
            width = min(len(ra), len(rb))
            if len(ra) != len(rb):
                problems.append(f"{name} {key} q={q}: widths {len(ra)} vs {len(rb)}")
            for j in range(width):
                if abs(ra[j] - rb[j]) > 1e-9:
                    problems.append(f"{name} eps={key[0]} level={LEVELS[key[1]-1]} q={q} "
                                    f"col={j+1}: {ra[j]} vs {rb[j]}")
    return problems


def vogelsang_f1(cache: str) -> list[float]:
    archive = zipfile.ZipFile(io.BytesIO(fetch(UNITROOT_ZIP, cache)))
    name = next(n for n in archive.namelist() if n.upper().endswith("PVAL/FF1.C"))
    text = archive.read(name).decode("latin-1")
    body = text[text.index("double *f1("):text.index("double *f2(")]
    values = {int(i): float(v) for i, v in re.findall(r"\*\(x\+(\d+)\)=\(([-\d.]+)\)", body)}
    return [values[i] for i in range(1, 100)]


def tspdlib_ls(cache: str) -> tuple[tuple, dict, tuple, dict]:
    one = fetch(TSPDLIB.format(name="lm_1br.src"), cache).decode()
    two = fetch(TSPDLIB.format(name="lm_2br.src"), cache).decode()
    a1 = tuple(float(v) for v in re.search(r"cv = (-[\d.]+)~(-[\d.]+)~(-[\d.]+);", one).groups())
    rows = re.search(r"crit = \{(.*?)\};", one, re.S).group(1).split(",")
    c1 = {round(0.1 * (i + 1), 1): tuple(float(v) for v in row.split())
          for i, row in enumerate(rows)}
    a2 = tuple(float(v) for v in re.search(r"cv = (-[\d.]+)~(-[\d.]+)~(-[\d.]+);", two).groups())
    mats = {}
    for level in ("1", "5", "10"):
        block = re.search(rf"mat_cv{level}\s*=\s*\{{(.*?)\}};", two, re.S).group(1)
        mats[level] = [[float(v) for v in row.split()] for row in block.split(",")]
    cells = {}
    for i, lam1 in enumerate((0.2, 0.4, 0.6)):
        for j, lam2 in enumerate((0.4, 0.6, 0.8)):
            if lam1 < lam2:
                cells[(lam1, lam2)] = tuple(mats[level][i][j] for level in ("1", "5", "10"))
    return a1, c1, a2, cells


def literal(value, indent: int = 4) -> str:
    return repr(value)


def write_module(path: str, supf, seq, dmax, f1, ls) -> None:
    a1, c1, a2, cells = ls

    def table(name, tables):
        lines = [f"{name}: dict[float, dict[str, tuple[tuple[float, ...], ...]]] = {{"]
        for eps in EPSILONS:
            lines.append(f"    {eps}: {{")
            for signif, level in enumerate(LEVELS, start=1):
                rows = ", ".join("(" + ", ".join(f"{v:.2f}" for v in row) + ",)"
                                 for row in tables[(eps, signif)])
                lines.append(f'        "{level}": ({rows}),')
            lines.append("    },")
        lines.append("}")
        return "\n".join(lines)

    body = f'''"""Published critical values of the break, bubble and break-unit-root tests.

Generated by ``scripts/generate_unitroot_breaks_oracles.py tables`` from the
primary sources below; do not edit by hand. Every Bai--Perron entry was
compared with a second, independent transcription (Perron's GAUSS code and
the CRAN package mbreaks) and the Lee--Strazicich one-break table with the
journal page; the script refuses to write on any disagreement.

Bai, J. and Perron, P. (1998). Estimating and testing linear models with
    multiple structural changes. Econometrica 66, 47-78, Table I.
Bai, J. and Perron, P. (2003). Critical values for multiple structural change
    tests. Econometrics Journal 6, 72-78, Tables 1-2.
    Transcriptions: BRCODE.SRC (getcv1, getcv2, getdmax) in
    https://blogs.bu.edu/perron/files/2020/05/m-Break.zip and
    https://github.com/cran/mbreaks/tree/master/R/{{supF,supF_next,Dmax}}.
    Tables are indexed [trimming epsilon][level][q - 1][column]; the statistic
    they refer to is the F statistic of Bai and Perron multiplied by q (the
    GAUSS code prints it "scaled by q"). SUPF columns are k = 1, 2, ...;
    SEQUENTIAL columns are l = 0, 1, ..., 9 for supF(l+1|l); DMAX columns are
    UDmax and WDmax for M = DMAX_BREAKS[epsilon] breaks.
Perron, P. and Vogelsang, T. J. (1992). Nonstationarity and level shifts with
    an application to purchasing power parity. JBES 10, 301-320. Model 0 (mean
    shift) asymptotic distribution of min t_alpha, common to the IO and AO
    statistics: percentiles 1, 2, ..., 99 from Vogelsang's PVAL program
    (FF1.C, function f1), https://blogs.bu.edu/perron/files/2020/05/unit-root.zip.
Lee, J. and Strazicich, M. C. (2013). Minimum LM unit root test with one
    structural break. Economics Bulletin 33(4), 2483-2492, Table 1 (T = 100).
Lee, J. and Strazicich, M. C. (2003). Minimum Lagrange multiplier unit root
    test with two structural breaks. REStat 85(4), 1082-1089, Table 2 (T = 100),
    as transcribed in https://github.com/aptech/tspdlib/blob/master/src/lm_2br.src.
"""

from __future__ import annotations

BP_LEVELS = {LEVELS!r}
BP_EPSILONS = {EPSILONS!r}
# Largest number of breaks M for which the Dmax columns are tabulated.
DMAX_BREAKS = {{0.05: 5, 0.10: 5, 0.15: 5, 0.20: 3, 0.25: 2}}

{table("SUPF", supf)}

{table("SEQUENTIAL", seq)}

{table("DMAX", dmax)}

# Percentiles 1..99 of the asymptotic min t_alpha distribution (PV 1992, model 0).
PV_MODEL0_PERCENTILES = (
{chr(10).join("    " + ", ".join(f"{v:.6f}" for v in f1[i:i + 6]) + "," for i in range(0, 99, 6))}
)

# Lee-Strazicich one break (2013, Table 1): 1%, 5%, 10%.
LS1_CRASH = {a1!r}
LS1_BREAK = {{{", ".join(f"{k}: {v!r}" for k, v in c1.items())}}}
# Lee-Strazicich two breaks (2003, Table 2): 1%, 5%, 10%; model C by (lambda1, lambda2).
LS2_CRASH = {a2!r}
LS2_BREAK = {{{", ".join(f"{k!r}: {v!r}" for k, v in cells.items())}}}
'''
    with open(path, "w") as handle:
        handle.write(body)


def tables_command(args) -> int:
    brcode = zipfile.ZipFile(io.BytesIO(fetch(PERRON_ZIP, args.cache)))
    name = next(n for n in brcode.namelist() if n.upper().endswith("BRCODE.SRC"))
    source = brcode.read(name).decode("latin-1").replace("\r", "")
    problems = []
    merged = {}
    for proc, kind in (("getcv1", "supF"), ("getcv2", "supF_next"), ("getdmax", "Dmax")):
        gauss = gauss_tables(source, proc)
        cran = mbreaks_tables(kind, args.cache)
        found = compare(kind, gauss, cran)
        problems.extend(found)
        merged[kind] = cran
        print(f"{kind}: {len(cran)} level/trim tables; {len(found)} disagreement(s)")
    f1 = vogelsang_f1(args.cache)
    print("PV model 0 min t_alpha 1/5/10%:", f1[0], f1[4], f1[9])
    a1, c1, a2, cells = tspdlib_ls(args.cache)
    if a1 != LS_2013_A or c1 != LS_2013_C:
        problems.append(f"LS 2013 table differs from tspdlib: {a1} {c1}")
    print("LS crash 1/2 breaks:", a1, a2, "model C cells:", cells)
    for line in problems:
        print("DISAGREEMENT", line)
    if problems:
        return 1
    if args.write:
        write_module(args.write, merged["supF"], merged["supF_next"], merged["Dmax"], f1,
                     (a1, c1, a2, cells))
        print("wrote", args.write)
    return 0


def psy_command(args) -> int:
    """Bounded seeded comparison with Phillips, Shi and Yu (CFDP 1843, Table 1(b)/(c))."""
    import torch

    from openecon.econometrics.unitroot import bubbles

    torch.set_num_threads(1)
    published = {(100, 0.4): {"sadf": (0.72, 1.05, 1.66), "gsadf": (1.16, 1.48, 2.08)},
                 (200, 0.2): {"sadf": (0.97, 1.30, 1.86), "gsadf": (1.64, 1.88, 2.46)}}
    for (total, fraction), expected in published.items():
        window = int(fraction * total)
        generator = torch.Generator().manual_seed(args.seed)
        draws = bubbles.null_paths(total, args.replications, generator)
        stats = bubbles.recursive_statistics(draws, window, 0)
        for name in ("sadf", "gsadf"):
            q = torch.quantile(stats[name], torch.tensor([0.9, 0.95, 0.99], dtype=torch.float64))
            print(f"T={total} r0={fraction} {name}: simulated {[round(v, 3) for v in q.tolist()]} "
                  f"published {expected[name]} (B={args.replications}, seed={args.seed})")
    return 0


def _wilson(hits: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    centre = (hits + z * z / 2) / (total + z * z)
    half = z * ((hits * (total - hits) / total + z * z / 4) ** 0.5) / (total + z * z)
    return centre - half, centre + half


def size_command(args) -> int:
    """Declared bounded rejection-rate grid (5% level), seeded; failures are counted, never dropped.

    Lee-Strazicich one break (A, C) and Clemente one shift (IO, AO) under an iid
    Gaussian random walk; PSY GSADF with Monte Carlo and wild-bootstrap critical
    values under (a) an iid random walk, (b) a random walk whose innovation
    standard deviation triples at T/2, and (c) an explosive episode (rho = 1.04
    for observations 50..69, then a return to the pre-episode level).
    """
    import numpy as np
    import pandas as pd
    import torch

    import openecon as oe
    from openecon.econometrics.unitroot import bubbles

    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    total, outer = 100, args.datasets

    def report(name, hits, failures):
        low, high = _wilson(hits, outer)
        print(f"{name}: {hits}/{outer} = {hits / outer:.3f} (Wilson 95% {low:.3f}-{high:.3f}); "
              f"failures {failures}", flush=True)

    for name, function, options in (
            ("LS one break, model A", oe.lee_strazicich, {"model": "crash"}),
            ("LS one break, model C", oe.lee_strazicich, {"model": "break"}),
            ("CMR one shift, IO (asymptotic)", oe.clemente, {"model": "io"}),
            ("CMR one shift, AO (asymptotic)", oe.clemente, {"model": "ao"})):
        hits = failures = 0
        for _ in range(outer):
            frame = pd.DataFrame({"y": rng.standard_normal(total).cumsum()})
            try:
                result = function(frame, "y", **options)
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            hits += int(result.attrs["statistic"] < result.attrs["critical_values"]["5%"])
        report(name, hits, failures)
    window = bubbles.default_window(total)
    generator = torch.Generator().manual_seed(args.seed)
    mc = bubbles.recursive_statistics(bubbles.null_paths(total, 1999, generator), window, 0)
    mc_cv = float(torch.quantile(mc["gsadf"], 0.95))
    print(f"PSY GSADF Monte Carlo 95% critical value (T={total}, w={window}, B=1999): {mc_cv:.3f}")

    def path(kind):
        e = rng.standard_normal(total)
        if kind == "variance break":
            e[total // 2:] *= 3.0
        y = e.cumsum()
        if kind == "explosive":
            y = np.zeros(total)
            y[0] = 10.0
            for i in range(1, total):
                y[i] = (1.04 if 50 <= i < 70 else 1.0) * y[i - 1] + e[i]
            y[70:] -= y[70] - y[49]
        return y

    for kind in ("iid", "variance break", "explosive"):
        hits_mc = hits_wb = failures = 0
        for _ in range(outer):
            frame = pd.DataFrame({"y": path(kind)})
            try:
                result = oe.psy(frame, "y", calibration="wild_bootstrap", replications=args.boot,
                                seed=int(rng.integers(0, 2**31)))
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            gsadf = result.attrs["gsadf"]
            hits_mc += int(gsadf > mc_cv)
            hits_wb += int(gsadf > result.attrs["critical_values"]["95%"])
        report(f"PSY GSADF Monte Carlo cv, {kind}", hits_mc, failures)
        report(f"PSY GSADF wild bootstrap (B={args.boot}), {kind}", hits_wb, failures)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tables = sub.add_parser("tables")
    tables.add_argument("--cache", default=os.path.join(os.getcwd(), ".breaks-cache"))
    tables.add_argument("--write", default=None)
    psy = sub.add_parser("psy")
    psy.add_argument("--replications", type=int, default=5000)
    psy.add_argument("--seed", type=int, default=20261010)
    size = sub.add_parser("size")
    size.add_argument("--datasets", type=int, default=400)
    size.add_argument("--boot", type=int, default=199)
    size.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    commands = {"tables": tables_command, "psy": psy_command, "size": size_command}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
