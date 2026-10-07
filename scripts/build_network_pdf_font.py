"""Build the PDF's TrueType font from the bundled OFL Barlow WOFF2.

Build tool only: ``uv run --no-project --with fonttools --with brotli
scripts/build_network_pdf_font.py``. No fonttools dependency ships at runtime.
The source has CFF cubic outlines; merely removing WOFF2 compression would
produce OpenType/CFF, which must not be embedded as PDF FontFile2.
"""
from base64 import b64encode
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ASSETS = Path(__file__).resolve().parents[1] / "packages/openecon-charts/src/openecon_charts/assets"


def main():
    source = TTFont(ASSETS / "Barlow.woff2", recalcTimestamp=False)
    order, glyph_set = source.getGlyphOrder(), source.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, max_err=1, reverse_direction=True))
        glyphs[name] = pen.glyph()
    builder = FontBuilder(source["head"].unitsPerEm, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(source.getBestCmap())
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(source["hmtx"].metrics)
    builder.setupHorizontalHeader(ascent=source["hhea"].ascent, descent=source["hhea"].descent)
    builder.setupNameTable({})
    builder.font["name"] = source["name"]
    builder.font["OS/2"] = source["OS/2"]
    builder.setupPost()
    builder.font["head"].created = source["head"].created
    builder.font["head"].modified = source["head"].modified
    builder.font.recalcTimestamp = False
    target = ASSETS / "Barlow.ttf"
    builder.save(target)
    converted = TTFont(target)
    assert converted.sfntVersion == "\x00\x01\x00\x00" and "glyf" in converted
    assert converted.getBestCmap() == source.getBestCmap()
    assert converted["hmtx"].metrics == source["hmtx"].metrics
    payload = b64encode(target.read_bytes()).decode("ascii")
    (ASSETS / "network-font.js").write_text(
        '/* Barlow Regular, SIL OFL1.1. Cubic outlines converted to quadratic\n'
        ' * with at most 1 font-unit error; see build_network_pdf_font.py. */\n'
        '(function(root){"use strict";root.OpenEconNetworkFont={name:"Barlow-Regular",'
        'license:"SIL Open Font License 1.1",base64:"' + payload + '"};})(window);\n',
        encoding="utf-8",
    )
    print(f"Verified TrueType: {len(order)} glyphs, {target.stat().st_size} bytes.")


if __name__ == "__main__":
    main()
