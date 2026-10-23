"""Generate SVG art for @amjadiqbal/tanstack-ai-rag (TanStack-channel family, cyan accent)."""
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "Menlo, 'SF Mono', Consolas, monospace"
NAME = "@amjadiqbal/tanstack-ai-rag"

DARK = dict(bg1="#0B1220", bg2="#0E2233", text="#F8FAFC", accent="#22D3EE", sub="#A5F3FC",
            chip="#16213A", chipstroke="#24324F", chiptext="#CBD5E1",
            foot="#64748B", panel="#0F172A", panelstroke="#1F4A5C", kw="#F0ABFC", fn="#67E8F9",
            str="#86EFAC", plain="#E2E8F0", dim="#64748B")
LIGHT = dict(bg1="#F8FAFC", bg2="#E3F4F8", text="#0F172A", accent="#0891B2", sub="#0E7490",
             chip="#FFFFFF", chipstroke="#CBD5E1", chiptext="#334155",
             foot="#64748B", panel="#0F172A", panelstroke="#67E8F9", kw="#F0ABFC", fn="#67E8F9",
             str="#86EFAC", plain="#E2E8F0", dim="#64748B")


def icon_body():
    """512x512 icon: a document sliced into chunks, the chunks land as points in vector space,
    the query finds its nearest neighbours and the answer cites them as [1]."""
    return """
  <defs>
    <linearGradient id="ibg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0F172A"/><stop offset="1" stop-color="#0B2A36"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="112" fill="url(#ibg)"/>
  <!-- document -->
  <path d="M64 92 H168 L204 128 V300 H64 Z" fill="#1E293B" stroke="#A5F3FC" stroke-width="6" stroke-linejoin="round"/>
  <path d="M168 92 V128 H204" fill="none" stroke="#A5F3FC" stroke-width="6" stroke-linejoin="round"/>
  <!-- chunk slices (dashed cut lines) -->
  <g stroke="#22D3EE" stroke-width="4" stroke-dasharray="8 7" stroke-linecap="round">
    <path d="M52 168 H216"/><path d="M52 236 H216"/>
  </g>
  <g fill="#475569">
    <rect x="86" y="140" width="92" height="10" rx="5"/>
    <rect x="86" y="186" width="96" height="10" rx="5"/><rect x="86" y="208" width="70" height="10" rx="5"/>
    <rect x="86" y="254" width="88" height="10" rx="5"/><rect x="86" y="276" width="60" height="10" rx="5"/>
  </g>
  <!-- chunks flow into vector space -->
  <g stroke="#334155" stroke-width="5" fill="none" stroke-linecap="round">
    <path d="M216 130 C 260 130, 270 150, 300 160"/>
    <path d="M216 202 C 256 202, 276 214, 318 222"/>
    <path d="M216 270 C 260 270, 290 300, 330 320"/>
  </g>
  <!-- vector space -->
  <circle cx="366" cy="236" r="112" fill="#0B1B26" stroke="#1F4A5C" stroke-width="5"/>
  <g fill="#64748B">
    <circle cx="300" cy="166" r="9"/><circle cx="420" cy="170" r="8"/><circle cx="330" cy="320" r="9"/>
    <circle cx="438" cy="292" r="8"/><circle cx="290" cy="246" r="7"/>
  </g>
  <!-- nearest neighbours of the query -->
  <path d="M378 228 L 318 222 M378 228 L 404 268" stroke="#22D3EE" stroke-width="4" stroke-dasharray="6 6" stroke-linecap="round"/>
  <circle cx="318" cy="222" r="11" fill="#22D3EE"/>
  <circle cx="404" cy="268" r="11" fill="#22D3EE"/>
  <circle cx="318" cy="222" r="20" fill="none" stroke="#67E8F9" stroke-width="4"/>
  <!-- query point -->
  <path d="M378 204 l7 15 l16 2 l-12 11 l3 16 l-14 -8 l-14 8 l3 -16 l-12 -11 l16 -2 z" fill="#F8FAFC"/>
  <!-- citation -->
  <rect x="64" y="352" width="200" height="72" rx="20" fill="#1E293B" stroke="#22D3EE" stroke-width="5"/>
  <rect x="88" y="374" width="66" height="10" rx="5" fill="#94A3B8"/><rect x="88" y="394" width="44" height="10" rx="5" fill="#94A3B8"/>
  <text x="214" y="402" text-anchor="middle" font-family="Menlo, 'SF Mono', Consolas, monospace" font-size="34" font-weight="700" fill="#22D3EE">[1]</text>
  <path d="M264 388 C 300 388, 300 250, 300 236" fill="none" stroke="#22D3EE" stroke-width="4" stroke-linecap="round" stroke-dasharray="2 9"/>
"""


def icon_svg():
    return f'<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">{icon_body()}</svg>\n'


def chips(x, y, labels, c, size=15, h=32, gap=10, cw=8.6):
    out, cx = [], x
    for lab in labels:
        w = round(len(lab) * cw + 36)
        out.append(f'<g transform="translate({cx},{y})"><rect width="{w}" height="{h}" rx="{h/2}" fill="{c["chip"]}" stroke="{c["chipstroke"]}"/>'
                   f'<text x="{w/2}" y="{h*0.66:.1f}" text-anchor="middle">{lab}</text></g>')
        cx += w + gap
    return f'<g font-family="{SANS}" font-size="{size}" font-weight="600" fill="{c["chiptext"]}">' + "".join(out) + "</g>"


def code_card(x, y, w, c, lines, size=15, lh=24):
    """Dark code card (always dark, also on the light banner). lines = list of [(text, colour-key)]."""
    h = 28 + lh * len(lines)
    g = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{c["panel"]}" stroke="{c["panelstroke"]}" stroke-width="2"/>',
         f'<g font-family="{MONO}" font-size="{size}">']
    for i, parts in enumerate(lines):
        spans = "".join(f'<tspan fill="{c[k]}">{t}</tspan>' for t, k in parts)
        g.append(f'<text x="{x+20}" y="{y+30+lh*i}" xml:space="preserve">{spans}</text>')
    g.append("</g>")
    return "".join(g)


SHORT = [
    [("const", "kw"), (" rag = ", "plain"), ("createRag", "fn"), ("({", "plain")],
    [("  embedding: ", "plain"), ("openaiEmbedding", "fn"), ("(model),", "plain")],
    [("  store: ", "plain"), ("pgvectorStore", "fn"), ("({ db: pool }),", "plain")],
    [("})", "plain")],
    [("chat", "fn"), ("({ adapter, messages, tools: [rag.", "plain"), ("tool", "fn"), ("()] })", "plain")],
]

LONG = [
    [("const", "kw"), (" rag = ", "plain"), ("createRag", "fn"), ("({ embedding: ", "plain"), ("openaiEmbedding", "fn"),
     ("(", "plain"), ("'text-embedding-3-small'", "str"), ("),", "plain")],
    [("                         store: ", "plain"), ("pgvectorStore", "fn"), ("({ db: pool }) })", "plain")],
    [("await", "kw"), (" rag.", "plain"), ("ingest", "fn"), ("([{ id: ", "plain"), ("'docs/install.md'", "str"), (", content, format: ", "plain"), ("'markdown'", "str"), (" }])", "plain")],
    [("chat", "fn"), ("({ adapter, messages, tools: [rag.", "plain"), ("tool", "fn"), ("()] })", "plain"),
     ("  // answers cite [1] [2]", "dim")],
]


def banner(c):
    return f'''<svg width="1200" height="300" viewBox="0 0 1200 300" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c["bg1"]}"/><stop offset="1" stop-color="{c["bg2"]}"/></linearGradient></defs>
  <rect width="1200" height="300" fill="url(#bg)"/>
  <g transform="translate(60,75) scale({150/512:.5f})">{icon_body()}</g>
  <text x="238" y="98" font-family="{MONO}" font-size="16" font-weight="600" fill="{c["accent"]}">for TanStack AI 1.x</text>
  <text x="235" y="160" font-family="{SANS}" font-size="64" font-weight="700" fill="{c["text"]}">TanStack AI RAG</text>
  <text x="238" y="198" font-family="{SANS}" font-size="22" font-weight="500" fill="{c["sub"]}">Chunk, embed, store, retrieve, cite</text>
  {chips(238, 220, ["pgvector", "in-memory", "retrieval tool", "rerank"], c)}
  {code_card(800, 62, 360, c, SHORT, size=12, lh=22)}
  <text x="1160" y="275" text-anchor="end" font-family="{SANS}" font-size="15" fill="{c["foot"]}">amjadiqbal/tanstack-ai-rag</text>
</svg>
'''


def social(c):
    return f'''<svg width="1280" height="640" viewBox="0 0 1280 640" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c["bg1"]}"/><stop offset="1" stop-color="{c["bg2"]}"/></linearGradient></defs>
  <rect width="1280" height="640" fill="url(#bg)"/>
  <g transform="translate(80,80) scale({176/512:.5f})">{icon_body()}</g>
  <text x="292" y="116" font-family="{MONO}" font-size="20" font-weight="600" fill="{c["accent"]}">{NAME}</text>
  <text x="288" y="196" font-family="{SANS}" font-size="78" font-weight="700" fill="{c["text"]}">TanStack AI RAG</text>
  <text x="292" y="244" font-family="{SANS}" font-size="28" font-weight="500" fill="{c["sub"]}">Retrieval-augmented chat for TanStack AI, with citations</text>
  {chips(80, 300, ["markdown + text chunking", "embeddings via TanStack AI embed()", "pgvector (Postgres)", "in-memory"], c, size=18, h=40, gap=12, cw=10.2)}
  {chips(80, 352, ["retrieval tool for chat()", "cited sources", "optional rerank()"], c, size=18, h=40, gap=12, cw=10.2)}
  {code_card(80, 420, 1120, c, LONG, size=16, lh=30)}
  <text x="1200" y="614" text-anchor="end" font-family="{SANS}" font-size="20" fill="{c["foot"]}">Free and open source · MIT · by Amjad</text>
</svg>
'''


def write(name, svg):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)


write("icon.svg", icon_svg())
write("banner-dark.svg", banner(DARK))
write("banner-light.svg", banner(LIGHT))
write("social-preview.svg", social(DARK))
print("wrote SVGs to", OUT)
