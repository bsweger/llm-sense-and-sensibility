"""
Shared visual theme — Murderbot-inspired palette.

Imported by all demo pages so the app reads as a single piece. Tweak here and
every page updates together.
"""

# Alternating pill backgrounds for chunked sequence views (e.g. tokenization).
# Color carries no semantic meaning — it just visually chunks adjacent items.
# Kept light so dark text on top stays legible.
# Order chosen to keep adjacent pills distinguishable for red-green colorblind
# viewers (moss and red are not neighbors in the cycle) and to avoid the
# stone/gunmetal grays sitting next to each other.
TOKEN_COLORS = [
    "#e7e5e4",  # stone — corporate interior beige
    "#fecaca",  # red — alert / warning indicator
    "#d4d4d8",  # gunmetal — SecUnit armor
    "#fed7aa",  # amber — HUD targeting overlay
    "#b6c8a5",  # muted moss — Preservation Alliance survey team
]

# Semantic accents reused across demos.
HUD_ACCENT = "#9a3412"  # burnt amber — HUD typography (ids, "winning" labels)
HIGHLIGHT_BG = "#fed7aa"  # HUD amber — "currently focused" marker
BODY_TEXT = "#1f2937"  # near-black — default body text on light pills/rows
