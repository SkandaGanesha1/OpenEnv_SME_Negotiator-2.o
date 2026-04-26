"""Problem framing aligned with Razorpay *Fix My Itch* (public listing).

See: https://www.razorpay.com/m/fix-my-itch/ — B2B Services itch
*Why can't SMEs negotiate favorable payment terms with large buyers?* (itch score **82.8**).
"""

from __future__ import annotations

# One paragraph: real-world motivation (not graded — grading is in graders.py).
RAZORPAY_ITCH_BLURB = (
    "Real-world context (Razorpay Fix My Itch, B2B Services, itch score 82.8): "
    "Small suppliers often receive buyer payment terms of 60–90+ days while paying their own suppliers in ~30 days, "
    "widening working-capital gaps and forcing costly short-term financing (often ~18–24% APR class). "
    "Large buyers may resist shorter cycles. This environment simulates multi-round negotiation; "
    "terminal scores are computed by deterministic rules in sme_negotiator_env/graders.py."
)
