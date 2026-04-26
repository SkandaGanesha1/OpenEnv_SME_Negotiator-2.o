"""
Pre-flight validation for NegotiationActions before they reach the environment.

Usage::

    handler = ActionHandler()
    ok, err = handler.validate(price=price, payment_days=days, ...)
    if not ok:
        return error_response(err)
    wired_type = handler.map_ui_action(ui_action)
"""
from __future__ import annotations


class ActionHandler:
    """Stateless validator for NegotiationAction field values."""

    # UI-only alias that maps to "propose" on the wire
    _UI_TO_WIRE: dict[str, str] = {"counter_offer": "propose"}

    def map_ui_action(self, ui_action: str) -> str:
        """Map Gradio dropdown value to ``NegotiationAction.action_type`` literal."""
        a = (ui_action or "propose").strip().lower()
        return self._UI_TO_WIRE.get(a, a)

    def validate(
        self,
        *,
        price: float,
        payment_days: int,
        use_treds: bool = False,
        propose_dynamic_discounting: bool = False,
        dynamic_discount_annual_rate: float = 0.0,
        action_type: str = "propose",
    ) -> tuple[bool, str]:
        """
        Validate raw field values *before* constructing a NegotiationAction.

        Returns:
            (True, "")  — all checks pass
            (False, error_message)  — first failure found
        """
        if price < 0:
            return False, f"Price must be ≥ 0 (got ₹{price:.2f}/unit)."

        if not (0 <= payment_days <= 365):
            return False, (
                f"Payment days must be between 0 and 365 (got {payment_days})."
            )

        if use_treds and payment_days < 1:
            return False, "TReDS financing requires at least 1 payment day."

        if propose_dynamic_discounting:
            if not (0.0 <= dynamic_discount_annual_rate <= 0.95):
                return False, (
                    f"Dynamic discount annual rate must be 0.00–0.95 "
                    f"(got {dynamic_discount_annual_rate:.3f})."
                )

        return True, ""

    def format_error(self, error: str) -> str:
        """Wrap a validation message for display in the Gradio chat panel."""
        return f"⚠️ **Validation error:** {error}"
