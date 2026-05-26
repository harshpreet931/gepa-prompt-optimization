"""Payment-intent classification dataset for the GEPA demo.

Each item has the shape DefaultAdapter expects:
    {
        "input": "<customer message>",
        "answer": "### <label>",   # substring-matched against model output
        "additional_context": {"explanation": "<why this label>"},
    }

To use your own task, build a list of dicts with the same shape and import it
from run_gepa.py instead of LABELLED.
"""
from __future__ import annotations

LABELS = ["refund", "dispute", "payment_failed", "card_help", "subscription", "other"]

LABELLED: list[dict] = [
    {
        "input": "My card was charged twice for the same purchase, please reverse one.",
        "answer": "### refund",
        "additional_context": {"explanation": "Duplicate charge — customer wants money back."},
    },
    {
        "input": "I never received the product I paid for and the seller is unresponsive.",
        "answer": "### dispute",
        "additional_context": {"explanation": "Goods not delivered → chargeback dispute."},
    },
    {
        "input": "Trying to pay but the page keeps saying transaction declined.",
        "answer": "### payment_failed",
        "additional_context": {"explanation": "Payment attempt failing at gateway."},
    },
    {
        "input": "How do I add a new debit card to my account?",
        "answer": "### card_help",
        "additional_context": {"explanation": "Card management / onboarding question."},
    },
    {
        "input": "Please cancel my monthly plan starting next month.",
        "answer": "### subscription",
        "additional_context": {"explanation": "Subscription cancellation request."},
    },
    {
        "input": "What time does customer support open on weekends?",
        "answer": "### other",
        "additional_context": {"explanation": "General info, not a transactional issue."},
    },
    {
        "input": "Refund for order #4471 still hasn't shown up after 10 days.",
        "answer": "### refund",
        "additional_context": {"explanation": "Outstanding refund — money back expected."},
    },
    {
        "input": "The merchant overcharged me by ₹500 and refuses to fix it.",
        "answer": "### dispute",
        "additional_context": {"explanation": "Merchant refusing — escalate as dispute."},
    },
    {
        "input": "Card declined again at checkout, my balance is fine.",
        "answer": "### payment_failed",
        "additional_context": {"explanation": "Repeated decline despite sufficient funds."},
    },
    {
        "input": "My credit card expires next month, where do I update it?",
        "answer": "### card_help",
        "additional_context": {"explanation": "Updating expiring card details."},
    },
    {
        "input": "Why did you charge me twice for the Pro subscription this month?",
        "answer": "### subscription",
        "additional_context": {"explanation": "Double billing on a recurring plan."},
    },
    {
        "input": "Is there a mobile app for managing my account?",
        "answer": "### other",
        "additional_context": {"explanation": "Product information question."},
    },
    {
        "input": "I want my money back for the order I cancelled yesterday.",
        "answer": "### refund",
        "additional_context": {"explanation": "Cancelled order — refund expected."},
    },
    {
        "input": "The hotel charged me for a night I didn't stay; they won't help.",
        "answer": "### dispute",
        "additional_context": {"explanation": "Merchant disputing charge — chargeback."},
    },
    {
        "input": "UPI payment keeps failing with error CR-04, what should I do?",
        "answer": "### payment_failed",
        "additional_context": {"explanation": "UPI failure with specific error code."},
    },
    {
        "input": "Lost my wallet — need to block all my saved cards immediately.",
        "answer": "### card_help",
        "additional_context": {"explanation": "Urgent card-block / security request."},
    },
    {
        "input": "Pause my premium plan for 3 months, will resume in March.",
        "answer": "### subscription",
        "additional_context": {"explanation": "Pause vs cancel — still subscription mgmt."},
    },
    {
        "input": "Do you support international transactions to Singapore?",
        "answer": "### other",
        "additional_context": {"explanation": "Capability / policy question."},
    },
]


def init_dataset() -> tuple[list[dict], list[dict]]:
    """Return (trainset, valset) — a roughly 50/50 split by label."""
    by_label: dict[str, list[dict]] = {}
    for ex in LABELLED:
        label = ex["answer"].removeprefix("### ")
        by_label.setdefault(label, []).append(ex)

    train, val = [], []
    for items in by_label.values():
        mid = len(items) // 2
        train.extend(items[:mid] if mid > 0 else items[:1])
        val.extend(items[mid:] if mid > 0 else items)
    return train, val
