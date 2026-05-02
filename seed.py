"""
Run this once to populate the database with test cases:
    python seed.py
"""
from lib.db import SessionLocal, TestCase, create_tables

TEST_CASES = [
    # ── Clear cases: should pass with any reasonable prompt ───────────────────
    {
        "user_message": "Where is my order #12345?",
        "expected_function_name": "lookup_order",
        "expected_params": {"order_id": "12345"},
    },
    {
        "user_message": "Track order AB-7788 for me.",
        "expected_function_name": "lookup_order",
        "expected_params": {"order_id": "AB-7788"},
    },
    {
        "user_message": "I want a refund for order #45678 because it arrived damaged.",
        "expected_function_name": "process_refund",
        "expected_params": {"order_id": "45678", "reason": "arrived damaged"},
    },
    {
        "user_message": "Please refund order 77777. I ordered the wrong item.",
        "expected_function_name": "process_refund",
        "expected_params": {"order_id": "77777", "reason": "ordered wrong item"},
    },
    {
        "user_message": "Your bot is not helping. Escalate this to a human.",
        "expected_function_name": "escalate_to_human",
        "expected_params": {"reason": "customer requested escalation to a human agent"},
    },
    {
        "user_message": "I need help with my order but I do not have the order number.",
        "expected_function_name": "escalate_to_human",
        "expected_params": {"reason": "customer needs order help but does not have an order number"},
    },

    # ── Targets wrong_function: weak prompts call a plausible-but-wrong tool ─
    # Agent sees "order 55555" and calls lookup_order, but repeated contact → escalate
    {
        "user_message": "I have contacted support three times this week about order 55555 and nothing has been resolved.",
        "expected_function_name": "escalate_to_human",
        "expected_params": {"reason": "customer has contacted support multiple times without resolution"},
    },
    # Agent calls lookup_order to "check status", but non-delivery → it's a refund dispute
    {
        "user_message": "Order 33333 shows as delivered but I never received it.",
        "expected_function_name": "process_refund",
        "expected_params": {"order_id": "33333", "reason": "order marked delivered but never received"},
    },

    # ── Targets missing_param: reason is implicit, weak prompts omit it ───────
    # "I'd like a refund" — no reason stated; process_refund still requires reason param
    {
        "user_message": "I would like a refund on order 88888.",
        "expected_function_name": "process_refund",
        "expected_params": {"order_id": "88888", "reason": "customer requested refund"},
    },
    # Terse message — easy to miss extracting both params
    {
        "user_message": "Wrong charge on order 44444.",
        "expected_function_name": "process_refund",
        "expected_params": {"order_id": "44444", "reason": "incorrect charge"},
    },

    # ── No-tool cases: model must reply in plain text, not call any function ──
    # A weak prompt that always tries to "do something" will call escalate_to_human here
    {
        "user_message": "Hello! What can you help me with today?",
        "expected_function_name": None,
        "expected_params": None,
    },
    {
        "user_message": "I am really frustrated with my recent experience.",
        "expected_function_name": None,
        "expected_params": None,
    },
    {
        "user_message": "What is your return policy?",
        "expected_function_name": None,
        "expected_params": None,
    },
    {
        "user_message": "Thanks for your help, have a great day!",
        "expected_function_name": None,
        "expected_params": None,
    },
    {
        "user_message": "I might want to make a change to my order but I haven't decided yet.",
        "expected_function_name": None,
        "expected_params": None,
    },
    {
        "user_message": "Do you have a phone number I can call for support?",
        "expected_function_name": None,
        "expected_params": None,
    },
]


def main():
    create_tables()
    db = SessionLocal()
    try:
        existing = db.query(TestCase).count()
        if existing > 0:
            print(f"Skipping seed — {existing} test cases already exist.")
            return

        for data in TEST_CASES:
            db.add(TestCase(**data))
        db.commit()
        print(f"Seeded {len(TEST_CASES)} test cases.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
