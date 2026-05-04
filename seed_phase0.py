"""
Seed Phase 0 baseline: Customer Service scenario + 16 stable test cases.
Run once after wiping the DB:
    python seed_phase0.py
"""
from lib.db import SessionLocal, Scenario, TestCase, create_tables


# Customer service tool schemas used to seed the baseline DB scenario.
CUSTOMER_SERVICE_TOOLS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up the current status and details of a customer order. Use this when "
            "the customer asks where their order is, when it will arrive, or what the "
            "status of a specific order ID is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID provided by the customer, e.g. '12345' or 'AB-7788'",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "process_refund",
        "description": (
            "Initiate a refund for a customer order. Use this only when the customer "
            "explicitly requests a refund, mentions being charged incorrectly, or wants "
            "to return an item. Do not use this just because the customer is unhappy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to be refunded",
                },
                "reason": {
                    "type": "string",
                    "description": "The reason for the refund as described by the customer",
                },
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate the conversation to a human support agent. Use this when: the "
            "customer explicitly asks for a human, the bot cannot resolve the issue, "
            "the customer has contacted support multiple times, or the customer does "
            "not have an order number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the conversation is being escalated to a human agent",
                }
            },
            "required": ["reason"],
        },
    },
]


PHASE0_TEST_CASES = [
    # ── Clear cases (6) ───────────────────────────────────────────────────────
    {"user_message": "Where is my order #12345?",
     "expected_function_name": "lookup_order",
     "expected_params": {"order_id": "12345"}},
    {"user_message": "Track order AB-7788 for me.",
     "expected_function_name": "lookup_order",
     "expected_params": {"order_id": "AB-7788"}},
    {"user_message": "I want a refund for order #45678 because it arrived damaged.",
     "expected_function_name": "process_refund",
     "expected_params": {"order_id": "45678", "reason": "arrived damaged"}},
    {"user_message": "Please refund order 77777. I ordered the wrong item.",
     "expected_function_name": "process_refund",
     "expected_params": {"order_id": "77777", "reason": "ordered wrong item"}},
    {"user_message": "Your bot is not helping. Escalate this to a human.",
     "expected_function_name": "escalate_to_human",
     "expected_params": {"reason": "customer requested escalation to a human agent"}},
    {"user_message": "I need help with my order but I do not have the order number.",
     "expected_function_name": "escalate_to_human",
     "expected_params": {"reason": "customer needs order help but does not have an order number"}},

    # ── Wrong function traps (2) ──────────────────────────────────────────────
    {"user_message": "I have contacted support three times this week about order 55555 and nothing has been resolved.",
     "expected_function_name": "escalate_to_human",
     "expected_params": {"reason": "customer has contacted support multiple times without resolution"}},
    {"user_message": "Order 33333 shows as delivered but I never received it.",
     "expected_function_name": "process_refund",
     "expected_params": {"order_id": "33333", "reason": "order marked delivered but never received"}},

    # ── Missing param traps (2) ───────────────────────────────────────────────
    {"user_message": "I would like a refund on order 88888.",
     "expected_function_name": "process_refund",
     "expected_params": {"order_id": "88888", "reason": "customer requested refund"}},
    {"user_message": "Wrong charge on order 44444.",
     "expected_function_name": "process_refund",
     "expected_params": {"order_id": "44444", "reason": "incorrect charge"}},

    # ── No-tool cases (6) ─────────────────────────────────────────────────────
    {"user_message": "Hello! What can you help me with today?",
     "expected_function_name": None, "expected_params": None},
    {"user_message": "I am really frustrated with my recent experience.",
     "expected_function_name": None, "expected_params": None},
    {"user_message": "What is your return policy?",
     "expected_function_name": None, "expected_params": None},
    {"user_message": "Thanks for your help, have a great day!",
     "expected_function_name": None, "expected_params": None},
    {"user_message": "I might want to make a change to my order but I haven't decided yet.",
     "expected_function_name": None, "expected_params": None},
    {"user_message": "Do you have a phone number I can call for support?",
     "expected_function_name": None, "expected_params": None},
]


def main():
    create_tables()
    db = SessionLocal()
    try:
        # Step 1: ensure Customer Service scenario exists
        scenario = db.query(Scenario).filter_by(name="Customer Service").first()
        if scenario is None:
            scenario = Scenario(
                name="Customer Service",
                description=(
                    "A customer service agent for an e-commerce platform that helps customers with "
                    "order status, refunds, and escalations. Only call a tool when the customer "
                    "clearly needs one of these actions — do not call tools for greetings or vague questions."
                ),
                tools_json=CUSTOMER_SERVICE_TOOLS,
            )
            db.add(scenario)
            db.commit()
            db.refresh(scenario)
            print(f"Created scenario: id={scenario.id}, name={scenario.name}")
        else:
            print(f"Found existing scenario: id={scenario.id}, name={scenario.name}")

        # Step 2: idempotent re-seed of test cases
        existing_count = db.query(TestCase).filter_by(scenario_id=scenario.id).count()
        if existing_count > 0:
            print(f"Scenario already has {existing_count} test cases, deleting them first...")
            db.query(TestCase).filter_by(scenario_id=scenario.id).delete()
            db.commit()

        for data in PHASE0_TEST_CASES:
            db.add(TestCase(scenario_id=scenario.id, **data))
        db.commit()
        print(f"Seeded {len(PHASE0_TEST_CASES)} test cases for scenario id={scenario.id}.")
        print(f"\nNext: use scenario_id={scenario.id} when creating prompt versions.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
