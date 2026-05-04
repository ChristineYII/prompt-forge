SCENARIO_DESCRIPTION = (
    "A customer service agent for an e-commerce platform that helps customers with "
    "order status, refunds, and escalations. Only call a tool when the customer "
    "clearly needs one of these actions — do not call tools for greetings or vague questions."
)

# Each dict matches the JSON Schema format the Vertex AI SDK expects for function declarations.
TOOL_SCHEMAS: list[dict] = [
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
