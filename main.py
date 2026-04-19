from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from opentelemetry import trace
import json
import re

app = BedrockAgentCoreApp()

# ---------------------------------------------------------
# OpenTelemetry tracer
# ---------------------------------------------------------

tracer = trace.get_tracer("loan-orchestration")

# ---------------------------------------------------------
# Lazy agent loader
# ---------------------------------------------------------

_agent_cache = {}

def get_agent(name: str) -> Agent:
    if name not in _agent_cache:
        _agent_cache[name] = Agent(model="meta.llama3-8b-instruct-v1:0")
    return _agent_cache[name]

# ---------------------------------------------------------
# Extractors
# ---------------------------------------------------------

def extract_message(result):
    if hasattr(result, "message"):
        return result.message
    if isinstance(result, dict) and "message" in result:
        return result["message"]
    return result

def extract_decision(result) -> str:
    msg = extract_message(result)
    text = str(msg).strip()

    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Try JSON object anywhere
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict) and "category" in parsed:
                return str(parsed["category"]).strip().lower()
        except Exception:
            pass

    # Try whole string as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "category" in parsed:
            return str(parsed["category"]).strip().lower()
    except Exception:
        pass

    # Keyword fallback
    keywords = [
        "policy",
        "interest_rates",
        "customer_details",
        "assessment",
        "full_workflow",
    ]
    lower = text.lower()
    for k in keywords:
        if k in lower:
            return k

    return lower

# ---------------------------------------------------------
# System prompts
# ---------------------------------------------------------

def supervisor_prompt():
    return (
        "You are a routing classifier for a banking loan system.\n"
        "Your job is to classify ANY user request into EXACTLY one category.\n\n"
        "MAP REQUESTS AS FOLLOWS:\n"
        "- Borrowing money, loan steps, loan process → \"full_workflow\"\n"
        "- Rules, eligibility, LVR, documentation → \"policy\"\n"
        "- Interest rates, repayments, pricing, loan amounts → \"interest_rates\"\n"
        "- Income, credit score, personal profile → \"customer_details\"\n"
        "- Approval, borrowing capacity, risk → \"assessment\"\n\n"
        "Prefer JSON:\n"
        "{\"category\": \"<one_of_the_categories>\"}\n"
        "If not possible, plain text category is acceptable."
    )

def policy_prompt():
    return (
        "You are a home loan policy specialist at a bank.\n"
        "Explain lending rules, LVR limits, borrowing capacity rules, "
        "required documents, and eligibility criteria in a friendly, human way.\n"
        "Use realistic but demo-safe values and examples."
    )

def rates_prompt():
    return (
        "You are a home loan banker explaining interest rates to a customer.\n"
        "Provide realistic demo interest rates for home loans: fixed, variable, "
        "and comparison rates.\n"
        "Answer conversationally, like: "
        "\"Right now, for an owner-occupied home loan, we could offer around 5.2% variable...\""
    )

def crm_prompt():
    return (
        "You are a CRM system at a bank.\n"
        "Return fictional but realistic customer data in JSON only:\n"
        "income, expenses, credit_score, liabilities, employment_status, existing_loans.\n"
        "Do NOT include explanations, only JSON."
    )

def assessment_prompt():
    return (
        "You are a home loan banker assessing a customer's borrowing capacity.\n"
        "You are friendly and conversational.\n"
        "You receive customer data, policy rules, and interest rates.\n"
        "Respond like a person at the bank, e.g.:\n"
        "\"Based on your income and expenses, you could comfortably borrow around $X at Y% over Z years, "
        "with approximate repayments of $R per month.\"\n"
        "Explain your reasoning briefly and include a clear yes/no style outcome."
    )

# ---------------------------------------------------------
# Traced agent call
# ---------------------------------------------------------

def traced_agent_call(agent_name: str, agent, system_prompt: str, user_message: str):
    with tracer.start_as_current_span(f"agent.{agent_name}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.user_message", user_message[:200])

        prompt = f"SYSTEM: {system_prompt}\nUSER: {user_message}"
        response = agent(prompt)

        msg = extract_message(response)
        span.set_attribute("agent.response_length", len(str(msg)))

        # For downstream use, return as string
        if isinstance(msg, dict):
            msg = json.dumps(msg)
        return str(msg)

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------

@app.entrypoint
def invoke(payload):
    with tracer.start_as_current_span("loan_workflow") as span:
        user_message = payload.get("prompt") or payload.get("input") or ""
        customer_id = payload.get("customer_id", "unknown")

        span.set_attribute("request.customer_id", customer_id)
        span.set_attribute("request.user_message", user_message[:200])

        # Supervisor classification
        with tracer.start_as_current_span("supervisor.classify") as sup_span:
            supervisor = get_agent("supervisor")
            sup_prompt = f"SYSTEM: {supervisor_prompt()}\nUSER: {user_message}"
            sup_response = supervisor(sup_prompt)

            decision = extract_decision(sup_response)
            sup_span.set_attribute("supervisor.decision", decision)
            sup_span.set_attribute("supervisor.raw_output", str(sup_response)[:300])

        # POLICY
        if decision == "policy":
            result = traced_agent_call(
                "policy",
                get_agent("policy"),
                policy_prompt(),
                user_message,
            )
            return {"result": result}

        # INTEREST RATES
        if decision == "interest_rates":
            result = traced_agent_call(
                "rates",
                get_agent("rates"),
                rates_prompt(),
                user_message,
            )
            return {"result": result}

        # CUSTOMER DETAILS
        if decision == "customer_details":
            result = traced_agent_call(
                "crm",
                get_agent("crm"),
                crm_prompt(),
                user_message,
            )
            return {"result": result}

        # ASSESSMENT
        if decision == "assessment":
            crm = traced_agent_call(
                "crm",
                get_agent("crm"),
                crm_prompt(),
                "Retrieve customer details for this customer.",
            )
            policy = traced_agent_call(
                "policy",
                get_agent("policy"),
                policy_prompt(),
                "Summarise key lending rules.",
            )
            rates = traced_agent_call(
                "rates",
                get_agent("rates"),
                rates_prompt(),
                "Summarise current demo interest rates.",
            )

            assessment_input = (
                f"Customer Data (JSON):\n{crm}\n\n"
                f"Policy Rules:\n{policy}\n\n"
                f"Interest Rates:\n{rates}\n\n"
                "Now, act like a banker and tell the customer whether they can borrow, "
                "roughly how much, at what sort of rate, and what their approximate "
                "repayments would be over 20 or 30 years."
            )

            result = traced_agent_call(
                "assessment",
                get_agent("assessment"),
                assessment_prompt(),
                assessment_input,
            )
            return {"result": result}

        # FULL WORKFLOW
        if decision == "full_workflow":
            crm = traced_agent_call(
                "crm",
                get_agent("crm"),
                crm_prompt(),
                "Retrieve customer details for this customer.",
            )
            policy = traced_agent_call(
                "policy",
                get_agent("policy"),
                policy_prompt(),
                "Summarise key lending rules.",
            )
            rates = traced_agent_call(
                "rates",
                get_agent("rates"),
                rates_prompt(),
                "Summarise current demo interest rates.",
            )

            assessment_input = (
                f"Customer Data (JSON):\n{crm}\n\n"
                f"Policy Rules:\n{policy}\n\n"
                f"Interest Rates:\n{rates}\n\n"
                "The customer is asking about getting a home loan. "
                "Act like a banker and walk them through:\n"
                "- Whether they are likely to be approved\n"
                "- Roughly how much they could borrow\n"
                "- An example rate (e.g. around 5%)\n"
                "- Example repayments (e.g. $X per month over 20 or 30 years)\n"
                "Answer in a friendly, conversational way."
            )

            result = traced_agent_call(
                "assessment",
                get_agent("assessment"),
                assessment_prompt(),
                assessment_input,
            )
            return {"result": result}

        # FALLBACK
        return {
            "result": (
                "I couldn't classify your request. Try asking about home loan policy, "
                "interest rates, your details, or say something like "
                "\"Can you assess if I can borrow for a home loan?\""
            )
        }

# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run()
