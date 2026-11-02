"""
field_explanations.py

Plain-English explanations for input features used in CredifyAI.
"""

FIELD_EXPLANATIONS = {
    "dti": {
        "label": "Debt-to-Income (DTI)",
        "description": "The percentage of your income used to repay all debts.",
        "why_it_matters": (
            "Higher DTI means less free income each month, "
            "making it harder to absorb financial shocks."
        ),
        "healthy_range": "< 30% is generally considered healthy"
    },
    "credit_utilization": {
        "label": "Credit Utilization",
        "description": "How much of your available credit you are currently using.",
        "why_it_matters": (
            "High utilization suggests financial strain and "
            "leaves little room for emergencies."
        ),
        "healthy_range": "< 30% is generally considered healthy"
    },
    "emi_to_income": {
        "label": "EMI-to-Income Ratio",
        "description": "The portion of your income spent on loan EMIs.",
        "why_it_matters": (
            "This is one of the strongest indicators of repayment stress."
        ),
        "healthy_range": "< 30% is typically manageable"
    },
    "loan_to_income": {
        "label": "Loan-to-Income Ratio",
        "description": "Total loan exposure compared to your annual income.",
        "why_it_matters": (
            "Higher values indicate long-term debt burden."
        ),
        "healthy_range": "< 3 is generally acceptable"
    },
    "active_loan_count": {
        "label": "Active Loan Count",
        "description": "The number of loans currently active.",
        "why_it_matters": (
            "Managing multiple loans increases repayment complexity."
        ),
        "healthy_range": "Lower is generally better"
    },
    "delinquency_count": {
        "label": "Delinquency Count",
        "description": "Number of times loan payments were missed or delayed.",
        "why_it_matters": (
            "Past delinquency is a strong signal of future repayment issues."
        ),
        "healthy_range": "0 is ideal"
    },
}
