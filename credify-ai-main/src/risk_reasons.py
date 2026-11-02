"""
risk_reasons.py

Human-readable risk explanation rules for CredifyAI.
Enhanced with structured data for premium UI generation.
"""

def generate_risk_reasons(features: dict):
    """
    Generate structured, directional reasons explaining risk outcome.
    """
    reasons = []

    if features["emi_to_income"] > 0.4:
        reasons.append({"feature": "EMI Ratio", "direction": "up", "type": "risk", "impact": 18, "text": "High EMI Ratio detected"})
    elif features["emi_to_income"] < 0.2:
        reasons.append({"feature": "EMI Ratio", "direction": "down", "type": "safe", "impact": 10, "text": "Low, healthy EMI Ratio"})

    if features["loan_to_income"] > 3:
        reasons.append({"feature": "Loan Exposure", "direction": "up", "type": "risk", "impact": 15, "text": "High total loan burden"})

    if features["credit_utilization"] > 50:
        reasons.append({"feature": "Credit Utilization", "direction": "up", "type": "risk", "impact": 22, "text": "High revolving credit usage"})
    else:
        reasons.append({"feature": "Credit Utilization", "direction": "down", "type": "safe", "impact": 12, "text": "Healthy credit utilization"})

    if features["dti"] > 40:
        reasons.append({"feature": "Debt-to-Income", "direction": "up", "type": "risk", "impact": 20, "text": "Overall debt stress is high"})

    if features["delinquency_count"] > 0:
        reasons.append({"feature": "Delinquency", "direction": "up", "type": "risk", "impact": 25, "text": "Past missed payments recorded"})
    else:
        reasons.append({"feature": "Delinquency", "direction": "down", "type": "safe", "impact": 5, "text": "Perfect repayment history"})

    # Sort so risk items appear at the top
    reasons.sort(key=lambda x: (x["type"] == "safe", -x["impact"]))
    
    return reasons[:4] # Return top 4 insights

def generate_insights(features: dict, prob_high_risk: float):
    """Generate structured advisory insight text."""
    pros = []
    cons = []
    
    if features["credit_utilization"] <= 30:
        pros.append("Low credit utilization is significantly lowering your risk.")
    elif features["credit_utilization"] >= 50:
        cons.append("High credit utilization is elevating your risk. Target <30%.")
        
    if features["delinquency_count"] == 0:
        pros.append("Maintaining zero delinquency strengthens your profile.")
    else:
        cons.append("Recent delinquencies heavily impact your score.")
        
    if features["emi_to_income"] <= 0.3:
        pros.append("Excellent EMI ratio shows strong repayment capacity.")
    elif features["emi_to_income"] >= 0.45:
        cons.append("Reducing EMI burden will rapidly improve your eligibility.")

    if features["dti"] <= 30:
        pros.append("Healthy overall debt-to-income balance.")
    elif features["dti"] >= 50:
        cons.append("High total debt obligations are raising your risk tier.")

    if not pros:
        pros.append("Maintain consistent payment history to slowly build profile strength.")
    if not cons:
        cons.append("No critical risk factors detected. Continue current financial habits.")
        
    return {"pros": pros, "cons": cons}

def generate_simulations(features: dict, prob_high_risk: float, current_band: str):
    """Generate What-If scenarios."""
    sims = []
    
    if current_band != "Low Risk":
        if features["credit_utilization"] > 50:
            sims.append("If credit utilization drops below 30%, your risk could reduce to Low Risk.")
        elif features["emi_to_income"] > 0.4:
            sims.append("If your EMI ratio drops below 0.30, your risk tier could improve.")
        elif features["delinquency_count"] > 0:
            sims.append("As recent delinquencies age past 12 months, your risk score will naturally stabilize.")
            
    if not sims:
        sims.append("Continue current financial habits to maintain or improve this score.")
        
    return sims
