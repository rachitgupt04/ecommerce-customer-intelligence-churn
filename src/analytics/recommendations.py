"""
Prescriptive Business Recommendations & Retention Playbooks.
Maps calibrated churn probabilities, RFM segments, and monetary value into
economically justified, ROI-positive customer retention interventions.
"""

from typing import Dict, Any
from src.utils.config import RISK_THRESHOLDS


class RetentionRecommendationEngine:
    """Derives operational retention strategies tailored to customer risk and value profiles."""

    @staticmethod
    def get_risk_tier(churn_probability: float) -> str:
        """Categorizes churn probability into business operational risk tiers."""
        if churn_probability >= RISK_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif churn_probability >= RISK_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        else:
            return "LOW"

    @classmethod
    def generate_recommendation(
        cls,
        segment: str,
        risk_level: str,
        churn_prob: float,
        monetary: float,
        recency: float,
        top_risk_factor: str = "",
    ) -> Dict[str, Any]:
        """
        Derives an operational business retention strategy based on empirical customer metrics.
        Ensures retention costs are economically justified relative to customer LTV.
        """
        is_high_value = any(term in str(segment).lower() for term in ["champions", "vip", "loyal", "at-risk high-value"]) or (monetary >= 300.0)

        if is_high_value and risk_level == "HIGH":
            action_title = "VIP Priority Intervention & Concierge Outreach"
            strategy = (
                f"Customer has generated high cumulative revenue (${monetary:,.2f}) but shows critical churn risk "
                f"({churn_prob:.1%}). Immediate proactive intervention required: Assign dedicated concierge agent, "
                f"dispatch an executive appreciation note with a margin-safe retention credit ($30-$40), and audit "
                f"delivery friction ({top_risk_factor})."
            )
            recommended_budget = "$30 - $45"
            delivery_channel = "Direct Account Specialist Email / Priority WhatsApp"
            operational_urgency = "URGENT (Action within 48 Hours)"

        elif is_high_value and risk_level == "MEDIUM":
            action_title = "Proactive VIP Relationship Nurturing"
            strategy = (
                f"High-value customer (${monetary:,.2f}) showing moderate inactivity ({recency:.0f} days). "
                f"Incentivize reactivation via exclusive early access to top category restocks and complimentary "
                f"freight subsidy on their next order."
            )
            recommended_budget = "$15 - $25"
            delivery_channel = "Personalized VIP Newsletter & SMS"
            operational_urgency = "Moderate (Within 7 Days)"

        elif is_high_value and risk_level == "LOW":
            action_title = "Brand Loyalty Amplification & Referral"
            strategy = (
                f"Highly engaged, low-risk customer. Maximize retention and lifetime value through loyalty club "
                f"invitations, tiered rewards, and peer referral incentives. Do not offer margin-diluting discounts."
            )
            recommended_budget = "$5 - $10"
            delivery_channel = "In-App Loyalty Portal & Email"
            operational_urgency = "Scheduled Quarterly"

        elif not is_high_value and risk_level == "HIGH":
            action_title = "Automated Digital Win-Back Sequence"
            strategy = (
                f"Lower spending customer (${monetary:,.2f}) with high churn risk ({churn_prob:.1%}). "
                f"Manual high-touch outreach is not ROI-positive. Enroll into automated 3-stage drip email workflow "
                f"featuring category-specific recommendations and an expiring 10% coupon on orders above $30."
            )
            recommended_budget = "$1 - $3 (Automated digital)"
            delivery_channel = "Automated Marketing Automation (Email/Push)"
            operational_urgency = "Automated System Trigger"

        elif not is_high_value and risk_level == "MEDIUM":
            action_title = "Cross-Category Discovery & Re-Engagement"
            strategy = (
                f"Customer purchase cadence is slowing down ({recency:.0f} days since order). Trigger algorithmic "
                f"recommendations of trending products in complementary categories with high review ratings."
            )
            recommended_budget = "$0.50 - $1.50"
            delivery_channel = "Dynamic Recommendation Email"
            operational_urgency = "Bi-Weekly Marketing Cycle"

        else:
            action_title = "Standard Nurture & Organic Engagement"
            strategy = (
                f"Stable behavioral profile with low churn risk ({churn_prob:.1%}). Maintain standard organic "
                f"engagement, seasonal product drops, and educational content. No intervention cost required."
            )
            recommended_budget = "$0.00 (Organic)"
            delivery_channel = "Standard Weekly Newsletter"
            operational_urgency = "Routine Cadence"

        return {
            "action_title": action_title,
            "strategy": strategy,
            "recommended_budget": recommended_budget,
            "delivery_channel": delivery_channel,
            "operational_urgency": operational_urgency,
            "is_high_value": is_high_value,
        }

