import json
from datetime import datetime

from services.emergency_rag import EmergencyRAGService
from services.urban_planner import UrbanPlanner


class CityOperationsAgent:
    """
    Agentic AI for City Operations
    Orchestrates multiple CitySense360 modules
    """

    def __init__(self):
        self.emergency_rag = EmergencyRAGService()
        self.urban_planner = UrbanPlanner()

    def classify_event(self, event_text: str) -> str:
        text = event_text.lower()

        if any(k in text for k in ["accident", "collision", "fire", "injury", "crowd"]):
            return "emergency"

        if any(k in text for k in ["aqi", "pollution", "smog", "air quality"]):
            return "pollution"

        if any(k in text for k in ["traffic", "congestion", "jam"]):
            return "traffic"

        if any(k in text for k in ["planning", "city layout", "urban design"]):
            return "planning"

        return "general"


    def handle_event(self, event_text: str) -> dict:
        event_type = self.classify_event(event_text)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "event_text": event_text,
            "actions": [],
        }

        # Emergency handling (Module 6)
        if event_type == "emergency":
            response = self.emergency_rag.handle_incident(event_text)
            report["actions"].append({
                "type": "EMERGENCY_RESPONSE",
                "output": response["response"]
            })

        # Pollution advisory
        elif event_type == "pollution":
            response = self.emergency_rag.handle_incident(event_text)
            report["actions"].append({
                "type": "POLLUTION_ADVISORY",
                "output": response["response"]
            })

        # Traffic advisory
        elif event_type == "traffic":
            report["actions"].append({
                "type": "TRAFFIC_ADVISORY",
                "output": (
                    "Traffic congestion detected. "
                    "Recommend rerouting and adaptive signal optimization."
                )
            })

        # Urban planning
        elif event_type == "planning":
            image, image_path = self.urban_planner.generate_plan(
                prompt=event_text,
                filename="agent_generated_city_plan.png",
            )
            report["actions"].append({
                "type": "URBAN_PLANNING",
                "output": "Urban planning visualization generated",
                "artifact": image_path
            })

        else:
            report["actions"].append({
                "type": "MONITORING",
                "output": "No immediate action required."
            })

        return report


if __name__ == "__main__":
    agent = CityOperationsAgent()

    event = """
    CCTV detected a vehicle collision at a busy intersection.
    Crowd gathering observed and possible injuries reported.
    """

    result = agent.handle_event(event)
    print(json.dumps(result, indent=2))
