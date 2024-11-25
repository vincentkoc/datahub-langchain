"""Platform extension registration for DataHub"""
import requests
from typing import Optional, Dict, List
from ..config import ObservabilityConfig

# Define all supported platforms
SUPPORTED_PLATFORMS: List[Dict] = [
    {
        "name": "LangChain",
        "displayName": "LangChain",
        "type": "OTHERS",
        "logoUrl": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langchain-ipuhh4qo1jz5ssl4x0g2a.png/langchain-dp1uxj2zn3752pntqnpfu2.png?_a=DAJFJtWIZAAC",
        "info": "LangChain Framework for LLM Applications",
        "category": "ML_FRAMEWORK"
    },
    {
        "name": "LangSmith",
        "displayName": "LangSmith",
        "type": "OTHERS",
        "logoUrl": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langsmith-ipuhh4qo1jz5ssl4x0g2a.png/langsmith-dp1uxj2zn3752pntqnpfu2.png?_a=DAJFJtWIZAAC",
        "info": "LangSmith Observability Platform",
        "category": "OBSERVABILITY"
    }
]

class DataHubPlatformExtender:
    """Handles registration of custom platforms in DataHub"""

    def __init__(self, gms_server: str, token: Optional[str] = None):
        self.gms_server = gms_server
        self.token = token
        self.headers = {
            "Content-Type": "application/json"
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def register_platform(self, platform_info: dict) -> None:
        """Register a platform in DataHub"""
        payload = {
            "entity": {
                "value": {
                    "com.linkedin.metadata.snapshot.DataPlatformSnapshot": {
                        "urn": f"urn:li:dataPlatform:{platform_info['name'].lower()}",
                        "aspects": [
                            {
                                "com.linkedin.dataplatform.DataPlatformInfo": {
                                    "datasetNameDelimiter": "/",
                                    "name": platform_info["name"],
                                    "displayName": platform_info["displayName"],
                                    "type": platform_info.get("type", "OTHERS"),
                                    "logoUrl": platform_info.get("logoUrl", ""),
                                    "info": platform_info.get("info", ""),
                                    "platform": platform_info["name"].lower(),
                                    "category": platform_info.get("category", "OTHERS")
                                }
                            }
                        ]
                    }
                }
            }
        }

        response = requests.post(
            f"{self.gms_server}/entities?action=ingest",
            json=payload,
            headers=self.headers
        )

        if response.status_code != 200:
            raise Exception(f"Failed to register platform {platform_info['name']}: {response.text}")
        print(f"✓ Successfully registered {platform_info['name']} platform")

    def register_all_platforms(self) -> None:
        """Register all supported platforms"""
        for platform in SUPPORTED_PLATFORMS:
            try:
                self.register_platform(platform)
            except Exception as e:
                print(f"Failed to register {platform['name']}: {e}")
