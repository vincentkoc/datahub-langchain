from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class LineageEdge:
    """Represents a lineage relationship between entities"""
    source_id: str
    target_id: str
    edge_type: str
    metadata: Optional[Dict] = None

class LineageTracker:
    """Tracks lineage relationships between entities"""

    def __init__(self):
        self.edges: List[LineageEdge] = []
        self._entity_relationships = {}

    def add_edge(self, source_id: str, target_id: str, edge_type: str, metadata: Dict = None) -> None:
        """Add a lineage edge"""
        edge = LineageEdge(source_id, target_id, edge_type, metadata)
        self.edges.append(edge)

        # Track relationships for quick lookup
        if source_id not in self._entity_relationships:
            self._entity_relationships[source_id] = {"upstream": [], "downstream": []}
        if target_id not in self._entity_relationships:
            self._entity_relationships[target_id] = {"upstream": [], "downstream": []}

        self._entity_relationships[source_id]["downstream"].append(edge)
        self._entity_relationships[target_id]["upstream"].append(edge)

    def get_upstream(self, entity_id: str) -> List[LineageEdge]:
        """Get upstream lineage for an entity"""
        return self._entity_relationships.get(entity_id, {}).get("upstream", [])

    def get_downstream(self, entity_id: str) -> List[LineageEdge]:
        """Get downstream lineage for an entity"""
        return self._entity_relationships.get(entity_id, {}).get("downstream", [])

    def get_lineage_graph(self) -> Dict:
        """Get complete lineage graph"""
        return {
            "nodes": list(self._entity_relationships.keys()),
            "edges": [vars(edge) for edge in self.edges]
        }
