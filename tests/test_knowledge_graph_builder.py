import json
import unittest
from tools.knowledge_graph_builder import knowledge_graph_builder

class TestKnowledgeGraphBuilder(unittest.TestCase):
    def test_knowledge_graph_builder_concept_limit(self):
        # Create 501 concepts
        concepts = [{"id": f"c{i}", "label": f"Concept {i}"} for i in range(501)]
        relationships = []

        result = knowledge_graph_builder(concepts, relationships)
        expected_json = json.dumps({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

        self.assertEqual(result, expected_json)

    def test_knowledge_graph_builder_relationship_limit(self):
        # Create 3001 relationships
        concepts = []
        relationships = [{"source": "c1", "target": "c2", "type": "related_to"} for _ in range(3001)]

        result = knowledge_graph_builder(concepts, relationships)
        expected_json = json.dumps({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

        self.assertEqual(result, expected_json)

if __name__ == '__main__':
    unittest.main()
