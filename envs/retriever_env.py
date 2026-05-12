from .base_env import BaseEnv


class RetrieverEnv(BaseEnv):
    def __init__(self, parameters=None):
        super().__init__()
        self.parameters = parameters or {}
        self.corpus = self.parameters.get("corpus", [])
        self.stressor_mode = self.parameters.get("stressor_mode", "none")
        self.replay_outputs = self.parameters.get("replay_outputs", {})
        self.counterfactual_outputs = self.parameters.get("counterfactual_outputs", {})

    def _score(self, query, title):
        # Metadata retrieval: rank by title only.
        q_tokens = [t for t in query.lower().split() if t]
        t_low = title.lower()
        score = 0
        for tok in q_tokens:
            if tok in t_low:
                score += 1
        return score

    def search_corpus(self, query, top_k=3):
        scored = []
        for d in self.corpus:
            s = self._score(query, d.get("title", ""))
            scored.append((s, d))
        scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        hits = []
        for s, d in scored[:max(1, top_k)]:
            snippet = d.get("text", "")[:100] + ("..." if len(d.get("text", "")) > 100 else "")
            hits.append({
                "id": d.get("id", ""),
                "title": d.get("title", ""),
                "snippet": snippet,
                "score": s,
            })
        return {
            "success": True,
            "data": {
                "query": query,
                "total_hits": len(scored),
                "returned": len(hits),
                "hits": hits,
            },
        }

    def read_doc(self, doc_id):
        for d in self.corpus:
            if d.get("id") == doc_id:
                return {
                    "success": True,
                    "data": {
                        "doc_id": d.get("id", ""),
                        "title": d.get("title", ""),
                        "content": d.get("text", ""),
                        "word_count": len(d.get("text", "").split()),
                        "source": "local_corpus",
                    },
                }
        return {"success": False, "message": "Document not found."}
