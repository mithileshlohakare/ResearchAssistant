from typing import Any, Dict, List


class WebResearchAgent:
    def run(self, user_goal: str, problem_type: str) -> Dict[str, Any]:
        topic = self._extract_topic(user_goal)
        candidates = self._build_candidates(topic=topic, problem_type=problem_type)
        ranked = self._rank_candidates(candidates)

        best_candidate = ranked[0] if ranked else None

        return {
            "status": "completed",
            "topic": topic,
            "problem_type": problem_type,
            "candidate_count": len(ranked),
            "best_candidate": best_candidate,
            "candidates": ranked
        }

    def _extract_topic(self, user_goal: str) -> str:
        goal = user_goal.lower().strip()

        prefixes = [
            "build a ",
            "build an ",
            "create a ",
            "create an ",
            "predict ",
            "forecast ",
            "detect ",
            "classify ",
            "train a ",
            "train an ",
        ]

        topic = goal
        for prefix in prefixes:
            if topic.startswith(prefix):
                topic = topic[len(prefix):]

        replacements = [
            " model",
            " system",
            " dataset",
            " prediction",
            " detector",
            " classifier",
        ]

        for item in replacements:
            topic = topic.replace(item, "")

        return topic.strip().title()

    def _build_candidates(self, topic: str, problem_type: str) -> List[Dict[str, Any]]:
        topic_slug = topic.lower().replace(" ", "-")

        candidates = [
            {
                "source": "Kaggle",
                "title": f"{topic} dataset",
                "url": f"https://www.kaggle.com/search?q={topic_slug}",
                "format": "csv/parquet/images",
                "license_known": False,
                "relevance_score": 0.92,
                "notes": "Strong general ML dataset source with competitions and tabular/image datasets."
            },
            {
                "source": "Hugging Face",
                "title": f"{topic} datasets",
                "url": f"https://huggingface.co/datasets?search={topic_slug}",
                "format": "dataset repo",
                "license_known": True,
                "relevance_score": 0.89,
                "notes": "Useful for NLP, vision, audio, and structured dataset repositories."
            },
            {
                "source": "UCI ML Repository",
                "title": f"{topic} dataset search",
                "url": f"https://archive.ics.uci.edu/ml/search?query={topic_slug}",
                "format": "csv/tabular",
                "license_known": True,
                "relevance_score": 0.84,
                "notes": "Trusted academic source for classical ML datasets."
            },
            {
                "source": "GitHub",
                "title": f"{topic} dataset repositories",
                "url": f"https://github.com/search?q={topic_slug}+dataset&type=repositories",
                "format": "csv/json/code",
                "license_known": False,
                "relevance_score": 0.75,
                "notes": "Useful fallback for public research repos and dataset mirrors."
            }
        ]

        if problem_type == "time_series":
            candidates.insert(0, {
                "source": "Yahoo Finance / Market Data",
                "title": f"{topic} historical market data",
                "url": f"https://finance.yahoo.com/",
                "format": "time_series",
                "license_known": False,
                "relevance_score": 0.96,
                "notes": "Strong source for stock and market historical time series."
            })

        if problem_type == "detection":
            candidates.insert(0, {
                "source": "Roboflow Universe",
                "title": f"{topic} computer vision datasets",
                "url": f"https://universe.roboflow.com/search?q={topic_slug}",
                "format": "images/annotations",
                "license_known": False,
                "relevance_score": 0.95,
                "notes": "Strong source for detection and annotated vision datasets."
            })

        return candidates

    def _rank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda item: item.get("relevance_score", 0),
            reverse=True
        )