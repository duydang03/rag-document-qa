from app.generation.llm import generate_answer


def rewrite_query(question: str) -> str:
    """Expand selected BDD100K questions with document-specific keywords."""

    q = question.lower()

    if "time-of-day" in q or "time of day" in q:
        return (
            f"{question} "
            "BDD100K time-of-day categories Dawn/Dusk Daytime Night"
        )

    if "weather categories" in q or "weather conditions" in q:
        return (
            f"{question} "
            "BDD100K weather categories Clear Partly Cloudy "
            "Overcast Rainy Snowy Foggy"
        )

    if "scene categories" in q or "scene types" in q:
        return (
            f"{question} "
            "BDD100K scene categories city street tunnel highway "
            "residential parking gas station"
        )

    return question
