from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.workflow.graph import app


load_dotenv()

logger = get_logger(__name__)


def format_response(result):
    """Extract response from workflow result."""

    if isinstance(result, dict) and "generation" in result:
        return result["generation"]

    if isinstance(result, dict) and "answer" in result:
        return result["answer"]

    return str(result)


def run_query(question: str):
    """
    Execute the Adaptive RAG workflow for a question.
    """

    logger.info("Processing question: %s", question)

    result = None

    for output in app.stream({"question": question}):

        for key, value in output.items():

            logger.info("Workflow node completed: %s", key)

            result = value

    if result:
        answer = format_response(result)

        logger.info("Question processed successfully")

        return answer

    logger.warning("No response generated")

    return "No response generated."


def main():

    print("Adaptive RAG System")
    print("Type 'quit' to exit.\n")

    while True:

        try:

            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q", ""]:
                break

            print("Processing...")

            answer = run_query(question)

            print(f"\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as exc:

            logger.exception(
                "Unexpected error while processing question"
            )

            print(f"Error: {exc}")


if __name__ == "__main__":
    main()