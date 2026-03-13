import json
from debate import debate_answer
import random

dataset = []

with open("test.jsonl", "r") as f:
    for line in f:
        dataset.append(json.loads(line))


# Uncomment to view length of dataset
# print("Loaded:", len(dataset))

# Uncomment to view sample dataset question and answer
# print(dataset[:3])

def main():
    results = []

    for item in dataset[:150]:  # limited to only 150 questions and answers for testing
        question = item["question"]
        answers = [item["options"][letter] for letter in ["A", "B", "C", "D", "E"]]
        correct_letter = item["answer_idx"]

        try:
            result = debate_answer(question, answers)

            result["correct"] = correct_letter

            result["initial_correct"] = result["initial"] == correct_letter
            result["final_correct"] = result["final"] == correct_letter

            result["correction"] = (
                    not result["initial_correct"] and result["final_correct"]
            )

            result["harm"] = (
                    result["initial_correct"] and not result["final_correct"]
            )

            results.append(result)

        except RuntimeError as e:
            print("Stopped early:", e)
            break

    total = len(results)

    initial_accuracy = sum(r["initial_correct"] for r in results) / total
    final_accuracy = sum(r["final_correct"] for r in results) / total
    influence_rate = sum(r["changed"] for r in results) / total
    correction_rate = sum(r["correction"] for r in results) / total
    harm_rate = sum(r["harm"] for r in results) / total

    print("\n===== RESULTS =====")
    print("Total Questions:", total)
    print("Initial Accuracy:", round(initial_accuracy, 3))
    print("Final Accuracy:", round(final_accuracy, 3))
    print("Influence Rate:", round(influence_rate, 3))
    print("Correction Rate:", round(correction_rate, 3))
    print("Harm Rate:", round(harm_rate, 3))

    # Uncomment to view a sample response from diagnostician -> critic -> diagnostician
    # print(results[:4])


if __name__ == "__main__":
    main()
