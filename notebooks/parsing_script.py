import pandas as pd
import ast
import re

# Load the CSV file
df = pd.read_csv('../datasets/submission (3).csv', header=None, names=['id', 'content'])

# Function to extract the assistant's answer

def extract_first_answer(content):
    try:
        # Regular expression pattern to find the assistant's answer list
        pattern = r"assistant\n(\[.*?\])"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            raw_answer = match.group(1).strip()

            # Try parsing it as a list using ast.literal_eval
            try:
                answer_list = ast.literal_eval(raw_answer)
                if isinstance(answer_list, list) and len(answer_list) > 0:
                    return answer_list[0]
            except (SyntaxError, ValueError):
                pass  # Fall back to manual extraction if parsing fails

        # Alternative extraction method if list parsing fails
        pattern_alt = r"assistant\n\[\s*'([А-Я])'"
        match_alt = re.search(pattern_alt, content)
        if match_alt:
            return match_alt.group(1)

    except Exception as e:
        print(f"Error processing content: {e}")

    return content  # Return None if extraction fails

# Apply the function to extract answers
df['answer'] = df['content'].apply(extract_first_answer)

# Drop rows where answer extraction failed
df = df.dropna(subset=['answer'])

# Format the output as 'id: answer'
df['formatted_output'] = df['id'].astype(str) + ': ' + df['answer']

# Display the results
for output in df['formatted_output']:
    print(output)

df = df[1:]
df.loc[:, ["id", "answer"]].rename(columns={"answer": "correct_answers"}).to_csv("../datasets/submissions_postprocessed.csv", index=False)
