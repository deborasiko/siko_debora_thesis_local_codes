import pandas as pd
import spacy
import ast

def substituteSubjectWithHypernym(dataset_path, hypernym_path, output_file_path):
    # Load original data
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()  # clean up column names

    # Load hypernym mappings
    with open(hypernym_path, 'r') as f:
        hypernym_map = ast.literal_eval(f.read())

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    updated_rows = []

    for _, row in data.iterrows():
        premise = row['sentence_A']
        doc = nlp(premise)
        modified_tokens = []
        replaced = False

        for token in doc:
            if token.dep_ == "nsubj":
                subj = token.text.lower()
                if subj in hypernym_map and hypernym_map[subj]:
                    # Replace subject with its hypernym
                    replacement = hypernym_map[subj]
                    modified_tokens.append(replacement + token.whitespace_)
                    replaced = True
                else:
                    modified_tokens.append(token.text_with_ws)
            else:
                modified_tokens.append(token.text_with_ws)

        new_premise = ''.join(modified_tokens) if replaced else premise

        # Keep the rest of the columns the same
        updated_rows.append({
            'pair_ID': row['pair_ID'],
            'sentence_A': new_premise,
            'sentence_B': row['sentence_B'],
            'relatedness_score': row['relatedness_score'],
            'entailment_judgment': row['entailment_judgment']
        })

    # Save to new file, same structure
    pd.DataFrame(updated_rows).to_csv(output_file_path, index=False)

    print(f"Done! Output written to: {output_file_path}")

# Run it
substituteSubjectWithHypernym(
    dataset_path='../data_files/neutral_sick_data.txt',
    hypernym_path='../data_files/hyponyms.txt',
    output_file_path='../data_files/neutral_sick_hyponyms_data.txt'
)
