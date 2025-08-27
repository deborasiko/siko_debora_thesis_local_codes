import pandas as pd

def substituteWithEmptyPremise(dataset_path, output_file_path):
    data = pd.read_csv(dataset_path, sep='\t', engine='python')
    data.columns = data.columns.str.strip()

    updated_rows = []

    for _, row in data.iterrows():
        premise = row['sentence_A']
        hypothesis = row['sentence_B']

        updated_rows.append(
            {
                'pair_ID': row['pair_ID'],
                'sentence_A': "",
                'sentence_B': row['sentence_B'],
                'relatedness_score': row['relatedness_score'],
                'entailment_judgment': row['entailment_judgment']
            }
        )

        pd.DataFrame(updated_rows).to_csv(output_file_path, index=False)

        print(f"Done! Output written to: {output_file_path}")

substituteWithEmptyPremise(
    dataset_path='SICK_train.txt',
    output_file_path='only_hypothesis.txt'
)
