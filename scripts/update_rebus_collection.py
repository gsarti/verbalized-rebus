# Script to add new rebuses to the eureka-rebus dataset
# Usage (from root folder): python update_rebus_collection.py --new_path <new_rebus_path>
# <new_rebus_path> is a CSV file exported from Eureka5 containing the new rebuses. The dataset can be exported by
# filtering e.g. all rebus for the current year (deduplication is performed by the script).
# The script will merge the new rebuses with the existing eureka-rebus dataset.
# After updating the collection:
# - Rerun process_data.py to update the derived datasets.
# - Update the last update date of the dataset in eureka-rebus/README.md.
# - Add an minor entry to eureka-rebus/CHANGELOG.md specifying the number of added rebuses and the date of the update.
# - Commit and push changes to the eureka-rebus repository.
# - Add a tag to the dataset using the selected minor version: 
#       huggingface_hub.create_tag("gsarti/eureka-rebus", tag="X.X", repo_type="dataset")
import argparse
import pandas as pd

def update_rebus_collection(args: argparse.Namespace):
    old = pd.read_csv(args.old_path, escapechar="\\")
    new = pd.read_csv(args.new_path, escapechar="\\")

    # Make sure to filter only new rebuses
    diff = new[~new['FRASE'].isin(old['FRASE'])]
    print(f"{len(diff)} new rebuses found. Adding them to {args.old_path}...")
    updated = pd.concat([old, diff], ignore_index=True)
    updated.to_csv(args.old_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_path", type=str, required=True)
    parser.add_argument("--old_path", type=str, default="eureka-rebus/rebus.csv")
    args = parser.parse_args()
    update_rebus_collection(args)