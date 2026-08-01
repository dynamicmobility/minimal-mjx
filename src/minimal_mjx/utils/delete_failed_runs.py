import argparse
import sys
import wandb


def _confirm(prompt: str) -> bool:
    """Asks the user a yes/no question, re-prompting until the answer is valid."""
    while True:
        try:
            answer = input(f"{prompt} [y/n]: ").strip().lower()
        except EOFError:
            # No interactive stdin available; treat as a decline
            print()
            return False

        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        print("Please answer 'y' or 'n'.")


def delete_failed_runs(entity: str, project: str, assume_yes: bool = False) -> None:
    """Finds and deletes all failed/crashed runs in a specified W&B project,

    printing the number of associated logged artifacts for each run. Each
    deletion is confirmed interactively unless ``assume_yes`` is set.
    """
    # Initialize the public API
    api = wandb.Api()

    path = f"{entity}/{project}"
    print(f"Fetching runs from: {path}...")

    try:
        # Retrieve all runs for the project path
        runs = api.runs(path)
    except Exception as e:
        print(f"Error connecting to project path '{path}': {e}", file=sys.stderr)
        return

    deleted_count = 0
    skipped_count = 0

    for run in runs:
        # Filter for failed or crashed states
        if run.state in ["failed", "crashed"]:
            try:
                # Retrieve and count the logged artifacts for this run
                logged_artifacts = list(run.logged_artifacts())
                artifact_count = len(logged_artifacts)
            except Exception:
                artifact_count = 0

            print(f"Run: {run.name} ({run.id})")
            print(f"  - State: {run.state}")
            print(f"  - Logged Artifacts: {artifact_count}")

            if not assume_yes and not _confirm(
                f"Delete this run and its {artifact_count} artifact(s)?"
            ):
                print("  - Skipped.")
                print("-" * 40)
                skipped_count += 1
                continue

            print("  - Deleting...")
            print("-" * 40)

            # Deletes the run and all of its associated artifacts
            run.delete(delete_artifacts=True)
            deleted_count += 1

    print(
        f"\nExecution finished. Successfully deleted {deleted_count} runs "
        f"({skipped_count} skipped)."
    )


def main() -> None:
    # Set up the command-line interface argument parser
    parser = argparse.ArgumentParser(
        description="Delete failed and crashed W&B runs along with their logged artifacts."
    )

    # Positional arguments require inputs; change to flags if you want defaults
    parser.add_argument(
        "-e",
        "--entity",
        type=str,
        required=True,
        help="Your W&B team name or personal username.",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        required=True,
        help="The specific W&B project name.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Delete without asking for confirmation on each run.",
    )

    # Parse inputs from the console
    args = parser.parse_args()

    # Run the main task
    delete_failed_runs(
        entity=args.entity, project=args.project, assume_yes=args.yes
    )


if __name__ == "__main__":
    main()