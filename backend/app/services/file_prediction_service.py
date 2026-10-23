from pathlib import Path

from ml.src.code_metrics import analyze_file
from ml.src.git_metrics import extract_repo, aggregate
from backend.app.services.model_service import model_service


class FilePredictionService:

    def extract_metrics(self, file_path: str) -> dict:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if path.suffix != ".py":
            raise ValueError(
                "Only Python files are supported."
            )

        repo_root = self._find_git_root(path)

        if repo_root is None:
            raise ValueError(
                "File must belong to a Git repository."
            )

        relative_path = path.relative_to(
            repo_root
        ).as_posix()

        code = analyze_file(
            path,
            repo_root.name,
            relative_path,
        )

        if code is None:
            raise ValueError(
                "Unable to extract code metrics."
            )

        git_records = extract_repo(
            repo_root,
            repo_root.name,
        )

        git_data = aggregate(git_records)

        file_git = git_data[
            git_data["file_path"] == relative_path
        ]

        if file_git.empty:
            git_metrics = {
                "commit_count": 0,
                "author_count": 0,
                "lines_added": 0,
                "lines_deleted": 0,
                "churn": 0,
            }
        else:
            row = file_git.iloc[0]

            git_metrics = {
                "commit_count": int(
                    row["commit_count"]
                ),
                "author_count": int(
                    row["author_count"]
                ),
                "lines_added": int(
                    row["lines_added"]
                ),
                "lines_deleted": int(
                    row["lines_deleted"]
                ),
                "churn": int(
                    row["churn"]
                ),
            }

        return {
            "loc": code["loc"],
            "cyclomatic_complexity":
                code["cyclomatic_complexity"],
            "function_count":
                code["num_functions"],
            "class_count":
                code["num_classes"],
            **git_metrics,
        }

    def predict_file(self, file_path: str) -> dict:
        metrics = self.extract_metrics(file_path)

        prediction = model_service.predict(
            metrics
        )

        return {
            "file_path": file_path,
            "metrics": metrics,
            **prediction,
        }

    def _find_git_root(
        self,
        path: Path,
    ) -> Path | None:

        current = path.parent

        while current != current.parent:
            if (current / ".git").exists():
                return current

            current = current.parent

        return None


file_prediction_service = FilePredictionService()
