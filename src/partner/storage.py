# Human-readable persistence for a Session (docs/BUILD_PLAN.md milestone 2).
# Save/load to sessions/ so a record is readable outside the application.
import json
from dataclasses import dataclass
from pathlib import Path
from partner.models import Session, SessionStatus


@dataclass
class StorageJson:
    file_path: Path

    def save_session(self, session: Session) -> None:
        session_json = {}
        session_json["goal"] = session.goal
        session_json["project_path"] = str(session.project_path)
        session_json["constraints"] = session.constraints
        session_json["acceptance_criteria"] = session.acceptance_criteria
        session_json["status"] = session.status.value
        session_json["evidence"] = session.evidence
        session_json["explanation_checkpoint"] = session.explanation_checkpoint
        session_json["next_action"] = session.next_action

        with open(self.file_path, "w") as f:
            json.dump(session_json, f, indent=4)
            
    def load_session(self) -> Session:
        with open(Path(self.file_path), "r") as f:
            self.session_json = json.load(f)
        return Session(
            goal=self.session_json["goal"],
            project_path=Path(self.session_json["project_path"]),
            constraints=self.session_json["constraints"],
            acceptance_criteria=self.session_json["acceptance_criteria"],
            status=SessionStatus(self.session_json["status"]),
            evidence=self.session_json["evidence"],
            explanation_checkpoint=self.session_json["explanation_checkpoint"],
            next_action=self.session_json["next_action"],
        )
        