

from __future__ import annotations

import argparse
from pathlib import Path


BLACKLIST_PATH = Path(__file__).resolve().parents[1] / "audio_corruption" / "blacklist.txt"


def load_blacklist(blacklist_path: Path) -> set[str]:
	"""Load consultation IDs from the blacklist file."""
	if not blacklist_path.exists():
		raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")

	ids: set[str] = set()
	with blacklist_path.open("r", encoding="utf-8") as f:
		for line in f:
			consultation_id = line.strip()
			if consultation_id:
				ids.add(consultation_id)
	return ids


def remove_blacklisted_txt_files(target_dir: Path, blacklist_ids: set[str]) -> int:
	"""Delete .txt files whose stem is listed in blacklist_ids."""
	if not target_dir.exists() or not target_dir.is_dir():
		raise NotADirectoryError(f"Target directory not found or not a directory: {target_dir}")

	removed_count = 0
	for file_path in target_dir.rglob("*.txt"):
		if file_path.stem in blacklist_ids:
			file_path.unlink()
			removed_count += 1
			print(f"Removed: {file_path}")

	return removed_count


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Remove blacklisted .txt files from a target directory recursively."
	)
	parser.add_argument(
		"target_directory",
		type=Path,
		help="Directory to scan recursively for .txt files to remove.",
	)
	args = parser.parse_args()

	blacklist_ids = load_blacklist(BLACKLIST_PATH)
	removed_count = remove_blacklisted_txt_files(args.target_directory, blacklist_ids)
	print(f"Done. Removed {removed_count} file(s).")


if __name__ == "__main__":
	main()