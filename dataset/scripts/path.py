from pathlib import Path
from collections import defaultdict

# Maximum number of files to display per folder
MAX_FILES = 2

# Maximum number of folders with the same first 3 letters
MAX_SAME_PREFIX = 2
PREFIX_LENGTH = 3


def print_tree(directory: Path, prefix=""):
    try:
        items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    # Separate folders and files
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]

    # Keep only two folders for each 3-letter prefix
    grouped = defaultdict(list)
    for folder in folders:
        key = folder.name[:PREFIX_LENGTH].lower()
        grouped[key].append(folder)

    selected_folders = []
    skipped_folders = 0

    for key in sorted(grouped):
        group = grouped[key]
        selected_folders.extend(group[:MAX_SAME_PREFIX])
        skipped_folders += max(0, len(group) - MAX_SAME_PREFIX)

    # Keep order alphabetical
    selected_folders.sort(key=lambda x: x.name.lower())

    # Keep only first two files
    selected_files = files[:MAX_FILES]
    skipped_files = max(0, len(files) - MAX_FILES)

    display_items = selected_folders + selected_files

    for index, item in enumerate(display_items):
        is_last = index == len(display_items) - 1
        connector = "└── " if is_last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension)

    if skipped_folders:
        print(prefix + f"├── ... ({skipped_folders} more similar folders)")
    if skipped_files:
        print(prefix + f"└── ... ({skipped_files} more files)")


if __name__ == "__main__":
    root = Path(__file__).parent.resolve()

    print(root.name)
    print_tree(root)