import os
import re

README_FILE = "README.md"
TABLE_HEADER = "| Folder Name"  # Line that marks the start of the table

def get_all_folders():
    return sorted([f for f in os.listdir('.') if os.path.isdir(f) and re.match(r'^[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}$', f)])

def read_documented_folders():
    if not os.path.exists(README_FILE):
        return []

    documented = set()
    with open(README_FILE, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\|\s*(\w{3}\d{2}_\d{2}-\d{2}-\d{2})\s*\|", line)
            if match:
                documented.add(match.group(1))
    return documented

def prompt_for_descriptions(folders):
    entries = []
    for folder in folders:
        description = input(f"Enter description for {folder}: ").strip()
        # Extract date and time
        parts = folder.split('_')
        date_str = parts[0]
        time_str = parts[1].replace('-', ':')
        date_fmt = f"20{date_str[3:5]}-{month_to_number(date_str[:3])}-{date_str[5:]}"
        entries.append(f"| {folder} | {date_fmt} | {time_str} | {description} |")
    return entries

def month_to_number(month_abbr):
    months = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    return months.get(month_abbr, '??')

def append_to_readme(entries):
    if not os.path.exists(README_FILE):
        # If no README exists, create one with header
        with open(README_FILE, 'w', encoding='utf-8') as f:
            f.write("# Experiment Logs\n\n")
            f.write("## Folder Overview\n\n")
            f.write("| Folder Name | Date | Time | Description |\n")
            f.write("|-------------|------|------|-------------|\n")
            for entry in entries:
                f.write(entry + "\n")
    else:
        with open(README_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find where to insert entries
        try:
            table_end = max(i for i, line in enumerate(lines) if line.startswith("|"))
        except ValueError:
            # Table doesn't exist, recreate header
            lines.append("\n## Folder Overview\n\n")
            lines.append("| Folder Name | Date | Time | Description |\n")
            lines.append("|-------------|------|------|-------------|\n")
            table_end = len(lines) - 1

        for entry in entries:
            lines.insert(table_end + 1, entry + "\n")
            table_end += 1

        with open(README_FILE, 'w', encoding='utf-8') as f:
            f.writelines(lines)

def main():
    all_folders = get_all_folders()
    documented = read_documented_folders()
    missing = [f for f in all_folders if f not in documented]

    if not missing:
        print("✅ All folders are already documented.")
        return

    print(f"📝 Found {len(missing)} undocumented folders.")
    new_entries = prompt_for_descriptions(missing)
    append_to_readme(new_entries)
    print("✅ README.md has been updated.")

if __name__ == "__main__":
    main()
