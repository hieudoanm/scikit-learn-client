from pathlib import Path
import json
import csv

x = range(1, 1001)
y = [item * 2 for item in x]
demo = [{"x": x_value, "y": y_value} for x_value, y_value in zip(x, y)]

json_path = "./demo.json"
csv_path = "./demo.csv"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(demo, file, indent=2)

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["x", "y"])
    writer.writeheader()
    writer.writerows(demo)

print(f"Created {json_path} and {csv_path} with {len(demo)} rows.")


for filename in (json_path, csv_path):
    path = Path(filename)
    stat = path.stat()

    print(
        json.dumps(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "format": path.suffix.lstrip("."),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "last_modified": stat.st_mtime,
                "exists": path.is_file(),
            },
            indent=2,
        )
    )
