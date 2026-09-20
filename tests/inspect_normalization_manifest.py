import os
import json

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CLIENTS = [
    "client_1",
    "client_2",
    "client_3",
]


def manifest_path(client_id):
    return os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "graphs",
        client_id,
        "normalization_manifest.json",
    )


def describe(obj, indent=0):

    prefix = " " * indent

    if isinstance(obj, dict):

        print(
            f"{prefix}DICT ({len(obj)} keys)"
        )

        for key, value in obj.items():

            print(
                f"{prefix}KEY: {key}"
            )

            describe(
                value,
                indent + 4
            )

    elif isinstance(obj, list):

        print(
            f"{prefix}LIST ({len(obj)} items)"
        )

        if len(obj) > 0:

            print(
                f"{prefix}FIRST ITEM TYPE: "
                f"{type(obj[0]).__name__}"
            )

            # Only inspect first item so output
            # doesn't become enormous.
            describe(
                obj[0],
                indent + 4
            )

    else:

        print(
            f"{prefix}VALUE TYPE: "
            f"{type(obj).__name__}"
        )


print()
print("=" * 70)
print("RFGN NORMALIZATION MANIFEST STRUCTURE")
print("=" * 70)

for client_id in CLIENTS:

    print()
    print("=" * 70)
    print(client_id.upper())
    print("=" * 70)

    path = manifest_path(
        client_id
    )

    print(
        f"Path: {path}"
    )

    if not os.path.exists(path):

        print(
            "MANIFEST NOT FOUND"
        )

        continue

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    print()
    print(
        "JSON structure:"
    )

    describe(
        manifest
    )

    print()
    print(
        "Top-level keys:"
    )

    if isinstance(
        manifest,
        dict,
    ):

        for key in manifest.keys():

            print(
                f"  - {key}"
            )

print()
print("=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)