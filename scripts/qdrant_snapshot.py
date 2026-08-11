
"""
OmniBrain Qdrant Snapshot & Restore Utility.

Commands:

    Backup:
        python scripts/qdrant_snapshot.py backup

    List snapshots:
        python scripts/qdrant_snapshot.py list

    Restore:
        python scripts/qdrant_snapshot.py restore <collection> <snapshot_file>

The script works with the local Qdrant instance used by OmniBrain.
"""

import argparse
from pathlib import Path
from typing import List

import requests

from omnibrain.vectorstore.collections import (
    IMAGE_COLLECTION,
    TEXT_COLLECTION,
)
from omnibrain.vectorstore.qdrant_client import (
    QdrantClientWrapper,
)


SNAPSHOT_DIR = Path("snapshots")


def get_client():
    """Return the configured Qdrant client."""
    return QdrantClientWrapper().client()


def get_qdrant_url() -> str:
    """Return the Qdrant REST URL."""
    return "http://localhost:6333"


def get_collections() -> List[str]:
    """Return OmniBrain collections that should be backed up."""
    return [
        TEXT_COLLECTION,
        IMAGE_COLLECTION,
    ]


def create_backup() -> None:
    """Create snapshots for all OmniBrain collections."""

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = get_client()

    print("=" * 60)
    print("Creating Qdrant snapshots")
    print("=" * 60)

    for collection_name in get_collections():

        print(
            f"\nCollection: {collection_name}"
        )

        if not client.collection_exists(
            collection_name
        ):
            print(
                "SKIPPED - collection does not exist."
            )
            continue

        snapshot = client.create_snapshot(
            collection_name=collection_name
        )

        print(
            f"Snapshot created: {snapshot.name}"
        )

        print(
            "Snapshot is stored by Qdrant."
        )

    print(
        "\nBackup creation completed."
    )


def list_backups() -> None:
    """List snapshots available for each collection."""

    client = get_client()

    print("=" * 60)
    print("Available Qdrant snapshots")
    print("=" * 60)

    for collection_name in get_collections():

        print(
            f"\n{collection_name}:"
        )

        if not client.collection_exists(
            collection_name
        ):
            print(
                "  Collection does not exist."
            )
            continue

        snapshots = client.list_snapshots(
            collection_name=collection_name
        )

        if not snapshots:
            print(
                "  No snapshots found."
            )
            continue

        for snapshot in snapshots:
            print(
                f"  - {snapshot.name}"
            )


def download_snapshot(
    collection_name: str,
    snapshot_name: str,
) -> Path:
    """Download a Qdrant snapshot to the local snapshots directory."""

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = (
        f"{get_qdrant_url()}"
        f"/collections/{collection_name}"
        f"/snapshots/{snapshot_name}"
    )

    output_path = (
        SNAPSHOT_DIR / snapshot_name
    )

    print(
        f"Downloading snapshot:"
    )
    print(
        f"  {snapshot_name}"
    )

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )

    print(
        f"Saved to: {output_path}"
    )

    return output_path


def download_all_snapshots() -> None:
    """Download the newest snapshot for every collection."""

    client = get_client()

    for collection_name in get_collections():

        if not client.collection_exists(
            collection_name
        ):
            continue

        snapshots = client.list_snapshots(
            collection_name=collection_name
        )

        if not snapshots:
            print(
                f"No snapshot available for "
                f"{collection_name}"
            )
            continue

        latest = snapshots[-1]

        download_snapshot(
            collection_name=collection_name,
            snapshot_name=latest.name,
        )


def restore_snapshot(
    collection_name: str,
    snapshot_file: str,
) -> None:
    """Restore a local snapshot file into Qdrant."""

    snapshot_path = Path(
        snapshot_file
    )

    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: "
            f"{snapshot_path}"
        )

    url = (
        f"{get_qdrant_url()}"
        f"/collections/{collection_name}"
        f"/snapshots/upload"
        f"?priority=snapshot"
    )

    print("=" * 60)
    print("Restoring Qdrant snapshot")
    print("=" * 60)

    print(
        f"Collection: {collection_name}"
    )

    print(
        f"Snapshot: {snapshot_path}"
    )

    with snapshot_path.open(
        "rb"
    ) as snapshot:

        response = requests.post(
            url,
            files={
                "snapshot": (
                    snapshot_path.name,
                    snapshot,
                    "application/octet-stream",
                )
            },
            timeout=300,
        )

    response.raise_for_status()

    print(
        "Snapshot restored successfully."
    )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "OmniBrain Qdrant "
            "Snapshot & Restore Utility"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "backup",
            "list",
            "download",
            "restore",
        ],
        help="Snapshot operation to perform",
    )

    parser.add_argument(
        "collection",
        nargs="?",
        help="Qdrant collection name",
    )

    parser.add_argument(
        "snapshot",
        nargs="?",
        help="Snapshot file/name",
    )

    args = parser.parse_args()

    if args.command == "backup":

        create_backup()

    elif args.command == "list":

        list_backups()

    elif args.command == "download":

        download_all_snapshots()

    elif args.command == "restore":

        if not args.collection:
            parser.error(
                "restore requires a collection name"
            )

        if not args.snapshot:
            parser.error(
                "restore requires a snapshot file"
            )

        restore_snapshot(
            collection_name=args.collection,
            snapshot_file=args.snapshot,
        )


if __name__ == "__main__":
    main()