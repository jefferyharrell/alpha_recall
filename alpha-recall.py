#!/usr/bin/env python3
"""
Command-line tool for interacting with the alpha-recall memory system.
Usage:
    python alpha-recall.py <verb> [options]

Example:
    python alpha-recall.py create-entity --entity "Sparkle" --type "Cat"
    python alpha-recall.py add-observation --entity "Sparkle" --observation "Sparkle is a cat."
    python alpha-recall.py create-relation --entity "Sparkle" --to-entity "Kylee Peña" --as-type "belongs_to"
    python alpha-recall.py get-entity --entity "Sparkle" --depth 1

Supported verbs:
    create-entity
    add-observation
    create-relation
    get-entity

"""
import argparse
import asyncio
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alpha_recall.db.neo4j_db import Neo4jDatabase

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="alpha-recall command-line tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-entity
    p_create = subparsers.add_parser("create-entity", help="Create an entity")
    p_create.add_argument("--entity", required=True, help="Entity name")
    p_create.add_argument("--type", required=False, help="Entity type")

    # add-observation
    p_obs = subparsers.add_parser("add-observation", help="Add an observation to an entity")
    p_obs.add_argument("--entity", required=True, help="Entity name")
    p_obs.add_argument("--observation", required=True, help="Observation text")

    # create-relation
    p_rel = subparsers.add_parser("create-relation", help="Create a relationship between entities")
    p_rel.add_argument("--entity", required=True, help="Source entity name")
    p_rel.add_argument("--to-entity", required=True, help="Target entity name")
    p_rel.add_argument("--as-type", required=True, help="Relationship type")

    # get-entity
    p_get = subparsers.add_parser("get-entity", help="Get entity and relationships")
    p_get.add_argument("--entity", required=True, help="Entity name")
    p_get.add_argument("--depth", type=int, default=1, help="Relationship depth (default: 1)")

    # delete-entity
    p_delete = subparsers.add_parser("delete-entity", help="Delete an entity and all its relationships")
    p_delete.add_argument("--entity", required=True, help="Entity name to delete")

    return parser.parse_args()

async def main():
    args = parse_args()
    db = Neo4jDatabase()
    await db.connect()

    def serialize_for_json(obj):
        # Neo4j Node/Relationship objects have .keys() and can be cast to dict
        # Also handle lists and dicts recursively
        from collections.abc import Mapping
        if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
            try:
                return {k: serialize_for_json(obj[k]) for k in obj.keys()}
            except Exception:
                return str(obj)
        elif isinstance(obj, Mapping):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_for_json(x) for x in obj]
        else:
            return obj

    def pretty_print(output):
        if isinstance(output, (dict, list)):
            print(json.dumps(serialize_for_json(output), indent=2, ensure_ascii=False))
        else:
            print(output)

    if args.command == "create-entity":
        result = await db.create_entity(args.entity, args.type)
        pretty_print(result)
    elif args.command == "add-observation":
        result = await db.add_observation(args.entity, args.observation)
        pretty_print(result)
    elif args.command == "create-relation":
        result = await db.create_relationship(args.entity, args.to_entity, args.as_type)
        pretty_print(result)
    elif args.command == "get-entity":
        result = await db.get_entity(args.entity, args.depth)
        pretty_print(result)
    elif args.command == "delete-entity":
        result = await db.delete_entity(args.entity)
        pretty_print(result)
    else:
        print(f"Unknown command: {args.command}")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
