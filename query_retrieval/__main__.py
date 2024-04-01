"""Main module for the query retrieval system."""

import argparse
import sys

from .query_retrieval import run_query_retrieval


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LlamaIndex-based query retrieval on local document files."
    )
    parser.add_argument("--query", type=str, help="Query to be used for retrieval.")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data",
        help="Directory containing the documents to be parsed.",
    )
    parser.add_argument(
        "--indexing_mode",
        type=str,
        choices=["summary", "vector", "keyword"],
        default="vector",
        help="Choose the indexing mode: summary, vector, or keyword table.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature value for the LLM. Range from 0 to 1.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Set the chunk size for processing.",
    )
    parser.add_argument(
        "--similarity_top_k",
        type=int,
        default=2,
        help="Set the top K similarities to consider.",
    )

    args = parser.parse_args()

    # Check arguments validity
    if args.temperature < 0 or args.temperature > 1:
        parser.print_help()
        print("\nerror: argument --temperature: must be between 0 and 1.\n")
        sys.exit(1)

    if args.query is None or args.query.strip() == "":
        parser.print_help()
        print("\nerror: the argument 'query' cannot be an empty string.\n")
        sys.exit(1)

    return args


def main():
    """Main function for the query retrieval system."""
    args = parse_arguments()
    run_query_retrieval(
        query=args.query,
        indexing_mode=args.indexing_mode,
        data_folder=args.data_folder,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        similarity_top_k=args.similarity_top_k,
    )


if __name__ == "__main__":
    main()
