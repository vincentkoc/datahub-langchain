import json
import os
import argparse
from dotenv import load_dotenv
import requests

load_dotenv()

def pretty_print_json(data):
    print(json.dumps(data, indent=2))

def query_datahub(query, variables=None):
    """Execute a GraphQL query against DataHub"""
    server_url = os.getenv("DATAHUB_GMS_URL", "http://localhost:8080").split("#")[0].strip()
    token = os.getenv("DATAHUB_TOKEN")

    headers = {
        "Content-Type": "application/json"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            f"{server_url}/api/graphql",
            headers=headers,
            json={
                "query": query,
                "variables": variables or {}
            }
        )

        print(f"Request URL: {server_url}/api/graphql")
        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return {}

        return response.json()
    except Exception as e:
        print(f"Error querying DataHub: {str(e)}")
        return {}

def search_runs():
    """Search for all LLM runs"""
    query = """
    query searchDatasets {
        search(input: {
            type: DATASET,
            query: "platform:llm",
            start: 0,
            count: 10,
            orFilters: [{
                and: [{
                    field: "platform",
                    value: "llm"
                }]
            }]
        }) {
            total
            searchResults {
                entity {
                    urn
                    type
                    ... on Dataset {
                        name
                        properties {
                            description
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths {
                            path
                        }
                        relationships(input: {
                            types: ["RunsOn", "Uses", "PartOf"],
                            direction: OUTGOING,
                            start: 0,
                            count: 10
                        }) {
                            total
                            relationships {
                                type
                                entity {
                                    urn
                                    type
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    return query_datahub(query)

def get_run_details(run_urn):
    """Get detailed information about a specific run"""
    query = """
    query getDataset($urn: String!) {
        dataset(urn: $urn) {
            urn
            name
            properties {
                description
                customProperties {
                    key
                    value
                }
            }
            browsePaths {
                path
            }
            relationships(input: {
                types: ["RunsOn", "Uses", "PartOf"],
                direction: OUTGOING,
                start: 0,
                count: 10
            }) {
                total
                relationships {
                    type
                    entity {
                        urn
                        type
                    }
                }
            }
        }
    }
    """
    return query_datahub(query, {"urn": run_urn})

def search_llm_data(session, server_url):
    """Search for LLM-related data in DataHub"""
    print("\n=== Searching for LLM Types ===")

    query = """
    query {
        search(input: {type: DATASET, query: "llm", start: 0, count: 10}) {
            total
            searchResults {
                entity {
                    urn
                    type
                    ... on Dataset {
                        name
                        properties {
                            description
                            customProperties {
                                key
                                value
                            }
                        }
                    }
                }
            }
        }
    }
    """

    response = session.post(
        f"{server_url}/api/graphql",
        json={"query": query}
    )

    print(f"Request URL: {server_url}/api/graphql")
    print(f"Response status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def main():
    parser = argparse.ArgumentParser(description='Query DataHub API')
    parser.add_argument('--action', choices=['search', 'details'], default='search',
                      help='Action to perform')
    parser.add_argument('--urn', help='URN for detailed lookup')

    args = parser.parse_args()

    if args.action == 'search':
        print("\n=== Searching for LLM Runs ===")
        result = search_runs()
        pretty_print_json(result)
    elif args.action == 'details' and args.urn:
        print(f"\n=== Getting Details for {args.urn} ===")
        result = get_run_details(args.urn)
        pretty_print_json(result)
    else:
        print("Please specify --urn when using --action details")

if __name__ == "__main__":
    main()
