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
    query searchLLMEntities {
        search(input: {
            type: ALL,
            query: "*",
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
                        browsePaths
                    }
                    ... on MLModel {
                        name
                        description
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                    ... on DataJob {
                        name
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                    ... on DataFlow {
                        name
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                }
                searchScore
            }
        }
    }
    """
    return query_datahub(query)

def get_run_details(run_urn):
    """Get detailed information about a specific run"""
    query = """
    query getEntity($urn: String!) {
        entity(urn: $urn) {
            urn
            type
            ... on DataJob {
                name
                properties {
                    customProperties {
                        key
                        value
                    }
                }
                inputOutput {
                    inputDatasets {
                        dataset { urn }
                    }
                    outputDatasets {
                        dataset { urn }
                    }
                }
                status {
                    status
                    startTime
                    endTime
                }
                browsePaths
            }
            ... on MLModel {
                name
                description
                properties {
                    customProperties {
                        key
                        value
                    }
                }
                institutionalMemory {
                    elements {
                        url
                        description
                        created {
                            time
                        }
                    }
                }
                deprecation {
                    deprecated
                    decommissionTime
                }
            }
        }
    }
    """
    return query_datahub(query, {"urn": run_urn})

def search_by_type(entity_type, platform="llm"):
    """Search for entities of a specific type"""
    query = """
    query searchByType($type: EntityType!, $platform: String!) {
        search(input: {
            type: $type,
            query: "*",
            start: 0,
            count: 10,
            orFilters: [{
                and: [{
                    field: "platform",
                    value: $platform
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
                        browsePaths
                    }
                    ... on MLModel {
                        name
                        description
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                    ... on DataJob {
                        name
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                    ... on DataFlow {
                        name
                        properties {
                            customProperties {
                                key
                                value
                            }
                        }
                        browsePaths
                    }
                }
            }
        }
    }
    """
    return query_datahub(query, {
        "type": entity_type,
        "platform": platform
    })

def main():
    parser = argparse.ArgumentParser(description='Query DataHub API')
    parser.add_argument('--action', choices=['search', 'details', 'models', 'runs', 'chains'],
                      default='search', help='Action to perform')
    parser.add_argument('--urn', help='URN for detailed lookup')
    parser.add_argument('--type', choices=['DATASET', 'MLMODEL', 'DATAJOB', 'DATAFLOW'],
                      help='Entity type to search for')

    args = parser.parse_args()

    if args.action == 'search':
        print("\n=== Searching for LLM Entities ===")
        result = search_runs()
        pretty_print_json(result)
    elif args.action == 'details' and args.urn:
        print(f"\n=== Getting Details for {args.urn} ===")
        result = get_run_details(args.urn)
        pretty_print_json(result)
    elif args.action in ['models', 'runs', 'chains']:
        entity_type = {
            'models': 'MLMODEL',
            'runs': 'DATAJOB',
            'chains': 'DATAFLOW'
        }[args.action]
        print(f"\n=== Searching for {args.action} ===")
        result = search_by_type(entity_type)
        pretty_print_json(result)
    else:
        print("Please specify --urn when using --action details")

if __name__ == "__main__":
    main()
