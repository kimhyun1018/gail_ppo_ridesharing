import os
import json
import csv
import time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


REALIZED_DEMAND_DIR = "/tf/rl_project/data/realized_demand_eval"
OFFLINE_BENCHMARK_DIR = "/tf/rl_project/data/offline_benchmark"
SUMMARY_CSV_PATH = os.path.join(OFFLINE_BENCHMARK_DIR, "offline_benchmark_summary.csv")


def manhattan_distance(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])


def load_episode_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_data_model_from_episode(episode_data, vehicle_capacity=6, time_horizon=None):
    """
    Build OR-Tools data model from one realized-demand episode JSON.
    """
    data = {}

    passengers = sorted(episode_data["passengers"], key=lambda p: p["name"])
    starting_location = tuple(episode_data["initial_shuttle_state"]["position"])

    pickups = [tuple(p["origin"]) for p in passengers]
    dropoffs = [tuple(p["destination"]) for p in passengers]
    generate_steps = [int(p["request_time"]) for p in passengers]

    all_locations = [starting_location] + pickups + dropoffs
    num_passengers = len(passengers)

    if time_horizon is None:
        max_request = max(generate_steps) if generate_steps else 0
        total_direct_distance = sum(
            manhattan_distance(tuple(p["origin"]), tuple(p["destination"]))
            for p in passengers
        )
        time_horizon = max(100, max_request + total_direct_distance + 100)

    data["episode_id"] = episode_data["episode_id"]
    data["passengers"] = passengers
    data["starting_location"] = starting_location
    data["locations"] = all_locations
    data["num_locations"] = len(all_locations)
    data["num_passengers"] = num_passengers
    data["vehicle_capacity"] = vehicle_capacity
    data["vehicle_starts"] = [0]
    data["vehicle_ends"] = [0]
    data["num_vehicles"] = 1

    # depot = 0, pickups = +1, dropoffs = -1
    data["demands"] = [0] + [1] * num_passengers + [-1] * num_passengers

    # Important:
    # offline DARP knows all requests, but pickups cannot happen before request_time
    data["generate_steps"] = [0] + generate_steps + generate_steps

    data["distance_matrix"] = [
        [manhattan_distance(loc1, loc2) for loc2 in all_locations]
        for loc1 in all_locations
    ]

    data["time_horizon"] = time_horizon
    return data


def solve_darp_episode(data):
    """
    Solve one offline DARP episode and return standardized results.
    """
    manager = pywrapcp.RoutingIndexManager(
        data["num_locations"],
        data["num_vehicles"],
        data["vehicle_starts"],
        data["vehicle_ends"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [data["vehicle_capacity"]],
        True,
        "Capacity"
    )

    time_dimension_name = "Time"
    routing.AddDimension(
        transit_callback_index,
        data["time_horizon"],   # waiting slack
        data["time_horizon"],   # max route duration
        False,
        time_dimension_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)

    # Enforce node availability windows
    for node_index in range(data["num_locations"]):
        index = manager.NodeToIndex(node_index)
        time_window = (data["generate_steps"][node_index], data["time_horizon"])
        time_dimension.CumulVar(index).SetRange(*time_window)

    # Pickup-dropoff pairing and precedence
    for i in range(data["num_passengers"]):
        pickup_node = i + 1
        dropoff_node = i + 1 + data["num_passengers"]

        pickup_index = manager.NodeToIndex(pickup_node)
        dropoff_index = manager.NodeToIndex(dropoff_node)

        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index))
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index)
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30

    start_solve = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    computation_time = time.time() - start_solve

    if not solution:
        return {
            "episode_id": data["episode_id"],
            "method": "offline_ortools",
            "model_path": None,
            "total_reward": None,
            "steps": None,
            "env_time": None,
            "num_passengers_generated": data["num_passengers"],
            "num_passengers_served": 0,
            "passengers": [],
            "computation_time": computation_time,
            "total_travel_distance": None,
            "total_wait_time_route": None,
            "route_events": [],
            "solved": False,
            "error": "No solution found."
        }

    return build_standardized_result(
        data=data,
        manager=manager,
        routing=routing,
        solution=solution,
        time_dimension=time_dimension,
        computation_time=computation_time,
    )


def build_standardized_result(data, manager, routing, solution, time_dimension, computation_time):
    """
    Build result JSON with RL-matching core structure plus optional extras.
    """
    total_distance = 0
    total_wait_time_route = 0
    route_load = 0
    route_events = []

    passenger_results = []

    # Passenger-level standardized metrics
    for i, p in enumerate(data["passengers"], start=1):
        pickup_node = i
        dropoff_node = i + data["num_passengers"]

        pickup_index = manager.NodeToIndex(pickup_node)
        dropoff_index = manager.NodeToIndex(dropoff_node)

        pickup_time = solution.Value(time_dimension.CumulVar(pickup_index))
        dropoff_time = solution.Value(time_dimension.CumulVar(dropoff_index))
        request_time = int(p["request_time"])

        waiting_time = pickup_time - request_time
        in_vehicle_time = dropoff_time - pickup_time
        service_time = dropoff_time - request_time

        passenger_results.append({
            "name": p["name"],
            "origin": list(p["origin"]),
            "destination": list(p["destination"]),
            "request_time": request_time,
            "pickup_time": pickup_time,
            "dropoff_time": dropoff_time,
            "waiting_time": waiting_time,
            "in_vehicle_time": in_vehicle_time,
            "service_time": service_time
        })

    # Route-level extras
    index = routing.Start(0)

    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        arrival_time = solution.Value(time_dimension.CumulVar(index))
        current_location = data["locations"][node_index]
        route_load += data["demands"][node_index]

        route_events.append({
            "node_id": node_index,
            "location": list(current_location),
            "load": route_load,
            "arrival_time": arrival_time
        })

        next_index = solution.Value(routing.NextVar(index))
        next_node_index = manager.IndexToNode(next_index)

        travel_time = data["distance_matrix"][node_index][next_node_index]
        total_distance += travel_time

        if not routing.IsEnd(next_index):
            next_arrival_time = solution.Value(time_dimension.CumulVar(next_index))
            wait_time = max(0, next_arrival_time - arrival_time - travel_time)
            total_wait_time_route += wait_time

        index = next_index

    # Final depot node
    node_index = manager.IndexToNode(index)
    arrival_time = solution.Value(time_dimension.CumulVar(index))
    current_location = data["locations"][node_index]

    route_events.append({
        "node_id": node_index,
        "location": list(current_location),
        "load": route_load,
        "arrival_time": arrival_time
    })

    total_simulation_time = arrival_time

    result = {
        # Standardized core structure
        "episode_id": data["episode_id"],
        "method": "offline_ortools",
        "model_path": None,
        "total_reward": None,
        "steps": total_simulation_time,
        "env_time": total_simulation_time,
        "num_passengers_generated": data["num_passengers"],
        "num_passengers_served": len(passenger_results),
        "passengers": passenger_results,

        # Optional extras
        "computation_time": computation_time,
        "total_travel_distance": total_distance,
        "total_wait_time_route": total_wait_time_route,
        "route_events": route_events,
        "solved": True,
        "load_on_vehicle_at_end": route_load
    }

    return result


def save_episode_result(output_dir, episode_id, result):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"episode_{episode_id:04d}_offline.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def append_summary_row(summary_csv_path, result):
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)

    file_exists = os.path.exists(summary_csv_path)
    fieldnames = [
        "episode_id",
        "method",
        "solved",
        "steps",
        "env_time",
        "num_passengers_generated",
        "num_passengers_served",
        "total_travel_distance",
        "total_wait_time_route",
        "computation_time"
    ]

    with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row = {k: result.get(k) for k in fieldnames}
        writer.writerow(row)


def print_episode_result(result):
    print(f"\n=== Episode {result['episode_id']} ===")
    print(f"Method: {result['method']}")
    print(f"Solved: {result['solved']}")

    if not result["solved"]:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print(f"Steps: {result['steps']}")
    print(f"Env Time: {result['env_time']}")
    print(f"Passengers Served: {result['num_passengers_served']} / {result['num_passengers_generated']}")
    print(f"Total Travel Distance: {result['total_travel_distance']}")
    print(f"Route Waiting Time: {result['total_wait_time_route']}")
    print(f"Computation Time: {result['computation_time']:.2f} sec")

    print("\nPassenger Performance:")
    for p in result["passengers"]:
        print(
            f"{p['name']}: "
            f"request={p['request_time']}, "
            f"pickup={p['pickup_time']}, "
            f"dropoff={p['dropoff_time']}, "
            f"waiting={p['waiting_time']}, "
            f"in_vehicle={p['in_vehicle_time']}, "
            f"service={p['service_time']}"
        )


def run_offline_benchmark(
    realized_demand_dir=REALIZED_DEMAND_DIR,
    output_dir=OFFLINE_BENCHMARK_DIR,
    summary_csv_path=SUMMARY_CSV_PATH,
    vehicle_capacity=6,
    print_each_episode=True,
):
    os.makedirs(output_dir, exist_ok=True)

    episode_files = sorted(
        f for f in os.listdir(realized_demand_dir)
        if f.endswith(".json") and not f.endswith("_performance.json")
    )

    if not episode_files:
        print(f"No episode JSON files found in: {realized_demand_dir}")
        return

    for episode_file in episode_files:
        file_path = os.path.join(realized_demand_dir, episode_file)
        episode_data = load_episode_file(file_path)

        data = create_data_model_from_episode(
            episode_data=episode_data,
            vehicle_capacity=vehicle_capacity,
            time_horizon=None,
        )

        result = solve_darp_episode(data)
        save_episode_result(output_dir, episode_data["episode_id"], result)
        append_summary_row(summary_csv_path, result)

        if print_each_episode:
            print_episode_result(result)

    print(f"\nOffline benchmark results saved to: {output_dir}")
    print(f"Offline benchmark summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    run_offline_benchmark(
        realized_demand_dir="/tf/rl_project/data/realized_demand_eval",
        output_dir="/tf/rl_project/data/offline_benchmark",
        summary_csv_path="/tf/rl_project/data/offline_benchmark/offline_benchmark_summary.csv",
        vehicle_capacity=6,
        print_each_episode=True,
    )