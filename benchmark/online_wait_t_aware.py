import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


Position = Tuple[int, int]


@dataclass
class Passenger:
    name: str
    origin: Position
    destination: Position
    request_time: int
    pickup_time: Optional[int] = None
    dropoff_time: Optional[int] = None
    status: str = "future"   # future, waiting, onboard, served


@dataclass
class Shuttle:
    position: Position
    capacity: int
    passengers_onboard: List[str]


class WaitTimeAwareGreedyOnline:
    """
    Online wait-time-aware greedy heuristic.

    At each decision point, among currently actionable targets:
      - waiting passenger pickup
      - onboard passenger dropoff

    choose the one minimizing:

      pickups:  distance - alpha * waiting_time
      dropoffs: distance - beta  * in_vehicle_time

    Lower score is better.
    """

    def __init__(
        self,
        passengers: List[Passenger],
        grid_size: int = 10,
        shuttle_start: Position = (5, 5),
        shuttle_capacity: int = 6,
        max_steps: int = 1000,
        model_path: str = "heuristic://wait_time_aware_greedy_online",
        alpha: float = 0.5,
        beta: float = 0.25,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.model_path = model_path
        self.alpha = alpha
        self.beta = beta

        self.passengers: Dict[str, Passenger] = {p.name: p for p in passengers}
        self.passenger_order = [p.name for p in passengers]

        self.shuttle = Shuttle(
            position=shuttle_start,
            capacity=shuttle_capacity,
            passengers_onboard=[],
        )

        self.time = 0
        self.generated_count = 0
        self.served_count = 0
        self.total_reward = None  # keep standardized format

    # -----------------------------
    # Geometry / movement
    # -----------------------------
    @staticmethod
    def manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def move_one_step_toward(self, cur: Position, target: Position) -> Position:
        x, y = cur
        tx, ty = target

        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        elif y < ty:
            y += 1
        elif y > ty:
            y -= 1

        return (x, y)

    # -----------------------------
    # Passenger revelation
    # -----------------------------
    def reveal_new_passengers(self):
        for p in self.passengers.values():
            if p.status == "future" and p.request_time <= self.time:
                p.status = "waiting"
                self.generated_count += 1

    # -----------------------------
    # Current sets
    # -----------------------------
    def waiting_passengers(self) -> List[Passenger]:
        return [p for p in self.passengers.values() if p.status == "waiting"]

    def onboard_passengers(self) -> List[Passenger]:
        return [self.passengers[name] for name in self.shuttle.passengers_onboard]

    def all_served(self) -> bool:
        return all(p.status == "served" for p in self.passengers.values())

    # -----------------------------
    # Service actions
    # -----------------------------
    def process_pickups_and_dropoffs(self):
        # Dropoffs first
        for name in list(self.shuttle.passengers_onboard):
            p = self.passengers[name]
            if self.shuttle.position == p.destination:
                p.dropoff_time = self.time
                p.status = "served"
                self.shuttle.passengers_onboard.remove(name)
                self.served_count += 1

        # Then pickups
        if len(self.shuttle.passengers_onboard) < self.shuttle.capacity:
            available = self.shuttle.capacity - len(self.shuttle.passengers_onboard)
            pickup_candidates = [
                p for p in self.waiting_passengers()
                if p.origin == self.shuttle.position
            ]
            for p in pickup_candidates[:available]:
                p.pickup_time = self.time
                p.status = "onboard"
                self.shuttle.passengers_onboard.append(p.name)

    # -----------------------------
    # Greedy scoring
    # -----------------------------
    def choose_wait_time_aware_target(self) -> Optional[Tuple[str, str]]:
        """
        Returns:
            ("pickup", passenger_name) or ("dropoff", passenger_name)

        Score:
            pickup  -> distance - alpha * waiting_time
            dropoff -> distance - beta  * in_vehicle_time

        Lower score is better.
        Tie-breaking:
            1) lower score
            2) lower raw distance
            3) prefer dropoff over pickup
            4) lexicographic passenger name
        """
        candidates = []

        # Dropoffs for onboard passengers
        for p in self.onboard_passengers():
            dist = self.manhattan(self.shuttle.position, p.destination)
            in_vehicle_time = self.time - p.pickup_time if p.pickup_time is not None else 0
            score = dist - self.beta * in_vehicle_time
            candidates.append((score, dist, 0, "dropoff", p.name))

        # Pickups for waiting passengers if capacity available
        if len(self.shuttle.passengers_onboard) < self.shuttle.capacity:
            for p in self.waiting_passengers():
                dist = self.manhattan(self.shuttle.position, p.origin)
                waiting_time = self.time - p.request_time
                score = dist - self.alpha * waiting_time
                candidates.append((score, dist, 1, "pickup", p.name))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[4]))
        _, _, _, target_type, passenger_name = candidates[0]
        return (target_type, passenger_name)

    def get_target_position(self, target: Tuple[str, str]) -> Position:
        target_type, passenger_name = target
        p = self.passengers[passenger_name]
        return p.origin if target_type == "pickup" else p.destination

    # -----------------------------
    # Main simulation
    # -----------------------------
    def run_episode(self, episode_id: int = 1) -> Dict:
        # Initial reveal/service at t=0
        self.reveal_new_passengers()
        self.process_pickups_and_dropoffs()

        while self.time < self.max_steps:
            if self.all_served():
                break

            target = self.choose_wait_time_aware_target()

            if target is not None:
                target_pos = self.get_target_position(target)
                if self.shuttle.position != target_pos:
                    self.shuttle.position = self.move_one_step_toward(
                        self.shuttle.position, target_pos
                    )

            # Service immediately after movement at current time
            self.process_pickups_and_dropoffs()

            # Advance simulation clock
            self.time += 1

            # Reveal passengers that become available at the new current time
            self.reveal_new_passengers()

        result = {
            "episode_id": episode_id,
            "model_path": self.model_path,
            "total_reward": self.total_reward,
            "steps": self.time,
            "env_time": self.time,
            "num_passengers_generated": self.generated_count,
            "num_passengers_served": self.served_count,
            "passengers": [],
        }

        for name in self.passenger_order:
            p = self.passengers[name]

            waiting_time = (
                p.pickup_time - p.request_time
                if p.pickup_time is not None else None
            )
            in_vehicle_time = (
                p.dropoff_time - p.pickup_time
                if p.pickup_time is not None and p.dropoff_time is not None else None
            )
            service_time = (
                p.dropoff_time - p.request_time
                if p.dropoff_time is not None else None
            )

            result["passengers"].append({
                "name": p.name,
                "origin": list(p.origin),
                "destination": list(p.destination),
                "request_time": p.request_time,
                "pickup_time": p.pickup_time,
                "dropoff_time": p.dropoff_time,
                "waiting_time": waiting_time,
                "in_vehicle_time": in_vehicle_time,
                "service_time": service_time,
            })

        return result


# ---------------------------------------
# File I/O helpers
# ---------------------------------------
def load_episode_file(json_path: str) -> Tuple[int, Position, int, int, List[Passenger]]:
    with open(json_path, "r") as f:
        episode_data = json.load(f)

    episode_id = episode_data.get("episode_id", 1)

    initial_state = episode_data.get("initial_shuttle_state", {})
    shuttle_start = tuple(initial_state.get("position", [5, 5]))
    shuttle_capacity = initial_state.get("capacity", 6)

    grid_size = episode_data.get("grid_size", 10)

    passengers = []
    for p in episode_data["passengers"]:
        passengers.append(
            Passenger(
                name=p["name"],
                origin=tuple(p["origin"]),
                destination=tuple(p["destination"]),
                request_time=p["request_time"],
            )
        )

    return episode_id, shuttle_start, shuttle_capacity, grid_size, passengers


def solve_directory(
    input_dir: str,
    output_dir: str,
    max_steps: int = 1000,
    alpha: float = 0.5,
    beta: float = 0.25,
):
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    json_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".json")
    )

    for fname in json_files:
        in_path = os.path.join(input_dir, fname)

        episode_id, shuttle_start, shuttle_capacity, grid_size, passengers = load_episode_file(in_path)

        solver = WaitTimeAwareGreedyOnline(
            passengers=passengers,
            grid_size=grid_size,
            shuttle_start=shuttle_start,
            shuttle_capacity=shuttle_capacity,
            max_steps=max_steps,
            model_path="heuristic://wait_time_aware_greedy_online",
            alpha=alpha,
            beta=beta,
        )

        result = solver.run_episode(episode_id=episode_id)
        all_results.append(result)

        out_name = os.path.splitext(fname)[0] + "_wait_time_aware_greedy_online_result.json"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved {out_path}")

    summary_path = os.path.join(output_dir, "wait_time_aware_greedy_online_all_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved combined summary to {summary_path}")


if __name__ == "__main__":
    INPUT_DIR = "/tf/rl_project/data/realized_demand_eval"
    OUTPUT_DIR = "/tf/rl_project/data/online_wait_t_aware_benchmark"

    solve_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_steps=1000,
        alpha=0.5,   # stronger urgency for waiting pickups
        beta=0.25,   # moderate urgency for onboard dropoffs
    )