"""Shared nominal navigation commands; no simulator, IK, or physics changes.

Direction applies to the translation toward each waypoint. A reverse segment
uses negative Joint9 vx while yaw still denotes the robot's heading.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def wrap_angle(value):
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def route_directions(plan):
    route = plan["route"]
    directions = plan.get("route_directions", [1] * len(route))
    if not route or len(directions) != len(route):
        raise ValueError("Each navigation waypoint requires one direction")
    if any(direction not in (-1, 1) for direction in directions):
        raise ValueError("Navigation direction must be +1 or -1")
    if np.asarray(route).shape != (len(route), 3) or not np.isfinite(route).all():
        raise ValueError("Navigation waypoints must be finite [x, y, yaw] poses")
    return directions


def waypoint_command(xy, yaw, goal, direction=1):
    """Return vx, yaw-rate, reached with the original limits and tolerances."""
    if direction not in (-1, 1):
        raise ValueError("Navigation direction must be +1 or -1")
    goal = np.asarray(goal, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if goal.shape != (3,) or xy.shape != (2,) or not np.isfinite(np.r_[goal, xy, yaw]).all():
        raise ValueError("Navigation state and goal must be finite poses")
    error = goal[:2] - xy
    distance = float(np.linalg.norm(error))
    yaw_error = wrap_angle(goal[2] - yaw)
    if distance < .003 and abs(yaw_error) < .004:
        return 0., 0., True
    if distance >= .003:
        heading = math.atan2(error[1], error[0]) + (math.pi if direction < 0 else 0.)
        heading_error = wrap_angle(heading - yaw)
        if abs(heading_error) > .015:
            return 0., float(np.clip(2 * heading_error, -.6, .6)), False
        return direction * min(.25, distance / .04), float(np.clip(heading_error, -.1, .1)), False
    return 0., float(np.clip(2 * yaw_error, -.6, .6)), False


def compact_nominal_plan(source):
    """Replace only R1's far-side navigation; this is not a launch certificate."""
    selected = source.get("selected") or {}
    expected_goal = [1.817, -1.213, math.atan2(.65, -.75)]
    if (source.get("status") != "candidate_found"
            or any(abs(float(selected.get(k, float("inf"))) - v) > 1e-9
                   for k, v in [("radius", .65), ("distance", .75), ("z", .95)])
            or not np.allclose(source.get("nominal_robot", []), [4.1, .18, .5, math.pi], atol=1e-9, rtol=0)
            or not np.allclose(source["route"][-1], expected_goal, atol=1e-9, rtol=0)):
        raise ValueError("This reviewed route is limited to the unchanged R1 nominal candidate")
    plan = copy.deepcopy(source)
    plan["prior_navigation"] = {
        "route": plan["route"], "route_directions": route_directions(plan),
        "route_audit": plan.pop("route_audit", []),
    }
    plan["route"] = [
        [1.18, .18, math.pi],
        [1.18, -1.05, -math.pi / 2],
        list(source["route"][-1]),
    ]
    plan["route_directions"] = [1, 1, -1]
    plan["navigation_revision"] = "nominal_compact_reverse_v1"
    plan["navigation_validation_status"] = "requires_independent_validation"
    plan["limitations"] = list(plan.get("limitations", [])) + [
        "Navigation changed; prior route audit does not apply to this route.",
        "Source-geometry checks do not certify imported shapes, self collision, or physical execution.",
    ]
    route_directions(plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    plan = compact_nominal_plan(json.loads(source.read_text()))
    plan["navigation_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with Path(args.output).open("x") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
