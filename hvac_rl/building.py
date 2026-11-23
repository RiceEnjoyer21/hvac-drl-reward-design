from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sinergym

from hvac_rl.envs import ZONES

BUILDING_FILE = Path(sinergym.__file__).parent / "data" / "buildings" / "5ZoneAutoDXVAV.epJSON"
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@lru_cache(maxsize=1)
def load_epjson() -> dict:
    return json.loads(BUILDING_FILE.read_text())


def _by_zone(objs: dict, field: str) -> dict:
    return {v["zone_or_zonelist_or_space_or_spacelist_name"]: v[field] for v in objs.values()}


def _orientation(zone: str, d: dict) -> str:
    north_axis = d["Building"]["Building"].get("north_axis", 0.0)
    for wall in d["BuildingSurface:Detailed"].values():
        if (wall["zone_name"] == zone and wall["surface_type"] == "Wall"
                and wall["outside_boundary_condition"] == "Outdoors"):
            v = np.array([[p[f"vertex_{a}_coordinate"] for a in "xyz"] for p in wall["vertices"]])
            nx, ny, _ = np.cross(v[1] - v[0], v[2] - v[0])
            azimuth = (np.degrees(np.arctan2(nx, ny)) + north_axis) % 360
            return COMPASS[int(((azimuth + 22.5) % 360) // 45)]
    return "core"


def zones_table() -> pd.DataFrame:
    d = load_epjson()
    people = _by_zone(d["People"], "number_of_people")
    lights = _by_zone(d["Lights"], "lighting_level")
    equip = _by_zone(d["ElectricEquipment"], "design_level")
    rows = []
    for z in ZONES:
        area = d["Zone"][z]["volume"] / d["Zone"][z]["ceiling_height"]
        rows.append({
            "zone": z, "orientation": _orientation(z, d), "area_m2": area, "volume_m3": d["Zone"][z]["volume"],
            "people_max": people[z], "lights_W_m2": lights[z] / area, "equipment_W_m2": equip[z] / area,
        })
    return pd.DataFrame(rows).set_index("zone")


def zone_areas() -> dict[str, float]:
    return zones_table()["area_m2"].to_dict()


def schedule_profile(name: str, day_type: str = "WeekDays") -> pd.Series:
    fields = [f["field"] for f in load_epjson()["Schedule:Compact"][name]["data"]]
    values = np.zeros(24)
    active = matched = False
    hour = until = 0
    for f in fields:
        if isinstance(f, str) and f.startswith("For:"):
            key = f.lower()
            active = not matched and any(k in key for k in (day_type.lower(), "allotherdays", "alldays"))
            matched |= active
            hour = 0
        elif isinstance(f, str) and f.startswith("Until:"):
            until = int(f.split(":")[1])
        elif active and not isinstance(f, str):
            values[hour:until] = f
            hour = until
    return pd.Series(values, index=pd.RangeIndex(24, name="hour"), name=name)


def hvac_summary() -> dict:
    d = load_epjson()
    fan = [f["field"] for f in d["Schedule:Compact"]["FanAvailSched"]["data"]]
    oa = next(iter(d["Controller:OutdoorAir"].values()))
    clg = next(iter(d["Coil:Cooling:DX:TwoSpeed"].values()))
    return {
        "air_loop": next(iter(d["AirLoopHVAC"])),
        "cooling": f"DX two-speed, COP {clg['high_speed_gross_rated_cooling_cop']}/"
                   f"{clg['low_speed_gross_rated_cooling_cop']} (high/low)",
        "heating": "electric: central coil + reheat coil in each VAV box",
        "terminals": f"AirTerminal:SingleDuct:VAV:Reheat x{len(d['AirTerminal:SingleDuct:VAV:Reheat'])}",
        "economizer": f"{oa['economizer_control_type']}, {oa['economizer_minimum_limit_dry_bulb_temperature']}-"
                      f"{oa['economizer_maximum_limit_dry_bulb_temperature']} degC",
        "supply_fan_always_on": fan[1:] == ["For: AllDays", "Until: 24:00", 1.0],
        "timestep_min": 60 // d["Timestep"]["Timestep 1"]["number_of_timesteps_per_hour"],
    }


def _vertices(obj: dict) -> np.ndarray:
    if "vertices" in obj:
        return np.array([[p[f"vertex_{a}_coordinate"] for a in "xyz"] for p in obj["vertices"]])
    return np.array([[obj[f"vertex_{i}_{a}_coordinate"] for a in "xyz"] for i in range(1, obj["number_of_vertices"] + 1)])


def geometry(true_north: bool = True) -> pd.DataFrame:
    d = load_epjson()
    theta = np.radians(d["Building"]["Building"].get("north_axis", 0.0) if true_north else 0.0)
    rot = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    surf = d["BuildingSurface:Detailed"]
    rows = [{"name": n, "kind": s["surface_type"], "zone": s["zone_name"],
             "boundary": s["outside_boundary_condition"], "vertices": _vertices(s)} for n, s in surf.items()]
    rows += [{"name": n, "kind": "Window", "zone": surf[s["building_surface_name"]]["zone_name"],
              "boundary": "Outdoors", "vertices": _vertices(s)} for n, s in d["FenestrationSurface:Detailed"].items()]
    rows += [{"name": n, "kind": "Shading", "zone": surf[s["base_surface_name"]]["zone_name"],
              "boundary": "Outdoors", "vertices": _vertices(s)} for n, s in d["Shading:Zone:Detailed"].items()]
    df = pd.DataFrame(rows)
    df["vertices"] = [v @ rot.T for v in df["vertices"]]
    return df
