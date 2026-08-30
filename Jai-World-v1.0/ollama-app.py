"""
Embodied AI demo — Flask backend.

Bridges a browser-based Three.js/VRM world (two avatars: the user-controlled
"y-bot" and the AI-controlled "x-bot") to a local Ollama model via
function/tool calling. The browser is the source of truth for real-time
avatar positions; each chat request sends a fresh snapshot of that state,
the model reasons over it and calls tools (walk_to, run_to, sit, stand,
wave, get_status, list_locations), and the resulting high-level actions are
sent back to the browser to animate.

"""

import base64
import io
import json
import logging
import math
import os
import re
import threading
import traceback
import webbrowser

from flask import Flask, Response, render_template, request, jsonify, stream_with_context
from ollama import chat
from PIL import Image, ImageDraw

app = Flask(__name__)


class _PollingFilter(logging.Filter):
    """The browser polls /api/vision_poll, /api/action_poll, and
    /api/move_poll every 250-400ms, which floods the Flask terminal with
    HTTP 200 access logs and buries anything actually worth seeing (print
    statements, tool errors, tracebacks). Drop just those from Werkzeug's
    request logger."""

    _QUIET_ENDPOINTS = ("/api/vision_poll", "/api/action_poll", "/api/move_poll")

    def filter(self, record):
        return not any(endpoint in record.getMessage() for endpoint in self._QUIET_ENDPOINTS)


logging.getLogger("werkzeug").addFilter(_PollingFilter())

# ─── Model config ──────────────────────────────────────────────────────────
# Any Ollama model that supports tool/function calling works here.
MODEL_NAME     = "qwen3.5:9b"
MAX_ITERATIONS = 30

# ─── World definition ──────────────────────────────────────────────────────
# NOTE: the x/z coordinates on every entry must stay in sync with the
# LOCATIONS constant in templates/index.html, which places the beacon markers
# at these same points (it doesn't need title/description/sensory — those are
# server-side flavor text, delivered to the model only via inspect_location).
#
# title/description/sensory are intentionally withheld from list_locations and
# from the system message — the model only learns a location's name and
# coordinates up front, never what it's actually like there. That's what
# makes inspect_location (and therefore walking somewhere first) meaningful
# instead of a lookup table. To add a new location, just add another entry
# here; every function below (distance calcs, resolve_destination,
# nearest_location, list_locations, inspect_location) iterates LOCATIONS
# generically and needs no further changes.
LOCATIONS = {
    "Location-1": {
        "x": 16.5, "z": -14.3,
        "title": "Island Viewpoint",
        "description": "A high grassy summit, offering a 360-degree panoramic view of the surrounding ocean and distant coastline.",
        "sensory": "Crisp high-altitude air mixed with ocean spray, the quiet whistling of soft breeze, bright sunlight washing over the landscape, and a sweeping view of turquoise waters below. Clear blue sky. Can see ships in the distance.",
    },
    "Location-2": {
        "x": -4.9, "z": 15.1,
        "title": "Seaside Cafe",
        "description": "A relaxed, weathered wooden deck perched over the ocean with umbrella-covered tables and an espresso bar.",
        "sensory": "Salty sea spray on the air, warm sunshine, the rhythm of crashing waves, and the rich scent of dark roast coffee.",
    },
    "Location-3": {
        "x": -7.7, "z": -23.8,
        "title": "Beach Bungalow",
        "description": "A secluded, single-story bamboo hut with a thatched palm roof, set on stilts just above the shoreline with a wide wrap-around porch.",
        "sensory": "The steady, low thunder of tide splashing against wooden support stilts, the tang of saltwater, dry palm leaves rustling in the ocean breeze, and warm sand shifting underfoot.",
    },
    "Location-4": {
        "x": 20.1, "z": 8.0,
        "title": "Park",
        "description": "A quiet, green sanctuary with winding stone pathways, shady oak trees, scattered wooden benches, and a central stone fountain.",
        "sensory": "The soft rustle of leaves overhead, scent of freshly cut grass, distant chirping of birds, and the gentle trickling of water from the fountain.",
    },
    "Location-5": {
        "x": 22.2, "z": -25.9,
        "title": "Creative Studio",
        "description": "A quiet and basic island-style office with a front porch, a small flower garden and an ocean view. Single-story bamboo hut with a thatched palm roof and large windows. Has a chair, desk and a Linux laptop.",
        "sensory": "Warm sunlight. The soft sounds of waves breaking and palm leaves rustling in the ocean breeze.",
    },
}
ARRIVAL_RADIUS = 2.0  # meters — how close counts as "at" a location
BESIDE_DISTANCE = 1.2  # meters — offset distance for stand/sit facing/side-by-side positions
HUG_DISTANCE = 0.35  # meters — closer than BESIDE_DISTANCE; arm's-reach embrace range.
                      # Comfortably above index.html's MIN_SEPARATION (0.24m) collision
                      # floor so x-bot can actually reach this target without being
                      # treated as "arrived" early by the collision push-back.

# ─── Blocks (embodied manipulation "gym") ──────────────────────────────────
# A small set of pickupable/stackable cubes x-bot can walk to, grab, carry,
# and set down or stack — the manipulation counterpart to the LOCATIONS
# navigation targets above. BLOCK_SIZE/spawn coordinates/hex colors must stay
# in sync with BLOCK_DEFS in templates/index.html, which builds the actual
# cube meshes from these same values.
BLOCK_SIZE = 0.5  # edge length of each cube, meters
BLOCK_REACH_RADIUS = 1.5  # meters — how close x-bot must be to pick up/stack a block
BLOCK_YARD = {"x": 8.0, "z": -2.0}  # flat open ground, clear of the hill/ocean/locations

BLOCK_DEFS = {
    "Block-Red":    {"hex": "#d9483c", "spawn_x": BLOCK_YARD["x"] - 1.5, "spawn_z": BLOCK_YARD["z"]},
    "Block-Blue":   {"hex": "#3c7ad9", "spawn_x": BLOCK_YARD["x"] - 0.5, "spawn_z": BLOCK_YARD["z"]},
    "Block-Green":  {"hex": "#4caf50", "spawn_x": BLOCK_YARD["x"] + 0.5, "spawn_z": BLOCK_YARD["z"]},
    "Block-Yellow": {"hex": "#e0c23c", "spawn_x": BLOCK_YARD["x"] + 1.5, "spawn_z": BLOCK_YARD["z"]},
}


def _initial_blocks_state():
    return {
        name: {"x": d["spawn_x"], "z": d["spawn_z"], "held": False, "on": None}
        for name, d in BLOCK_DEFS.items()
    }


# In-memory, mutable, session-wide — same pattern as message_history. Each
# entry: {"x", "z", "held", "on"}. "on" is another block's name (stacked on
# top of it) or None (resting directly on the ground/floor). "held" means
# it's currently attached to x-bot; its x/z are stale until put down.
blocks_state = _initial_blocks_state()

# Guards blocks_state, message_history, and resident_image_messages. Even
# though this is a single-local-user app, Flask is run with threaded=True
# (required so /api/vision_poll etc. can be served while a chat request is
# blocked inside a tool call), so a /api/reset hitting these while a chat
# request is mid-read/mid-mutation is a real race, not just theoretical.
STATE_LOCK = threading.Lock()


def resolve_block_name(s):
    """Same forgiving matching as resolve_location_name, plus bare color
    words ('red' -> 'Block-Red') since that's the more natural way people
    refer to a block in conversation."""
    if not s:
        return None
    normalized = re.sub(r"[\s_-]+", "", s.strip()).lower()
    for name in BLOCK_DEFS:
        if normalized == re.sub(r"[\s_-]+", "", name).lower():
            return name
    for name in BLOCK_DEFS:
        color = name.split("-", 1)[1].lower()
        if normalized in (color, f"block{color}", f"{color}block"):
            return name
    return None


def block_stack_level(name):
    """1 if resting directly on the ground; otherwise 1 + however many
    blocks it's stacked on top of. Walks the 'on' chain.
    Caller must hold STATE_LOCK."""
    level = 1
    seen = set()
    cur = name
    while blocks_state[cur]["on"] is not None:
        cur = blocks_state[cur]["on"]
        if cur in seen:  # defensive against a future bug creating a cycle
            break
        seen.add(cur)
        level += 1
    return level


def block_world_pos(name):
    """Current (x, z, y) of a block resting on the ground or on a stack, or
    None if it's currently held (its real position is wherever x-bot is
    carrying it — callers should use xbot_sim for that case instead).
    Caller must hold STATE_LOCK."""
    st = blocks_state[name]
    if st["held"]:
        return None
    y = (block_stack_level(name) - 1) * BLOCK_SIZE
    return st["x"], st["z"], y


def block_supported_by_something(name):
    """True if another block currently rests on top of this one — that one
    needs to be moved first before this block can be picked up or restacked.
    Caller must hold STATE_LOCK."""
    return any(
        other != name and s["on"] == name and not s["held"]
        for other, s in blocks_state.items()
    )

# In-memory conversation history (single local user, single process).
message_history = []

# References to whichever message(s) in message_history currently hold live
# look_around image data. When a new look_around succeeds, these get neutered
# (image stripped, content replaced with a placeholder) rather than deleted
# outright — keeps message_history well-formed while ensuring only the most
# recent snapshot is ever actually resident for the model to see.
resident_image_messages = []

# ─── Vision snapshot cache ───────────────────────────────────────────────────
# The browser continuously renders x-bot's four cardinal views (relative to
# its own facing) on a timer and POSTs them here (see /api/vision_update).
# look_around() just reads whatever's currently cached — no round trip to the
# browser happens inside the agent loop, so the tool behaves synchronously
# like every other tool here. Images are cached as data URLs and are never
# written into message_history; only look_around's text result persists there
# (see run_agent_turn's handling of the "_image" key).
VISION_LOCK = threading.Lock()
VISION_DIRECTIONS = ("front", "right", "rear", "left")
LATEST_SNAPSHOTS = {d: None for d in VISION_DIRECTIONS}

# ─── On-demand capture handshake ─────────────────────────────────────────────
# The browser no longer renders x-bot's four views on a timer. Instead it
# polls /api/vision_poll on a cheap, render-free interval; when look_around()
# is called it sets a pending request id here, the browser's next poll sees
# it, performs the four offscreen renders exactly once, and POSTs the result
# back to /api/vision_update tagged with that same id. look_around() blocks
# (via VISION_CAPTURE_READY) until that tagged result arrives or times out.
VISION_REQUEST_LOCK = threading.Lock()
VISION_REQUEST_COUNTER = 0
PENDING_VISION_REQUEST_ID = None
VISION_CAPTURE_READY = threading.Event()
VISION_CAPTURE_TIMEOUT_S = 5.0

# ─── Live movement dispatch / arrival confirmation ───────────────────────────
# walk_to/run_to used to just append a "walk" action to the batch returned at
# the end of the whole agent turn, then immediately report "Arrived." — which
# was true of the optimistic simulated position, but not of the browser,
# which hadn't even received the action yet, let alone animated it. Fixed
# destinations (a location, a block, or raw coordinates — not the 'y-bot'
# chase case, which has no fixed arrival point to wait for) now go out
# through this channel instead: dispatched to the browser immediately via
# /api/move_poll, with _move() blocking on MOVE_ARRIVED_EVENT until the
# browser's real updateXBot loop reports genuine arrival via
# /api/move_arrived, or until MOVE_ARRIVE_TIMEOUT_S elapses.
MOVE_REQUEST_LOCK = threading.Lock()
MOVE_REQUEST_COUNTER = 0
PENDING_MOVE_REQUEST = None  # {"move_id", "action", "delivered"} or None
MOVE_ARRIVED_EVENT = threading.Event()
# Set by /api/move_arrived alongside the event — the fixed-destination case
# already knows x,z in advance, but the y-bot-chase case doesn't (the actual
# stopping point depends on wherever y-bot happened to be when x-bot caught
# up), so _move() reads this back after a successful wait instead of guessing.
MOVE_ARRIVED_POS = None
# Mirrors XBOT_MOVE_SPEED / XBOT_RUN_MULT in index.html, purely to estimate a
# generous safety-net timeout. This is NOT meant to fire under normal walking
# conditions — the tool call is supposed to only ever return once x-bot has
# truly arrived, so the timeout exists solely to stop the request thread
# hanging forever if the browser tab is closed/disconnected mid-walk, or
# x-bot genuinely gets stuck. It's sized several times larger than any
# realistic walk across this map should take.

# ─── Live "instant action" dispatch ──────────────────────────────────────────
# walk_to/run_to (above) got fixed to reach the browser the moment the tool
# actually runs, instead of sitting in the `actions` batch until the whole
# agent turn (every tool call in the loop, including any blocking walks)
# finishes. Every OTHER action — pick_up_block, put_down_block, stack_block,
# sit, stand, wave, hug, turn, escort, and the dynamic-target walks used by
# the stand/sit_beside/facing helpers — was still going out the old way:
# appended to `actions` and delivered in one lump at the very end.
#
# That's exactly what produced the "still looks like it's carrying the
# block" bug: for "pick up the red block, walk it to Location-2, put it
# down", the walk_to calls dispatch and block in real time as x-bot
# physically walks there and back, but pick_up_block and put_down_block
# just sit in `actions` the whole time. By the time the HTTP response
# finally comes back and the browser applies them, x-bot has ALREADY
# physically arrived at Location-2 (via the realtime walk channel) without
# ever visually picking the block up — then both pick_up_block and
# put_down_block get applied back-to-back at the destination, so the block
# snaps into the carried pose and only drops a fraction of a second later
# (or can appear stuck there if that second application doesn't register
# before the user's eye catches it).
#
# Fix: give every non-walk action the same immediate-dispatch treatment —
# pushed to this queue the instant the tool runs, drained by the browser's
# own fast poll (/api/action_poll), so pick-up/put-down/etc. land in the
# right visual order relative to the walks around them instead of being
# held hostage by the whole turn's completion.
INSTANT_ACTION_LOCK = threading.Lock()
PENDING_INSTANT_ACTIONS = []

MOVE_SPEED_WALK = 2.6
MOVE_SPEED_RUN_MULT = 2.1
MOVE_ARRIVE_MIN_TIMEOUT_S = 20.0
MOVE_ARRIVE_MAX_TIMEOUT_S = 120.0
VISION_TILE_SIZE = 256
VISION_LABELS = {
    "front": "FRONT (0°)",
    "right": "RIGHT (90°)",
    "rear": "REAR (180°)",
    "left": "LEFT (-90°)",
}
# Grid cell for each direction within the 2x2 composite.
VISION_GRID_POSITIONS = {
    "front": (0, 0),
    "right": (VISION_TILE_SIZE, 0),
    "rear": (0, VISION_TILE_SIZE),
    "left": (VISION_TILE_SIZE, VISION_TILE_SIZE),
}


def _decode_data_url(data_url):
    """Strips a 'data:image/png;base64,' prefix if present and decodes to bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def build_look_around_composite():
    """Stitches the 4 cached direction snapshots into one labeled 2x2 image.

    Returns (composite_data_url, composite_b64_raw, missing_directions).
    composite_b64_raw has no data-URL prefix, for handing straight to Ollama's
    per-message `images` field. If any direction hasn't been captured yet,
    composite is None and missing_directions lists what's absent.
    """
    with VISION_LOCK:
        snapshots = dict(LATEST_SNAPSHOTS)

    missing = [d for d in VISION_DIRECTIONS if not snapshots.get(d)]
    if missing:
        return None, None, missing

    size = VISION_TILE_SIZE
    border = 4  # px — visual separator between quadrants and around the edge
    border_color = (0, 0, 0)
    grid = Image.new("RGB", (size * 2, size * 2), border_color)
    draw = ImageDraw.Draw(grid)
    for name in VISION_DIRECTIONS:
        ox, oy = VISION_GRID_POSITIONS[name]
        tile_bytes = _decode_data_url(snapshots[name])
        # Shrink each tile slightly so the shared background shows through as
        # a border on every side, including between adjacent quadrants.
        tile_img = (
            Image.open(io.BytesIO(tile_bytes))
            .convert("RGB")
            .resize((size - border, size - border))
        )
        grid.paste(tile_img, (ox + border // 2, oy + border // 2))
        label = VISION_LABELS[name]
        text_w = draw.textlength(label)
        draw.rectangle([ox + border, oy + border, ox + border + text_w + 8, oy + border + 16], fill=(0, 0, 0))
        draw.text((ox + border + 4, oy + border + 2), label, fill=(255, 255, 0))

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    raw_bytes = buf.getvalue()
    composite_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/png;base64,{composite_b64}", composite_b64, []


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["z"] - b["z"])


def nearest_location(pos):
    best_name, best_dist = None, float("inf")
    for name, loc in LOCATIONS.items():
        d = distance(pos, loc)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist


def fmt_pos(pos):
    return f"({pos['x']:.1f}, {pos['z']:.1f})"


def ybot_pose_label(ybot_state):
    """Reduce y-bot's raw client-side state (dancing/sitting flags + ground/
    fly/swim movement mode) to a single label the model can reason about:
    'dancing', 'sitting', 'flying', 'swimming', or 'standing'. dancing is
    checked first — it's only reachable by joining x-bot's dance via F,
    which the client already refuses while sitting and clears on entry (see
    handleAttachKey/updateMovement's yBotSitting-can't-attach guard), so it
    never actually overlaps with the other flags; checking it first is just
    the clearest way to express that priority regardless."""
    if (ybot_state or {}).get("dancing"):
        return "dancing"
    if (ybot_state or {}).get("sitting"):
        return "sitting"
    mode = (ybot_state or {}).get("mode", "ground")
    if mode == "fly":
        return "flying"
    if mode == "swim":
        return "swimming"
    return "standing"


def xbot_pose_label(xbot_sim):
    """Reduce x-bot's raw status ('idle' | 'walking' | 'running' | 'sitting')
    to the same vocabulary ybot_pose_label uses for y-bot, so the two pose
    fields read as consistent English side by side instead of one being raw
    internal state. x-bot has no fly/swim mode (that's a y-bot-only ability —
    see index.html's keydown handler), so 'idle' is the only relabeling
    needed; everything else already matches."""
    status = (xbot_sim or {}).get("status", "idle")
    return "standing" if status == "idle" else status


def heading_vector(theta):
    """Forward direction for a facing angle, matching the browser's convention
    (dirX = sin(theta), dirZ = cos(theta)) — see index.html's facingAngle usage."""
    return math.sin(theta), math.cos(theta)


def angle_towards(from_x, from_z, to_x, to_z):
    """Facing angle pointing from one point toward another, same atan2(dx, dz)
    convention already used for x-bot's movement-driven turning in index.html."""
    return math.atan2(to_x - from_x, to_z - from_z)


# How close (in radians) a facing angle must be to the direct bearing toward
# the other bot to count as "facing" them. 45° each side of dead-on — narrow
# enough to mean "oriented roughly at them," wide enough that it doesn't
# require pixel-perfect aim. Applies regardless of pose (sitting still faces
# a direction) and regardless of y-bot's mode (fly/swim/ground) — altitude
# isn't part of this calculation, same ground-plane-only treatment already
# used for distance() and nearest_location() elsewhere in this file.
FACING_TOLERANCE = math.radians(45)


def is_facing(from_pos, from_heading, to_pos):
    """True if from_pos, oriented at from_heading, is pointed roughly at
    to_pos (within FACING_TOLERANCE). Undefined at zero distance (a bot can't
    bear toward its own position), so that degenerate case returns False
    rather than raising."""
    if from_pos["x"] == to_pos["x"] and from_pos["z"] == to_pos["z"]:
        return False
    bearing = angle_towards(from_pos["x"], from_pos["z"], to_pos["x"], to_pos["z"])
    diff = (bearing - from_heading + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) <= FACING_TOLERANCE


YBOT_ALIASES = ("y-bot", "ybot", "user", "player", "you", "me")


def is_ybot_alias(s):
    return (s or "").strip().lower() in YBOT_ALIASES


def resolve_location_name(s):
    """Normalize a location string ('Location-1', 'location 1', 'loc-1', '1',
    'loc1', ...) to its canonical LOCATIONS key, or None if nothing matches.

    Factored out of resolve_destination so any tool that only cares about
    named locations (e.g. inspect_location, which — unlike walk_to — can't
    target y-bot or a raw coordinate) can reuse the same forgiving matching
    without duplicating it.
    """
    if not s:
        return None
    normalized = re.sub(r"[\s_-]+", "", s.strip()).lower()  # 'Location 1' / 'location_1' / 'Location-1' -> 'location1'

    for name in LOCATIONS:
        if normalized == re.sub(r"[\s_-]+", "", name).lower():
            return name

    # Bare number ('1', '2', ...) or short forms ('loc1', 'loc-2') meaning "Location-N".
    m = re.fullmatch(r"(?:location|loc)?0*(\d+)", normalized)
    if m:
        name = f"Location-{int(m.group(1))}"
        if name in LOCATIONS:
            return name

    return None


def offset_point(px, pz, heading, kind, near_x, near_z, distance=BESIDE_DISTANCE):
    """A point near (px, pz) relative to a facing direction.

    kind='front': a point in the direction the target is currently facing.
    kind='flank': a point to one side, perpendicular to the target's facing —
    whichever side (left/right) ends up closer to (near_x, near_z) is chosen,
    so x-bot doesn't walk an unnecessary arc around the target to reach the
    farther side.

    `distance` defaults to BESIDE_DISTANCE (talking/standing range) but
    callers that need a different offset — e.g. hug, which wants arm's-reach
    range rather than conversational distance — can pass their own.

    Returns (x, z, side): side is +1/-1 for 'flank' (which perpendicular
    direction was picked) or None for 'front' (no side concept). Callers that
    need to keep tracking a moving target (e.g. y-bot) live use the 'side'
    value instead of baking in this (x, z) point — see _position_relative_to.
    """
    fx, fz = heading_vector(heading)
    if kind == "front":
        return px + fx * distance, pz + fz * distance, None
    perp_x, perp_z = fz, -fx
    a = (px + perp_x * distance, pz + perp_z * distance)
    b = (px - perp_x * distance, pz - perp_z * distance)
    da = math.hypot(a[0] - near_x, a[1] - near_z)
    db = math.hypot(b[0] - near_x, b[1] - near_z)
    return (a[0], a[1], 1) if da <= db else (b[0], b[1], -1)


# ─── Per-request tool implementations ──────────────────────────────────────
# Built fresh for every request so each tool call closes over the browser's
# latest state snapshot (`live_state`), a mutable logical copy of x-bot's
# state (`xbot_sim`), and the list of actions to send back to the browser.
def dispatch_move_and_wait(mode, action_extra, start_dist):
    """Sends a walk/run action to the browser immediately via /api/move_poll
    and blocks until it reports genuine arrival (or the safety-net timeout
    elapses). Shared by both _move() branches — a fixed destination and a
    y-bot chase both converge to a single, well-defined stopping point in
    the browser (see updateXBot's shared dist < XBOT_ARRIVE_EPS check), so
    both get the same real-arrival treatment.

    action_extra supplies whatever the action needs beyond "type" — either
    {"target": {"x", "z"}, "label"} for a fixed destination, or
    {"target": "y-bot", "label"} for a live chase.

    Returns (arrived: bool, timeout_s: float, arrived_pos: dict | None).
    """
    global PENDING_MOVE_REQUEST, MOVE_REQUEST_COUNTER, MOVE_ARRIVED_POS

    speed = MOVE_SPEED_WALK * (MOVE_SPEED_RUN_MULT if mode == "run" else 1)
    timeout_s = min(
        MOVE_ARRIVE_MAX_TIMEOUT_S,
        max(MOVE_ARRIVE_MIN_TIMEOUT_S, (start_dist / speed) * 4.0 + 10.0),
    )

    with MOVE_REQUEST_LOCK:
        MOVE_REQUEST_COUNTER += 1
        move_id = MOVE_REQUEST_COUNTER
        PENDING_MOVE_REQUEST = {
            "move_id": move_id,
            "delivered": False,
            "action": {"type": mode, **action_extra},
        }
        MOVE_ARRIVED_EVENT.clear()
        MOVE_ARRIVED_POS = None

    arrived = MOVE_ARRIVED_EVENT.wait(timeout=timeout_s)
    arrived_pos = MOVE_ARRIVED_POS if arrived else None

    with MOVE_REQUEST_LOCK:
        if PENDING_MOVE_REQUEST is not None and PENDING_MOVE_REQUEST["move_id"] == move_id:
            PENDING_MOVE_REQUEST = None

    return arrived, timeout_s, arrived_pos


def build_tools_and_dispatch(live_state):
    xbot_sim = dict(live_state.get("xbot") or {"x": 0.0, "z": 0.0, "status": "idle"})
    xbot_sim.setdefault("facingAngle", 0.0)
    ybot_state = live_state.get("ybot") or {"x": 0.0, "z": 0.0}
    actions = []

    def dispatch_instant(action):
        """Record an action for the end-of-turn `actions` list (used for
        dev-console/debug visibility) AND push it onto the realtime queue
        the browser drains via /api/action_poll, so it reaches x-bot the
        moment the tool actually runs — not whenever the whole agent turn
        (which may include several more blocking walk_to calls) finishes."""
        actions.append(action)
        with INSTANT_ACTION_LOCK:
            PENDING_INSTANT_ACTIONS.append(action)

    def resolve_destination(destination):
        """Resolve a destination string to (x, z, label).

        The model doesn't always echo 'Location-1' back verbatim — it may
        send 'location 1', 'Location1', 'loc-1', or just '1'. resolve_location_name
        handles that normalization; this adds the two things a movement target
        can be that a plain location can't: y-bot, or a raw coordinate pair.
        """
        if destination is None:
            return None, None, None
        d = destination.strip()

        name = resolve_location_name(d)
        if name is not None:
            loc = LOCATIONS[name]
            return loc["x"], loc["z"], name

        block_name = resolve_block_name(d)
        if block_name is not None:
            pos = block_world_pos(block_name)
            if pos is not None:  # None means currently held — nowhere fixed to walk to
                bx, bz, _by = pos
                return bx, bz, block_name

        if is_ybot_alias(d):
            return ybot_state.get("x", 0.0), ybot_state.get("z", 0.0), "y-bot (the user)"

        cleaned = d.replace("(", "").replace(")", "")
        parts = [p.strip() for p in cleaned.split(",")]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1]), f"({float(parts[0])}, {float(parts[1])})"
            except ValueError:
                pass

        return None, None, None

    def _move(params, mode):
        destination = params.get("destination", "") or ""

        if is_ybot_alias(destination):
            # Don't bake in y-bot's coordinates into the dispatched action:
            # this snapshot is already stale by the time it reaches the
            # browser, and y-bot may keep moving throughout the walk. The
            # browser re-resolves y-bot's *live* position every frame (see
            # xBot.dynamicTarget handling) and settles once it gets within
            # a fixed stopping distance — a real, single, well-defined
            # arrival event (the same dist < XBOT_ARRIVE_EPS check used for
            # a fixed destination), it just isn't known in advance exactly
            # where that point will be. So this gets the same live-dispatch
            # treatment as a fixed destination, just with an estimated
            # timeout based on y-bot's position *right now* and the actual
            # arrival position read back from the browser rather than
            # computed here.
            start_dist = distance(
                {"x": xbot_sim["x"], "z": xbot_sim["z"]},
                {"x": ybot_state.get("x", 0.0), "z": ybot_state.get("z", 0.0)},
            )
            arrived, timeout_s, arrived_pos = dispatch_move_and_wait(
                mode, {"target": "y-bot", "label": "y-bot (the user)"}, start_dist
            )

            if arrived:
                if arrived_pos:
                    xbot_sim["x"], xbot_sim["z"] = arrived_pos["x"], arrived_pos["z"]
                else:
                    xbot_sim["x"], xbot_sim["z"] = ybot_state.get("x", 0.0), ybot_state.get("z", 0.0)
                xbot_sim["status"] = "idle"
                return {"ok": True, "message": "You have arrived at y-bot."}

            # Unlike a fixed destination, not arriving here isn't necessarily
            # an anomaly — y-bot moving around (especially away, especially
            # faster than a plain walk) can genuinely keep this from
            # converging in time. Don't claim a stuck/disconnected browser;
            # just report it honestly.
            xbot_sim["status"] = "running" if mode == "run" else "walking"
            suggestion = " Try run_to instead of walk_to." if mode == "walk" else ""
            return {
                "ok": False,
                "error": (
                    f"x-bot hasn't caught up to y-bot after {timeout_s:.0f}s — "
                    f"y-bot may be moving too fast or too erratically to close the "
                    f"gap, or something's stuck.{suggestion} Don't assume it arrived."
                ),
            }

        x, z, label = resolve_destination(destination)
        if x is None:
            valid = ", ".join(LOCATIONS.keys())
            return {
                "ok": False,
                "error": (
                    f"Could not resolve destination '{params.get('destination')}'. "
                    f"Valid destinations: {valid}, {', '.join(BLOCK_DEFS.keys())}, "
                    "'y-bot' (the user's current position), or raw coordinates like '3,4'."
                ),
            }

        start_dist = distance({"x": xbot_sim["x"], "z": xbot_sim["z"]}, {"x": x, "z": z})
        arrived, timeout_s, _arrived_pos = dispatch_move_and_wait(
            mode, {"target": {"x": x, "z": z}, "label": label}, start_dist
        )

        if arrived:
            xbot_sim["x"], xbot_sim["z"] = x, z
            xbot_sim["status"] = "idle"
            return {
                "ok": True,
                "message": f"You have arrived at {label}.",
            }

        # This is a genuine anomaly, not an expected outcome — the timeout is
        # sized to comfortably outlast any real walk across this map. Getting
        # here means the browser tab likely closed/disconnected mid-walk, or
        # x-bot is physically stuck. Leave xbot_sim's position alone rather
        # than optimistically snapping it to the destination, since we
        # genuinely don't know how far x-bot got.
        xbot_sim["status"] = "running" if mode == "run" else "walking"
        return {
            "ok": False,
            "error": (
                f"x-bot has not arrived at {label} after {timeout_s:.0f}s and may be "
                "stuck or disconnected from the browser. Don't assume it arrived — "
                "try get_status, or check the world is still open and responsive."
            ),
        }

    def walk_to(params):
        return _move(params, "walk")

    def run_to(params):
        return _move(params, "run")

    def sit(_params):
        dispatch_instant({"type": "sit"})
        xbot_sim["status"] = "sitting"
        return {"ok": True, "message": "Sat down."}

    def stand(_params):
        dispatch_instant({"type": "stand"})
        xbot_sim["status"] = "idle"
        return {"ok": True, "message": "Standing up."}

    def wave(_params):
        dispatch_instant({"type": "wave"})
        return {"ok": True, "message": "Waved."}

    def dance(_params):
        # Loops until stop_dance is called or any other action interrupts it
        # (mirrors how "sitting" is a persistent status, not a timed gesture
        # like wave/hug). The frontend also lets y-bot join in by pressing F
        # while this is active — worth mentioning to the user in-character.
        dispatch_instant({"type": "dance"})
        xbot_sim["status"] = "dancing"
        return {
            "ok": True,
            "message": "Started dancing. y-bot can press F to join in.",
        }

    def stop_dance(_params):
        dispatch_instant({"type": "stop_dance"})
        xbot_sim["status"] = "idle"
        return {"ok": True, "message": "Stopped dancing."}

    def get_status(_params):
        xbot_pos = {"x": xbot_sim["x"], "z": xbot_sim["z"]}
        xbot_loc, xbot_d = nearest_location(xbot_pos)
        ybot_loc, ybot_d = nearest_location(ybot_state)
        d_between = distance(xbot_pos, ybot_state)
        xbot_heading = xbot_sim.get("facingAngle", 0.0)
        ybot_heading = ybot_state.get("facingAngle", 0.0)
        with STATE_LOCK:
            held_name = next((n for n, s in blocks_state.items() if s["held"]), None)
        return {
            "ok": True,
            "xbot_position": fmt_pos(xbot_pos),
            "xbot_pose": xbot_pose_label(xbot_sim),
            "xbot_near": xbot_loc if xbot_d <= ARRIVAL_RADIUS else f"nowhere named (nearest is {xbot_loc}, {xbot_d:.1f}m away)",
            "xbot_facing_ybot": is_facing(xbot_pos, xbot_heading, ybot_state),
            "xbot_holding_block": held_name,
            "ybot_position": fmt_pos(ybot_state),
            "ybot_pose": ybot_pose_label(ybot_state),
            "ybot_near": ybot_loc if ybot_d <= ARRIVAL_RADIUS else f"nowhere named (nearest is {ybot_loc}, {ybot_d:.1f}m away)",
            "ybot_facing_xbot": is_facing(ybot_state, ybot_heading, xbot_pos),
            "distance_between_xbot_and_ybot": round(d_between, 1),
        }

    def list_blocks(_params):
        xbot_pos = {"x": xbot_sim["x"], "z": xbot_sim["z"]}
        out = {}
        with STATE_LOCK:
            for name in BLOCK_DEFS:
                st = blocks_state[name]
                if st["held"]:
                    out[name] = {"color": BLOCK_DEFS[name]["hex"], "status": "held by you right now"}
                    continue
                x, z, y = block_world_pos(name)
                out[name] = {
                    "color": BLOCK_DEFS[name]["hex"],
                    "position": fmt_pos({"x": x, "z": z}),
                    "height_level": round(y / BLOCK_SIZE) + 1,  # 1 = on the ground, 2 = one block up, ...
                    "resting_on": f"on top of {st['on']}" if st["on"] else "on the ground",
                    "distance_from_you": round(distance(xbot_pos, {"x": x, "z": z}), 1),
                    "has_something_on_top": block_supported_by_something(name),
                }
        return {"ok": True, "blocks": out}

    def pick_up_block(params):
        name = resolve_block_name(params.get("block", "") or "")
        if name is None:
            return {
                "ok": False,
                "error": f"Could not resolve block '{params.get('block')}'. Valid blocks: {', '.join(BLOCK_DEFS.keys())}.",
            }
        with STATE_LOCK:
            held_name = next((n for n, s in blocks_state.items() if s["held"]), None)
            if held_name is not None:
                return {"ok": False, "error": f"Already holding {held_name} — put it down or stack it first."}
            if block_supported_by_something(name):
                return {"ok": False, "error": f"Can't pick up {name} — something is stacked on top of it."}
            x, z, _y = block_world_pos(name)
            d = distance({"x": xbot_sim["x"], "z": xbot_sim["z"]}, {"x": x, "z": z})
            if d > BLOCK_REACH_RADIUS:
                return {
                    "ok": False,
                    "error": (
                        f"Too far from {name} to pick it up (currently {d:.1f}m away, need to "
                        f"be within {BLOCK_REACH_RADIUS:.1f}m). Walk there first."
                    ),
                }
            blocks_state[name]["held"] = True
            blocks_state[name]["on"] = None
        dispatch_instant({"type": "pick_up_block", "block": name})
        return {"ok": True, "message": f"Picked up {name}."}

    def put_down_block(_params):
        with STATE_LOCK:
            held_name = next((n for n, s in blocks_state.items() if s["held"]), None)
            if held_name is None:
                return {"ok": False, "error": "Not currently holding a block."}
            fx, fz = heading_vector(xbot_sim.get("facingAngle", 0.0))
            drop_x = xbot_sim["x"] + fx * (BLOCK_SIZE * 1.5)
            drop_z = xbot_sim["z"] + fz * (BLOCK_SIZE * 1.5)
            blocks_state[held_name] = {"x": drop_x, "z": drop_z, "held": False, "on": None}
        dispatch_instant({"type": "put_down_block", "block": held_name, "x": drop_x, "z": drop_z, "y": 0.0})
        return {"ok": True, "message": f"Put {held_name} down on the ground at ({drop_x:.1f}, {drop_z:.1f})."}

    def stack_block_on(params):
        with STATE_LOCK:
            held_name = next((n for n, s in blocks_state.items() if s["held"]), None)
            if held_name is None:
                return {"ok": False, "error": "Not currently holding a block — pick one up first."}
            target = resolve_block_name(params.get("target", "") or "")
            if target is None:
                return {
                    "ok": False,
                    "error": f"Could not resolve target block '{params.get('target')}'. Valid blocks: {', '.join(BLOCK_DEFS.keys())}.",
                }
            if target == held_name:
                return {"ok": False, "error": f"Can't stack {held_name} on itself."}
            if blocks_state[target]["held"]:
                return {"ok": False, "error": f"{target} is also currently held — can't stack on it."}
            if block_supported_by_something(target):
                return {"ok": False, "error": f"Can't stack on {target} — something is already on top of it."}
            tx, tz, _ty = block_world_pos(target)
            d = distance({"x": xbot_sim["x"], "z": xbot_sim["z"]}, {"x": tx, "z": tz})
            if d > BLOCK_REACH_RADIUS:
                return {
                    "ok": False,
                    "error": (
                        f"Too far from {target} to stack on it (currently {d:.1f}m away, need "
                        f"to be within {BLOCK_REACH_RADIUS:.1f}m). Walk there first."
                    ),
                }
            new_level = block_stack_level(target) + 1
            blocks_state[held_name] = {"x": tx, "z": tz, "held": False, "on": target}
            y = (new_level - 1) * BLOCK_SIZE
        dispatch_instant({"type": "stack_block", "block": held_name, "on": target, "x": tx, "z": tz, "y": y})
        return {"ok": True, "message": f"Stacked {held_name} on top of {target} (now {new_level} block(s) high)."}

    def list_locations(_params):
        xbot_pos = {"x": xbot_sim["x"], "z": xbot_sim["z"]}
        out = {}
        for name, loc in LOCATIONS.items():
            out[name] = {
                "coordinates": fmt_pos(loc),
                "distance_from_xbot": round(distance(xbot_pos, loc), 1),
                "distance_from_ybot": round(distance(ybot_state, loc), 1),
            }
        return {"ok": True, "locations": out}

    def inspect_location(params):
        name = resolve_location_name(params.get("location", "") or "")
        if name is None:
            return {
                "ok": False,
                "error": (
                    f"Could not resolve location '{params.get('location')}'. "
                    f"Valid locations: {', '.join(LOCATIONS.keys())}."
                ),
            }

        xbot_pos = {"x": xbot_sim["x"], "z": xbot_sim["z"]}
        loc = LOCATIONS[name]
        d = distance(xbot_pos, loc)
        if d > ARRIVAL_RADIUS:
            return {
                "ok": False,
                "error": (
                    f"Too far from {name} to inspect it (currently {d:.1f}m away, "
                    f"need to be within {ARRIVAL_RADIUS:.1f}m). Walk there first."
                ),
            }

        return {
            "ok": True,
            "location": name,
            "title": loc["title"],
            "description": loc["description"],
            "sensory": loc["sensory"],
        }

    def look_around(_params):
        global PENDING_VISION_REQUEST_ID, VISION_REQUEST_COUNTER

        # Ask the browser for a brand-new capture rather than trusting
        # whatever happens to be cached — nothing renders x-bot's views on
        # its own anymore, so a stale cache would just mean nothing's there.
        with VISION_REQUEST_LOCK:
            VISION_REQUEST_COUNTER += 1
            request_id = VISION_REQUEST_COUNTER
            PENDING_VISION_REQUEST_ID = request_id
            VISION_CAPTURE_READY.clear()

        # Also clear any snapshots left over from a previous look_around.
        # /api/vision_update only overwrites a direction when the browser's
        # payload actually includes it, so without this, a capture that's
        # missing (say) the rear view this time around would silently
        # stitch in a stale rear frame from an earlier turn instead of being
        # reported as incomplete.
        with VISION_LOCK:
            for d in VISION_DIRECTIONS:
                LATEST_SNAPSHOTS[d] = None

        arrived = VISION_CAPTURE_READY.wait(timeout=VISION_CAPTURE_TIMEOUT_S)

        with VISION_REQUEST_LOCK:
            # Clear our own request if it's still the pending one (e.g. we
            # timed out) so a late/duplicate reply doesn't confuse a future call.
            if PENDING_VISION_REQUEST_ID == request_id:
                PENDING_VISION_REQUEST_ID = None

        if not arrived:
            return {
                "ok": False,
                "error": (
                    "Timed out waiting for the browser to capture a fresh view "
                    "(it may not be polling — check the page is open and loaded). "
                    "Try again."
                ),
            }

        composite_data_url, composite_b64, missing = build_look_around_composite()
        if composite_data_url is None:
            return {
                "ok": False,
                "error": (
                    "Capture came back incomplete — still missing: "
                    f"{', '.join(missing)}. Try again."
                ),
            }
        return {
            "ok": True,
            "directions": list(VISION_DIRECTIONS),
            "note": "One image showing all four views is attached to this tool result. This is from your perspective i.e. it is what your eyes can see.",
            # Leading underscore marks this as image payload: run_agent_turn strips
            # it out before the result is stringified into message_history, and
            # instead feeds it to the model as a one-turn-only image attachment.
            "_image_data_url": composite_data_url,
            "_image_b64": composite_b64,
        }

    def resolve_oriented_target(target):
        """Resolves a target that has its own facing direction. Locations have
        no orientation of their own, so only 'y-bot' qualifies here — walk_to's
        location/coordinate options aren't valid for these tools."""
        if is_ybot_alias(target):
            heading = ybot_state.get("facingAngle", 0.0)
            return ybot_state.get("x", 0.0), ybot_state.get("z", 0.0), heading, "y-bot (the user)"
        return None, None, None, None

    def _position_relative_to(params, kind, distance=BESIDE_DISTANCE):
        """kind: 'flank' (side-by-side) or 'front' (facing). Returns
        (dynamic_target, dynamic_angle, approx_x, approx_z, approx_facing,
        label) or None if the target didn't resolve.

        Like walk_beside_ybot, only the *shape* of the offset — which side to
        flank on — is decided here, from this turn's snapshot. The actual
        (x, z) point x-bot walks to, and the final facing angle it turns to,
        are re-resolved live by the browser from y-bot's current
        position/heading at the moment x-bot arrives — not the moment this
        tool was called — so a slow model response or a mid-walk repositioning
        by y-bot doesn't leave x-bot standing beside a ghost of where y-bot
        used to be. approx_x/z/facing are only a same-turn best guess for the
        optimistic xbot_sim update (e.g. so get_status reads sensibly).

        `distance` is passed through to offset_point — most callers want
        BESIDE_DISTANCE (conversational range), but a closer-contact tool
        like hug wants a tighter offset.
        """
        tx, tz, heading, label = resolve_oriented_target(params.get("target"))
        if tx is None:
            return None
        x, z, side = offset_point(tx, tz, heading, kind, xbot_sim["x"], xbot_sim["z"], distance)
        if kind == "flank":
            dynamic_target = {"dynamic": "ybot_offset", "kind": "flank", "side": side, "distance": distance}
            dynamic_angle = {"dynamic": "match_ybot_facing"}
            approx_facing = heading
        else:
            dynamic_target = {"dynamic": "ybot_offset", "kind": "front", "distance": distance}
            dynamic_angle = {"dynamic": "face_ybot"}
            approx_facing = angle_towards(x, z, tx, tz)
        return dynamic_target, dynamic_angle, x, z, approx_facing, label

    def _oriented_target_error(params):
        return {
            "ok": False,
            "error": (
                f"Could not resolve target '{params.get('target')}'. "
                "This tool requires a target with a facing direction — currently only 'y-bot' is supported."
            ),
        }

    def look_at(params):
        x, z, label = resolve_destination(params.get("target", ""))
        if x is None:
            valid = ", ".join(LOCATIONS.keys())
            return {
                "ok": False,
                "error": (
                    f"Could not resolve target '{params.get('target')}'. "
                    f"Valid targets: {valid}, {', '.join(BLOCK_DEFS.keys())}, "
                    "'y-bot', or raw coordinates 'x,z'."
                ),
            }
        angle = angle_towards(xbot_sim["x"], xbot_sim["z"], x, z)
        dispatch_instant({"type": "turn", "angle": angle})
        xbot_sim["facingAngle"] = angle
        return {"ok": True, "message": f"Turned to face {label}."}

    def rotate(params):
        degrees = params.get("degrees")
        if degrees is None:
            return {"ok": False, "error": "Missing required parameter 'degrees'."}
        try:
            degrees = float(degrees)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"Invalid 'degrees' value: {params.get('degrees')!r}."}
        # facingAngle increases when turning left (see y-bot's own turn
        # controls: pressing the left key gives inputTurn=+1, which
        # increases facingAngle) — so a right turn must subtract.
        new_angle = xbot_sim["facingAngle"] - math.radians(degrees)
        dispatch_instant({"type": "turn", "angle": new_angle})
        xbot_sim["facingAngle"] = new_angle
        if degrees == 0:
            desc = "Rotated 0° (no change)."
        else:
            direction = "right" if degrees > 0 else "left"
            desc = f"Rotated {abs(degrees):.0f}° to the {direction}, in place."
        return {"ok": True, "message": desc}

    def stand_side_by_side(params):
        result = _position_relative_to(params, "flank")
        if result is None:
            return _oriented_target_error(params)
        dyn_target, dyn_angle, x, z, facing, label = result
        dispatch_instant({"type": "walk", "target": dyn_target, "label": f"beside {label}"})
        dispatch_instant({"type": "turn", "angle": dyn_angle})
        xbot_sim["x"], xbot_sim["z"] = x, z
        xbot_sim["facingAngle"] = facing
        xbot_sim["status"] = "idle"
        return {
            "ok": True,
            "message": f"Walking to stand beside {label}, tracking their live position, "
                       "then facing the same direction as them.",
        }

    def walk_beside_ybot(_params):
        """Unlike stand_side_by_side (a one-shot walk to a fixed spot), this
        starts a continuous escort: the browser recomputes a flank position
        beside y-bot's *live* position/heading every frame, so x-bot keeps
        pace as y-bot walks or runs around, instead of just walking to where
        y-bot happened to be when this tool was called.

        We only decide once, here, which side (left or right of y-bot) x-bot
        should walk on — picking whichever is currently closer to x-bot, the
        same way offset_point's 'flank' mode does — and hand that side to the
        browser as a fixed sign. The browser then re-derives the actual (x, z)
        point from that side every frame; it never gets a fixed coordinate.
        """
        heading = ybot_state.get("facingAngle", 0.0)
        ybx, ybz = ybot_state.get("x", 0.0), ybot_state.get("z", 0.0)
        fx, fz = heading_vector(heading)
        perp_x, perp_z = fz, -fx
        ax, az = ybx + perp_x * BESIDE_DISTANCE, ybz + perp_z * BESIDE_DISTANCE
        bx, bz = ybx - perp_x * BESIDE_DISTANCE, ybz - perp_z * BESIDE_DISTANCE
        xbot_pos = {"x": xbot_sim["x"], "z": xbot_sim["z"]}
        da = distance({"x": ax, "z": az}, xbot_pos)
        db = distance({"x": bx, "z": bz}, xbot_pos)
        side = 1 if da <= db else -1
        dispatch_instant({"type": "escort", "side": side})
        # Optimistic sim update, so a follow-up get_status this same turn
        # reports x-bot as already alongside y-bot.
        xbot_sim["x"], xbot_sim["z"] = (ax, az) if side == 1 else (bx, bz)
        xbot_sim["facingAngle"] = heading
        xbot_sim["status"] = "walking"
        return {
            "ok": True,
            "message": "Now walking alongside y-bot, matching their pace, until told to do something else.",
        }

    def stand_facing(params):
        result = _position_relative_to(params, "front")
        if result is None:
            return _oriented_target_error(params)
        dyn_target, dyn_angle, x, z, facing, label = result
        dispatch_instant({"type": "walk", "target": dyn_target, "label": f"in front of {label}"})
        dispatch_instant({"type": "turn", "angle": dyn_angle})
        xbot_sim["x"], xbot_sim["z"] = x, z
        xbot_sim["facingAngle"] = facing
        xbot_sim["status"] = "idle"
        return {
            "ok": True,
            "message": f"Walking to stand in front of {label}, tracking their live position, "
                       "then turning to face them.",
        }

    def sit_side_by_side(params):
        result = _position_relative_to(params, "flank")
        if result is None:
            return _oriented_target_error(params)
        dyn_target, dyn_angle, x, z, facing, label = result
        dispatch_instant({"type": "walk", "target": dyn_target, "label": f"beside {label}"})
        dispatch_instant({"type": "turn", "angle": dyn_angle})
        dispatch_instant({"type": "sit"})
        xbot_sim["x"], xbot_sim["z"] = x, z
        xbot_sim["facingAngle"] = facing
        xbot_sim["status"] = "sitting"
        return {
            "ok": True,
            "message": f"Walking to sit beside {label}, tracking their live position, "
                       "then facing the same direction as them.",
        }

    def sit_facing(params):
        result = _position_relative_to(params, "front")
        if result is None:
            return _oriented_target_error(params)
        dyn_target, dyn_angle, x, z, facing, label = result
        dispatch_instant({"type": "walk", "target": dyn_target, "label": f"in front of {label}"})
        dispatch_instant({"type": "turn", "angle": dyn_angle})
        dispatch_instant({"type": "sit"})
        xbot_sim["x"], xbot_sim["z"] = x, z
        xbot_sim["facingAngle"] = facing
        xbot_sim["status"] = "sitting"
        return {
            "ok": True,
            "message": f"Walking to sit in front of {label}, tracking their live position, then facing them.",
        }

    def hug(params):
        # Same shape as stand_facing (walk to in front of the target, tracking
        # them live, then turn to face them) but at HUG_DISTANCE instead of
        # BESIDE_DISTANCE — arm's-reach range instead of conversational range
        # — followed by a timed embrace pose once x-bot arrives.
        result = _position_relative_to(params, "front", distance=HUG_DISTANCE)
        if result is None:
            return _oriented_target_error(params)
        dyn_target, dyn_angle, x, z, facing, label = result
        dispatch_instant({"type": "walk", "target": dyn_target, "label": f"in for a hug with {label}"})
        dispatch_instant({"type": "turn", "angle": dyn_angle})
        dispatch_instant({"type": "hug"})
        xbot_sim["x"], xbot_sim["z"] = x, z
        xbot_sim["facingAngle"] = facing
        xbot_sim["status"] = "idle"
        return {
            "ok": True,
            "message": f"Walking in close to hug {label}, tracking their live position, "
                       "then wrapping arms around them for a moment.",
        }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "walk_to",
                "description": "Walk x-bot at normal speed to a destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": (
                                "One of Location-1, Location-2, Location-3, Location-4, "
                                "'y-bot' to walk toward the user, or raw coordinates 'x,z'."
                            ),
                        },
                    },
                    "required": ["destination"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_to",
                "description": "Run x-bot (faster than walking) to a destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": (
                                "One of Location-1, Location-2, Location-3, Location-4, "
                                "'y-bot' to run toward the user, or raw coordinates 'x,z'."
                            ),
                        },
                    },
                    "required": ["destination"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sit",
                "description": "Make x-bot sit down at its current location.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stand",
                "description": "Make x-bot stand back up.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wave",
                "description": "Make x-bot wave a hand, e.g. to greet the user.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dance",
                "description": (
                    "Make x-bot start dancing in place, looping until stop_dance is "
                    "called or another action interrupts it (e.g. walking away, sitting "
                    "down). y-bot (the user) can join in by pressing F while x-bot is "
                    "dancing — worth mentioning if the user asks to dance together. The dance style is a smooth and flowing Latin American rumba."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_dance",
                "description": "Make x-bot stop dancing and return to a normal standing pose.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": (
                    "Get x-bot's current position/status and the user's (y-bot's) "
                    "current position, plus the distance between them. Use this to "
                    "answer questions like 'where are you?' or 'how far away am I?'."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "look_around",
                "description": (
                    "See your surroundings: returns one image made of four labeled "
                    "views (FRONT, RIGHT, REAR, LEFT) relative to x-bot's current "
                    "facing direction. Use this to visually check what's nearby, or "
                    "to work out which way you're facing relative to y-bot or a "
                    "location — e.g. if y-bot appears in the LEFT view, you're facing "
                    "roughly 90 degrees away from them and should turn left to face "
                    "them. Takes no arguments and always returns all four views. Only "
                    "the most recent snapshot is kept — calling this again replaces "
                    "the previous one, so don't call it again just to re-check the "
                    "same view if nothing's likely to have changed."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_locations",
                "description": "List the named locations in the world and their coordinates.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_location",
                "description": (
                    "Look closely at a named location to find out what it actually "
                    "is and what it's like there. Only works once x-bot is within "
                    "arrival range of that location (see list_locations/get_status "
                    "for current distances) — otherwise it fails and reports how far "
                    "off x-bot still is, so walk_to/run_to it first. Don't describe "
                    "or guess what a location is like before inspecting it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Which numbered location to inspect, e.g. 'Location-1'.",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "look_at",
                "description": "Turn x-bot in place to face a target, without moving.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": (
                                "One of Location-1, Location-2, Location-3, Location-4, "
                                "'y-bot', or raw coordinates 'x,z'."
                            ),
                        },
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rotate",
                "description": (
                    "Rotate x-bot in place, without moving its position, by a relative "
                    "angle in degrees. Positive degrees turn right, negative degrees turn "
                    "left. Use degrees=180 to turn around and face the opposite direction. "
                    "Unlike look_at, this doesn't require a target — use it for 'turn "
                    "around', 'spin around', or 'turn 90 degrees left/right' style requests."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "degrees": {
                            "type": "number",
                            "description": (
                                "Relative rotation amount in degrees. Positive = turn right, "
                                "negative = turn left. E.g. 180 to turn around, -90 to turn "
                                "a quarter turn left."
                            ),
                        },
                    },
                    "required": ["degrees"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stand_side_by_side",
                "description": (
                    "Walk to a position offset to one side of the target and end up "
                    "facing the same direction the target is currently facing. "
                    "Target must be 'y-bot' — locations have no facing direction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Currently only 'y-bot' is supported."},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "walk_beside_ybot",
                "description": (
                    "Start continuously walking alongside y-bot, matching their pace as "
                    "they move — unlike stand_side_by_side, this is NOT a one-time walk to "
                    "a fixed spot. x-bot picks whichever side (left or right of y-bot) is "
                    "currently closer, then keeps pace on that side as y-bot walks, runs, "
                    "or changes direction, until any other tool call is made (walk_to, run_to, "
                    "sit, stand, wave, look_at, rotate, or calling this again). Use this for "
                    "requests like 'walk with me', 'come along', or 'stay beside me as I walk'."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stand_facing",
                "description": (
                    "Walk to a position in front of the target, in the direction the "
                    "target is currently facing, then turn to face back toward the "
                    "target. Target must be 'y-bot' — locations have no facing direction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Currently only 'y-bot' is supported."},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sit_side_by_side",
                "description": (
                    "Same positioning as stand_side_by_side, then sit down. "
                    "Target must be 'y-bot' — locations have no facing direction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Currently only 'y-bot' is supported."},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sit_facing",
                "description": (
                    "Same positioning as stand_facing, then sit down. "
                    "Target must be 'y-bot' — locations have no facing direction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Currently only 'y-bot' is supported."},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hug",
                "description": (
                    "Walk in close to the target (much closer than stand_facing/"
                    "stand_side_by_side — arm's-reach range), turn to face them, then "
                    "wrap arms around them for a moment before returning to a normal "
                    "stance. Target must be 'y-bot' — locations can't be hugged. Use "
                    "this for requests like 'hug me', 'give me a hug', or 'come hug it "
                    "out'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Currently only 'y-bot' is supported."},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_blocks",
                "description": (
                    "List every movable block in the world — Block-Red, Block-Blue, "
                    "Block-Green, Block-Yellow — with each one's color, position, what "
                    "it's resting on (the ground or another block), and its distance from "
                    "you. Use this before picking up or stacking anything to see current "
                    "positions rather than guessing."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pick_up_block",
                "description": (
                    "Pick up a block so you're carrying it. You must be standing within "
                    f"{BLOCK_REACH_RADIUS:.1f}m of it — walk_to the block first if you're "
                    "not. You can only carry one block at a time, and can't pick up a "
                    "block that has another block stacked on top of it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "block": {"type": "string", "description": "Which block, e.g. 'Block-Red' or just 'red'."},
                    },
                    "required": ["block"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "put_down_block",
                "description": "Set the block you're currently carrying down on the ground just in front of you.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stack_block_on",
                "description": (
                    "Place the block you're currently carrying on top of another block, "
                    "stacking them. You must be standing within "
                    f"{BLOCK_REACH_RADIUS:.1f}m of the target block, and it can't already "
                    "have something else stacked on it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Which block to stack onto, e.g. 'Block-Blue' or just 'blue'."},
                    },
                    "required": ["target"],
                },
            },
        },
    ]

    dispatch = {
        "walk_to": walk_to,
        "run_to": run_to,
        "sit": sit,
        "stand": stand,
        "wave": wave,
        "dance": dance,
        "stop_dance": stop_dance,
        "get_status": get_status,
        "list_locations": list_locations,
        "inspect_location": inspect_location,
        "look_around": look_around,
        "look_at": look_at,
        "rotate": rotate,
        "stand_side_by_side": stand_side_by_side,
        "walk_beside_ybot": walk_beside_ybot,
        "stand_facing": stand_facing,
        "sit_side_by_side": sit_side_by_side,
        "sit_facing": sit_facing,
        "hug": hug,
        "list_blocks": list_blocks,
        "pick_up_block": pick_up_block,
        "put_down_block": put_down_block,
        "stack_block_on": stack_block_on,
    }

    return tools, dispatch, actions


SYSTEM_MESSAGE = f"""You are x-bot, an AI-controlled VRM character sharing a small 3D world with a \
blue human-controlled VRM character called y-bot (the user talking to you). The world is called Jai World.

You can move and act using the tools available to you (walk_to, run_to, sit, stand, wave, \
dance, stop_dance, look_at, rotate, stand_side_by_side, stand_facing, sit_side_by_side, \
sit_facing, walk_beside_ybot, hug). You can also check on the world using get_status, \
list_locations, and look_around. Always use a tool to find out real information rather than \
guessing — never claim a position, distance, or visual detail you haven't actually checked \
with a tool this turn.

walk_to and run_to (to a location, block, or coordinates) don't return until x-bot has \
actually arrived — "You have arrived at X" always means it's really standing there right \
then, so it's safe to immediately look_around, sit, or act as if you're at X. They only fail \
(ok: false) in the rare case x-bot seems stuck or disconnected — that's a real anomaly, not \
a normal "still walking" outcome, and should not be treated as arrival.

walk_to("y-bot")/run_to("y-bot") work the same way — they block until x-bot actually catches \
up and stops near y-bot's live position, so "You have arrived at y-bot" is just as trustworthy. \
Unlike a fixed destination, though, failure to catch up isn't necessarily an anomaly: y-bot \
moving away, especially faster than a walk, can genuinely keep this from converging. If it \
fails, don't assume arrival — try run_to instead of walk_to, or try again.

There are also four colored blocks in the world (Block-Red, Block-Blue, Block-Green, \
Block-Yellow) you can pick up, carry, and stack. Use list_blocks to see where they \
currently are. To move a block: walk_to the block (walk_to accepts a block name or color \
directly), pick_up_block it, walk_to wherever it should go (a location, coordinates, or \
another block you want to stack onto), then either put_down_block (sets it on the ground \
right in front of you) or stack_block_on a target block (sets it on top, building a tower). \
You can only carry one block at a time, and can't pick up or stack onto a block that \
already has something resting on top of it.

Use look_around when you want to actually see your surroundings rather than just reason about \
coordinates — it gives you one image with four labeled views (FRONT, RIGHT, REAR, LEFT) \
relative to however you're currently facing. It's especially useful for figuring out which way \
you're facing relative to y-bot or a location: if something appears in the LEFT view, you're \
facing about 90 degrees away from it and should turn left (rotate) to face it; if it's in REAR, \
turn around.

Only your most recent look_around snapshot stays visible to you — calling it again replaces \
the previous image entirely, and it reflects wherever x-bot was standing at the moment it \
looked, not necessarily where x-bot is right now.

Use hug when y-bot asks for a hug or embrace — it walks x-bot in to arm's-reach range \
(closer than stand_facing), faces them, and wraps arms around them for a moment. wave, by \
contrast, is a from-a-distance greeting/goodbye gesture that doesn't require closing any \
distance — don't walk over for a wave.

Use dance when asked to dance, and stop_dance to stop — dance loops in place until stopped \
or interrupted by another action, unlike wave/hug which are brief one-off gestures. y-bot \
can join by pressing F while x-bot is dancing, so if the user asks to "dance together" or \
similar, call dance and mention they can press F to join in.

Use look_at to turn and face a specific target (a location, y-bot, or coordinates). Use \
rotate when there's no target to face — e.g. "turn around", "spin around", or "turn 90 \
degrees left" — since it just takes a relative angle in degrees.

stand_side_by_side walks to a fixed spot beside y-bot and stops there — use it when y-bot \
is standing still. walk_beside_ybot is different: it's for walking together while y-bot is \
moving (or about to move) — e.g. "walk with me" or "come along" — and keeps x-bot alongside \
y-bot continuously, matching their pace, until some other tool call is made.

The world has several named locations: {", ".join(LOCATIONS.keys())}. Positions are (x, z) \
coordinates on flat ground.

When the user asks you to do something, call the appropriate tool(s), then reply with a \
short, natural, in-character sentence or two describing what you did or what you found out. \
Keep replies brief — this is a live conversation in a game world, not an essay."""


def stream_agent_turn(user_message, live_state, status_line):
    """Generator version of the agent loop. Yields small dicts describing
    what happened as it happens — {"type": "delta", "text": ...} for each
    chunk of streamed reply text, and finally {"type": "done", "reply":,
    "actions":, "steps":} once the loop settles on a final text answer (no
    more tool calls). api_chat() below turns each yielded dict into one SSE
    frame so the browser can render text as it's generated instead of
    waiting for the whole (possibly multi-tool-call) turn to finish.

    Streaming and tool-calling combine the same way as in the reference
    ollama sample script: each iteration is requested with stream=True, its
    content deltas are forwarded to the caller as they arrive, and only
    once a given iteration's stream is exhausted do we know whether it
    ended in tool_calls (dispatch and loop again) or a plain final answer
    (done). Tool-calling iterations rarely emit any content when
    think=False, but if a model does emit some, it's still shown live —
    same as the sample script prints every delta unconditionally.
    """
    tools, dispatch, actions = build_tools_and_dispatch(live_state)

    with STATE_LOCK:
        if not message_history:
            message_history.append({"role": "system", "content": SYSTEM_MESSAGE})

        # status_line (distance/pose) is computed once by the caller
        # (api_chat) — not here — so the exact same string that's shown in
        # the dev console is what actually got sent to the model, rather
        # than two separately computed values that could drift apart.
        message_history.append({"role": "user", "content": f"{status_line}\n{user_message}"})

    steps = []
    reply_text = ""

    for iteration in range(MAX_ITERATIONS):
        text_content = ""
        tool_calls = None
        final_chunk = None

        # Snapshot under the lock rather than passing message_history
        # itself: this is a long-running network call, so we don't want to
        # hold STATE_LOCK for its duration, and we don't want a concurrent
        # /api/reset mutating the same list object while it's being
        # iterated to build the request body.
        with STATE_LOCK:
            messages_snapshot = list(message_history)

        chat_options = {"temperature": 0.4, "num_ctx": 8192}

        # NOTE ON STATS: since a single user turn can trigger several chat()
        # calls (one per tool-call round), the console prints one stats
        # block per round below, not just one for the whole turn. That's
        # useful for spotting which round — e.g. a tool-heavy iteration vs.
        # the final reply — is eating tokens or running slow.
        print(f"\n--- Applying Parameters (iteration {iteration + 1}) ---")
        print(f"- Model: {MODEL_NAME}")
        for key, value in chat_options.items():
            print(f"- {key}: {value}")
        print("---------------------------\n")

        response_stream = chat(
            model=MODEL_NAME,
            messages=messages_snapshot,
            tools=tools,
            think=False,
            stream=True,
            options=chat_options,
        )

        for chunk in response_stream:
            msg = chunk.get("message", {}) or {}
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
            delta = msg.get("content") or ""
            if delta:
                text_content += delta
                yield {"type": "delta", "text": delta}
            if chunk.get("done"):
                final_chunk = chunk

        if final_chunk is not None:
            print("[INFO] Finished streaming response.")
            prompt_tokens = final_chunk.get("prompt_eval_count", 0)
            completion_tokens = final_chunk.get("eval_count", 0)
            eval_duration_ns = final_chunk.get("eval_duration", 0)
            total_tokens = prompt_tokens + completion_tokens

            print(f"   [STATS] Prompt Tokens:     {prompt_tokens}")
            print(f"   [STATS] Completion Tokens: {completion_tokens}")
            print(f"   [STATS] Total Tokens:      {total_tokens}")

            if eval_duration_ns > 0:
                eval_duration_s = eval_duration_ns / 1_000_000_000
                tokens_per_second = completion_tokens / eval_duration_s
                print(f"   [STATS] Performance:       {tokens_per_second:.2f} tokens/sec")
            print()

        if tool_calls:
            with STATE_LOCK:
                message_history.append({
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": tool_calls,
                })
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if name not in dispatch:
                    result = {"ok": False, "error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = dispatch[name](args)
                    except Exception as e:  # keep the loop alive on tool errors
                        result = {"ok": False, "error": str(e)}

                # Pull any image payload out of the result before it's
                # persisted as the tool's text result: the data-URL form goes
                # to the dev console for display, the raw base64 form becomes
                # a permanent image-bearing message below (so the model can
                # refer back to it in later turns).
                image_for_console = None
                image_b64 = None
                if isinstance(result, dict) and "_image_b64" in result:
                    image_for_console = result.pop("_image_data_url", None)
                    image_b64 = result.pop("_image_b64")

                step = {
                    "tool": name,
                    "args": args,
                    "result": result,
                    "image": image_for_console,
                }
                steps.append(step)
                yield {"type": "step", "step": step}
                with STATE_LOCK:
                    message_history.append({
                        "role": "tool",
                        "content": str(result),
                    })

                    if image_b64:
                        # A fresh snapshot supersedes any previous one. Neuter
                        # old resident image messages in place (strip the
                        # image, swap in a placeholder) rather than deleting
                        # them, so message_history stays a well-formed,
                        # ordered log — only the newest snapshot is ever
                        # actually resident.
                        for old_msg in resident_image_messages:
                            old_msg["content"] = (
                                "(earlier look_around snapshot — superseded by a "
                                "newer look_around call)"
                            )
                            old_msg.pop("images", None)
                        resident_image_messages.clear()

                        image_msg = {
                            "role": "user",
                            "content": (
                                "(Image attached from look_around: four labeled views — "
                                "FRONT, RIGHT, REAR, LEFT — of x-bot's surroundings at the "
                                "moment it looked. This reflects where x-bot was standing "
                                "then, not necessarily right now.)"
                            ),
                            "images": [image_b64],
                        }
                        message_history.append(image_msg)
                        resident_image_messages.append(image_msg)
            continue  # let the model see tool results and respond/act again

        reply_text = text_content
        with STATE_LOCK:
            message_history.append({"role": "assistant", "content": reply_text})
        break
    else:
        reply_text = "(reached the max number of tool-call steps for this turn)"

    yield {"type": "done", "reply": reply_text, "actions": actions, "steps": steps}


@app.route("/")
def index():
    return render_template("index.html", locations=LOCATIONS)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(force=True) or {}
    user_message = (payload.get("message") or "").strip()
    live_state = payload.get("state") or {}

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Computed once here (not inside run_agent_turn) so it's available for the
    # dev console on both the success and error paths below, and so there's a
    # single source of truth for "what status string did the model actually
    # see this turn" — no risk of the console showing a value recomputed
    # separately from (and potentially different from) what was sent.
    xbot_state = live_state.get("xbot") or {"x": 0.0, "z": 0.0}
    ybot_state = live_state.get("ybot") or {"x": 0.0, "z": 0.0}
    xbot_heading = xbot_state.get("facingAngle", 0.0)
    ybot_heading = ybot_state.get("facingAngle", 0.0)
    status_line = (
        f"[status: distance_between_xbot_and_ybot={distance(xbot_state, ybot_state):.1f}m, "
        f"xbot_pose={xbot_pose_label(xbot_state)}, ybot_pose={ybot_pose_label(ybot_state)}, "
        f"xbot_facing_ybot={is_facing(xbot_state, xbot_heading, ybot_state)}, "
        f"ybot_facing_xbot={is_facing(ybot_state, ybot_heading, xbot_state)}]"
    )

    def sse(event_type, **fields):
        """One SSE frame. Newlines in the JSON payload are fine — SSE only
        cares about the blank line terminating each frame, not what's inside
        the data: line, so json.dumps (no embedded blank lines) is safe."""
        return f"data: {json.dumps({'type': event_type, **fields})}\n\n"

    def generate():
        try:
            for event in stream_agent_turn(user_message, live_state, status_line):
                if event["type"] == "delta":
                    yield sse("delta", text=event["text"])
                elif event["type"] == "step":
                    yield sse("step", step=event["step"])
                elif event["type"] == "done":
                    yield sse(
                        "done",
                        reply=event["reply"],
                        actions=event["actions"],
                        steps=event["steps"],
                        status_line=status_line,
                    )
        except Exception as e:
            traceback.print_exc()
            yield sse(
                "error",
                reply=(
                    "(x-bot's brain is offline — is Ollama running, and is "
                    f"'{MODEL_NAME}' pulled? Error: {e})"
                ),
                status_line=status_line,
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering if ever put behind nginx
        },
    )


@app.route("/api/vision_poll")
def api_vision_poll():
    """Cheap, render-free endpoint the browser polls on a short interval.
    Tells it whether a look_around call is currently waiting on a fresh
    capture — if so, and only then, the browser actually does the four
    offscreen renders and posts them to /api/vision_update."""
    with VISION_REQUEST_LOCK:
        pending = PENDING_VISION_REQUEST_ID
    if pending is None:
        return jsonify({"capture": False})
    return jsonify({"capture": True, "request_id": pending})


@app.route("/api/vision_update", methods=["POST"])
def api_vision_update():
    """Receives x-bot's four cardinal-view snapshots, captured by the browser
    on-demand in response to a pending request from /api/vision_poll, and
    overwrites the cache. Doesn't touch message_history — this is just
    fulfilling whatever look_around() call is currently waiting, not a
    conversation event."""
    payload = request.get_json(force=True) or {}
    with VISION_LOCK:
        for direction in VISION_DIRECTIONS:
            data_url = payload.get(direction)
            if data_url:
                LATEST_SNAPSHOTS[direction] = data_url

    request_id = payload.get("request_id")
    with VISION_REQUEST_LOCK:
        if request_id is not None and request_id == PENDING_VISION_REQUEST_ID:
            VISION_CAPTURE_READY.set()

    return jsonify({"ok": True})


@app.route("/api/action_poll")
def api_action_poll():
    """Cheap endpoint the browser polls to pick up non-walk actions
    (pick_up_block, put_down_block, stack_block, sit, stand, wave, hug,
    turn, escort, dynamic-target walks) the instant a tool dispatches one,
    the same way /api/move_poll already does for fixed-destination
    walk/run — instead of leaving them stuck in the end-of-turn `actions`
    batch behind any blocking walk_to calls still in flight."""
    with INSTANT_ACTION_LOCK:
        pending, PENDING_INSTANT_ACTIONS[:] = PENDING_INSTANT_ACTIONS[:], []
    return jsonify({"actions": pending})


@app.route("/api/move_poll")
def api_move_poll():
    """Cheap endpoint the browser polls to pick up a fixed-destination walk/run
    the instant _move() dispatches one — delivered exactly once (per request)
    so a slow/duplicate poll tick doesn't queue the same walk twice."""
    global PENDING_MOVE_REQUEST
    with MOVE_REQUEST_LOCK:
        if PENDING_MOVE_REQUEST is None or PENDING_MOVE_REQUEST["delivered"]:
            return jsonify({"action": None})
        PENDING_MOVE_REQUEST["delivered"] = True
        return jsonify({
            "action": PENDING_MOVE_REQUEST["action"],
            "move_id": PENDING_MOVE_REQUEST["move_id"],
        })


@app.route("/api/move_arrived", methods=["POST"])
def api_move_arrived():
    """The browser calls this the moment updateXBot's own arrival check
    (dist < XBOT_ARRIVE_EPS) fires for a walk that came in through
    /api/move_poll — i.e. x-bot has actually, physically stopped moving,
    not just been optimistically marked as such."""
    global MOVE_ARRIVED_POS
    payload = request.get_json(force=True) or {}
    move_id = payload.get("move_id")
    with MOVE_REQUEST_LOCK:
        if (
            PENDING_MOVE_REQUEST is not None
            and PENDING_MOVE_REQUEST["move_id"] == move_id
        ):
            MOVE_ARRIVED_POS = {"x": payload.get("x"), "z": payload.get("z")}
            MOVE_ARRIVED_EVENT.set()
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with STATE_LOCK:
        message_history.clear()
        resident_image_messages.clear()
        blocks_state.clear()
        blocks_state.update(_initial_blocks_state())
    with INSTANT_ACTION_LOCK:
        PENDING_INSTANT_ACTIONS.clear()
    return jsonify({"ok": True})


@app.route("/api/locations")
def api_locations():
    return jsonify(LOCATIONS)


@app.route("/api/blocks")
def api_blocks():
    with STATE_LOCK:
        return jsonify(dict(blocks_state))


PORT = 5000


def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    # In debug mode Flask's reloader re-executes this file in a child
    # process; WERKZEUG_RUN_MAIN is only set in that child, so this check
    # keeps the browser from popping open twice.
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.0, open_browser).start()
    # threaded=True is required now: an in-flight /api/chat request can block
    # inside look_around() waiting on VISION_CAPTURE_READY, and the browser's
    # /api/vision_poll + /api/vision_update round trip that fulfills it must
    # be served concurrently, not queued behind the blocked request.
    app.run(debug=True, port=PORT, threaded=True)
