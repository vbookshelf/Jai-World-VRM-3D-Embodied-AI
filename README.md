# Jai World - VRM Embodied AI
Interact with your VRM Embodied agent in a 3D virtual world. A local Flask app that supports both Ollama and OpenRouter.

This app is an experiment that explores embodied AI, AI spatial navigation, and virtual character interaction.
Tech stack: Three.js + HTML + CSS + JS + Flask + Ollama/OpenRouter

- You control Y-bot, a 3D character in a virtual world.
- You interact with X-bot, an AI controlled character.
- Chat with X-bot and ask it to perform various actions.
- Take walks with X-bot or ask for a hug.
- X-bot uses tool calling to execute actions.
- X-bot is vision capable.
- The UI has a dev console that shows the inner workings of the agent - responses, tool calls, images etc.
- Use vibe coding (Claude Sonnet) to modify any aspect of the app - change the world design, change the model etc.
- Easily replace X-bot or Y-bot with your own VRM characters.
- In Thai, Jai means heart, mind, or spirit.


<br>

<img src="images/image1.png" alt="Screenshot showing x-bot and y-bot" height="500">
<p>X-bot (left) and Y-bot (user controlled)</p>

<br>

<img src="images/image2.png" alt="Screenshot showing x-bot and y-bot" height="500">
<p>View of Jai World</p>

<br>

<img src="images/image3.png" alt="Screenshot showing dev console open" height="500">
<p>Dev console showing tool calls</p>

<br>

<img src="images/image5.png" alt="Screenshot showing x-bot and y-bot walking side by side" height="500">
<p>"Walking together" feature</p>

<br>

## How to run this app

To render the 3D graphics at a high frame rate your computer needs to have graphics acceleration. Without it the frame rate may be very slow. On an M4 Macbook Air this app runs at around 60 FPS.

The following instructions are the same for both Mac and Windows. 

```

You need to have the following installed:
- Optional: Ollama with the qwen3.5:9b (6.6 GB) downloaded (only required if not using OpenRouter)
- UV package manager

- Download the project folder and unzip
- Cd into the project folder
- Terminal: uv sync

If using OpenRouter:
- Open the my-api-keys.env.txt file and paste in your OpenRouter API Key.
- Change the file name from my-api-keys.env.txt to my-api-keys.env
- Terminal: uv run python openrouter-app.py

If using Ollama:
- Ensure that Ollama is running.
- Terminal: uv run python ollama-app.py

- The app will open in your browser.
- If using Ollama there will be a slight delay before the first chat response because the model has to load.

```

<br>

## What is a vrm character?
A vrm character is a 3D virtual avatar saved in a universal file format (.vrm) that can be easily moved and used across different games, virtual reality platforms, and streaming software.

Think of it like an MP3 file for 3D models. Just as an MP3 plays on any audio device, a VRM file standardizes a character's skeleton, facial expressions, and textures so you don't have to rebuild, re-texture, or re-rig the avatar every time you switch to a new virtual world.

I like to think of a vrm character as a robot that exposes an API - you can use code to move the joints and limbs.

Pre-built vrm characters can be downloaded from the web. The characters used in this app were downloaded from Mixamo.com in .fbx format. They were then converted to .vrm format. The conversion process is explained here:<br>
https://github.com/vbookshelf/mixamo-vrm-characters

<br>

## How does this app work?

- The 3D world is built using three.js
- The vrm characters (x-bot and y-bot) are loaded into this world
- This is a standard flask app - python backend with a web based (html, css) frontend
- Y-bot is manually controlled by the user
- X-bot is controlled by an AI model served either via Ollama running locally (qwen3.5:9b) or via OpenRouter (qwen3.5-flash-02-23)
- When the user gives instructions to x-bot, function calls are used to execute those instructions (run, walk, sit etc.)

<br>

## How is X-bot able to see the world?

- Status info is sent with every prompt:
```
[status: distance_between_xbot_and_ybot=19.7m,
xbot_pose=standing,
ybot_pose=sitting,
xbot_facing_ybot=False,
ybot_facing_xbot=False]
```
- X-bot can call a get_status tool that returns useful info:
```
tool_call: get_status({})
{
  "distance_between_xbot_and_ybot": 1.2,
  "ok": true,
  "xbot_facing_ybot": false,
  "xbot_holding_block": null,
  "xbot_near": "nowhere named (nearest is Location-2, 12.7m away)",
  "xbot_pose": "sitting",
  "xbot_position": "(-13.0, 5.3)",
  "ybot_facing_xbot": false,
  "ybot_near": "nowhere named (nearest is Location-2, 11.6m away)",
  "ybot_pose": "standing",
  "ybot_position": "(-12.7, 6.5)"
}
```
- X-bot can also call tools to get the coordinates of the numbered locations and of the coloured blocks.
- X-bot is vision capable. It can call a look_around tool that returns a composite image with four views - front, rear, left, right. Imagine four cameras located on X-bot's head. Each time a new image is captured, the previous image is removed from the chat history.<br>
  <img src="images/image4.png" alt="Image showing the 4 views that x-bot sees" height="400">

<br>

## Procedural Movement vs Pre-Made Animations

This app uses procedural movement. This meaans that the code calculates body movements and limb positions in real-time using math and runtime data like speed and direction. In contrast, pre-made animations use fixed, pre-recorded sequences. On Mixamo.com you can explore applying pre-made animations to different characters.

The Rumba dance routine is a mixamo animation.

<br>

## Y-bot controls

- Use the arrow keys or WASD keys to move.
- Press the space bar to take off and fly.
- Press the down arrow key to land.
- When swimming under water, press the space bar to go up.
- Alt+S or Option+S to sit down.
- Tank controls are enabled by default. To turn them off click the button in the top left. Then you can move the camera around freely.

## Walking together
- This capability is useful for "AI as a virtual tour guide" applications.
- When walking together either x-bot or y-bot can take control.
- If x-bot joins y-bot, then the arrow keys or WASD keys will move both bots. Click F to disconnect.
- Click F to attach y-bot to x-bot. Then when x-bot moves y-bot will move automatically. Click F to disconnect. You can also click F to run to and join x-bot when it's already moving.
- If x-bot is dancing, also click F to join in. 


## X-bot capabilities
- Behave like an AI assistant.
- Explore by going to the named locations (Location-1, Location-2 etc.)
- Pick up, move and stack blocks.
- Dance the Rumba.
- If you need more info on any app capabilities try asking X-bot what it can do.

<br>

## System prompt

This is the system prompt:

```
SYSTEM_MESSAGE = f"""You are x-bot, an AI-controlled VRM character sharing a small 3D world with a \
blue human-controlled VRM character called y-bot (the user talking to you). The world is called Jai World.

You can move and act using the tools available to you (walk_to, run_to, sit, stand, wave, \
look_at, rotate, stand_side_by_side, stand_facing, sit_side_by_side, sit_facing, \
walk_beside_ybot, hug). You can also check on the world using get_status, list_locations, \
and look_around. Always use a tool to find out real information rather than guessing — never \
claim a position, distance, or visual detail you haven't actually checked with a tool this turn.

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

```


<br>

## Notes
- Model reasoning is currently turned off. You can turn it on to improve performance, but this will also increase latency.
- I suggest testing bot the Ollama and OpenRouter versions. There's a real difference in latency and intelligence. Seeing this will help you build intuition regarding differences between small and large models.
- Qwen3.5-Flash cost on OpenRouter -> In: $0.065 per 1M,  Out: $0.26 per 1M
- I used Qwen3.8-Max (free version) to generate the three.js 3D world. After that all vibe coding was done using Claude Sonnet 5.0 Medium (free version).

<br>

## References
- Mixamo vrm Characters<br>
https://github.com/vbookshelf/mixamo-vrm-characters

- Mixamo<br>
https://www.mixamo.com/

- Mixamo licensing conditions:<br>
https://community.adobe.com/questions-696/mixamo-faq-licensing-royalties-ownership-eula-and-tos-589400

- Juru Lab Agent Sandbox<br>
https://huggingface.co/datasets/vbookshelf/Juru-Lab-Agent-Sandbox-HYA

<br>

## Versions

Version 1.0<br>
28-August-2026<br>
First release.


<br>
