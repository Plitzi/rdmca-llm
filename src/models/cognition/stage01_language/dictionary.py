"""Tiered child-first dictionary bank, shared by the definitions and grammar sources
(stage 1). A level includes every tier ≤ its number, so vocabulary AND definitions
grow per level (the identical-structure principle: same source, size scales with
level). entry: word -> (pos, definition)  ·  pos: 'n' noun · 'v' verb · 'a' adjective.
"""

from __future__ import annotations

DICT_TIER1: dict[str, tuple] = {
    "car": ("n", "a road vehicle with wheels and an engine that people drive"),
    "dog": ("n", "an animal with four legs that many people keep as a pet"),
    "cat": ("n", "a small furry animal that people keep as a pet"),
    "sun": ("n", "the bright star in the sky that gives us light and warmth"),
    "moon": ("n", "the round light we see in the sky at night"),
    "rain": ("n", "water that falls from the clouds"),
    "tree": ("n", "a tall plant with a trunk, branches and leaves"),
    "house": ("n", "a building where people live"),
    "book": ("n", "pages with words and pictures that you read"),
    "water": ("n", "the clear liquid we drink and that fills rivers and seas"),
    "food": ("n", "what people and animals eat to live and grow"),
    "friend": ("n", "someone you like and enjoy spending time with"),
    "hand": ("n", "the part of your body at the end of your arm, with fingers"),
    "school": ("n", "a place where children go to learn"),
    "bird": ("n", "an animal with wings and feathers that can usually fly"),
    "fish": ("n", "an animal that lives in water and swims"),
    "ball": ("n", "a round object you throw, kick or catch in games"),
    "door": ("n", "the part of a building you open to go in or out"),
    "run": ("v", "move quickly using your legs, faster than walking"),
    "eat": ("v", "put food in your mouth and swallow it"),
    "sleep": ("v", "rest with your eyes closed, the way you do at night"),
    "play": ("v", "do something fun, like a game"),
    "read": ("v", "look at words and understand what they say"),
    "walk": ("v", "move along on your feet at a normal speed"),
    "help": ("v", "do something useful for someone"),
    "jump": ("v", "push yourself up into the air with your legs"),
    "give": ("v", "let someone have something"),
    "happy": ("a", "feeling good and pleased"),
    "sad": ("a", "feeling unhappy"),
    "big": ("a", "large in size"),
    "small": ("a", "little in size"),
    "hot": ("a", "having a high temperature, the opposite of cold"),
    "cold": ("a", "having a low temperature, the opposite of hot"),
    "fast": ("a", "moving quickly"),
    "slow": ("a", "moving with little speed"),
    "kind": ("a", "friendly and caring toward others"),
}
DICT_TIER2: dict[str, tuple] = {
    "river": ("n", "a long line of water that flows across the land to the sea"),
    "mountain": ("n", "a very high hill of rock and earth"),
    "doctor": ("n", "a person whose job is to help sick people get better"),
    "machine": ("n", "a thing built from parts that does work using power"),
    "music": ("n", "sounds put together in a way that is nice to listen to"),
    "money": ("n", "the coins and notes people use to buy things"),
    "language": ("n", "the words and rules people use to speak and write"),
    "weather": ("n", "what the air outside is like, such as sunny or rainy"),
    "build": ("v", "make something by putting parts together"),
    "learn": ("v", "get to know something new by studying or practising"),
    "remember": ("v", "keep something in your mind and bring it back"),
    "explain": ("v", "make something clear by telling about it"),
    "brave": ("a", "ready to face danger or pain without being too afraid"),
    "honest": ("a", "telling the truth and not cheating"),
    "heavy": ("a", "weighing a lot, hard to lift"),
    "quiet": ("a", "making little or no noise"),
}
DICT_TIERS = [DICT_TIER1, DICT_TIER2]  # index 0 → level 1, index 1 → levels ≥2
