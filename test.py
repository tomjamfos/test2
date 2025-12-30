import random

# Story Generator - Creates a unique short story each time

# Story components
characters = [
    "a weary detective", "a young programmer", "an old librarian",
    "a mysterious stranger", "a curious child", "a retired astronaut"
]

settings = [
    "in a foggy city", "at an abandoned train station", "in a quiet coffee shop",
    "on a remote island", "in a futuristic lab", "at midnight in the park"
]

conflicts = [
    "discovered a hidden message", "found a glowing artifact",
    "heard a strange melody", "noticed time was moving backwards",
    "encountered their future self", "stumbled upon a secret door"
]

twists = [
    "it was all a dream within a dream", "they had been there before",
    "nothing was as it seemed", "they were the one they were searching for",
    "time looped back to the beginning", "the answer was in the question itself"
]

emotions = [
    "fear", "wonder", "curiosity", "nostalgia", "hope", "confusion"
]

resolutions = [
    "walked away with a smile", "understood everything at last",
    "decided to start over", "kept the secret forever",
    "chose to forget", "embraced the unknown"
]

def generate_story():
    """Generate a random short story"""
    character = random.choice(characters)
    setting = random.choice(settings)
    conflict = random.choice(conflicts)
    emotion = random.choice(emotions)
    twist = random.choice(twists)
    resolution = random.choice(resolutions)

    story = f"""
{'='*60}
                    THE RANDOM STORY
{'='*60}

Once, {character} found themselves {setting}.

In a moment filled with {emotion}, they {conflict}.

Everything changed when they realized that {twist}.

In the end, they {resolution}.

{'='*60}
    """
    return story

def main():
    print("Welcome to the Story Generator!")
    print("\nGenerating your unique story...\n")

    story = generate_story()
    print(story)

    # Story statistics
    total_possibilities = (
        len(characters) * len(settings) * len(conflicts) *
        len(emotions) * len(twists) * len(resolutions)
    )

    print(f"\nThis is one of {total_possibilities:,} possible stories!")
    print("\nRun again for a different story.")

if __name__ == "__main__":
    main()
