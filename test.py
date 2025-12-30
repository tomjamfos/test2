import random
import time

# Matrix Falling Rain Effect

def matrix_rain():
    """Create a matrix-style falling rain effect for 30 seconds"""

    # Matrix characters (mix of katakana, numbers, and symbols)
    chars = [
        'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ',
        'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ', 'ﾄ',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '@', '#', '$', '%', '&', '*', '+', '=', '|', ':'
    ]

    # Terminal dimensions (approximate)
    width = 80
    height = 24

    # Initialize columns with random starting positions
    columns = []
    for i in range(width):
        columns.append({
            'y': random.randint(-height, 0),
            'speed': random.choice([1, 2, 3]),
            'chars': [random.choice(chars) for _ in range(height)]
        })

    start_time = time.time()
    duration = 30  # seconds

    print("\033[2J")  # Clear screen
    print("\033[?25l")  # Hide cursor

    try:
        while time.time() - start_time < duration:
            # Clear screen and move cursor to top
            print("\033[H", end='')

            # Build the frame
            frame = [[' ' for _ in range(width)] for _ in range(height)]

            # Update and draw each column
            for col_idx, col in enumerate(columns):
                # Move column down
                col['y'] += col['speed']

                # Reset if column is off screen
                if col['y'] > height + 10:
                    col['y'] = random.randint(-height, -1)
                    col['speed'] = random.choice([1, 2, 3])
                    col['chars'] = [random.choice(chars) for _ in range(height)]

                # Draw the column
                for i in range(height):
                    y_pos = col['y'] - i
                    if 0 <= y_pos < height:
                        # Brightest at the head
                        if i == 0:
                            frame[y_pos][col_idx] = f"\033[97m{col['chars'][i]}\033[0m"  # Bright white
                        elif i < 3:
                            frame[y_pos][col_idx] = f"\033[92m{col['chars'][i]}\033[0m"  # Bright green
                        elif i < 8:
                            frame[y_pos][col_idx] = f"\033[32m{col['chars'][i]}\033[0m"  # Green
                        else:
                            frame[y_pos][col_idx] = f"\033[2;32m{col['chars'][i]}\033[0m"  # Dim green

                # Randomly change some characters
                if random.random() < 0.1:
                    idx = random.randint(0, len(col['chars']) - 1)
                    col['chars'][idx] = random.choice(chars)

            # Print the frame
            for row in frame:
                print(''.join(row))

            time.sleep(0.05)  # Control speed

        # Show elapsed time message
        print("\033[2J\033[H")  # Clear screen
        print("\033[92m" + "="*60)
        print(" " * 15 + "MATRIX RAIN COMPLETE")
        print("="*60 + "\033[0m")
        print(f"\nRan for {duration} seconds")

    finally:
        print("\033[?25h")  # Show cursor again
        print("\033[0m")    # Reset colors

if __name__ == "__main__":
    print("\033[92mStarting Matrix Rain Effect...\033[0m")
    print("Running for 30 seconds...\n")
    time.sleep(1)
    matrix_rain()
