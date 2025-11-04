"""
Run A_fluvpy_main.py in a loop
"""
import subprocess
import sys
import time


def run_fluvpy_multiple_times(num_iterations):
    """
    Runs A_fluvpy_main.py for the specified number of iterations in a loop

    Args:
        num_iterations: The number of times to run the script
    """

    print(f"Starting loop execution of main.py for {num_iterations} iterations")

    # Record the start time
    start_time = time.time()

    # Loop for the specified number of iterations
    for i in range(num_iterations):
        print(f"\n==== Iteration {i+1}/{num_iterations} ====")

        # Use subprocess to call the external program
        try:
            # Execute A_fluvpy_main_TI64-64-25.py (Note: The script name in the command is 'main.py')
            subprocess.run(["python", "A_fluvpy_main均质-训练数据.py"], check=True)
            print(f"Iteration {i+1} completed")
        except subprocess.CalledProcessError as e:
            print(f"Execution failed: {e}")
        except FileNotFoundError:
            print("Error: main.py not found. Please ensure the file exists and the path is correct.")
            break

    # Calculate the total time elapsed
    total_time = time.time() - start_time
    print(f"\nLoop execution finished. Total iterations: {num_iterations}")
    print(f"Total time elapsed: {total_time:.2f} seconds")

if __name__ == "__main__":
    # Get the number of iterations from command-line arguments, default to 1
    iterations = 10

    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print("Error: Please enter a valid number of iterations (integer)")
            sys.exit(1)

    # Run the loop
    run_fluvpy_multiple_times(iterations)