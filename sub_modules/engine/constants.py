"""
constants.py
"""
import numpy as np
import hashlib

# Global constants
DEG2RAD = 3.141592654 / 180.0  # Angle to radian conversion factor
EPSLON = 1.0e-10  # Numerical precision control

# Global random state variables - initial definition
random_state = None  # For saving random state
initial_seed = 123456  # Initial seed value

# Substructure random seed offsets - ensure different substructure types use different random sequences
TRIBUTARY_SEED_OFFSET = 10000
LEVEE_SEED_OFFSET = 20000
CREVASSE_SEED_OFFSET = 30000
# Channel seed offset
CHANNEL_SEED_OFFSET = 40000

def get_channel_seed(icc, ic):
    """
    Generate fully random channel seed - enhanced randomness to avoid low-frequency signal similarity.

    Args:
        icc: Channel complex index
        ic: Channel index

    Returns:
        Highly randomized seed value
    """
    # Get base seed
    base_seed = initial_seed

    # Use multiple large primes and complex operations to ensure high randomness of seeds
    prime1 = 982451653  # Large prime 1
    prime2 = 15485863   # Large prime 2
    prime3 = 104729     # Large prime 3
    prime4 = 7919       # Large prime 4
    prime5 = 6700417    # Large prime 5
    prime6 = 4785074    # Large prime 6

    # Complex seed generation formula - avoid linear patterns
    temp_seed = (
        base_seed * prime1 +
        icc * prime2 +
        ic * prime3 +
        (icc * ic) * prime4 +
        (icc ** 2) * prime5 +
        (ic ** 2) * prime6 +
        (icc * ic * (icc + ic)) * 982451 +
        CHANNEL_SEED_OFFSET * prime1
    ) % 2147483647

    # Use hash to increase randomness
    hash_input = f"{base_seed}_{icc}_{ic}_{temp_seed}_{CHANNEL_SEED_OFFSET}".encode()
    hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)

    # Second round hash enhancement
    second_hash_input = f"channel_{hash_value}_{icc * ic}_{temp_seed}".encode()
    second_hash = int(hashlib.sha256(second_hash_input).hexdigest()[:8], 16)

    # Final seed combination
    final_seed = (temp_seed + hash_value + second_hash) % 2147483647

    # Ensure seed is positive
    if final_seed <= 0:
        final_seed = abs(final_seed) + 1

    return final_seed


def set_global_seed(seed):
    """Set global seed and initialize random state."""
    global initial_seed, random_state
    initial_seed = seed
    np.random.seed(seed)
    random_state = np.random.get_state()
    return random_state

def get_tributary_seed(icc, ic):
    """Get deterministic seed for tributary."""
    return initial_seed + TRIBUTARY_SEED_OFFSET + (icc * 100) + ic

def get_levee_seed(icc, ic):
    """Get deterministic seed for levee."""
    return initial_seed + LEVEE_SEED_OFFSET + (icc * 100) + ic

def get_crevasse_seed(icc, ic):
    """Get deterministic seed for crevasse splay."""
    return initial_seed + CREVASSE_SEED_OFFSET + (icc * 150) + ic

def save_random_state():
    """Save current random state."""
    global random_state
    random_state = np.random.get_state()

def restore_random_state():
    """Restore global random state."""
    global random_state
    if random_state is not None:
        np.random.set_state(random_state)



def set_global_params(params):
    """Set global parameters."""
    global global_params
    global_params = params

def get_global_param(key, default=None):
    """Get global parameter."""
    return global_params.get(key, default)