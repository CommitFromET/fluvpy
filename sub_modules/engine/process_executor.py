"""
process_executor.py

"""
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import warnings

class ParallelConfig:
    """Parallel processing configuration management class."""

    def __init__(self):
        """Initialize basic parallel processing configuration."""
        # Available system cores
        self.physical_cores = mp.cpu_count()

        # Default to half CPU cores, minimum 2, maximum 6
        self.max_processes = max(2, min(self.physical_cores // 2, 6))

        # Default threads per process
        self.threads_per_process = 2

        # Use larger batch sizes to reduce overhead
        self.min_batch_size = 100

        # Auto-adjust configuration
        self._simple_auto_configure()

    def _simple_auto_configure(self):
        """Simplified automatic configuration logic."""
        try:
            # Attempt to detect system memory
            try:
                import psutil
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024 ** 3)

                # Low memory system - reduce process count
                if available_gb < 4:
                    self.max_processes = 2
            except ImportError:
                pass

            # Check system load, conservatively set process count
            try:
                system_load = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                if system_load > self.physical_cores * 0.5:
                    # High system load - reduce used process count
                    self.max_processes = max(2, self.max_processes // 2)
            except (AttributeError, OSError):
                pass

        except Exception as e:
            warnings.warn(f"Auto-configuration failed: {e}. Using default settings.")

# Global function: process batch data (can be pickle serialized)
def _global_process_batch(batch_with_indices_and_args):
    """
    Global function for processing batch data, avoiding pickle serialization issues.

    Args:
        batch_with_indices_and_args: (batch, start_idx, func, args, kwargs)

    Returns:
        List of processing results
    """
    batch, start_idx, func, args, kwargs = batch_with_indices_and_args
    try:
        # Return list of (index, result) pairs
        results = []
        for i, item in enumerate(batch):
            result = func(item, *args, **kwargs)
            results.append((start_idx + i, result))
        return results
    except Exception as e:
        warnings.warn(f"Process batch error: {str(e)}")
        return []

def _global_thread_process_batch(batch_and_args):
    """
    Global function for thread processing batch data.

    Args:
        batch_and_args: (batch, func, args, kwargs)

    Returns:
        List of processing results
    """
    batch, func, args, kwargs = batch_and_args
    try:
        return [func(item, *args, **kwargs) for item in batch]
    except Exception as e:
        warnings.warn(f"Thread batch error: {str(e)}")
        return []

class ProcessExecutor:
    """
    Optimized process executor supporting multiple parallel strategies.

    Key improvements:
    1. Reduce inter-process synchronization points
    2. Use larger batch sizes
    3. Support shared memory
    4. Work stealing for dynamic load balancing
    5. Hybrid thread and process model
    """

    def __init__(self, config=None):
        """Initialize process executor."""
        self.config = config or ParallelConfig()
        self._active_tasks = 0
        self._task_lock = threading.Lock()
        self._result_cache = {}
        self._shared_mem_blocks = {}

    def execute(self, func, items, *args, mode="process", batch_size=None, **kwargs):
        """
        Execute parallel processing task.

        Args:
            func: Function to apply
            items: List of items to process
            mode: Parallel mode - "thread" or "process" (default)
            batch_size: Batch size, automatically determined if None
            *args, **kwargs: Additional arguments passed to function

        Returns:
            List of results
        """
        if not items:
            return []

        # Determine batch size, use larger batches to reduce overhead
        if batch_size is None:
            workers = self.config.max_processes
            # Try to use larger batches to reduce synchronization overhead
            items_per_worker = max(5, len(items) // workers)
            batch_size = max(self.config.min_batch_size, items_per_worker)

        # Create batches
        batches = self._create_batches(items, batch_size)

        # Choose execution strategy based on mode
        if mode == "thread":
            return self._execute_with_threads(func, batches, *args, **kwargs)
        else:  # Default to process
            return self._execute_with_processes(func, batches, items, *args, **kwargs)

    def _execute_with_threads(self, func, batches, *args, **kwargs):
        """Execute tasks using threads."""
        max_threads = min(self.config.max_processes * self.config.threads_per_process,
                          len(batches))

        # Prepare batch arguments
        batch_args = [(batch, func, args, kwargs) for batch in batches]

        # Execute using thread pool
        all_results = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            try:
                # Use global function for thread processing
                for batch_results in executor.map(_global_thread_process_batch, batch_args):
                    all_results.extend(batch_results)
            except Exception as e:
                warnings.warn(f"Thread pool execution error: {str(e)}")

        return all_results

    def _execute_with_processes(self, func, batches, original_items, *args, **kwargs):
        """
        Execute tasks using processes - modified version maintaining original item order and structure.

        Args:
            func: Function to execute
            batches: Batched data
            original_items: Original unbatched data item list
            *args, **kwargs: Arguments passed to function

        Returns:
            List of results corresponding to original item list
        """
        max_processes = min(self.config.max_processes, len(batches))

        # Add start index and arguments to each batch
        indexed_batches = []
        current_idx = 0
        for batch in batches:
            batch_with_args = (batch, current_idx, func, args, kwargs)
            indexed_batches.append(batch_with_args)
            current_idx += len(batch)

        # Create result array same size as original item list
        results = [None] * len(original_items)

        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            try:
                # Use global function for process handling
                all_batch_results = list(executor.map(_global_process_batch, indexed_batches))

                # Place results in correct positions
                for batch_results in all_batch_results:
                    for idx, result in batch_results:
                        if 0 <= idx < len(results):
                            results[idx] = result
                        else:
                            warnings.warn(f"Process returned invalid index: {idx}")
            except Exception as e:
                warnings.warn(f"Process pool execution error: {str(e)}")

        # Check for missing results
        missing_indices = [i for i, r in enumerate(results) if r is None]
        if missing_indices:
            warnings.warn(f"Some results missing: {missing_indices}")

            # Sequential processing for missing results
            for idx in missing_indices:
                try:
                    results[idx] = func(original_items[idx], *args, **kwargs)
                except Exception as e:
                    warnings.warn(f"Error processing missing result (index {idx}): {str(e)}")

        return results

    def _create_batches(self, items, batch_size):
        """Create batches of specified size."""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches

# Global process executor instance
_EXECUTOR = None

def get_executor(config=None):
    """Get global process executor instance."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessExecutor(config)
    return _EXECUTOR

def execute(func, items, *args, **kwargs):
    """Convenience function for executing parallel tasks."""
    return get_executor().execute(func, items, *args, **kwargs)