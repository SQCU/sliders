# d_tdict_async.py
# uv run python d_tdict_async.py >> d_tdict_async.txt 2>&1
"""
A high-performance, safe, asynchronous data loader using a multi-process pipeline.

This module is designed to solve a core problem in Python-based data loading for
machine learning: the Global Interpreter Lock (GIL) and I/O blocking. A standard
data loading loop that performs I/O (e.g., reading files), data transformation
(e.g., tokenization), and tensor conversion in the main training process can
become a significant bottleneck, starving the GPU.

This implementation creates a dedicated, asynchronous pipeline that runs in
separate processes, keeping the GPU fed with a constant stream of pre-processed
data batches.

Key Patterns Implemented:
1.  **Multi-Process Pipeline**:
    - A pool of `_cpu_worker_process` instances pulls raw data, runs a user-
      defined `preprocessing_fn`, and batches the resulting tensors into shared,
      pinned CPU memory.
    - A single `_gpu_transfer_process` moves these completed batches from pinned
      CPU memory to the target GPU, fully asynchronously.

2.  **Context Manager Handle (`BatchHandle`)**:
    - The user never directly touches the underlying memory pools. Instead, they
      receive a `BatchHandle` context manager.
    - This pattern guarantees that the memory buffer associated with a batch is
      automatically and safely released back to the pipeline pool as soon as
      the user's `with` block is exited, preventing memory leaks and deadlocks.

3.  **Forceful, Clean Shutdown (`__exit__` method)**:
    - This is the most complex and critical part of the loader. Python's
      `multiprocessing` library has non-obvious behavior that can easily cause
      a program to hang indefinitely upon exit if not handled correctly.
    - We implement a "controlled demolition" pattern using `q.cancel_join_thread()`
      to mitigate the default pythonic behavior of 'leaking' multiple hardware threads 
      in any execution context which can be partitioned or interrupted.
"""


import itertools
import torch
import multiprocessing as mp
import time
from queue import Empty
from tensordict import TensorDict

# ==============================================================================
# ==                       I. Sentinel Signal Objects                         ==
# ==============================================================================
# Using dedicated classes for signals makes the control flow explicit and type-safe,
# avoiding any ambiguity with using `None` as a sentinel.

class LoaderTerminationSignal:
    """Signal that the data stream is exhausted and the pipeline is empty."""
    pass

class LoaderStalledSignal:
    """Signal that the pipeline is temporarily empty, but not terminated."""
    pass

# ==============================================================================
# ==                    II. The Public BatchHandle Class                      ==
# ==============================================================================
# This is the object the user receives. It's a context manager that guarantees
# the buffer it represents is released back to the pipeline upon exit.

class BatchHandle:
    """A context manager that provides safe, temporary access to a GPU buffer."""
    def __init__(self, loader_instance, gpu_buffer_ref, buffer_index):
        """
        Initializes the handle. The internal_key is no longer needed.

        Args:
            loader_instance: A reference to the parent SafePipelinedTensorDictLoader.
            gpu_buffer_ref: A direct reference to the TensorDict on the target device.
            buffer_index: The integer index of the buffer this handle manages.
        """
        self._loader = loader_instance
        self._gpu_buffer_ref = gpu_buffer_ref
        self._buffer_index = buffer_index

    def __enter__(self):
        """
        REVISED: Called upon entering the 'with' block. Returns the actual
        TensorDict data container.
        """
        return self._gpu_buffer_ref

    def __exit__(self, exc_type, exc_value, traceback):
        """
        (Unchanged) Called upon exiting the 'with' block. Guarantees the buffer is released.
        """
        self._loader._release_buffer_by_index(self._buffer_index)

# ==============================================================================
# ==                  III. Internal Autonomous Worker Functions               ==
# ==============================================================================
# These functions run in separate processes and form the pipeline's engine.

def _cpu_worker_process(
    # --- Operands ---
    raw_data_queue: mp.Queue,
    free_indices_queue: mp.Queue,
    cpu_filled_buffer_queue: mp.Queue,
    batch_size_queue: mp.Queue,
    cpu_pinned_pool: list,
    # The preprocessor now returns a dictionary of tensors.
    preprocessing_fn: callable,
    partial_batch_mode: str,
    sentinel_value: object
    # No longer needs internal_key
):
    """
    Worker that calls the user's preprocessor (which returns a Dict[str, Tensor]),
    wraps each item into a TensorDict, and then batches them by stacking.
    """
    current_bs = 1
    # The buffer will now hold single-item TensorDicts.
    batch_buffer = []

    while True:
        try:
            current_bs = batch_size_queue.get_nowait()
        except Empty:
            pass

        raw_item = raw_data_queue.get()

        if isinstance(raw_item, LoaderTerminationSignal):
            if batch_buffer:
                idx = free_indices_queue.get()
                if not isinstance(idx, LoaderTerminationSignal):
                    # Stack the list of TensorDicts to create one batched TensorDict.
                    stacked_td = torch.stack(batch_buffer, dim=0)
                    cpu_pinned_pool[idx] = stacked_td.pin_memory()
                    cpu_filled_buffer_queue.put(idx)

            cpu_filled_buffer_queue.put(sentinel_value)
            break

        # --- REVISED: Normal Processing Path ---
        # 1. Preprocessor returns a dictionary of tensors.
        user_dict = preprocessing_fn(raw_item)
        # 2. Immediately wrap it in a non-batched TensorDict.
        item_td = TensorDict(user_dict, batch_size=[])
        batch_buffer.append(item_td)

        if len(batch_buffer) >= current_bs:
            idx = free_indices_queue.get()
            if isinstance(idx, LoaderTerminationSignal):
                cpu_filled_buffer_queue.put(sentinel_value)
                break

            # 3. Stack the list of TensorDicts. This correctly creates a
            #    batched TensorDict where each key points to a batched tensor.
            stacked_td = torch.stack(batch_buffer, dim=0)

            # 4. Place the final batched TensorDict in the shared pool.
            cpu_pinned_pool[idx] = stacked_td.pin_memory()
            cpu_filled_buffer_queue.put(idx)
            batch_buffer = []

def _gpu_transfer_process(
    cpu_filled_buffer_queue: mp.Queue,
    ready_for_consumption_queue: mp.Queue,
    cpu_pinned_pool: list,  # This will be a Manager.list
    gpu_pool: list,         # This will be a Manager.list
    device: torch.device,
    sentinel_value: object
):
    """
    An autonomous process that moves data from filled CPU buffers to GPU buffers.
    It does NOT release buffers; it only signals when a transfer is complete.
    """
    while True:
        # Block until a CPU buffer is ready for transfer
        filled_idx = cpu_filled_buffer_queue.get()
        if isinstance(filled_idx, LoaderTerminationSignal): # Shutdown signal
            ready_for_consumption_queue.put(sentinel_value)
            break

        # Initiate the non-blocking transfer
        source_td = cpu_pinned_pool[filled_idx]

        if source_td is None:
            # This block should not be hit with the fixes in place.
            print(f"[GPU Worker ERROR] Encountered None at index {filled_idx}. Shutting down.")
            ready_for_consumption_queue.put(sentinel_value)
            break

        gpu_pool[filled_idx] = source_td.to(device, non_blocking=True, num_threads=4)
        
        # For CUDA devices, use events for precise synchronization
        sync_event = None
        if device.type == 'cuda':
            sync_event = torch.cuda.Event()
            sync_event.record()

        # Signal that the batch is ready for consumption, providing the sync event
        ready_for_consumption_queue.put((filled_idx, sync_event))


# ==============================================================================
# ==                   IV. The Main Public Loader Class                       ==
# ==============================================================================

class SafePipelinedTensorDictLoader:
    """Manages a safe, multi-process data loading pipeline."""

    def __init__(self, raw_data_iterator, preprocessing_fn, device, buffer_count=3,
        num_cpu_workers=2, partial_batch_mode='send'):
        self.device = torch.device(device)
        self.buffer_count = buffer_count
        self.partial_batch_mode = partial_batch_mode
        
        # The internal data key is no longer necessary.
        # self._INTERNAL_DATA_KEY = "_payload"
        self.SENTINEL = LoaderTerminationSignal()

        ctx = mp.get_context("spawn")
        # Create a Manager to handle shared state between processes.
        self.manager = ctx.Manager()

        # Create communication channels (queues)
        self.raw_data_queue = ctx.Queue()
        self.free_indices_queue = ctx.Queue(self.buffer_count)
        self.cpu_filled_buffer_queue = ctx.Queue(self.buffer_count)
        self.ready_for_consumption_queue = ctx.Queue(self.buffer_count)
        self.batch_size_queue = ctx.Queue(1)

        # Pre-populate the free queue with all available buffer indices
        for i in range(self.buffer_count):
            self.free_indices_queue.put(i)

        # Pre-allocate buffer pools
        self._preallocate_pools()
        self.source_iterator = raw_data_iterator
        
        # --- Spawn and start autonomous worker processes ---
        self.cpu_workers = []
        for _ in range(num_cpu_workers):
            p = ctx.Process(
                target=_cpu_worker_process,
                args=(self.raw_data_queue, self.free_indices_queue, self.cpu_filled_buffer_queue,
                      self.batch_size_queue, self.cpu_pinned_pool, preprocessing_fn,
                      self.partial_batch_mode, self.SENTINEL
                      ),
                daemon=True
            )
            p.start()
            self.cpu_workers.append(p)
            
        self.gpu_worker = ctx.Process(
            target=_gpu_transfer_process,
            args=(self.cpu_filled_buffer_queue, self.ready_for_consumption_queue,
                  self.cpu_pinned_pool, self.gpu_pool, self.device, self.SENTINEL),
            daemon=True
        )
        self.gpu_worker.start()

        # Fill the raw data queue to kick off processing
        for item in raw_data_iterator:
            self.raw_data_queue.put(item)
        # Add termination sentinels for each CPU worker
        for _ in self.cpu_workers:
            self.raw_data_queue.put(self.SENTINEL)

    def _preallocate_pools(self):
        """Uses a Manager to create lists that can be shared across processes."""
        self.cpu_pinned_pool = self.manager.list([None] * self.buffer_count)
        self.gpu_pool = self.manager.list([None] * self.buffer_count)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # 1. Unblock any CPU workers waiting for raw data.
        #    Send one sentinel for each worker to ensure they all get the message.
        for _ in self.cpu_workers:
            try:
                self.raw_data_queue.put_nowait(self.SENTINEL)
            except Exception as E:
                print(E)
                pass
            try:
                self.free_indices_queue.put_nowait(self.SENTINEL)
            except Exception as E:
                print(E)
                pass

        # 2. Wait for the worker processes to finish.
        #    They should now exit their loops upon receiving a sentinel.
        #    A short timeout prevents hanging here if a worker is truly stuck.
        for p in self.cpu_workers:
            p.join(timeout=0.1)
        self.gpu_worker.join(timeout=0.1)

        print("[Loader] Closing queues...")
        for q in [self.raw_data_queue, self.cpu_filled_buffer_queue, self.ready_for_consumption_queue, self.free_indices_queue, self.batch_size_queue]:
            q.cancel_join_thread()#the python manual is wrong. you must use this in 100% of all programs!
            q.close()

        # 3. Wait for processes to exit cleanly, with a timeout.
        #    We join them without calling terminate() first.
        for p in self.cpu_workers:
            p.join(timeout=0.1)
        self.gpu_worker.join(timeout=0.1)

        # 4. As a final safety measure, forcefully terminate any process that
        #    defied the graceful shutdown sequence.
        for p in self.cpu_workers:
            if p.is_alive():
                print(f"[Loader] Forcing termination of stubborn worker: {p.pid}")
                p.terminate()
        if self.gpu_worker.is_alive():
            print(f"[Loader] Forcing termination of stubborn GPU worker: {self.gpu_worker.pid}")
            self.gpu_worker.terminate()
        
        # 5. Close the manager to release its resources.
        self.manager.shutdown()

        print("[Loader] Shutdown complete.")

    def set_batch_size(self, new_bs: int):
        """Commands the CPU workers to adopt a new batch size."""
        # Clear any old message first
        try: self.batch_size_queue.get_nowait()
        except Empty: pass
        self.batch_size_queue.put(new_bs)

    def get_next_batch(self):
        """
        REVISED: Requests the next available batch. The returned handle now
        provides the full TensorDict.
        """
        try:
            item = self.ready_for_consumption_queue.get_nowait()
            if isinstance(item, LoaderTerminationSignal):
                return item

            idx, sync_event = item
            if sync_event:
                sync_event.synchronize()
            
            # REVISED: Create the handle without the internal key.
            return BatchHandle(
                self,
                self.gpu_pool[idx],
                idx
            )

        except Empty:
            return LoaderStalledSignal()

    def _release_buffer_by_index(self, index: int):
        """Internal method called by BatchHandle to return a buffer to the pool."""
        self.free_indices_queue.put(index)

# ==============================================================================
# ==                 V. Standalone Test Harness (`main`)                      ==
# ==============================================================================

# 1. Define a simple test iterator
def beans_iterator(num_items: int):
    """Yields 'beans' with more vowels each time."""
    base = "beans"
    for i in range(1, num_items + 1):
        # Creates 'beans', 'beeaans', 'beeeaaans', etc.
        yield base[:1] + 'e'*i + 'a'*i + base[2:]
        #time.sleep(0.05) # Keep this low to ensure we're testing the pipeline, not the source

def nonfinite_beans_iterator():
    """
    Yields 'beans' with more vowels each time, indefinitely.
    This simulates an infinite or very large data stream.
    """
    base = "beans"
    # Use itertools.count to create an infinite generator
    for i in itertools.count(1):
        yield base[:1] + 'e'*i + 'a'*i + base[2:]
        # A tiny sleep can prevent this from completely hogging a CPU core if the
        # rest of the pipeline is ever blocked.

# 2. Define a simple test preprocessing function
def simple_text_preprocessor(text_string: str) -> TensorDict:
    """Converts a string to a tensor of ASCII values."""
    # Max length for padding
    max_len = 4096
    # Convert string to a list of ASCII values
    ascii_values = [ord(c) for c in text_string]
    # Pad or truncate to max_len
    padded_values = (ascii_values + [0] * max_len)[:max_len]
    return {"input_ids": torch.tensor(padded_values, dtype=torch.int32)}

if __name__ == "__main__":
    print("--- Testing SafePipelinedTensorDictLoader ---")

    # 3. Setup test parameters
    WORK_QUOTA = 1024
    INITIAL_BATCH_SIZE = 16
    END_BATCH_SIZE = 128
    DEVICE = 'cpu' # Use CPU for this test; logic is device-agnostic
    
    processed_items_count = 0
    # Create an iterator instance *before* passing it to the loader
    source_iterator = beans_iterator(2*WORK_QUOTA)

    # 4. Demonstrate the correct, safe usage pattern
    print(f"Attempting to process up to {WORK_QUOTA} items...")
    print(f"Batch size will lerp from {INITIAL_BATCH_SIZE} to {END_BATCH_SIZE}.")
    
    with SafePipelinedTensorDictLoader(
        raw_data_iterator=source_iterator,
        preprocessing_fn=simple_text_preprocessor,
        device=DEVICE,
        buffer_count=6,
        num_cpu_workers=2
    ) as loader:

        loader.set_batch_size(INITIAL_BATCH_SIZE)
        
        # STYLE GUIDE: To process a fixed number of items from a large or
        # infinite stream, use a `while True` loop and check the work
        # quota as the primary exit condition. This decouples the consumer's
        # needs from the producer's state.
        while True:
            t_start_wait = time.perf_counter()
            handle = loader.get_next_batch()

            # SECONDARY Condition: The data stream ended unexpectedly.
            if isinstance(handle, LoaderTerminationSignal):
                print("[Main Loop] Received termination signal. Exiting gracefully.")
                break
            if isinstance(handle, LoaderStalledSignal):
                # The pipeline is working but has no output right now.
                time.sleep(0.01)
                continue

            t_end_wait = time.perf_counter()
            wait_time_ms = (t_end_wait - t_start_wait) * 1000

            # --- Use the handle in a 'with' block for state management ---
            with handle as on_device_batch:
                
                payload_tensor = on_device_batch['input_ids']
                t_start_work = time.perf_counter()

                #print(f"how confusing... what kind of object did we pass ourselves...?\nrepr:{repr(on_device_batch)}")
                #print(f"we're about to call `on_device_batch.shape[0]`...\nwhat's on_device_batch.shape?\n{on_device_batch.shape}")
                batch_size = payload_tensor.shape[0]                
                # Simulate a computationally intensive task
                time.sleep(0.05)
                
                t_end_work = time.perf_counter()
                work_time_ms = (t_end_work - t_start_work) * 1000
                
                print(
                    f"[Main Loop] BS: {batch_size:<2} | "
                    f"Shape:{on_device_batch.shape} | "
                    f"Wait: {wait_time_ms:6.2f}ms | Work: {work_time_ms:6.2f}ms | "
                    f"Processed: {processed_items_count + batch_size}"
                )
                
                processed_items_count += batch_size

            # PRIMARY Condition: The work quota has been met or exceeded.
            if processed_items_count >= WORK_QUOTA:
                print(f"\n[Main Loop] Work quota of {WORK_QUOTA} met. Initiating shutdown.")
                break # Exit the loop. The `with` block will handle process termination.

            # --- Batch Scheduler Logic ---
            # Calculate the progress through the dataset
            progress = min(processed_items_count / WORK_QUOTA, 1.0)
            # Linearly interpolate the batch size
            new_bs = int(INITIAL_BATCH_SIZE + (END_BATCH_SIZE - INITIAL_BATCH_SIZE) * progress)
            new_bs = max(1, new_bs) # Ensure batch size is at least 1
            
            loader.set_batch_size(new_bs)

            #print(f"[Main Loop] Buffer released. Total items processed: {processed_items_count}")
            
    print("\n--- Test Complete ---")
    print(f"Final count of processed items: {processed_items_count}")