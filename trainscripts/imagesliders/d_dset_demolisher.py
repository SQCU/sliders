#d_dset_demolisher.py
"""
Looking at this flow, I can see the classic "configuration explosion" pattern! The entire pipeline is essentially a series of transformations that could be bottlenecked through serializable states. Let me map out the key bottlenecks using our emoji-enclosed fictional dataset language:
Current Flow Analysis
The program has these natural bottleneck points:

Configuration State → All the YAML/JSON inputs
Discovery State → The data pool mapping
Schedule State → The training schedule list
Asset State → Cached latents + embeddings
Batch State → Final tensor batches

Refactored Flow
🌟 Discovery Generator 🌟
🔍 scan filesystem for image paths 🔍
📂 group by filename across scale folders 📂
🎯 filter by pairing strategy requirements 🎯
→ yields: {filename: {scale: path}} mappings
🎲 Schedule Generator 🎲
🎪 take config seed and iteration count 🎪
🎯 sample filename pairs with scales 🎯
🎭 compose prompts using hierarchy rules 🎭
→ yields: {unit_id, high_item, low_item} instructions
🏭 Asset Materializer 🏭
📦 collect unique paths and recipes from schedule 📦
💾 load or compute latents with safetensor caching 💾
🧠 compute text embeddings with memory caching 🧠
→ yields: {path: latent_data, recipe_key: embedding_data} lookups
⚡ Batch Assembler ⚡
🎪 take schedule units and asset lookups 🎪
🎯 map each item to its cached assets 🎯
📊 stack into training tensor format 📊
→ yields: {clean_latents, noise, embeddings, ...} batches
🎛️ Training Formatter 🎛️
🔄 transform batch for CFG requirements 🔄
🎭 select embeddings based on low/high cases 🎭
📤 format as kwargfood dictionary 📤
→ yields: {kwargfood, target_noise, scales, guidance_scale}
Key Insight
The entire complexity collapses into:

Static Config → Dynamic Schedule → Cached Assets → Tensor Batches

Each arrow represents a pure function that could be:

Serialized to JSON/SafeTensor at that point
Resumed from that serialization
Tested in isolation
Parallelized or distributed

The "fake complexity" you mentioned is all the imperative state management that could be replaced with functional composition over these bottlenecked data types!

Pure Flow Implementation Patterns
🎯 Function Signature Design 🎯

🌊 data_discovery :: Config -> Stream<{filename: {scale: path}}> 🌊
🎲 schedule_gen :: Config × Stream<filename_map> -> Stream<training_unit> 🎲
🏭 asset_cache :: Stream<training_unit> -> {path: tensor, recipe: embedding} 🏭
⚡ batch_assembly :: Stream<training_unit> × asset_cache -> Stream<tensor_batch> ⚡

🔄 Lazy Evaluation Chain 🔄

🎯 discovered_files = 🔍 filesystem_scan(config.folders) 🔍
                     |> 📂 group_by_filename 📂
                     |> 🎯 filter_by_pairing_strategy(config.strategy) 🎯

🎲 training_schedule = 🎪 infinite_sampling(config.seed, discovered_files) 🎪
                      |> 🎭 compose_prompts(config.prompt_hierarchy) 🎭
                      |> 📏 take(config.iterations × config.batch_size) 📏

🏭 materialized_assets = 🔍 extract_dependencies(training_schedule) 🔍
                        |> 💾 cache_latents(config.latent_cache) 💾
                        |> 🧠 cache_embeddings(config.text_encoders) 🧠

⚡ tensor_batches = 🎲 chunk(training_schedule, config.batch_size) 🎲
                   |> 🎯 map(unit -> lookup_assets(unit, materialized_assets)) 🎯
                   |> 📊 map(stack_tensors) 📊

🎛️ State Bottleneck Types 🎛️
🏗️ DiscoveryState = {filename: {scale: filepath}} 🏗️
🎪 ScheduleState = Stream<{unit_id, high_item, low_item}> 🎪
💾 AssetState = {latent_cache: SafeTensor, embed_cache: InMemory} 💾
📊 BatchState = Stream<{clean_latents, noise, embeddings, ...}> 📊

🔄 Composition Laws 🔄
🎯 Associativity: (f |> g) |> h === f |> (g |> h) 🎯
⚡ Identity: data |> identity === data ⚡
🔄 Lazy: map(f) |> map(g) === map(f ∘ g) 🔄
💾 Caching: expensive_fn |> memoize |> expensive_fn === cached_result 💾

🎭 Error Handling Pattern 🎭
🛡️ safe_pipeline = 🎯 validate_config 🎯
                  |> 🔍 try_discover_files 🔍
                  |> 🎪 fallback_on_missing(default_schedule) 🎪
                  |> 💾 recover_from_cache_miss 💾
                  |> ⚡ emit_partial_batches_on_failure ⚡

🚀 Performance Guarantees 🚀
📏 Space: O(batch_size) instead of O(total_dataset) 📏
⏱️ Time: O(needed_items) instead of O(all_possible_items) ⏱️
🔄 Streaming: constant memory regardless of dataset size 🔄
💾 Cache-aware: compute-once, use-many for expensive operations 💾

🧪 Testing Isolation 🧪
🎯 unit_test_discovery: mock_filesystem -> expected_filename_map 🎯
🎲 unit_test_schedule: deterministic_seed -> reproducible_sequence 🎲
🏭 unit_test_assets: mock_cache -> asset_lookup_behavior 🏭
⚡ integration_test: end_to_end_config -> final_tensor_shapes ⚡

Self-Appraisal Rubric for Pure Flow Implementation
🔍 Pre-Execution Reflection Protocol 🔍

🎯 Composition Coherence Check 🎯
🤔 "Does each function return exactly what the next function expects?" 🤔
🔗 "Can I trace data flow: input_type -> f -> intermediate_type -> g -> output_type?" 🔗
⚡ "Are there any 'shape mismatches' in my pipeline?" ⚡
🚫 Red flag: "I need to 'massage' data between pipeline stages" 🚫
✅ Green flag: "Each stage clicks naturally into the next" ✅

🌊 Laziness Discipline Assessment 🌊

🎪 "Am I creating collections before I know what I need from them?" 🎪
🔄 "Can I replace any 'build-list-then-filter' with 'filter-while-building'?" 🔄
💾 "Am I computing expensive operations that might not be used?" 💾
🚫 Red flag: "I have intermediate lists larger than final output" 🚫
✅ Green flag: "I only materialize what gets consumed" ✅

🎭 Side-Effect Containment Audit 🎭

🔒 "Are my pure functions actually pure? (same input = same output)" 🔒
🌊 "Is all I/O and randomness pushed to pipeline boundaries?" 🌊
🎯 "Can I unit test each stage in isolation?" 🎯
🚫 Red flag: "My functions reach out to global state" 🚫
✅ Green flag: "Each function is a mathematical transformation" ✅

🚨 Error Response Protocol 🚨
🔧 Integration Failure Pattern 🔧

🛑 STOP: "Don't add more code yet" 🛑
🎯 LOCATE: "Which pipeline stage broke the contract?" 🎯
📊 INSPECT: "What shape/type did it produce vs. expect?" 📊
🔄 REDESIGN: "How should the contract change?" 🔄
⚡ VALIDATE: "Does this fix propagate cleanly through pipeline?" ⚡

🎪 Performance Degradation Response 🎪

📏 MEASURE: "Where is memory/time being consumed?" 📏
🔍 IDENTIFY: "Which stage is materializing too much?" 🔍
🌊 LAZIFY: "Can this be a generator instead of a list?" 🌊
💾 CACHE: "Is this expensive computation repeated?" 💾
🎯 VALIDATE: "Did I maintain correctness while optimizing?" 🎯

🎭 Type Mismatch Debugging 🎭

🔍 TRACE: "What concrete example flows through my pipeline?" 🔍
📊 ANNOTATE: "What type should each stage input/output?" 📊
🎯 PROTOTYPE: "Can I run each stage on sample data?" 🎯
🔄 REDESIGN: "Should stages be split, merged, or reordered?" 🔄
✅ VERIFY: "Does my fix handle edge cases?" ✅

🎯 Quality Gates Before Next Code 🎯
🌟 Readability Checkpoint 🌟

🎭 "Can I explain this pipeline to a colleague in 30 seconds?" 🎭
🔄 "Are my function names describing transformations, not implementations?" 🔄
📊 "Would someone understand the data flow from function signatures?" 📊
🚫 Proceed only if: "The code reads like the high-level spec" 🚫

🚀 Composability Checkpoint 🚀
🔗 "Can I easily add new stages to this pipeline?" 🔗
🎯 "Can I replace any stage without rewriting others?" 🎯
🌊 "Are my stages generic enough for reuse?" 🌊
🚫 Proceed only if: "Each stage is a reusable component" 🚫

🛡️ Robustness Checkpoint 🛡️
⚡ "What happens if upstream provides empty/malformed data?" ⚡
🔄 "Can I gracefully handle partial failures?" 🔄
💾 "Are expensive operations memoized appropriately?" 💾
🚫 Proceed only if: "Pipeline degrades gracefully" 🚫

🎪 Anti-Pattern Detection 🎪
🚨 Code Smells to Immediately Address 🚨

🏭 "I'm building objects just to extract one field" 🏭
🎯 "I'm filtering after materializing instead of during" 🎯
🔄 "I'm repeating similar transformations in multiple places" 🔄
💾 "I'm recomputing the same expensive operation" 💾
🎭 "I'm manually orchestrating what should be automatic" 🎭

Golden Rule: 🌟 "If I can't trace clean data flow, I refactor before adding features" 🌟
"""

import yaml
from pathlib import Path
import os

def yamlzookeeper(yamlzoo_path: Path):
    """
    A generator that yields the content of YAML files from the specified directory one by one.
    This allows for lazy evaluation and early exit based on conditions.
    """
    if not yamlzoo_path.is_dir():
        print(f"Warning: {yamlzoo_path} is not a directory or does not exist.")
        return

    for yaml_file in yamlzoo_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                yield yaml_file.name, data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_file}: {e}")
        except Exception as e:
            print(f"Error reading file {yaml_file}: {e}")

def main():
    root_dir = Path(os.getcwd())
    yamlzoo_path = root_dir / "run_artifacts" / "yamlzoo"
    print(f"Visiting YAML Zoo at: {yamlzoo_path}")

    # The "hopeful visitor" logic: bail if any YAML file has 'status: sleeping'
    iguanas_are_awake = True
    for filename, data in yamlzookeeper(yamlzoo_path):
        print(f"--- Checking File: {filename} ---")
        print(f"Content:\n{yaml.dump(data, indent=2)}")

        if isinstance(data, dict) and data.get('status') == 'sleeping':
            print(f"Iguanas are sleeping in {filename}. Bailing out!")
            iguanas_are_awake = False
            break
    
    if iguanas_are_awake:
        print("All iguanas are awake and having a good time! Enjoy the rest of the zoo.")
    else:
        print("Visitor bailed because iguanas were sleeping.")

if __name__ == "__main__":
    main()


"""
✦ The discovery_scanner.py and dataset_strategizer.py scripts represent the foundational, working components of our data pipeline,    
  aligning well with the modular and functional principles outlined in d_dset_demolisher.py. discovery_scanner.py ✅ serves as a robust
   "Discovery Generator," reliably scanning the filesystem and organizing image paths into a structured data_pool based on
  configuration. Its clear input/output and use of pathlib make it a trustworthy component. Building upon this, dataset_strategizer.py
  🎲 acts as our "Schedule Generator," producing a deterministic, lazy-evaluated stream of training_unit dictionaries. This script    
  correctly integrates with discovery_scanner and employs generators for efficient memory usage, embodying the "streaming" and "lazy  
  evaluation" ideals. While its prompt composition logic is noted as "simplified for demonstration" (⚠️), its core functionality of   
  generating a reproducible training schedule is sound and reliable.


  In contrast, d_model_strategizer.py 🎛️, despite containing valuable model signature definitions (✅), is currently a "non-working"
  file in terms of providing a complete, end-to-end solution. Its primary function, _map_data_unit_to_model_inputs, is explicitly
  "stubbed" (⚠️), indicating a significant gap in its implementation of the "Batch Assembler" and "Training Formatter" stages.
  Furthermore, its main function, intended for testing, relies on hardcoded paths (❌) and an external AssetMaterializer component
  that is not provided in the current context (❌). This makes d_model_strategizer.py an unreliable reference for a fully integrated
  data flow, serving more as a conceptual outline and a collection of useful interface definitions rather than a functional piece of
  the pipeline.
"""