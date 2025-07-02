from tkinter import TOP
from Evolutionary.base_unit import base_unit_factory
from Evolutionary.train_unit import TrainUnit, run_train_session, simple_eval
from Evolutionary.evolutionary_algorithms import SimpleEvolutionaryAlg

def base_experiment():
    UNIT_POP_SIZE = 1000
    NUM_DATA_OBJ = 5
    TRAINS_PER_UNIT = 3000
    EPOCHS_PER_TRAIN = 3
    EXAMPLES_PER_TRAIN = 1
    NUM_GENERATIONS = 100
    MUTATION_PROB = 0.3
    TOP_Q = 0.1

    evolutionary_alg = SimpleEvolutionaryAlg(
        pop_size=UNIT_POP_SIZE,
        num_data_obj=NUM_DATA_OBJ,
        num_op_obj=NUM_DATA_OBJ * NUM_DATA_OBJ,
        epochs_per_train=EPOCHS_PER_TRAIN,
        examples_per_train=EXAMPLES_PER_TRAIN,
        trains_per_unit=TRAINS_PER_UNIT,
        mutation_probability=MUTATION_PROB,
        top_q_percent=TOP_Q
    )

    print("\nRunning evolutionary algorithm...")
    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{NUM_GENERATIONS} ---")
        evolutionary_alg.run_evolutionary_step()

    print("\nEvolutionary algorithm finished.")

def progressive_experiment():
    UNIT_POP_SIZE = 1000
    NUM_DATA_OBJ = 5
    TRAINS_PER_UNIT = 300
    EXAMPLES_PER_TRAIN = 1
    NUM_GENERATIONS = 100
    MUTATION_PROB = 0.1
    TOP_Q = 0.6

    evolutionary_alg = SimpleEvolutionaryAlg(
        pop_size=UNIT_POP_SIZE,
        num_data_obj=NUM_DATA_OBJ,
        num_op_obj=NUM_DATA_OBJ * NUM_DATA_OBJ,
        epochs_per_train=0,
        examples_per_train=EXAMPLES_PER_TRAIN,
        trains_per_unit=TRAINS_PER_UNIT,
        mutation_probability=MUTATION_PROB,
        top_q_percent=TOP_Q
    )

    epochs_per_train_schedule = [3, 5, 10, 15]

    print(f"\nRunning progressive evolutionary algorithm... epochs_per_train_schedule = {epochs_per_train_schedule}")
    for sched_idx, num_e in enumerate(epochs_per_train_schedule):
        evolutionary_alg.set_train_parameters(epochs_per_train=num_e)
        for generation in range(NUM_GENERATIONS):
            print(f"\n--- Schedule {sched_idx + 1}/{len(epochs_per_train_schedule)} Generation {generation + 1}/{NUM_GENERATIONS} ---")
            evolutionary_alg.run_evolutionary_step()
    print("\nEvolutionary algorithm finished.")

progressive_experiment()