import random
import copy
from typing import List, Optional, Tuple, Any

from Evolutionary.base_unit import base_unit_factory
from Evolutionary.abstract_unit import Mat
from Evolutionary.train_unit import TrainUnit, run_train_session, simple_eval
from Evolutionary.mutations import simple_mutate, splice_units

class SimpleEvolutionaryAlg:
    def __init__(self, 
                 pop_size: int, 
                 num_data_obj: int,
                 trains_per_unit: int,
                 examples_per_train: int,
                 epochs_per_train: int,
                 mutation_probability: float,
                 top_q_percent: float = 0.7,
                 num_op_obj: Optional[int] = None,
                ):
        self.pop_size = pop_size
        self.num_data_obj = num_data_obj
        self.num_op_obj = num_op_obj
        self.trains_per_unit = trains_per_unit
        self.examples_per_train = examples_per_train
        self.epochs_per_train = epochs_per_train
        self.mutation_probability = mutation_probability
        self.top_q_percent = top_q_percent
        self.population: List[Tuple[Any, float]] = [] # List of (BaseUnit, performance) tuples
        # Create a TrainUnit instance to get num_examples and num_epochs for static calls
        self.train_config = TrainUnit(base_unit_factory(self.num_data_obj), self.examples_per_train, self.epochs_per_train)
        self._initialize_population()
        print(f"Algorithm initialized with parameters: pop_size={self.pop_size}, num_data_obj={self.num_data_obj}, num_op_obj={self.num_op_obj}, trains_per_unit={self.trains_per_unit}, examples_per_train={self.examples_per_train}, epochs_per_train={self.epochs_per_train}, mutation_probability={self.mutation_probability}, top_q_percent={self.top_q_percent}")

    def set_train_parameters(self,
                 trains_per_unit: Optional[int] = None,
                 examples_per_train: Optional[int] = None,
                 epochs_per_train: Optional[int] = None,
                 mutation_probability: Optional[float] = None,
                 top_q_percent: Optional[float] = None):
        print(f"Updating training parameters: trains_per_unit={trains_per_unit}, examples_per_train={examples_per_train}, epochs_per_train={epochs_per_train}, mutation_probability={mutation_probability}, top_q_percent={top_q_percent}")
        if trains_per_unit is not None:
            self.trains_per_unit = trains_per_unit
        if examples_per_train is not None:
            self.examples_per_train = examples_per_train
        if epochs_per_train is not None:
            self.epochs_per_train = epochs_per_train
        if mutation_probability is not None:
            self.mutation_probability = mutation_probability
        if top_q_percent is not None:
            self.top_q_percent = top_q_percent
        
        # Re-initialize train_config with potentially updated parameters
        self.train_config = TrainUnit(base_unit_factory(self.num_data_obj), self.examples_per_train, self.epochs_per_train)

    def _initialize_population(self):
        print("Initializing population...")
        for i in range(self.pop_size):
            base_unit_class = base_unit_factory(self.num_data_obj, num_mat=self.num_op_obj)
            unit_to_train = base_unit_class(Mat(random.randrange(base_unit_class.NUM_OPERATOR_OBJ)))
            # Use the static method directly
            performance = self.train_config.evaluate_training_performance(
                unit_to_train,
                self.trains_per_unit,
                simple_eval
            )
            self.population.append((unit_to_train, performance))
            if i % 100 == 0: print(f"\rUnit {i + 1} / {self.pop_size}", end="")
        print("\nPopulation initialized.")

    def run_evolutionary_step(self):
        # Sort population by performance (descending)
        self.population.sort(key=lambda x: x[1], reverse=True)

        # Identify top performers
        num_top_performers = int(self.pop_size * self.top_q_percent)
        top_performers = [unit for unit, _ in self.population[:num_top_performers]]

        # Calculate and display performance metrics
        performances = [perf for _, perf in self.population]
        avg_performance = sum(performances) / self.pop_size
        max_unit_info = self.population[0]
        print(f"Average performance: {avg_performance:.6f}, Max performance: {max_unit_info[1]:.6f}")

        p = self.train_config.evaluate_training_performance(
                max_unit_info[0],
                self.trains_per_unit * 5,
                simple_eval,
            )
        print(f"validating max. Perf: {p:.4f}")


        new_population: List[Tuple[Any, float]] = []
        num_new_units = self.pop_size - num_top_performers

        # Create new units through mutation and splicing
        for i in range(num_new_units):
            # Choose a method: mutate or splice
            if random.random() < 0.5 and len(top_performers) > 0: # 50% chance to mutate, if top_performers exist
                unit_to_mutate = random.choice(top_performers)
                new_unit = simple_mutate(unit_to_mutate, self.mutation_probability)
            elif len(top_performers) >= 2: # Otherwise, try to splice if at least two top performers exist
                unit1, unit2 = random.sample(top_performers, 2)
                new_unit = splice_units(unit1, unit2)
            else: # Fallback to random unit from current population if not enough top performers for splice
                BaseUnit = base_unit_factory(self.num_data_obj)
                random_w_idx = random.randrange(BaseUnit.NUM_OPERATOR_OBJ)
                new_unit = BaseUnit(w=Mat(random_w_idx)) # Create a completely new random unit

            # Evaluate new unit and add to new_population
            # We need to create a BaseUnit instance from the 'new_unit' (which is a BaseUnit class generated by the factory)
            # and then pass it to the TrainUnit constructor.
            # The simple_mutate and splice_units functions return instances of BaseUnit, so we can directly pass them to TrainUnit.
            # Use the static method directly
            performance = self.train_config.evaluate_training_performance(
                new_unit,
                self.trains_per_unit,
                simple_eval,
            )
            new_population.append((new_unit, performance))
            if i % 100 == 0: print(f"\rCreating new unit {i + 1} / {num_new_units}", end="")
        print("\nNew population created.")

        top_performers_reevaluated = []
        for unit in top_performers:
            reevaluated_performance = self.train_config.evaluate_training_performance(
                new_unit,
                self.trains_per_unit,
                simple_eval,
            )
            top_performers_reevaluated.append((unit, reevaluated_performance))


        # Combine top performers and new units to form the next generation
        self.population = top_performers_reevaluated + new_population 