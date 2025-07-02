from __future__ import annotations
import random
from typing import Tuple, Optional

from Evolutionary.base_unit import base_unit_factory
from Evolutionary.abstract_unit import AbstractBaseUnit, Vec, Mat

# Define BaseUnitVecType outside the class to allow for proper type hinting

class TrainUnit:
    def __init__(self,
                 base_unit_class: type[AbstractBaseUnit],
                 num_examples: int,
                 num_epochs: int
                 ) -> None:
        assert base_unit_class.NUM_DATA_OBJ >= num_examples, "Can't have more examples that data objects"
        self.BaseUnit = base_unit_class
        self.num_examples = num_examples
        self.num_epochs = num_epochs
        self.num_data_obj = self.BaseUnit.NUM_DATA_OBJ

    @staticmethod
    def _static_generate_toy_dataset(num_examples: int, num_data_obj: int):
        dataset = []
        for _ in range(num_examples):
            input_vec = Vec(random.randrange(num_data_obj))
            output_vec = Vec(random.randrange(num_data_obj))
            dataset.append((input_vec, output_vec))
        return dataset

    @staticmethod
    def static_train(unit: AbstractBaseUnit, num_examples: int, num_epochs: int, num_data_obj: int) -> list[float]:
        toy_dataset = TrainUnit._static_generate_toy_dataset(num_examples, num_data_obj)
        accuracies = []
        for epoch in range(num_epochs):
            correct_predictions = 0
            for input_vec, target_vec in toy_dataset:
                # Forward pass
                output_vec = unit.forward(input_vec)

                # Calculate accuracy
                if output_vec.v_idx == target_vec.v_idx:
                    correct_predictions += 1

                back_info = unit.find_back_info(target_vec) # Loss analogue
                unit.update(back_info)
            
            accuracy = correct_predictions / num_examples
            accuracies.append(accuracy)
            #print(f"Epoch {epoch + 1}/{self.num_epochs}, Accuracy: {accuracy:.4f}")
        return accuracies

    def train(self, unit: AbstractBaseUnit) -> list[float]:
        # The original train method called init_data to get self.unit and self.toy_dataset.
        # Now, self.unit and self.toy_dataset are initialized in __init__ via init_data.
        # So, we can directly pass self.unit to the static_train method.
        return TrainUnit.static_train(unit, self.num_examples, self.num_epochs, self.num_data_obj)

    @staticmethod
    def static_evaluate_training_performance(unit: AbstractBaseUnit, trains_per_unit: int, evaluation_function, num_examples: int, num_epochs: int, num_data_obj: int) -> float:
        eval_score = 0
        for i in range(trains_per_unit):
            train_results = TrainUnit.static_train(unit, num_examples, num_epochs, num_data_obj)
            eval_score += evaluation_function(train_results)
        return eval_score / trains_per_unit

    def evaluate_training_performance(self, unit: AbstractBaseUnit, trains_per_unit: int, evaluation_function) -> float:
        # The original evaluate_training_performance called self.train(), which initialized self.unit.
        # Now, self.unit is available from __init__. We pass it directly to the static method.
        return TrainUnit.static_evaluate_training_performance(unit, trains_per_unit, evaluation_function, self.num_examples, self.num_epochs, self.num_data_obj)

def simple_eval(results):
    weights = list(range(1, len(results)))
    return sum([r * w for r, w in zip(results[1:], weights)]) / max(1, sum(weights)) # Weighted average - bias towards later steps

def run_train_session(
        num_units_to_try = 100,
        trains_per_unit = 100,
        num_data_obj = 10,
    ):
    results = [] # List of "success rates" for each unit. I.e. what proportion of the times it was able to train to the target accuracy
    for i in range(num_units_to_try):
        base_unit_class = base_unit_factory(num_data_obj)
        initial_unit = base_unit_class(Mat(random.randrange(base_unit_class.NUM_OPERATOR_OBJ)))
        trainer_instance = TrainUnit(base_unit_class, 1, 10) # num_examples=1, num_epochs=10
        if i % 100 == 0: print(f"\rEvaluating unit {i + 1} / {num_units_to_try}", end="")
        
        train_success_rate = TrainUnit.static_evaluate_training_performance(
            initial_unit,
            trains_per_unit,
            simple_eval,
            trainer_instance.num_examples,
            trainer_instance.num_epochs,
            trainer_instance.num_data_obj
        )
        results.append((initial_unit, train_success_rate)) # Store the unit and its success rate
    
    success_rates = [r[1] for r in results]
    print(f"average success rate: {sum(success_rates)/num_units_to_try}")
    print(f"max success rate: {max(success_rates)}")
    return results